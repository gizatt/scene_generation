from __future__ import print_function
from functools import partial
import time

import matplotlib.pyplot as plt
import numpy as np

import pydrake
import torch
import torch.distributions.constraints as constraints
import pyro
import pyro.distributions as dist
from pyro import poutine

from scene_generation.data.dataset_utils import DrawYamlEnvironmentPlanar

def get_planar_pose_w2_torch(p_w1, p_12):
    ''' p_w1: Pose 1 in world frame
        p_12: Pose 2 in Pose 1's frame
        Returns: Pose 2 in world frame. '''
    p_out = torch.zeros(3, dtype=p_w1.dtype)
    p_out[:] = p_w1[:]
    # Rotations just add
    p_out[2] += p_12[2]
    # Rotate p_12 by p_w1's rotation
    r = p_w1[2]
    p_out[0] += p_12[0]*torch.cos(r) - p_12[1]*torch.sin(r)
    p_out[1] += p_12[0]*torch.sin(r) + p_12[1]*torch.cos(r)
    return p_out

class OrNode(object):
    def __init__(self, production_rules, production_weights):
        if len(production_weights) != len(production_rules):
            raise ValueError("# of production rules and weights must match.")
        if len(production_weights) == 0:
            raise ValueError("Must have nonzero number of production rules.")
        self.production_rules = production_rules
        self.production_dist = dist.Categorical(production_weights)

    def sample_production_rule(self, site_prefix, obs=None):
        sampled_rule = pyro.sample(site_prefix + "_or_sample", self.production_dist, obs=obs)
        return self.production_rules[sampled_rule](site_prefix)


class AndNode(object):
    def __init__(self, production_rules):
        if len(production_rules) == 0:
            raise ValueError("Must have nonzero # of production rules.")
        self.production_rules = production_rules
        
    def sample_production_rule(self, site_prefix):
        return [x(site_prefix + "_rule_%04d" % i) for i, x in enumerate(self.production_rules)]


class ExhaustiveSetNode(object):
    def __init__(self, site_prefix, production_rules,
                 production_weights_hints = {},
                 remaining_weight = 1.):
        ''' Make a categorical distribution over
           every possible combination of production rules
           that could be active.

           Hints can be supplied in the form of a dictionary
           of (int tuple) : (float weight) pairs, and a float
           indicating the weight to distribute to the remaining
           pairs. These floats all indicate relative occurance
           weights. '''

        # Build the initial weights, taking the suggestion
        # weights into account.
        assert(remaining_weight >= 0.)
        num_combinations = 2**len(production_rules)
        init_weights = torch.ones(num_combinations) * remaining_weight
        for hint in production_weights_hints.keys():
            val = production_weights_hints[hint]
            assert(val >= 0.)
            combination_index = 0
            for index in hint:
                assert(isinstance(index, int) and index >= 0 and
                       index < len(production_rules))
                combination_index += 2**index
            init_weights[combination_index] = val
        init_weights /= torch.sum(init_weights)

        self.exhaustive_set_weights = pyro.param(
            site_prefix + "_exhaustive_set_weights",
            init_weights,
            constraint=constraints.simplex)

        self.production_dist = dist.Categorical(self.exhaustive_set_weights)
        self.site_prefix = ""
        self.production_rules = production_rules

    def sample_production_rule(self, site_prefix):
        selected_rules = pyro.sample(
            site_prefix + "_exhaustive_set_sample",
            self.production_dist)
        # Select out the appropriate rules
        output = []
        for k, rule in enumerate(self.production_rules):
            if (selected_rules >> k) & 1:
                output += rule(site_prefix + "_rule_%04d" % k)
        return output


class PlaceSetting(ExhaustiveSetNode):
    def __init__(self, pose):
        self.pose = pose
        # Represent each object's relative position to the
        # place setting origin with a diagonal Normal distribution.
        # So some objects will show up multiple
        # times here (left/right variants) where we know ahead of time
        # that they'll have multiple modes.
        # TODO(gizatt) GMMs? Guide will be even harder to write.
        self.object_types_by_name = {
            "plate": Plate,
            "cup": Cup,
            "left_fork": Fork,
            "right_fork": Fork
        }
        param_guesses_by_name = {
            "plate": ([0., 0.1, 0.], [0.01, 0.01, 1.]),
            "cup": ([0., 0.1 + 0.15, 0.], [0.05, 0.01, 1.]),
            "right_fork": ([0.15, 0.1, 0.], [0.01, 0.01, 0.01]),
            "left_fork": ([-0.15, 0.1, 0.], [0.01, 0.01, 0.01]),
        }
        self.distributions_by_name = {}
        self.params_by_name = {}
        production_rules = []
        name_to_ind = {}
        for k, object_name in enumerate(self.object_types_by_name.keys()):
            mean_init, var_init = param_guesses_by_name[object_name]
            mean = pyro.param("place_setting_%s_mean" % object_name,
                              torch.tensor(mean_init))
            var = pyro.param("place_setting_%s_var" % object_name,
                              torch.tensor(var_init),
                              constraint=constraints.positive)
            self.distributions_by_name[object_name] = dist.Normal(
                mean, var)
            self.params_by_name[object_name] = (mean, var)
            production_rules.append(partial(
                partial(self._produce_object, object_name=object_name)))
            name_to_ind[object_name] = k

        # Weight the "correct" rules very heavily
        production_weights_hints = {
            (name_to_ind["plate"], name_to_ind["cup"], name_to_ind["left_fork"]): 2.,
            (name_to_ind["plate"], name_to_ind["cup"], name_to_ind["right_fork"]): 2.,
            (name_to_ind["plate"], name_to_ind["right_fork"]): 2.,
            (name_to_ind["plate"], name_to_ind["left_fork"]): 2.,
            (name_to_ind["cup"],): 0.5,
            (name_to_ind["plate"],): 1.,
        }
        ExhaustiveSetNode.__init__(self, "place_setting", production_rules,
                                   production_weights_hints,
                                   remaining_weight=0.)

    def _produce_object(self, site_prefix, object_name):
        rel_pose = pyro.sample("%s_%s_pose" % (site_prefix, object_name),
                           self.distributions_by_name[object_name])
        abs_pose = get_planar_pose_w2_torch(self.pose, rel_pose)
        return [self.object_types_by_name[object_name](abs_pose)]


class TableNode(ExhaustiveSetNode):
    def __init__(self, num_place_setting_locations=6):
        self.pose = torch.tensor([0.5, 0.5, 0.])
        self.table_radius = pyro.param("table_radius", torch.tensor(0.4),
                                       constraint=constraints.positive)
        # Set-valued: a plate may appear at each location.
        production_rules = []
        for k in range(num_place_setting_locations):
            root_pose = torch.zeros(3)
            root_pose[2] = (k / float(num_place_setting_locations))*np.pi*2.
            root_pose[0] = self.table_radius * torch.cos(root_pose[2])
            root_pose[1] = self.table_radius * torch.sin(root_pose[2])
            # TODO(gizatt) Root pose could be a parameter. This turns this
            # into a GMM.
            production_rules.append(partial(
                self._produce_place_setting_at_pose,
                root_pose=root_pose))

        # Relative offset from root pose is drawn from a diagonal
        # Normal. It's rotated into the root pose frame at sample time.
        mean = pyro.param("table_place_setting_mean",
                          torch.tensor([0.0, 0., np.pi/2.]))
        var = pyro.param("table_place_setting_var",
                          torch.tensor([0.01, 0.01, 0.01]),
                          constraint=constraints.positive)
        self.place_setting_offset_dist = dist.Normal(mean, var)
        ExhaustiveSetNode.__init__(self, "table_node", production_rules)

    def _produce_place_setting_at_pose(self, site_prefix, root_pose):
        rel_offset = pyro.sample("%s_place_setting_offset" % site_prefix,
                             self.place_setting_offset_dist)
        # Rotate offset
        abs_offset = get_planar_pose_w2_torch(self.pose, rel_offset)
        return [PlaceSetting(pose=root_pose + abs_offset)]

    def _produce_two_place_settings(self, site_prefix):
        return [PlaceSetting(pose=torch.tensor([0., 0., 0.])),
                PlaceSetting(pose=torch.tensor([1., 0., 0.]))]

class TerminalNode(object):
    def __init__(self):
        Node.__init__(self, [], [])

class Plate(TerminalNode):
    def __init__(self, pose):
        self.pose = pose

    def generate_yaml(self):
        return {
            "class": "plate",
            "color": None,
            "img_path": "table_setting_assets/plate_red.png",
            "params": [0.2],
            "params_names": ["radius"],
            "pose": self.pose.tolist()
        }

class Cup(TerminalNode):
    def __init__(self, pose):
        self.pose = pose

    def generate_yaml(self):
        return {
            "class": "cup",
            "color": None,
            "img_path": "table_setting_assets/cup_water.png",
            "params": [0.05],
            "params_names": ["radius"],
            "pose": self.pose.tolist()
        }


class Fork(TerminalNode):
    def __init__(self, pose):
        self.pose = pose
    
    def generate_yaml(self):
        return {
            "class": "fork",
            "color": None,
            "img_path": "table_setting_assets/fork.png",
            "params": [0.02, 0.14],
            "params_names": ["width", "height"],
            "pose": self.pose.tolist()
        }

class ProbabilisticSceneGrammarModel():
    def __init__(self):
        pass

    def model(self, data=None):
        nodes = [TableNode()]
        all_terminal_nodes = []
        num_productions = 0
        iter_k = 0
        while len(nodes) > 0:
            node = nodes.pop(0)
            if isinstance(node, TerminalNode):
                # Instantiate
                print("Instantiating terminal node: ", node, " at pose ", node.pose)
                all_terminal_nodes.append(node)
            else:
                print("Expanding non-terminal node ", node)
                # Expand by picking a production rule
                new_node_classes = node.sample_production_rule("production_%04d" % num_productions)
                [nodes.append(x) for x in new_node_classes]
                num_productions += 1
            iter_k += 1

        return all_terminal_nodes

    def guide(self, data):
        pass


def convert_list_of_terminal_nodes_to_yaml_env(node_list):
    env = {"n_objects": len(node_list)}
    for k, node in enumerate(node_list):
        env["obj_%04d" % k] = node.generate_yaml()
    return env

if __name__ == "__main__":
    pyro.enable_validation(True)

    model = ProbabilisticSceneGrammarModel()

    plt.figure()
    for k in range(10):
        start = time.time()
        pyro.clear_param_store()
        trace = poutine.trace(model.model).get_trace()
        terminal_node_list = trace.nodes["_RETURN"]["value"]
        end = time.time()

        print("\n\n")
        print("Generated data in %f seconds." % (end - start))
        print("Full trace values:" )
        for node_name in trace.nodes.keys():
            if node_name in ["_INPUT", "_RETURN"]:
                continue
            print(node_name, ": ", trace.nodes[node_name]["value"].detach().numpy())

        yaml_env = convert_list_of_terminal_nodes_to_yaml_env(terminal_node_list)

        plt.gca().clear()
        DrawYamlEnvironmentPlanar(yaml_env, base_environment_type="table_setting", ax=plt.gca())
        plt.pause(1.0)
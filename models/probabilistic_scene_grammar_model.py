from __future__ import print_function
from functools import partial
import time

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.distributions.constraints as constraints
import pyro
import pyro.distributions as dist
from pyro import poutine

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
        return [x(site_prefix) for x in self.production_rules]


class ExhaustiveSetNode(object):
    def __init__(self, site_prefix, production_rules):
        # Make a categorical distribution over
        # every possible combination of production rules
        # that could be active.
        num_combinations = 2**len(production_rules)
        self.exhaustive_set_weights = pyro.param(
            site_prefix + "_exhaustive_set_weights",
            torch.ones(num_combinations) / num_combinations,
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
                output += rule(site_prefix)
        return output


class PlaceSetting(ExhaustiveSetNode):
    def __init__(self, site_prefix, pose):
        self.pose = pose
        # Represent each object's relative position to the
        # place setting origin with a diagonal Normal distribution.
        # So some objects will show up multiple
        # times here (left/right variants) where we know ahead of time
        # that they'll have multiple modes.
        # TODO(gizatt) GMMs? Guide will be even harder to write.
        self.object_types_by_name = {
            "dish": Dish,
            "cup": Cup,
            "left_fork": Fork,
            "right_fork": Fork
        }
        self.distributions_by_name = {}
        self.params_by_name = {}
        all_production_rules = []
        for object_name in self.object_types_by_name.keys():
            mean = pyro.param("%s_%s_mean" % (site_prefix, object_name),
                              torch.tensor([0., 0., 0.]))
            var = pyro.param("%s_%s_var" % (site_prefix, object_name),
                              torch.tensor([0.1, 0.1, 0.1]),
                              constraint=constraints.positive)
            self.distributions_by_name[object_name] = dist.Normal(
                mean, var)
            self.params_by_name[object_name] = (mean, var)
            all_production_rules.append(partial(
                partial(self._produce_object, object_name=object_name)))
        ExhaustiveSetNode.__init__(self, "place_setting", all_production_rules)

    def _produce_object(self, site_prefix, object_name):
        rel_pose = pyro.sample("%s_%s_pose" % (site_prefix, object_name),
                           self.distributions_by_name[object_name])
        abs_pose = get_planar_pose_w2_torch(self.pose, rel_pose)
        return [self.object_types_by_name[object_name](abs_pose)]


class RootNode(OrNode):
    def __init__(self):
        production_rules = [ self._produce_one_place_setting,
                             self._produce_two_place_settings]
        production_weights = torch.tensor([ 0.75, 0.25 ])
        OrNode.__init__(self, production_rules, production_weights)

    def _produce_one_place_setting(self, site_prefix):
        return [PlaceSetting("%s_place_setting_1" % site_prefix, pose=torch.tensor([0., 0., 0.]))]

    def _produce_two_place_settings(self, site_prefix):
        return [PlaceSetting("%s_place_setting_1" % site_prefix, pose=torch.tensor([0., 0., 0.])),
                PlaceSetting("%s_place_setting_1" % site_prefix, pose=torch.tensor([1., 0., 0.]))]

class TerminalNode(object):
    def __init__(self):
        Node.__init__(self, [], [])

class Dish(TerminalNode):
    def __init__(self, pose):
        self.pose = pose

class Cup(TerminalNode):
    def __init__(self, pose):
        self.pose = pose

class Fork(TerminalNode):
    def __init__(self, pose):
        self.pose = pose

class ProbabilisticSceneGrammarModel():
    def __init__(self):
        pass

    def model(self, data=None):
        nodes = [RootNode()]
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
                new_node_classes = node.sample_production_rule("rule_%04d" % num_productions)
                [nodes.append(x) for x in new_node_classes]
                num_productions += 1
            iter_k += 1

        return all_terminal_nodes

    def guide(self, data):
        pass


if __name__ == "__main__":
    pyro.enable_validation(True)

    model = ProbabilisticSceneGrammarModel()

    start = time.time()
    pyro.clear_param_store()
    out = model.model()
    end = time.time()

    print("Generated data in %f seconds:" % (end - start))
    print(out)

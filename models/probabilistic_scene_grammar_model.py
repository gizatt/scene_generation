from __future__ import print_function
from functools import partial
import time
import random
import sys
import yaml

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import pydrake
import torch
import torch.distributions.constraints as constraints
import pyro
import pyro.distributions as dist
from pyro import poutine

from scene_generation.data.dataset_utils import (
    DrawYamlEnvironmentPlanar, ProjectEnvironmentToFeasibility)


def chain_pose_transforms(p_w1, p_12):
    ''' p_w1: xytheta Pose 1 in world frame
        p_12: xytheta Pose 2 in Pose 1's frame
        Returns: xytheta Pose 2 in world frame. '''
    p_out = torch.zeros(3, dtype=p_w1.dtype)
    p_out[:] = p_w1[:]
    # Rotations just add
    p_out[2] += p_12[2]
    # Rotate p_12 by p_w1's rotation
    r = p_w1[2]
    p_out[0] += p_12[0]*torch.cos(r) - p_12[1]*torch.sin(r)
    p_out[1] += p_12[0]*torch.sin(r) + p_12[1]*torch.cos(r)
    return p_out

def invert_pose(pose):
    # TF^-1 = [R^t  -R.' T]
    p_out = torch.zeros(3, dtype=pose.dtype)
    r = pose[2]
    p_out[0] -= pose[0]*torch.cos(-r) - pose[1]*torch.sin(-r)
    p_out[1] -= pose[0]*torch.sin(-r) + pose[1]*torch.cos(-r)
    p_out[2] = -r
    return p_out


class ProductionRule(object):
    ''' Abstract interface for a production rule.
    Callable to perform the production, but also
    queryable for what nodes this connects and able to
    provide scoring for whether a candidate production
    is a good idea at all. '''
    def __init__(self, parent, products):
        self.parent = parent
        self.products = products
    def __call__(self):
        raise NotImplementedError()
    def score_products(self, products):
        raise NotImplementedError()

class RootNode(object):
    pass

class Node(object):
    def __init__(self, parent):
        self.parent = parent

class OrNode(Node):
    def __init__(self, parent, production_rules, production_weights):
        Node.__init__(self, parent)
        if len(production_weights) != len(production_rules):
            raise ValueError("# of production rules and weights must match.")
        if len(production_weights) == 0:
            raise ValueError("Must have nonzero number of production rules.")
        self.production_rules = production_rules
        self.production_dist = dist.Categorical(production_weights)
        
    def sample_production_rules(self, site_prefix, obs=None):
        sampled_rule = pyro.sample(site_prefix + "_or_sample", self.production_dist, obs=obs)
        return [self.production_rules[sampled_rule]]

    def score_production_rules(self, production_rules):
        if len(production_rules) != 1:
            print("Warning: invalid # of production rules for OrNode.")
            return -np.inf
        if production_rules[0] not in self.production_rules:
            print("Warning: rule not in OrNode production rules.")
        active_rule = torch.tensor(self.production_rules.index(production_rules[0]))
        return self.production_dist.log_prob(active_rule).sum()


class AndNode(Node):
    def __init__(self, parent, production_rules):
        Node.__init__(self, parent)
        if len(production_rules) == 0:
            raise ValueError("Must have nonzero # of production rules.")
        self.production_rules = production_rules
        
    def sample_production_rules(self, site_prefix):
        return self.production_rules

    def score_production_rules(self, production_rules):
        if production_rules != self.production_rules:
            print("Warning: invalid production rules given to AndNode.")
            return -np.inf
        else:
            return 0.0


class CovaryingSetNode(Node):
    def __init__(self, parent, site_prefix, production_rules,
                 production_weights_hints = {},
                 remaining_weight = 1.):
        ''' Make a categorical distribution over
           every possible combination of production rules
           that could be active, with a separate weight
           for each combination. (2^n weights!)

           Hints can be supplied in the form of a dictionary
           of (int tuple) : (float weight) pairs, and a float
           indicating the weight to distribute to the remaining
           pairs. These floats all indicate relative occurance
           weights. '''
        Node.__init__(self, parent)
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

    def sample_production_rules(self, site_prefix):
        selected_rules = pyro.sample(
            site_prefix + "_exhaustive_set_sample",
            self.production_dist)
        # Select out the appropriate rules
        output = []
        for k, rule in enumerate(self.production_rules):
            if (selected_rules >> k) & 1:
                output.append(rule)
        return output

    def score_production_rules(self, production_rules):
        selected_rules = 0
        for rule in production_rules:
            if rule not in self.production_rules:
                print("Warning: rule not in CovaryingSetNode production rules: ", rule)
                return -np.inf
            k = self.production_rules.index(rule)
            selected_rules += 2**k
        return self.production_dist.log_prob(torch.tensor(selected_rules)).sum()


class IndependentSetNode(Node):
    def __init__(self, parent, site_prefix, production_rules,
                 production_probs):
        ''' Make a categorical distribution over production rules
            that could be active, where each rule occurs
            independently of the others. Each production weight
            is a probability of that rule being active. '''
        Node.__init__(self, parent)
        if len(production_probs) != len(production_rules):
            raise ValueError("Must have same number of production probs "
                             "as rules.")
        self.production_dist = dist.Bernoulli(production_probs)
        self.site_prefix = ""
        self.production_rules = production_rules

    def sample_production_rules(self, site_prefix):
        active_rules = pyro.sample(
            site_prefix + "_independent_set_sample",
            self.production_dist)
        # Select out the appropriate rules
        output = []
        for k, rule in enumerate(self.production_rules):
            if active_rules[k] == 1:
                output.append(rule)
        return output

    def score_production_rules(self, production_rules):
        selected_rules = torch.zeros(len(self.production_rules))
        for rule in production_rules:
            if rule not in self.production_rules:
                print("Warning: rule not in IndependentSetNode production rules: ", rule)
                return -np.inf
            selected_rules[self.production_rules.index(rule)] = 1
        return self.production_dist.log_prob(selected_rules).sum()


class PlaceSetting(CovaryingSetNode):

    class ObjectProductionRule(ProductionRule):
        def __init__(self, parent, object_name, object_type, mean_init, var_init):
            ProductionRule.__init__(self, parent, products=[object_type])
            self.object_name = object_name
            self.object_type = object_type
            mean = pyro.param("place_setting_%s_mean" % object_name,
                              torch.tensor(mean_init))
            var = pyro.param("place_setting_%s_var" % object_name,
                              torch.tensor(var_init),
                              constraint=constraints.positive)
            self.offset_dist = dist.Normal(
                mean, var)

        def __call__(self, site_prefix):
            rel_pose = pyro.sample("%s_%s_pose" % (site_prefix, self.object_name),
                                   self.offset_dist)
            abs_pose = chain_pose_transforms(self.parent.pose, rel_pose)
            return [self.object_type(self, abs_pose)]

        def score_products(self, products):
            if len(products) != 1 or not isinstance(products[0], self.object_type):
                print("Warning: invalid input to scoreproducts for %sProductionRule: " % 
                      self.object_name, products)
            # Get relative offset of the PlaceSetting
            abs_pose = products[0].pose
            rel_pose = chain_pose_transforms(invert_pose(self.parent.pose), abs_pose)
            return self.offset_dist.log_prob(rel_pose).sum()

    def __init__(self, parent, pose):
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
            "left_knife": Knife,
            "left_spoon": Spoon,
            "right_fork": Fork,
            "right_knife": Knife,
            "right_spoon": Spoon,
        }
        param_guesses_by_name = {
            "plate": ([0., 0.16, 0.], [0.01, 0.01, 1.]),
            "cup": ([0., 0.16 + 0.15, 0.], [0.05, 0.01, 1.]),
            "right_fork": ([0.15, 0.16, 0.], [0.01, 0.01, 0.01]),
            "left_fork": ([-0.15, 0.16, 0.], [0.01, 0.01, 0.01]),
            "left_spoon": ([-0.15, 0.16, 0.], [0.01, 0.01, 0.01]),
            "right_spoon": ([0.15, 0.16, 0.], [0.01, 0.01, 0.01]),
            "left_knife": ([-0.15, 0.16, 0.], [0.01, 0.01, 0.01]),
            "right_knife": ([0.15, 0.16, 0.], [0.01, 0.01, 0.01]),
        }
        self.distributions_by_name = {}
        production_rules = []
        name_to_ind = {}
        for k, object_name in enumerate(self.object_types_by_name.keys()):
            mean_init, var_init = param_guesses_by_name[object_name]
            production_rules.append(
                self.ObjectProductionRule(
                    parent=self, object_name=object_name,
                    object_type=self.object_types_by_name[object_name],
                    mean_init=mean_init, var_init=var_init))
            # Build name mapping for convenienc of building the hint dictionary
            name_to_ind[object_name] = k

        # Weight the "correct" rules very heavily
        production_weights_hints = {
            (name_to_ind["plate"], name_to_ind["cup"], name_to_ind["left_fork"], name_to_ind["right_knife"], name_to_ind["right_spoon"]): 2.,
            (name_to_ind["plate"], name_to_ind["cup"], name_to_ind["left_fork"], name_to_ind["right_knife"]): 2.,
            (name_to_ind["plate"], name_to_ind["cup"], name_to_ind["left_fork"]): 2.,
            (name_to_ind["plate"], name_to_ind["cup"], name_to_ind["right_fork"]): 2.,
            (name_to_ind["plate"], name_to_ind["cup"], name_to_ind["right_fork"], name_to_ind["right_knife"]): 2.,
            (name_to_ind["plate"], name_to_ind["right_fork"]): 2.,
            (name_to_ind["plate"], name_to_ind["left_fork"]): 2.,
            (name_to_ind["cup"],): 0.5,
            (name_to_ind["plate"],): 1.,
        }
        CovaryingSetNode.__init__(self, parent, "place_setting", production_rules,
                                   production_weights_hints,
                                   remaining_weight=0.)


class Table(IndependentSetNode, RootNode):

    class PlaceSettingProductionRule(ProductionRule):
        def __init__(self, parent, root_pose):
            ProductionRule.__init__(self, parent, products=[PlaceSetting])
            # Relative offset from root pose is drawn from a diagonal
            # Normal. It's rotated into the root pose frame at sample time.
            mean = pyro.param("table_place_setting_mean",
                              torch.tensor([0.0, 0., np.pi/2.]))
            var = pyro.param("table_place_setting_var",
                              torch.tensor([0.01, 0.01, 0.01]),
                              constraint=constraints.positive)
            self.offset_dist = dist.Normal(mean, var)
            self.root_pose = root_pose

        def __call__(self, site_prefix):
            rel_offset = pyro.sample("%s_place_setting_offset" % site_prefix,
                                     self.offset_dist)
            # Rotate offset
            abs_offset = chain_pose_transforms(self.parent.pose, rel_offset)
            return [PlaceSetting(self, pose=self.root_pose + abs_offset)]

        def score_products(self, products):
            if len(products) != 1 or not isinstance(products[0], PlaceSetting):
                print("Warning: invalid input to scoreproducts for PlaceSettingProductionRule: ", products)
            # Get relative offset of the PlaceSetting
            abs_offset = products[0].pose - self.root_pose
            rel_offset = chain_pose_transforms(invert_pose(self.parent.pose), abs_offset)
            return self.offset_dist.log_prob(rel_offset).sum()

    def __init__(self, parent, num_place_setting_locations=4):
        self.pose = torch.tensor([0.5, 0.5, 0.])
        self.table_radius = pyro.param("table_radius", torch.tensor(0.45),
                                       constraint=constraints.positive)
        # Set-valued: a plate may appear at each location.
        production_rules = []
        for k in range(num_place_setting_locations):
            # TODO(gizatt) Root pose for each cluster could be a parameter.
            # This turns this into a GMM, sort of?
            root_pose = torch.zeros(3)
            root_pose[2] = (k / float(num_place_setting_locations))*np.pi*2.
            root_pose[0] = self.table_radius * torch.cos(root_pose[2])
            root_pose[1] = self.table_radius * torch.sin(root_pose[2])
            production_rules.append(self.PlaceSettingProductionRule(
                parent=self, root_pose=root_pose))
        IndependentSetNode.__init__(self, parent, "table_node", production_rules,
                                    torch.ones(num_place_setting_locations)*0.5)

class TerminalNode(Node):
    def __init__(self, parent):
        Node.__init__(self, parent)

class Plate(TerminalNode):
    def __init__(self, parent, pose, params=[0.2]):
        TerminalNode.__init__(self, parent)
        self.pose = pose
        self.params = params

    def generate_yaml(self):
        return {
            "class": "plate",
            "color": None,
            "img_path": "table_setting_assets/plate_red.png",
            "params": self.params,
            "params_names": ["radius"],
            "pose": self.pose.tolist()
        }

class Cup(TerminalNode):
    def __init__(self, parent, pose, params=[0.05]):
        TerminalNode.__init__(self, parent)
        self.pose = pose
        self.params = params

    def generate_yaml(self):
        return {
            "class": "cup",
            "color": None,
            "img_path": "table_setting_assets/cup_water.png",
            "params": self.params,
            "params_names": ["radius"],
            "pose": self.pose.tolist()
        }


class Fork(TerminalNode):
    def __init__(self, parent, pose, params=[0.02, 0.14]):
        TerminalNode.__init__(self, parent)
        self.pose = pose
        self.params = params

    def generate_yaml(self):
        return {
            "class": "fork",
            "color": None,
            "img_path": "table_setting_assets/fork.png",
            "params": self.params,
            "params_names": ["width", "height"],
            "pose": self.pose.tolist()
        }

class Knife(TerminalNode):
    def __init__(self, parent, pose, params=[0.015, 0.15]):
        TerminalNode.__init__(self, parent)
        self.pose = pose
        self.params = params
    
    def generate_yaml(self):
        return {
            "class": "knife",
            "color": None,
            "img_path": "table_setting_assets/knife.png",
            "params": self.params,
            "params_names": ["width", "height"],
            "pose": self.pose.tolist()
        }

class Spoon(TerminalNode):
    def __init__(self, parent, pose, params=[0.02, 0.12]):
        TerminalNode.__init__(self, parent)
        self.pose = pose
        self.params = params
    
    def generate_yaml(self):
        return {
            "class": "spoon",
            "color": None,
            "img_path": "table_setting_assets/spoon.png",
            "params": self.params,
            "params_names": ["width", "height"],
            "pose": self.pose.tolist()
        }

class ProbabilisticSceneGrammarModel():
    def __init__(self):
        pass

    def model(self, data=None):
        nodes = [Table(parent=None)]
        all_terminal_nodes = []
        all_nodes = []
        all_production_rules = []
        num_productions = 0
        iter_k = 0
        while len(nodes) > 0:
            node = nodes.pop(0)
            all_nodes.append(node)
            if isinstance(node, TerminalNode):
                # Instantiate
                all_terminal_nodes.append(node)
            else:
                # Expand by picking a production rule
                production_rules = node.sample_production_rules("production_%04d" % num_productions)
                for i, rule in enumerate(production_rules):
                    new_nodes = rule("production_%04d_sample_%04d" % (num_productions, i))
                    nodes += new_nodes
                all_production_rules += production_rules
                num_productions += 1
            iter_k += 1

        return all_terminal_nodes, all_production_rules, all_nodes

    def guide(self, data):
        pass


def convert_list_of_terminal_nodes_to_yaml_env(node_list):
    env = {"n_objects": len(node_list)}
    for k, node in enumerate(node_list):
        env["obj_%04d" % k] = node.generate_yaml()
    return env

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def build_networkx_parse_tree(all_terminal_nodes):
    ''' Represent the parse tree as a graph over two node
    types: Nodes, and ProductionRules. '''
    G = nx.DiGraph()
    label_dict = {}
    def add_node_and_parents_to_graph(node):
        # Recursive construction
        if node not in G:
            G.add_node(node)
            label_dict[node] = node.__class__.__name__
            rule = node.parent
            if rule is None:
                return
            if rule not in G:
                # New rule!
                G.add_node(rule)
                parent_node = rule.parent
                # All production rules must have parents
                if parent_node is None:
                    raise ValueError("Tree had a production rule with no parent!")
                add_node_and_parents_to_graph(parent_node)
                G.add_edge(parent_node, rule)
            G.add_edge(rule, node)

    pos_dict = {}
    for node in all_terminal_nodes:
        add_node_and_parents_to_graph(node)
        pos_dict[node] = node.pose[0:2].tolist()
    for node in G.nodes:
        if node in pos_dict.keys():
            # Already have the position
            continue
        if hasattr(node, "pose"):
            pos_dict[node] = node.pose[0:2].tolist()
        else:
            # Compute avg child pose over all children
            avg_child_pose = np.zeros(2)
            num_children = 0
            child_queue = list(G.successors(node))
            while len(child_queue) > 0:
                child = child_queue.pop(0)
                if hasattr(child, "pose"):
                    avg_child_pose += child.pose[0:2].detach().numpy()
                    num_children += 1
                child_queue += list(G.successors(child))
            pos_dict[node] = avg_child_pose / num_children
    return G, pos_dict, label_dict

def remove_production_rules_from_parse_tree(parse_tree):
    ''' For drawing '''
    new_tree = nx.DiGraph()
    for node in parse_tree:
        if isinstance(node, Node):
            new_tree.add_node(node)
    for node in parse_tree:
        if isinstance(node, ProductionRule):
            for child in list(parse_tree.successors(node)):
                new_tree.add_edge(node.parent, child)
    return new_tree

class_name_to_type = {
    "plate": Plate,
    "cup": Cup,
    "fork": Fork,
    "knife": Knife,
    "spoon": Spoon,
}
def terminal_nodes_from_yaml(yaml_env):
    all_terminal_nodes = []
    for k in range(yaml_env["n_objects"]):
        new_obj = yaml_env["obj_%04d" % k]
        if new_obj["class"] not in class_name_to_type.keys():
            raise NotImplementedError("Unknown class: ", new_obj["class"])
        all_terminal_nodes.append(class_name_to_type[new_obj['class']](
            parent=None,
            pose=torch.tensor(new_obj["pose"]),
            params=new_obj["params"]))
    return all_terminal_nodes

def score_tree(parse_tree):
    ''' Sum the log probabilities over the tree:
    For every node, score its set of its production rules.
    For every production rule, score its products.
    If the tree is infeasible / ill-posed, return -inf.

    Modifies the tree in-place, setting the "log_prob"
    attribute of each node.

    TODO(gizatt): My tree-building here echos a lot of the
    machinery from within pyro's execution trace checking,
    to the degree that I sanity check in the tests down below
    by comparing a pyro-log-prob-sum of a forward run of the model
    to the value I compute here. Can I better use Pyro's machinery
    to do less work?'''
    total_ll = 0.
    scores_by_node = {}

    for node in parse_tree.nodes:
        if isinstance(node, Node):
            # Sanity-check feasibility
            if node.parent is None and not isinstance(node, RootNode):
                node_score = -np.inf
            elif isinstance(node, TerminalNode):
                # TODO(gizatt): Eventually, TerminalNodes
                # may want to score their generated parameters.
                node_score = 0.
            else:
                # Score the kids
                node_score = node.score_production_rules(list(parse_tree.successors(node)))
        elif isinstance(node, ProductionRule):
            node_score = node.score_products(list(parse_tree.successors(node)))
        else:
            raise ValueError("Invalid node type in tree: ", type(node))
        scores_by_node[node] = node_score
        total_ll += node_score

    nx.set_node_attributes(parse_tree, scores_by_node, name="log_prob")
    return total_ll


def guess_place_setting(nodes):
    todo

def guess_table(nodes):
    todo

def greedily_guess_parse_tree_from_yaml(yaml_env):
    candidate_intermediate_node_generators = [guess_place_setting, guess_table]

    # Build a list of the terminal nodes.
    all_terminal_nodes = terminal_nodes_from_yaml(yaml_env)

    # Build the preliminary parse tree of all terminal nodes
    # being root nodes.
    parse_tree, pos_dict, label_dict = build_networkx_parse_tree(all_terminal_nodes)
    iter_k = 0
    while iter_k < 10:
        tree_ll = score_tree(parse_tree)
        print("At start of iter %d, tree score is %f" % (iter_k, tree_ll))
        log_probs = nx.get_node_attributes(parse_tree, name="log_prob")
        
        # Find the currently-infeasible nodes.
        infeasible_nodes = [key for key in log_probs.keys() if np.isinf(log_probs[key])]

        # Assert that they're all infeasible Nodes. Infeasible rules
        # should never appear by construction.
        for node in infeasible_nodes:
            assert(isinstance(node, Node))

        print("%d infeasible nodes" % len(infeasible_nodes))

        if len(infeasible_nodes) > 0:
            # Sample a subset of infeasible nodes.
            number_of_sampled_nodes = min(np.random.geometric(p=0.5), len(infeasible_nodes))
            sampled_nodes = random.sample(infeasible_nodes, number_of_sampled_nodes)

            possible_parent_nodes = []
            possible_parent_nodes_scores = []
            # Check all ways of adding this to the tree:
            #  - Append this to products of an existing rule.
            #  - Append a production rule to an already existing non-terminal node.
            for node in parse_tree.nodes():
                if isinstance(node, ProductionRule):
                    # Try adding this to the products of this rule
                    score = node.score_products( list(parse_tree.successors(node)) + sampled_nodes )
                    if not np.inf(score):
                        possible_parent_nodes.append(node)
                        possible_parent_nodes_scores.append(score)
                elif isinstance(node, AndNode):
                    # This can't ever work either, as an instantiated AndNode
                    # is already feasible by construction, and the output
                    # set isn't flexible.
                    pass
                elif isinstance(node, OrNode):
                    # This can never work, as instantiated OrNode is already
                    # feasible by construction, so adding another product
                    # isn't possible.
                    pass
                elif isinstance(node, CovaryingSetNode):
                    # Try adding this via any of this node's production rules
                    for rule in node.production_rules:
                        score = rule.score_products(sampled_nodes)
                        score += node.score_production_rules( list(parse_tree.successors(node)) + [rule] )
                        if not np.inf(score):
                            possible_parent_nodes.append(rule)
                            possible_parent_nodes_scores.append(score)
                elif isinstance(node, IndependentSetNode):
                    raise NotImplementedError()
                else:
                    pass
            # Make a new intermediate node with these as the products.
            for node_type in candidate_intermediate_node_generators:
                pass

            print("Possible parent nodes: ", possible_parent_nodes)
            print("Possible parent node scores: ", possible_parent_nodes_scores)

        else:
            # Consider possible mutations of the tree?
            print("Tree is feasible!")
            print("But if we have any root nodes that can also be generated by ")
            print("rules, we never tried to expand them upwards.")
            print("Perhaps an assumption works: there's only one root node, ever.")
            break
        iter_k += 1



if __name__ == "__main__":
    seed = 43
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    pyro.enable_validation(True)

    model = ProbabilisticSceneGrammarModel()

    plt.figure().set_size_inches(15, 10)
    for k in range(1):
        start = time.time()
        pyro.clear_param_store()
        trace = poutine.trace(model.model).get_trace()
        all_terminal_nodes, all_production_rules, all_nodes = trace.nodes["_RETURN"]["value"]
        end = time.time()

        print(bcolors.OKGREEN, "Generated data in %f seconds." % (end - start), bcolors.ENDC)
        print("Full trace values:" )
        for node_name in trace.nodes.keys():
            if node_name in ["_INPUT", "_RETURN"]:
                continue
            print(node_name, ": ", trace.nodes[node_name]["value"].detach().numpy())

        # Recover and print the parse tree
        G, pos_dict, label_dict = build_networkx_parse_tree(all_terminal_nodes)
        plt.subplot(2, 1, 1)
        plt.gca().clear()
        nx.draw_networkx(remove_production_rules_from_parse_tree(G), pos=pos_dict, labels=label_dict, font_weight='bold')
        plt.xlim(-0.2, 1.2)
        plt.ylim(-0.2, 1.2)
        plt.title("Parse tree")
        yaml_env = convert_list_of_terminal_nodes_to_yaml_env(all_terminal_nodes)
        print("Original tree score: ", score_tree(G))
        print("Pyro trace ll: ", trace.log_prob_sum())
        assert(abs(score_tree(G) - trace.log_prob_sum()) < 0.001)
        print("And that matches what pyro says.")

        # And then try to parse it
        greedily_guess_parse_tree_from_yaml(yaml_env)

        #with open("table_setting_environments_generated.yaml", "a") as file:
        #    yaml.dump({"env_%d" % int(round(time.time() * 1000)): yaml_env}, file)

        #yaml_env = ProjectEnvironmentToFeasibility(yaml_env, base_environment_type="table_setting",
        #                                           make_static=False)[0]

        try:
            plt.subplot(2, 2, 3)
            plt.gca().clear()
            DrawYamlEnvironmentPlanar(yaml_env, base_environment_type="table_setting", ax=plt.gca())
            plt.title("Generated scene")
            plt.subplot(2, 2, 4)
            plt.gca().clear()
            DrawYamlEnvironmentPlanar(yaml_env, base_environment_type="table_setting", ax=plt.gca())
            nx.draw_networkx(remove_production_rules_from_parse_tree(G), pos=pos_dict, with_labels=False, node_color=[0., 0., 0.], alpha=0.5, node_size=100)
            plt.title("Generated scene with parse tree")
            plt.show()
        except Exception as e:
            print("Exception ", e)
        except:
            print(bcolors.FAIL, "Caught ????, probably sim fault due to weird geometry.", bcolors.ENDC)
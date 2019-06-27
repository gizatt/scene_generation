from __future__ import print_function
from copy import deepcopy
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
    out = torch.empty(3)
    r = p_w1[2]
    out[0] = p_w1[0] + p_12[0]*torch.cos(r) - p_12[1]*torch.sin(r)
    out[1] = p_w1[1] + p_12[0]*torch.sin(r) + p_12[1]*torch.cos(r)
    out[2] = p_w1[2] + p_12[2]
    return out

def invert_pose(pose):
    # TF^-1 = [R^t  -R.' T]
    out = torch.empty(3)
    r = pose[2]
    out[0] = -(pose[0]*torch.cos(-r) - pose[1]*torch.sin(-r))
    out[1] = -(pose[0]*torch.sin(-r) + pose[1]*torch.cos(-r))
    out[2] = -r
    return out


class ProductionRule(object):
    ''' Abstract interface for a production rule.
    Callable to perform the production, but also
    queryable for what nodes this connects and able to
    provide scoring for whether a candidate production
    is a good idea at all. '''
    def __init__(self, products):
        self.products = products
    def __call__(self, parent, site_prefix):
        raise NotImplementedError()
    def score_products(self, parent, products):
        raise NotImplementedError()

class RootNode(object):
    pass

class Node(object):
    def __init__(self):
        pass

class OrNode(Node):
    def __init__(self, production_rules, production_weights):
        Node.__init__(self)
        if len(production_weights) != len(production_rules):
            raise ValueError("# of production rules and weights must match.")
        if len(production_weights) == 0:
            raise ValueError("Must have nonzero number of production rules.")
        self.production_rules = production_rules
        self.production_dist = dist.Categorical(production_weights)
        
    def sample_production_rules(self, parent, site_prefix, obs=None):
        sampled_rule = pyro.sample(site_prefix + "_or_sample", self.production_dist, obs=obs)
        return [self.production_rules[sampled_rule]]

    def score_production_rules(self, parent, production_rules):
        if len(production_rules) != 1:
            return torch.tensor(-np.inf)
        if production_rules[0] not in self.production_rules:
            print("Warning: rule not in OrNode production rules.")
        active_rule = torch.tensor(self.production_rules.index(production_rules[0]))
        return self.production_dist.log_prob(active_rule).sum()


class AndNode(Node):
    def __init__(self, production_rules):
        Node.__init__(self)
        if len(production_rules) == 0:
            raise ValueError("Must have nonzero # of production rules.")
        self.production_rules = production_rules
        
    def sample_production_rules(self, parent, site_prefix):
        return self.production_rules

    def score_production_rules(self, parent, production_rules):
        if production_rules != self.production_rules:
            return torch.tensor(-np.inf)
        else:
            return torch.tensor(-np.inf)


class CovaryingSetNode(Node):
    def __init__(self, site_prefix, production_rules,
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
        Node.__init__(self)
        # Build the initial weights, taking the suggestion
        # weights into account.
        assert(remaining_weight >= 0.)
        num_combinations = 2**len(production_rules) + 1
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

        self.production_dist = dist.Categorical(logits=torch.log(self.exhaustive_set_weights))
        self.site_prefix = ""
        self.production_rules = production_rules

    def sample_production_rules(self, parent, site_prefix):
        selected_rules = pyro.sample(
            site_prefix + "_exhaustive_set_sample",
            self.production_dist)
        # Select out the appropriate rules
        output = []
        for k, rule in enumerate(self.production_rules):
            if (selected_rules >> k) & 1:
                output.append(rule)
        return output

    def score_production_rules(self, parent, production_rules):
        selected_rules = 0
        for rule in production_rules:
            if rule not in self.production_rules:
                print("Warning: rule not in CovaryingSetNode production rules: ", rule)
                return torch.tensor(-np.inf)
            k = self.production_rules.index(rule)
            selected_rules += 2**k
        assert(selected_rules >= 0 and selected_rules <= len(self.exhaustive_set_weights))
        return self.production_dist.log_prob(torch.tensor(selected_rules)).sum()


class IndependentSetNode(Node):
    def __init__(self, site_prefix, production_rules,
                 production_probs):
        ''' Make a categorical distribution over production rules
            that could be active, where each rule occurs
            independently of the others. Each production weight
            is a probability of that rule being active. '''
        Node.__init__(self)
        if len(production_probs) != len(production_rules):
            raise ValueError("Must have same number of production probs "
                             "as rules.")
        self.production_dist = dist.Bernoulli(production_probs)
        self.site_prefix = ""
        self.production_rules = production_rules

    def sample_production_rules(self, parent, site_prefix):
        active_rules = pyro.sample(
            site_prefix + "_independent_set_sample",
            self.production_dist)
        # Select out the appropriate rules
        output = []
        for k, rule in enumerate(self.production_rules):
            if active_rules[k] == 1:
                output.append(rule)
        return output

    def score_production_rules(self, parent, production_rules):
        selected_rules = torch.zeros(len(self.production_rules))
        for rule in production_rules:
            if rule not in self.production_rules:
                print("Warning: rule not in IndependentSetNode production rules: ", rule)
                return torch.tensor(-np.inf)
            selected_rules[self.production_rules.index(rule)] = 1
        return self.production_dist.log_prob(selected_rules).sum()


class PlaceSetting(CovaryingSetNode):

    class ObjectProductionRule(ProductionRule):
        def __init__(self, object_name, object_type, mean_init, var_init):
            ProductionRule.__init__(self, products=[object_type])
            self.object_name = object_name
            self.object_type = object_type
            mean = pyro.param("place_setting_%s_mean" % object_name,
                              torch.tensor(mean_init))
            var = pyro.param("place_setting_%s_var" % object_name,
                              torch.tensor(var_init),
                              constraint=constraints.positive)
            self.offset_dist = dist.Normal(
                mean, var)

        def __call__(self, parent, site_prefix):
            rel_pose = pyro.sample("%s_%s_pose" % (site_prefix, self.object_name),
                                   self.offset_dist)
            abs_pose = chain_pose_transforms(parent.pose, rel_pose)
            return [self.object_type(abs_pose)]

        def score_products(self, parent, products):
            if len(products) != 1 or not isinstance(products[0], self.object_type):
                return torch.tensor(-np.inf)
            # Get relative offset of the PlaceSetting
            abs_pose = products[0].pose
            rel_pose = chain_pose_transforms(invert_pose(parent.pose), abs_pose)
            return self.offset_dist.log_prob(rel_pose).sum()

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
            #"left_knife": Knife,
            #"left_spoon": Spoon,
            #"right_fork": Fork,
            #"right_knife": Knife,
            #"right_spoon": Spoon,
        }
        param_guesses_by_name = {
            "plate": ([0., 0.16, 0.], [0.01, 0.01, 3.]),
            "cup": ([0., 0.16 + 0.15, 0.], [0.05, 0.01, 3.]),
            #"right_fork": ([0.15, 0.16, 0.], [0.01, 0.01, 0.01]),
            "left_fork": ([-0.15, 0.16, 0.], [0.01, 0.01, 0.01]),
            #"left_spoon": ([-0.15, 0.16, 0.], [0.01, 0.01, 0.01]),
            #"right_spoon": ([0.15, 0.16, 0.], [0.01, 0.01, 0.01]),
            #"left_knife": ([-0.15, 0.16, 0.], [0.01, 0.01, 0.01]),
            #"right_knife": ([0.15, 0.16, 0.], [0.01, 0.01, 0.01]),
        }
        self.distributions_by_name = {}
        production_rules = []
        name_to_ind = {}
        for k, object_name in enumerate(self.object_types_by_name.keys()):
            mean_init, var_init = param_guesses_by_name[object_name]
            production_rules.append(
                self.ObjectProductionRule(
                    object_name=object_name,
                    object_type=self.object_types_by_name[object_name],
                    mean_init=mean_init, var_init=var_init))
            # Build name mapping for convenienc of building the hint dictionary
            name_to_ind[object_name] = k

        # Weight the "correct" rules very heavily
        production_weights_hints = {
            #(name_to_ind["plate"], name_to_ind["cup"], name_to_ind["left_fork"], name_to_ind["right_knife"], name_to_ind["right_spoon"]): 2.,
            #(name_to_ind["plate"], name_to_ind["cup"], name_to_ind["left_fork"], name_to_ind["right_knife"]): 2.,
            #(name_to_ind["plate"], name_to_ind["cup"], name_to_ind["left_fork"]): 2.,
            #(name_to_ind["plate"], name_to_ind["cup"], name_to_ind["right_fork"]): 2.,
            #(name_to_ind["plate"], name_to_ind["cup"], name_to_ind["right_fork"], name_to_ind["right_knife"]): 2.,
            #(name_to_ind["plate"], name_to_ind["right_fork"]): 2.,
            (name_to_ind["plate"], name_to_ind["left_fork"]): 1.,
            #(name_to_ind["cup"],): 0.5,
            (name_to_ind["plate"], name_to_ind["cup"]): 1.,
            (name_to_ind["plate"],): 1.,
        }
        CovaryingSetNode.__init__(self, "place_setting", production_rules,
                                   production_weights_hints,
                                   remaining_weight=0.)


class Table(IndependentSetNode, RootNode):

    class PlaceSettingProductionRule(ProductionRule):
        def __init__(self, pose):
            ProductionRule.__init__(self, products=[PlaceSetting])
            # Relative offset from root pose is drawn from a diagonal
            # Normal. It's rotated into the root pose frame at sample time.
            mean = pyro.param("table_place_setting_mean",
                              torch.tensor([0.0, 0., np.pi/2.]))
            var = pyro.param("table_place_setting_var",
                              torch.tensor([0.01, 0.01, 0.1]),
                              constraint=constraints.positive)
            self.offset_dist = dist.Normal(mean, var)
            self.pose = pose

        def __call__(self, parent, site_prefix):
            rel_offset = pyro.sample("%s_place_setting_offset" % site_prefix,
                                     self.offset_dist)
            # Rotate offset
            pose_in_world = chain_pose_transforms(parent.pose, self.pose)
            abs_offset = chain_pose_transforms(pose_in_world, rel_offset)
            return [PlaceSetting(pose=abs_offset)]

        def score_products(self, parent, products):
            if len(products) != 1 or not isinstance(products[0], PlaceSetting):
                return torch.tensor(-np.inf)
            # Get relative offset of the PlaceSetting
            abs_offset = products[0].pose
            pose_in_world = chain_pose_transforms(parent.pose, self.pose)
            rel_offset = chain_pose_transforms(invert_pose(pose_in_world), abs_offset)
            return self.offset_dist.log_prob(rel_offset).sum()

    def __init__(self, num_place_setting_locations=4):
        self.pose = torch.tensor([0.5, 0.5, 0.])
        self.table_radius = pyro.param("table_radius", torch.tensor(0.45), constraint=constraints.positive)
        # Set-valued: a plate may appear at each location.
        production_rules = []
        for k in range(num_place_setting_locations):
            # TODO(gizatt) Root pose for each cluster could be a parameter.
            # This turns this into a GMM, sort of?
            r = torch.tensor((k / float(num_place_setting_locations))*np.pi*2.)
            pose = torch.empty(3)
            pose[0] = self.table_radius * torch.cos(r)
            pose[1] = self.table_radius * torch.sin(r)
            pose[2] = r
            production_rules.append(self.PlaceSettingProductionRule(
                pose=pose))
        IndependentSetNode.__init__(self, "table_node", production_rules,
                                    torch.ones(num_place_setting_locations)*0.5)

class TerminalNode(Node):
    def __init__(self):
        Node.__init__(self)

class Plate(TerminalNode):
    def __init__(self, pose, params=[0.2]):
        TerminalNode.__init__(self)
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
    def __init__(self, pose, params=[0.05]):
        TerminalNode.__init__(self)
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
    def __init__(self, pose, params=[0.02, 0.14]):
        TerminalNode.__init__(self)
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
    def __init__(self, pose, params=[0.015, 0.15]):
        TerminalNode.__init__(self)
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
    def __init__(self, pose, params=[0.02, 0.12]):
        TerminalNode.__init__(self)
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


def get_node_parent_or_none(parse_tree, node):
    parents = list(parse_tree.predecessors(node))
    if len(parents) == 0:
        return None
    elif len(parents) == 1:
        return parents[0]
    else:
        print("Bad parse tree: ", parse_tree)
        print("Node: ", node)
        print("Parents: ", parents)
        raise NotImplementedError("> 1 parent --> bad parse tree")


class ProbabilisticSceneGrammarModel():
    def __init__(self):
        pass

    def model(self, data=None):
        root_node = Table()
        input_nodes_with_parents = [ (None, root_node) ]  # (parent, node) order
        parse_tree = nx.DiGraph()
        parse_tree.add_node(root_node)
        num_productions = 0
        while len(input_nodes_with_parents) > 0:
            parent, node = input_nodes_with_parents.pop(0)
            if isinstance(node, TerminalNode):
                # Nothing more to do with this node
                pass
            else:
                # Expand by picking a production rule
                production_rules = node.sample_production_rules(parent, "production_%04d" % num_productions)
                for i, rule in enumerate(production_rules):
                    parse_tree.add_node(rule)
                    parse_tree.add_edge(node, rule)
                    new_nodes = rule(node, "production_%04d_sample_%04d" % (num_productions, i))
                    for new_node in new_nodes:
                        parse_tree.add_node(new_node)
                        parse_tree.add_edge(rule, new_node)
                        input_nodes_with_parents.append((rule, new_node))
                num_productions += 1
        return parse_tree

    def guide(self, data):
        pass


def convert_tree_to_yaml_env(parse_tree):
    terminal_nodes = []
    for node in parse_tree:
        if isinstance(node, TerminalNode):
            terminal_nodes.append(node)
    env = {"n_objects": len(terminal_nodes)}
    for k, node in enumerate(terminal_nodes):
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
    def add_node_and_parents_to_graph(node):
        # Recursive construction
        if node not in G:
            G.add_node(node)
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
    for node in all_terminal_nodes:
        add_node_and_parents_to_graph(node)
    return G

def remove_production_rules_from_parse_tree(parse_tree):
    ''' For drawing '''
    new_tree = nx.DiGraph()
    for node in parse_tree:
        if isinstance(node, Node):
            new_tree.add_node(node)
    for node in parse_tree:
        if isinstance(node, ProductionRule):
            parent = get_node_parent_or_none(parse_tree, node)
            assert(parent is not None)
            for child in list(parse_tree.successors(node)):
                new_tree.add_edge(parent, child)
    return new_tree

class_name_to_type = {
    "plate": Plate,
    "cup": Cup,
    "fork": Fork,
    "knife": Knife,
    "spoon": Spoon,
}
def terminal_nodes_from_yaml(yaml_env):
    terminal_nodes = []
    for k in range(yaml_env["n_objects"]):
        new_obj = yaml_env["obj_%04d" % k]
        if new_obj["class"] not in class_name_to_type.keys():
            raise NotImplementedError("Unknown class: ", new_obj["class"])
        terminal_nodes.append(class_name_to_type[new_obj['class']](
            pose=torch.tensor(new_obj["pose"]),
            params=new_obj["params"]))
    return terminal_nodes

def score_tree(parse_tree, assert_rooted=True):
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
    total_ll = torch.tensor(0.)
    scores_by_node = {}

    for node in parse_tree.nodes:
        parent = get_node_parent_or_none(parse_tree, node)
        if isinstance(node, Node):
            # Sanity-check feasibility
            if parent is None and assert_rooted and not isinstance(node, RootNode):
                node_score = torch.tensor(-np.inf)
            elif isinstance(node, TerminalNode):
                # TODO(gizatt): Eventually, TerminalNodes
                # may want to score their generated parameters.
                node_score = torch.tensor(0.)
            else:
                # Score the kids
                node_score = node.score_production_rules(parent, list(parse_tree.successors(node)))
        elif isinstance(node, ProductionRule):
            node_score = node.score_products(parent, list(parse_tree.successors(node)))
        else:
            raise ValueError("Invalid node type in tree: ", type(node))
        scores_by_node[node] = node_score
        total_ll = total_ll + node_score

    return total_ll, scores_by_node

def draw_parse_tree(parse_tree, ax=None, label_score=True, label_name=True, **kwargs):
    pruned_tree = remove_production_rules_from_parse_tree(parse_tree)
    score, scores_by_node = score_tree(parse_tree)
    pos_dict = {
        node: node.pose[:2].detach().numpy() for node in pruned_tree
    }
    label_dict = {}
    for node in pruned_tree.nodes:
        label_str = ""
        if label_name:
            label_str += node.__class__.__name__
        if label_score:
            score_of_node = scores_by_node[node].item()
            score_of_children = sum(scores_by_node[child] for child in parse_tree.successors(node))
            label_str += ": %2.02f / %2.02f" % (score_of_node, score_of_children)
        if label_name != "":
            label_dict[node] = label_str
    colors = np.array([scores_by_node[node].item() for node in pruned_tree.nodes])
    if len(colors) == 0:
        colors = None
    else:
        colors -= min(colors)
        colors /= max(colors)
    # Convert scores to colors
    if ax is None:
        ax = plt.gca()
    nx.draw_networkx(pruned_tree, ax=ax, pos=pos_dict, labels=label_dict,
                     node_color=colors, cmap='jet', font_weight='bold', **kwargs)
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.set_title("Score: %f" % score)


#def guess_table(child_nodes):
#   # Tables are always at the same place
#   table_node = Table(parent=None)
#   candidate_rules = []
#   candidate_lls = []
#   for rule in table_node.production_rules:
#       total_ll = rule.score_products(child_nodes) + table_node.score_production_rules([rule])
#       if not torch.isinf(total_ll):
#           candidate_rules.append(rule)
#           candidate_lls.append(total_ll)
#   return table_node, candidate_rules, candidate_lls

def guess_place_setting(parent, child_nodes):
    # Sample a place setting root pose at the average
    # location of child poses, oriented inwards toward
    # middle like we know place settings ought to be.
    init_pose = torch.zeros(3)
    for node in child_nodes:
        init_pose = init_pose + node.pose
    init_pose /= len(child_nodes)
    init_pose[2] = torch.atan2(init_pose[1], init_pose[0]) + np.pi/2.
    init_pose = dist.Normal(init_pose, torch.tensor([0.1, 0.1, 0.1])).sample()
    place_setting_node = PlaceSetting(pose=init_pose)

    candidate_rules = []
    candidate_lls = []
    for rule in place_setting_node.production_rules:
        total_ll = rule.score_products(place_setting_node, child_nodes) + place_setting_node.score_production_rules(parent, [rule])
        if not torch.isinf(total_ll):
            candidate_rules.append(rule)
            candidate_lls.append(total_ll)

    return place_setting_node, candidate_rules, candidate_lls


def repair_parse_tree_in_place(parse_tree, max_num_iters=100, ax=None):
    # Build that tree into a feasible one by repeatedly sampling
    # subsets of infeasible nodes and selecting among the rules
    # that could have generated them.

    candidate_intermediate_node_generators = [
        guess_place_setting
    ]

    iter_k = 0
    while iter_k < max_num_iters:
        score, scores_by_node = score_tree(parse_tree,  assert_rooted=True)
        print("At start of iter %d, tree score is %f" % (iter_k, score))
        if ax is not None:
            ax.clear()
            draw_parse_tree(parse_tree, ax=ax)
            plt.pause(0.01)
        
        # Find the currently-infeasible nodes.
        infeasible_nodes = [key for key in scores_by_node.keys() if torch.isinf(scores_by_node[key])]

        # Assert that they're all infeasible Nodes. Infeasible rules
        # should never appear by construction.
        for node in infeasible_nodes:
            # TODO(gizatt) I could conceivably just handle productionrules
            # as a special case. They can be crammed into the tree the
            # same way -- iterate over all nodes and see how they fit into
            # the available products.
            assert(isinstance(node, Node))

        if len(infeasible_nodes) > 0:
            possible_child_nodes = []
            possible_parent_nodes = []
            possible_parent_nodes_scores = []

            for inner_iter in range(5):
                # Sample a subset of infeasible nodes.
                number_of_sampled_nodes = min(np.random.geometric(p=0.5), len(infeasible_nodes))
                sampled_nodes = random.sample(infeasible_nodes, number_of_sampled_nodes)

                # Check all ways of adding this to the tree:
                #  - Append this to products of an existing rule.
                #  - Append a production rule to an already existing non-terminal node.
                for node in parse_tree.nodes():
                    parent = get_node_parent_or_none(parse_tree, node)
                    if isinstance(node, ProductionRule):
                        # Try adding this to the products of this rule
                        score = node.score_products(parent, list(parse_tree.successors(node)) + sampled_nodes)
                        if not torch.isinf(score):
                            possible_child_nodes.append(sampled_nodes)
                            possible_parent_nodes.append((parent, node))
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
                    elif isinstance(node, CovaryingSetNode) or isinstance(node, IndependentSetNode):
                        # Try adding this via any of this node's production rules
                        existing_rules = list(parse_tree.successors(node))
                        for rule in node.production_rules:
                            if rule in existing_rules:
                                continue
                            score = rule.score_products(node, sampled_nodes)
                            score = score + node.score_production_rules(parent, existing_rules + [rule])
                            if not torch.isinf(score):
                                possible_child_nodes.append(sampled_nodes)
                                possible_parent_nodes.append((node, rule))
                                possible_parent_nodes_scores.append(score)
                    else:
                        pass
                # - Make a new intermediate node and rule with these as the products.
                for node_sampler in candidate_intermediate_node_generators:
                    new_node, new_rules, scores = node_sampler(None, sampled_nodes)
                    for rule, score in zip(new_rules, scores):
                        possible_child_nodes.append(sampled_nodes)
                        possible_parent_nodes.append((new_node, rule))
                        possible_parent_nodes_scores.append(score)

            if len(possible_parent_nodes_scores) == 0:
                iter_k  += 1
                continue
            possible_parent_nodes_scores_raw = torch.stack(possible_parent_nodes_scores)
            # Normalize by subtracting off log(sum(exp(possible_parent_nodes_scores_raw)))
            # See https://en.wikipedia.org/wiki/LogSumExp
            maxval = possible_parent_nodes_scores_raw.max()
            normalization_factor = maxval + torch.log(torch.exp(possible_parent_nodes_scores_raw - maxval).sum())
            possible_parent_nodes_scores = torch.exp(possible_parent_nodes_scores_raw - normalization_factor)
            # Sample an action from that set
            ind = dist.Categorical(possible_parent_nodes_scores).sample()
            child_nodes = possible_child_nodes[ind]
            new_parent_node, new_prod_node = possible_parent_nodes[ind]
            best_score = possible_parent_nodes_scores_raw[ind]
            # print("Adding ", new_parent_node, " and rule ", new_prod_node, " as parent to ", child_nodes)
            if new_parent_node not in parse_tree.nodes:
                parse_tree.add_node(new_parent_node)
                assert(new_prod_node not in parse_tree.nodes)
                parse_tree.add_node(new_prod_node)
                parse_tree.add_edge(new_parent_node, new_prod_node)
            elif new_prod_node not in parse_tree.nodes:
                parse_tree.add_node(new_prod_node)
                parse_tree.add_edge(new_parent_node, new_prod_node)
            for child_node in child_nodes:
                parse_tree.add_edge(new_prod_node, child_node)
        else:
            # Whole tree is feasible!
            break
        iter_k += 1
    return score_tree(parse_tree)[0]

def optimize_parse_tree_hmc_in_place(parse_tree, ax=None):

    # Run HMC on the continuous paramaters (poses of the place settings)
    # for a few steps.
    continuous_params = []
    v_proposal_dists = []
    num_hmc_steps = 10
    num_dynamics_steps = 10
    epsilon_v = 5E-3
    epsilon_p = 5E-3
    proposal_variance = 1.0
    for node in parse_tree:
        if isinstance(node, PlaceSetting):
            node.pose.requires_grad = True
            continuous_params.append(node.pose)
            v_proposal_dists.append(dist.Normal(torch.zeros(3), torch.ones(3)*proposal_variance))
    for hmc_step in range(num_hmc_steps):
        if ax is not None:
            ax.clear()
            draw_parse_tree(parse_tree)
            plt.pause(0.1)
        initial_score, _ = score_tree(parse_tree)
        initial_score.backward(retain_graph=True)
        initial_param_vals = [p.detach().numpy().copy() for p in continuous_params]
        current_vs = [v_dist.sample() for v_dist in v_proposal_dists]
        initial_potential = (sum([torch.pow(v, 2) for v in current_vs])/2.).sum()
        # Simulate trajectory for a few steps
        # following https://arxiv.org/pdf/1206.1901.pdf page 14
        # First half-step the velocities
        for p, v in zip(continuous_params, current_vs):
            v.data = v - epsilon_v * -p.grad / 2.
        for step in range(num_dynamics_steps):
            # Step positions
            for p, v in zip(continuous_params, current_vs):
                p.data = p + epsilon_p * v
            # Update grads
            for p in continuous_params:
                p.grad.zero_()

            current_score, _ = score_tree(parse_tree)
            current_score.backward(retain_graph=True)

            # Step momentum normally, except at final step.
            if step < (num_dynamics_steps - 1):
                for p, v in zip(continuous_params, current_vs):
                    v.data = v - epsilon_v * -p.grad
            else:
                # Final half momentum step and momentum flip for final energy calc
                for p, v in zip(continuous_params, current_vs):
                    v.data = - (v - epsilon_v * -p.grad / 2.)

        proposed_score = current_score
        proposed_potential = (sum([torch.pow(v, 2) for v in current_vs])/2.).sum()

        # Accept or reject
        thresh = torch.exp((initial_score - proposed_score) +
                           (initial_potential - proposed_potential))
        if dist.Bernoulli(1. - min(thresh, 1.)).sample():
            # Poses are already updated in-place
            pass
        else:
            for p, p0 in zip(continuous_params, initial_param_vals):
                p.data = torch.tensor(p0)


class ParseTreeState():
    def __init__(self, source_tree):
        self.tree = source_tree.copy()
        self.poses_by_node = {}
        for node in self.tree.nodes:
            if hasattr(node, "pose"):
                self.poses_by_node[node] = node.pose.detach().numpy().copy()

    def rebuild_original_tree(self):
        # Restore attributes
        for node in self.poses_by_node.keys():
           node.pose.data = torch.tensor(self.poses_by_node[node])
        return self.tree


def prune_node_from_tree(parse_tree, victim_node):
    # Removes the node from the tree.
    # If it's a Node: also remove its production rule
    # if the rule isn't producing anything else, or
    # if removing this node makes the rule invalid.
    # Remove all of the node's rules as well.
    # If it's a ProductionRule: remove the ProductionRule,
    # and the parent node as well if this makes it
    # infeasible.
    if isinstance(victim_node, ProductionRule):
        parent = get_node_parent_or_none(parse_tree, victim_node)
        assert(parent)  # All Rules should always have parents
        parse_tree.remove_node(victim_node)
        # Clean up the parent Node if this made it invalid
        remaining_siblings = list(parse_tree.successors(parent))
        parent_parent = get_node_parent_or_none(parse_tree, parent)
        if (torch.isinf(
                parent.score_production_rules(parent_parent, remaining_siblings))):
            # This will fall down into the next case, which
            # is removing a Node.
            victim_node = parent

    if isinstance(victim_node, Node):
        parent = get_node_parent_or_none(parse_tree, victim_node)
        # Remove all child rules
        child_rules = list(parse_tree.successors(victim_node))
        for child_rule in child_rules:
            parse_tree.remove_node(child_rule)
        parse_tree.remove_node(victim_node)
        # Clean up the parent rule if this made it invalid
        if parent is not None:
            remaining_siblings = list(parse_tree.successors(parent))
            parent_parent = get_node_parent_or_none(parse_tree, parent)
            if (len(remaining_siblings) == 0 or
                np.isinf(parent.score_products(parent_parent, remaining_siblings))):
                parse_tree.remove_node(parent)


def guess_parse_tree_from_yaml(yaml_env, outer_iterations=10, ax=None):
    # Build an initial parse tree.
    parse_tree = nx.DiGraph()
    parse_tree.add_node(Table()) # Root node
    for terminal_node in terminal_nodes_from_yaml(yaml_env):
        parse_tree.add_node(terminal_node)

    for outer_k in range(outer_iterations):
        original_parse_tree_state = ParseTreeState(parse_tree)
        score, scores_by_node = score_tree(parse_tree)
        print("Starting iter %d at score %f" % (outer_k, score))

        if not torch.isinf(score):
            # Take random nodes (taking the less likely nodes more
            # frequently) and remove them.
            for remove_k in range(np.random.geometric(p=0.5)):
                removable_nodes = [
                    n for n in parse_tree.nodes if
                    (not isinstance(n, RootNode) and not isinstance(n, TerminalNode))
                ]
                if len(removable_nodes) == 0:
                    break
                scores_raw = torch.tensor([-scores_by_node[n] for n in removable_nodes])
                # Normalize by subtracting off log(sum(exp(possible_parent_nodes_scores_raw)))
                # See https://en.wikipedia.org/wiki/LogSumExp
                maxval = scores_raw.max()
                normalization_factor = maxval + torch.log(torch.exp(scores_raw - maxval).sum())
                scores_normed = torch.exp(scores_raw - normalization_factor)
                ind = dist.Categorical(scores_normed).sample()
                print("Pruning node ", removable_nodes[ind])
                prune_node_from_tree(parse_tree, removable_nodes[ind])

        repaired_score = repair_parse_tree_in_place(parse_tree, ax=None)
        if torch.isinf(repaired_score):
            print("\tRejecting due to failure to find a feasible repair.")
            parse_tree = original_parse_tree_state.rebuild_original_tree()
        else:
            optimize_parse_tree_hmc_in_place(parse_tree, ax=None)
            new_score, _ = score_tree(parse_tree)
            # Accept probability based on ratio of old score and current score
            accept_prob = min(1., torch.exp(new_score - score))
            if not dist.Bernoulli(accept_prob).sample():
                print("\tRejected step to score %f" % new_score)
                # TODO(gizatt) In-place mutation of trees that can't be deepcopied (due to torch tensor stuff)
                # has led to disgusting code here...
                # TODO(gizatt) This doesn't deal with the issue of poses getting updated
                # in the loop. Tree-cloning might be good to slip in first.
                parse_tree = original_parse_tree_state.rebuild_original_tree()

        score, _ = score_tree(parse_tree)
        ax.clear()
        draw_parse_tree(parse_tree, ax=ax)
        print("\tEnding iter %d at score %f" % (outer_k, score))
        plt.pause(1E-3)

    return parse_tree, score


if __name__ == "__main__":
    seed = 47
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
        parse_tree = trace.nodes["_RETURN"]["value"]
        end = time.time()

        print(bcolors.OKGREEN, "Generated data in %f seconds." % (end - start), bcolors.ENDC)
        print("Full trace values:" )
        for node_name in trace.nodes.keys():
            if node_name in ["_INPUT", "_RETURN"]:
                continue
            print(node_name, ": ", trace.nodes[node_name]["value"].detach().numpy())

        # Recover and print the parse tree
        plt.subplot(2, 1, 1)
        plt.gca().clear()
        draw_parse_tree(parse_tree)
        plt.xlim(-0.2, 1.2)
        plt.ylim(-0.2, 1.2)
        score, score_by_node = score_tree(parse_tree)
        print("Score by node: ", score_by_node)
        yaml_env = convert_tree_to_yaml_env(parse_tree)
        print("Our score: %f" % score)
        print("Trace score: %f" % trace.log_prob_sum())
        assert(abs(score - trace.log_prob_sum()) < 0.001)

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
            draw_parse_tree(parse_tree, label_name=False, label_score=False, alpha=0.25)
            plt.title("Generated scene with parse tree")
            plt.pause(0.5)
        except Exception as e:
            print("Exception ", e)
        except:
            print(bcolors.FAIL, "Caught ????, probably sim fault due to weird geometry.", bcolors.ENDC)

        plt.figure()
        for k in range(9):
            # And then try to parse it
            ax = plt.subplot(3, 3, k+1)
            plt.xlim(-0.2, 1.2)
            plt.ylim(-0.2, 1.2)

            guessed_parse_tree, score = guess_parse_tree_from_yaml(yaml_env, ax=plt.gca())

            print(guessed_parse_tree.nodes, guessed_parse_tree.edges)
            plt.title("Guessed parse tree with score %f" % score)
            plt.gca().clear()
            draw_parse_tree(guessed_parse_tree)
            plt.pause(1E-3)
        plt.show()
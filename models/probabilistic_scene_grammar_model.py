from __future__ import print_function
from copy import deepcopy
from functools import partial
import multiprocessing
import time
import random
import sys
import traceback
import yaml
import weakref

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import pydrake
from pydrake.multibody.inverse_kinematics import InverseKinematics
import torch
import torch.distributions.constraints as constraints
import pyro
import pyro.distributions as dist
from pyro import poutine
import torch.multiprocessing as mp
from multiprocessing.managers import SyncManager

from scene_generation.data.dataset_utils import (
    DrawYamlEnvironmentPlanar, DrawYamlEnvironment, DrawYamlEnvironmentPlanarForTableSettingPretty, ProjectEnvironmentToFeasibility,
    BuildMbpAndSgFromYamlEnvironment)
from scene_generation.models.probabilistic_scene_grammar_nodes import *
from scene_generation.models.probabilistic_scene_grammar_nodes_place_setting import *

from collections import Mapping, Set, Sequence

non_recurse_types = (str, int, bool, float, type)
def rebuild_object_recursively_with_detach(obj, verbose=False):
    if verbose:
        print("Iter to type ", type(obj), ": ", obj)
    if isinstance(obj, torch.Tensor):
        obj = obj.detach()
    # But still recurse into it, in case it has child attributes
    if isinstance(obj, non_recurse_types):
        return obj
    elif isinstance(obj, dict):
        new_dict = dict()
        for child_key, child_value in obj.items():
            new_key = rebuild_object_recursively_with_detach(child_key, verbose=verbose)
            new_value = rebuild_object_recursively_with_detach(child_value, verbose=verbose)
            new_dict[new_key] = new_value
        return new_dict
    elif isinstance(obj, list) or isinstance(obj, tuple):
        out_iterable = []
        for child in obj:
            out_iterable.append(rebuild_object_recursively_with_detach(child, verbose=verbose))
        return type(obj)(out_iterable)
    elif hasattr(obj, "__dict__"):
        for key in vars(obj).keys():
            child = getattr(obj, key)
            rebuilt_child = rebuild_object_recursively_with_detach(child, verbose=verbose)
            if child is not rebuilt_child:
                # Doing this as seldomly as possible makes it less likely that
                # we try to reassign and un-assignable attribute.
                setattr(obj, key, rebuilt_child)
        return obj
    elif isinstance(obj, weakref.ref) and obj() is not None:
        return weakref.ref(rebuild_object_recursively_with_detach(obj(), verbose=verbose))
    else:
        if verbose:
            print("Warning, detach is skipping object ", obj, " of type ", type(obj))
        return obj
    raise NotImplementedError("SHOULD NOT GET HERE")

class ParseTree(nx.DiGraph):
    def __init__(self):
        self.global_variable_store = GlobalVariableStore()
        nx.DiGraph.__init__(self)

    def copy(self):
        ''' Copy of topology, but reference of gvs. '''
        new_copy = nx.DiGraph.copy(self)
        new_copy.global_variable_store = self.global_variable_store
        return new_copy

    def get_global_variable_store(self):
        return self.global_variable_store

    def get_total_log_prob(self, assert_rooted=True, include_observed=True):
        ''' Sum the log probabilities over the tree:
        For every node, score its set of its production rules.
        For every production rule, score its products.
        If the tree is infeasible / ill-posed, will return -inf.

        TODO(gizatt): My tree-building here echos a lot of the
        machinery from within pyro's execution trace checking,
        to the degree that I sanity check in the tests down below
        by comparing a pyro-log-prob-sum of a forward run of the model
        to the value I compute here. Can I better use Pyro's machinery
        to do less work?'''
        all_scores = []
        scores_by_node = {}

        active_global_var_names = []
        for node in self.nodes:
            parent = self.get_node_parent_or_none(node)
            active_global_var_names += node.get_global_variable_names()
            if isinstance(node, Node):
                # Sanity-check feasibility
                if parent is None and assert_rooted and not isinstance(node, RootNode):
                    node_score = torch.tensor(-np.inf, dtype=torch.double)
                elif isinstance(node, TerminalNode):
                    # TODO(gizatt): Eventually, TerminalNodes
                    # may want to score their generated parameters.
                    node_score = torch.tensor(0., dtype=torch.double)
                else:
                    # Score the kids
                    node_score = node.score_production_rules(parent, list(self.successors(node)))
            elif isinstance(node, ProductionRule):
                products = list(self.successors(node))
                terminal_product_mask = [isinstance(p, TerminalNode) for p in products]
                if any(terminal_product_mask):
                    assert(all(terminal_product_mask))
                    if include_observed:
                        node_score = node.score_products(parent, products)
                    else:
                        node_score = torch.tensor(0.)
                else:
                    node_score = node.score_products(parent, products)
            else:
                raise ValueError("Invalid node type in tree: ", type(node))
            scores_by_node[node] = node_score
            all_scores.append(node_score)

        total_score = (torch.stack(all_scores).sum() +
                       self.global_variable_store.get_total_log_prob(
                        active_global_var_names))
        return total_score, scores_by_node

    def get_node_parent_or_none(self, node):
        parents = list(self.predecessors(node))
        if len(parents) == 0:
            return None
        elif len(parents) == 1:
            return parents[0]
        else:
            print("Bad parse tree: ", self)
            print("Node: ", node)
            print("Parents: ", parents)
            raise NotImplementedError("> 1 parent --> bad parse tree")

    def is_feasible(self, base_environment_type):
        yaml_env = convert_tree_to_yaml_env(self)
        if yaml_env["n_objects"] == 0:
            # Trivial short-circuit that avoids MBP barfing on empty environments
            return True
        builder, mbp, scene_graph, q0 = BuildMbpAndSgFromYamlEnvironment(
            yaml_env, base_environment_type)
        diagram = builder.Build()

        diagram_context = diagram.CreateDefaultContext()
        mbp_context = diagram.GetMutableSubsystemContext(
            mbp, diagram_context)

        ik = InverseKinematics(mbp, mbp_context)
        q_dec = ik.q()

        constraint_binding = ik.AddMinimumDistanceConstraint(0.001)
        try:
            return constraint_binding.evaluator().CheckSatisfied(q0)
        except Exception as e:
            print("Unhandled except in feasibility checking: ", e)
            return False
        except:
            print("Unhandled non-Exception problem in feasibility checking.")
            return False

def generate_unconditioned_parse_tree(initial_gvs=None, root_node=Table()):
    input_nodes_with_parents = [ (None, root_node) ]  # (parent, node) order
    parse_tree = ParseTree()
    if initial_gvs is not None:
        parse_tree.global_variable_store = initial_gvs
    parse_tree.add_node(root_node)
    while len(input_nodes_with_parents)>  0:
        parent, node = input_nodes_with_parents.pop(0)
        if isinstance(node, TerminalNode):
            # Nothing more to do with this node
            pass
        else:
            # Expand by picking a production rule
            node.sample_global_variables(parse_tree.get_global_variable_store())
            production_rules = node.sample_production_rules(parent)
            for i, rule in enumerate(production_rules):
                parse_tree.add_node(rule)
                parse_tree.add_edge(node, rule)
                rule.sample_global_variables(parse_tree.get_global_variable_store())
                new_nodes = rule.sample_products(node)
                for new_node in new_nodes:
                    parse_tree.add_node(new_node)
                    parse_tree.add_edge(rule, new_node)
                    input_nodes_with_parents.append((rule, new_node))

    return parse_tree

def resample_parse_tree_to_feasibility(old_parse_tree, base_environment_type, max_num_iters=1000):
    is_feasible = False
    num_iters = 0
    while not is_feasible and num_iters < max_num_iters:
        # Build tree into MBP and check feasibility
        is_feasible = old_parse_tree.is_feasible(base_environment_type)
        if not is_feasible:
            # Find the original parent node
            root_node = list(old_parse_tree.nodes)[0]
            while len(list(old_parse_tree.predecessors(root_node))) > 0:
                root_node = old_parse_tree.predecessors(root_node)[0]

            # Rebuild a parse tree with the same structure, but resampled nodes
            input_nodes_with_old_nodes = [ (root_node, root_node) ]  # (node, old_version_of_node) order
            new_parse_tree = ParseTree()
            new_parse_tree.global_variable_store = old_parse_tree.global_variable_store
            new_parse_tree.add_node(root_node)
            while len(input_nodes_with_old_nodes)>  0:
                new_node, old_node = input_nodes_with_old_nodes.pop(0)
                if isinstance(new_node, TerminalNode):
                    # Nothing more to do with this node
                    pass
                else:
                    # Get the old rules...
                    production_rules = old_parse_tree.successors(old_node)
                    # And add them to the new tree, resampling the produced nodes.
                    for i, rule in enumerate(production_rules):
                        new_parse_tree.add_node(rule)
                        new_parse_tree.add_edge(new_node, rule)
                        rule.sample_global_variables(new_parse_tree.get_global_variable_store())
                        new_child_nodes = rule.sample_products(new_node)
                        old_child_nodes = list(old_parse_tree.successors(rule))
                        # If the rule has an inconsistent number (or ordering) of children, we can't resample like this.
                        assert(len(new_child_nodes) == len(old_child_nodes))
                        for new_child_node, old_child_node in zip(new_child_nodes, old_child_nodes):
                            new_parse_tree.add_node(new_child_node)
                            new_parse_tree.add_edge(rule, new_child_node)
                            input_nodes_with_old_nodes.append((new_child_node, old_child_node))
            old_parse_tree = new_parse_tree
            num_iters += 1
    if num_iters == max_num_iters:
        raise NotImplementedError("Ran out of resampling iterations!")
    return old_parse_tree

def project_parse_tree_to_feasibility(old_parse_tree, base_environment_type, make_nonpenetration=True, make_static=False):
    # Build tree into MBP and check feasibility
    is_feasible = old_parse_tree.is_feasible(base_environment_type)
    if is_feasible:
        return old_parse_tree

    # Do feasibility projection
    yaml_env = convert_tree_to_yaml_env(old_parse_tree)
    new_yaml_env = ProjectEnvironmentToFeasibility(yaml_env, base_environment_type=base_environment_type,
        make_nonpenetration=make_nonpenetration, make_static=make_static)[-1]



def generate_hyperexpanded_parse_tree(root_node = Table()):
    # Make a fully expanded parse tree where
    # *every possible* non-terminal production rule and product is followed.
    input_nodes_with_parents = [ (None, root_node) ]  # (parent, node) order
    parse_tree = ParseTree()
    parse_tree.add_node(root_node)
    while len(input_nodes_with_parents)>  0:
        parent, node = input_nodes_with_parents.pop(0)
        if isinstance(node, TerminalNode):
            # Nothing more to do with this node
            pass
        else:
            # Activate all production rules.
            node.sample_global_variables(parse_tree.get_global_variable_store())
            for i, rule in enumerate(node.production_rules):
                # Always at least get the global vars for this rule established.
                rule.sample_global_variables(parse_tree.get_global_variable_store())
                # But don't actually expand into terminal nodes -- there are too many.
                if all([issubclass(prod, TerminalNode) for prod in rule.product_types]):
                    continue
                parse_tree.add_node(rule)
                parse_tree.add_edge(node, rule)
                new_nodes = rule.sample_products(node)
                for new_node in new_nodes:
                    if isinstance(new_node, TerminalNode):
                        continue
                    parse_tree.add_node(new_node)
                    parse_tree.add_edge(rule, new_node)
                    input_nodes_with_parents.append((rule, new_node))
    return parse_tree

def convert_tree_to_yaml_env(parse_tree):
    terminal_nodes = []
    for node in parse_tree:
        if isinstance(node, TerminalNode):
            terminal_nodes.append(node)
    env = {"n_objects": len(terminal_nodes)}
    for k, node in enumerate(terminal_nodes):
        env["obj_%04d" % k] = node.generate_yaml()
    env["global_variable_store"] = parse_tree.get_global_variable_store().generate_yaml()
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


def remove_production_rules_from_parse_tree(parse_tree):
    ''' For drawing '''
    new_tree = nx.DiGraph()
    for node in parse_tree:
        if isinstance(node, Node):
            new_tree.add_node(node)
    for node in parse_tree:
        if isinstance(node, ProductionRule):
            parent = parse_tree.get_node_parent_or_none(node)
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

def score_terminal_node_productions(parse_tree):
    total_score = torch.tensor(0.)
    for node in parse_tree.nodes:
        if isinstance(node, ProductionRule):
            parent = parse_tree.get_node_parent_or_none(node)
            assert(parent is not None)
            children = list(parse_tree.successors(node))
            mask = [isinstance(child, TerminalNode) for child in children]
            if any(mask):
                # No mixed productions of Terminal + Non-Terminal, for easier scoring.
                assert(all(mask))
                total_score += node.score_products(parent, children)
    return total_score

def draw_parse_tree(parse_tree, ax=None, label_score=False, label_name=False, node_class_to_color_dict={}, **kwargs):
    pruned_tree = remove_production_rules_from_parse_tree(parse_tree)

    if label_score:
        score, scores_by_node = parse_tree.get_total_log_prob()
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
    if label_score:
        colors = np.array([max(-1000., scores_by_node[node].item()) for node in pruned_tree.nodes]) 
        colors -= min(colors)
        colors /= max(colors)
    elif len(node_class_to_color_dict.keys()) > 0:
        colors = []
        for node in pruned_tree.nodes:
            if node.__class__.__name__ in node_class_to_color_dict.keys():
                colors.append(node_class_to_color_dict[node.__class__.__name__])
            else:
                colors.append([1., 0., 0.])
        colors = np.array(colors)
    else:
        colors = None
    # Convert scores to colors
    if ax is None:
        ax = plt.gca()

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    nx.draw_networkx(pruned_tree, ax=ax, pos=pos_dict, labels=label_dict,
                     node_color=colors, cmap='jet', font_weight='bold', **kwargs)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if label_score:
        ax.set_title("Score: %f" % score)


def repair_parse_tree_in_place(parse_tree, candidate_intermediate_nodes,
                               max_num_iters=100, ax=None, verbose=False):
    iter_k = 0
    while iter_k < max_num_iters:
        score, scores_by_node = parse_tree.get_total_log_prob(assert_rooted=True)
        if verbose:
            print("At start of iter %d, tree score is %f" % (iter_k, score))
        if ax is not None:
            ax.clear()
            DrawYamlEnvironmentPlanar(yaml_env, base_environment_type="table_setting", ax=ax)
            draw_parse_tree(parse_tree, label_name=True, label_score=True, ax=ax, alpha=0.75)
            plt.title("Iter %02d: score %f" % (iter_k, score.item()))
            plt.pause(0.1)
            plt.savefig('iter_%02d.png' % iter_k)
        
        # Find the currently-infeasible nodes.
        infeasible_nodes = [key for key in scores_by_node.keys() if torch.isinf(scores_by_node[key])]

        # Assert that they're all infeasible Nodes. Infeasible rules
        # should never appear by construction.
        for node in infeasible_nodes:
            # TODO(gizatt) I could conceivably just handle productionrules
            # as a special case. They can be crammed into the tree the
            # same way -- iterate over all nodes and see how they fit into
            # the available products.
            # TODO(gizatt) Indeed, I'll *need* to support this for trees with
            # AND nodes to ever be inferable.
            if not isinstance(node, Node):
                print("Found infeasible production rule %s: " % node.name, node)
                print("Full parse tree: ", parse_tree)
                print("Nodes: ", list(parse_tree.nodes))
                raise NotImplementedError()

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
                    parent = parse_tree.get_node_parent_or_none(node)
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
                for new_node in candidate_intermediate_nodes:
                    if new_node in parse_tree.nodes:
                        continue
                    if isinstance(new_node, AndNode) and len(new_node.production_rules) > 1:
                        print("WARNING: inference will never succeed, as only one prod rule is added at a time.")
                    for rule in new_node.production_rules:
                        rule.sample_global_variables(parse_tree.get_global_variable_store())
                        score = rule.score_products(new_node, sampled_nodes) + new_node.score_production_rules(None, [rule])
                        if not torch.isinf(score):
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
    return parse_tree.get_total_log_prob()[0]

def optimize_parse_tree_hmc_in_place(parse_tree, ax=None, verbose=False):

    # Run HMC on the poses of the place settings for a few steps.
    continuous_params = []
    v_proposal_dists = []
    num_hmc_steps = 10
    num_dynamics_steps = 10
    epsilon_v = 5E-3
    epsilon_p = 5E-3
    proposal_variance = 0.1
    for node in parse_tree:
        if isinstance(node, PlaceSetting):
            node.pose.requires_grad = True
            continuous_params.append(node.pose)
            v_proposal_dists.append(dist.Normal(torch.zeros(3), torch.ones(3)*proposal_variance))
    if len(v_proposal_dists) == 0:
        return
    for hmc_step in range(num_hmc_steps):
        if ax is not None:
            ax.clear()
            draw_parse_tree(parse_tree)
            plt.pause(0.1)
        initial_score, _ = parse_tree.get_total_log_prob()
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

            current_score, _ = parse_tree.get_total_log_prob()
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
        if (not np.isnan(thresh.detach().numpy()) and
                dist.Bernoulli(1. - max(min(thresh, 1.), 0.)).sample()):
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
        parent = parse_tree.get_node_parent_or_none(victim_node)
        assert(parent)  # All Rules should always have parents
        parse_tree.remove_node(victim_node)
        # Clean up the parent Node if this made it invalid
        remaining_siblings = list(parse_tree.successors(parent))
        parent_parent = parse_tree.get_node_parent_or_none(parent)
        if (torch.isinf(
                parent.score_production_rules(parent_parent, remaining_siblings))):
            # This will fall down into the next case, which
            # is removing a Node.
            victim_node = parent

    if isinstance(victim_node, Node):
        parent = parse_tree.get_node_parent_or_none(victim_node)
        # Remove all child rules
        child_rules = list(parse_tree.successors(victim_node))
        for child_rule in child_rules:
            parse_tree.remove_node(child_rule)
        parse_tree.remove_node(victim_node)
        # Clean up the parent rule if this made it invalid
        if parent is not None:
            remaining_siblings = list(parse_tree.successors(parent))
            parent_parent = parse_tree.get_node_parent_or_none(parent)
            if (len(remaining_siblings) == 0 or
                np.isinf(parent.score_products(parent_parent, remaining_siblings))):
                parse_tree.remove_node(parent)


def guess_parse_tree_from_yaml(yaml_env, guide_gvs=None, outer_iterations=2, num_attempts=2, ax=None, verbose=False, root_node_type=Table):
    best_tree = None
    best_score = -np.inf

    for attempt in range(num_attempts):
        # Build an initial parse tree.
        # Collect all possible non-terminal intermediate nodes
        hyper_parse_tree = generate_hyperexpanded_parse_tree(root_node=root_node_type())
        if guide_gvs is not None:
            hyper_parse_tree.global_variable_store = guide_gvs
        candidate_intermediate_nodes = []
        for node in hyper_parse_tree:
            if isinstance(node, Node) and hyper_parse_tree.get_node_parent_or_none(node) is not None:
                candidate_intermediate_nodes.append(node)

        parse_tree = ParseTree()
        parse_tree.global_variable_store = hyper_parse_tree.global_variable_store
        parse_tree.add_node(root_node_type()) # Root node
        for terminal_node in terminal_nodes_from_yaml(yaml_env):
            parse_tree.add_node(terminal_node)
        for outer_k in range(outer_iterations):
            original_parse_tree_state = ParseTreeState(parse_tree)
            score, scores_by_node = parse_tree.get_total_log_prob()
            if verbose:
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
                    if verbose:
                        print("Pruning node ", removable_nodes[ind])
                    prune_node_from_tree(parse_tree, removable_nodes[ind])

            repaired_score = repair_parse_tree_in_place(
                parse_tree, candidate_intermediate_nodes,
                ax=ax, verbose=verbose)
            if torch.isinf(repaired_score):
                if verbose:
                    print("\tRejecting due to failure to find a feasible repair.")
                parse_tree = original_parse_tree_state.rebuild_original_tree()
            else:
                optimize_parse_tree_hmc_in_place(parse_tree, ax=None, verbose=verbose)
                new_score, _ = parse_tree.get_total_log_prob()
                # Accept probability based on ratio of old score and current score
                accept_prob = min(1., torch.exp(new_score - score))
                if not dist.Bernoulli(accept_prob).sample():
                    if verbose:
                        print("\tRejected step to score %f" % new_score)
                    # TODO(gizatt) In-place mutation of trees that can't be deepcopied (due to torch tensor stuff)
                    # has led to disgusting code here...
                    # TODO(gizatt) This doesn't deal with the issue of poses getting updated
                    # in the loop. Tree-cloning might be good to slip in first.
                    parse_tree = original_parse_tree_state.rebuild_original_tree()

            score, _ = parse_tree.get_total_log_prob()
            if ax is not None:
                ax.clear()
                DrawYamlEnvironmentPlanar(yaml_env, base_environment_type="table_setting", ax=ax)
                draw_parse_tree(parse_tree, label_name=True, label_score=True, ax=ax, alpha=0.75)
                plt.pause(0.1)
            if verbose:
                print("\tEnding iter %d at score %f" % (outer_k, score))
        if score > best_score:
            best_tree = parse_tree
            best_score = score
        if verbose:
            print("\tEnding attempt %d at best score %f" % (attempt, best_score))

    return best_tree, best_score

def worker(i, env, guide_gvs, outer_iterations, num_attempts, output_queue, synchro_prims):
    try:
        torch.set_default_tensor_type(torch.DoubleTensor)
        done_event,  = synchro_prims
        tree, score = guess_parse_tree_from_yaml(
            env, guide_gvs=guide_gvs, 
            outer_iterations=outer_iterations,
            num_attempts=num_attempts, verbose=False)
        print("Finished tree with score %f" % score)
        # Detach all values in the tree so it can be communicated IPC
        tree = rebuild_object_recursively_with_detach(tree)
        for key in pyro.get_param_store().keys():
            pyro.get_param_store()._params[key].requires_grad = False
            pyro.get_param_store()._params[key].grad = None
        output_queue.put((i, tree))
        done_event.wait()
    except Exception as e:
        print("Parse tree guessing thread had exception: ", e)
        traceback.print_exc()
        output_queue.put((i, None))
        done_event.wait()

def guess_parse_trees_batch_async(envs, guide_gvs=None, outer_iterations=2, num_attempts=2):
    processes = []
    mp.set_start_method('spawn')
    output_queue = mp.SimpleQueue()
    done_event = mp.Event()
    synchro_prims = [done_event]
    n = len(envs)
    parse_trees = [None]*n
    for i, env in enumerate(envs):
        p = mp.Process(
            target=worker, args=(
                i, env, guide_gvs, outer_iterations, num_attempts, output_queue, synchro_prims))
        p.start()
        processes.append(p)
    for k in range(n):
        i, parse_tree = output_queue.get()
        parse_trees[i] = parse_tree
    done_event.set()
    for p in processes:
        p.join()
    return parse_trees

if __name__ == "__main__":
    #seed = 52
    #torch.manual_seed(seed)
    #np.random.seed(seed)
    #random.seed(seed)
    pyro.enable_validation(True)

    noalias_dumper = yaml.dumper.SafeDumper
    noalias_dumper.ignore_aliases = lambda self, data: True
    
    # Draw + plot a few generated environments and their trees
    plt.figure().set_size_inches(20, 20)
    for k in range(9):
        start = time.time()
        pyro.clear_param_store()
        trace = poutine.trace(generate_unconditioned_parse_tree).get_trace()
        parse_tree = trace.nodes["_RETURN"]["value"]
        end = time.time()

        print(bcolors.OKGREEN, "Generated data in %f seconds." % (end - start), bcolors.ENDC)
        #print("Full trace values:" )
        #for node_name in trace.nodes.keys():
        #    if node_name in ["_INPUT", "_RETURN"]:
        #        continue
        #    print(node_name, ": ", trace.nodes[node_name]["value"].detach().numpy())

        # Recover and print the parse tree
        plt.subplot(3, 3, k+1)
        plt.gca().clear()
        plt.gca().set_xlim(0.1, 0.9)
        plt.gca().set_ylim(0.1, 0.9)
        score, score_by_node = parse_tree.get_total_log_prob()
        #print("Score by node: ", score_by_node)

        # Resample it to feasibility
        #parse_tree = resample_parse_tree_to_feasibility(parse_tree, base_environment_type="table_setting")
        yaml_env = convert_tree_to_yaml_env(parse_tree)
        yaml_env = ProjectEnvironmentToFeasibility(yaml_env, base_environment_type="table_setting", make_nonpenetrating=True, make_static=False)[-1]
        DrawYamlEnvironmentPlanarForTableSettingPretty(yaml_env, ax=plt.gca())
        node_class_to_color_dict = {"Table":[0., 1., 0.], "PlaceSetting":[0., 0., 1.]}
        draw_parse_tree(parse_tree, label_name=False, label_score=False, alpha=0.25,
                        node_class_to_color_dict=node_class_to_color_dict)
        print("Our score: %f" % score)
        #print("Trace score: %f" % trace.log_prob_sum())
        plt.gca().set_xlim(0.1, 0.9)
        plt.gca().set_ylim(0.1, 0.9)
        assert(abs(score - trace.log_prob_sum()) < 0.001)
    plt.show()
    sys.exit(0)


    hyper_parse_tree = generate_hyperexpanded_parse_tree()
    guide_gvs = hyper_parse_tree.get_global_variable_store()
    for var_name in guide_gvs.keys():
        guide_gvs[var_name][0] = pyro.param(var_name + "_est",
                                            guide_gvs[var_name][0],
                                            constraint=guide_gvs[var_name][1].support)
    #print("Left fork var est: ", guide_gvs["place_setting_plate_var"][0])
    #guide_gvs["place_setting_plate_var"][0] *= 50

    start = time.time()
    pyro.clear_param_store()
    trace = poutine.trace(generate_unconditioned_parse_tree).get_trace()
    parse_tree = trace.nodes["_RETURN"]["value"]
    end = time.time()

    # Save out the param store to "nominal"
    pyro.get_param_store().save("place_setting_nominal_param_store.pyro")

    plt.figure().set_size_inches(15, 10)
    for k in range(200):
        start = time.time()
        pyro.clear_param_store()
        trace = poutine.trace(generate_unconditioned_parse_tree).get_trace(guide_gvs)
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
        score, score_by_node = parse_tree.get_total_log_prob()
        print("Score by node: ", score_by_node)
        yaml_env = convert_tree_to_yaml_env(parse_tree)
        print("Our score: %f" % score)
        print("Trace score: %f" % trace.log_prob_sum())
        #assert(abs(score - trace.log_prob_sum()) < 0.001)

        #with open("table_setting_environments_generated_simple.yaml", "a") as file:
        #    yaml.dump({"env_%d" % int(round(time.time() * 1000)): yaml_env}, file, Dumper=noalias_dumper)

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
#
        plt.show()
        #sys.exit(0)
        plt.pause(1E-3)
        #plt.figure()
        #for k in range(3):
        #    # And then try to parse it
        #    ax = plt.subplot(3, 1, k+1)
        #    plt.xlim(-0.2, 1.2)
        #    plt.ylim(-0.2, 1.2)
###
        #    guessed_parse_tree, score = guess_parse_tree_from_yaml(yaml_env, outer_iterations=2, ax=plt.gca())
###
        #    print(guessed_parse_tree.nodes, guessed_parse_tree.edges)
        #    plt.title("Guessed parse tree with score %f" % score)
        #    plt.gca().clear()
        #    draw_parse_tree(guessed_parse_tree)
        #    plt.pause(1E-3)
###
        #plt.show()
        #plt.pause(0.1)
        #trace = poutine.trace(model.model(observed_tree=guessed_parse_tree)).get_trace()
        #print("Trace log prob sum: %f" % trace.log_prob_sum())
        #print("full trace values: ")
        #for node_name in trace.nodes.keys():
        #    if node_name in ["_INPUT", "_RETURN"]:
        #        continue
        #    print(node_name, ": ", trace.nodes[node_name]["value"].detach().numpy())
        
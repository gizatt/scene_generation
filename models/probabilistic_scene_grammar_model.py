from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import time

import torch
import torch.distributions.constraints as constraints
import pyro
import pyro.distributions as dist
from pyro import poutine

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


class TerminalNode(object):
    def __init__(self):
        Node.__init__(self, [], [])


class PlaceSetting(ExhaustiveSetNode):
    def __init__(self, pose):
        production_rules = [ self._produce_place_setting,
                             self._produce_dish_and_cup ]
        ExhaustiveSetNode.__init__(self, "place_setting", production_rules)
        self.pose = pose

    def _produce_place_setting(self, site_prefix):
        return [Dish(self.pose)]

    def _produce_dish_and_cup(self, site_prefix):
        return [Dish(self.pose), Cup(self.pose + torch.tensor([0., 0.1, 0.]))]


class Dish(TerminalNode):
    def __init__(self, pose):
        self.pose = pose

class Cup(TerminalNode):
    def __init__(self, pose):
        self.pose = pose

class RootNode(OrNode):
    def __init__(self):
        production_rules = [ self._produce_one_place_setting,
                             self._produce_two_place_settings]
        production_weights = torch.tensor([ 0.75, 0.25 ])
        OrNode.__init__(self, production_rules, production_weights)

    def _produce_one_place_setting(self, site_prefix):
        return [PlaceSetting(pose=torch.tensor([0., 0., 0.]))]

    def _produce_two_place_settings(self, site_prefix):
        return [PlaceSetting(pose=torch.tensor([0., 0., 0.])),
                PlaceSetting(pose=torch.tensor([1., 0., 0.]))]


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

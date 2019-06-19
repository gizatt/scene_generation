from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import time

import torch
import pyro
import pyro.distributions as dist
from pyro import poutine


class ProductionRule(object):
    def __init__(self):
        raise NotImplementedError()
    def __call__(self, *args):
        return [x(*args) for x in self.products]

class AndRule(ProductionRule):
    def __init__(self, *args):
        self.products = args


class Node(object):
    def __init__(self, production_rules, production_weights):
        if len(production_weights) != len(production_rules):
            raise ValueError("# of production rules and weights must match.")

        self.production_rules = production_rules
        if len(self.production_rules) > 0:
            self.production_dist = dist.Categorical(production_weights)
        else:
            self.production_dist = None

        self.attributes = []

    def sample_production_rule(self, site_name, obs=None):
        if not self.production_dist:
            raise ValueError("Trying to sample from a terminal node.")
        sampled_rule = pyro.sample(site_name, self.production_dist, obs=obs)
        return self.production_rules[sampled_rule](*self.attributes)

class TerminalNode(Node):
    def __init__(self):
        Node.__init__(self, [], [])

class PlaceSetting(Node):
    def __init__(self):
        production_rules = [ AndRule(Dish), AndRule(Dish, Cup) ]
        production_weights = torch.tensor([0.2, 0.8])
        Node.__init__(self, production_rules, production_weights)

class Dish(TerminalNode):
    pass

class Cup(TerminalNode):
    pass

class RootNode(Node):
    def __init__(self):
        production_rules = [ AndRule(PlaceSetting), AndRule(PlaceSetting, PlaceSetting) ]
        production_weights = torch.tensor([ 0.75, 0.25 ])
        Node.__init__(self, production_rules, production_weights)

class ProbabilisticSceneGrammarModel():


    def __init__(self):
        pass

    def model(self, data=None):
        nodes = [RootNode()]
        num_productions = 0
        iter_k = 0
        while len(nodes) > 0:
            node = nodes.pop(0)
            if isinstance(node, TerminalNode):
                # Instantiate
                print("Instantiating terminal node: ", node)
            else:
                print("Expanding non-terminal node ", node)
                # Expand by picking a production rule
                new_node_classes = node.sample_production_rule("rule_%04d" % num_productions)
                [nodes.append(x) for x in new_node_classes]
                num_productions += 1
            iter_k += 1

        return pyro.sample("test", dist.Normal(0., 1.))

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

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

class GlobalVariableStore():
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.store = {}
    
    def sample_global_variable(self, name, dist):
        if name not in self.store.keys():
            self.store[name] = [pyro.sample(name, dist), dist]
        return self.store[name][0]
    
    def keys(self):
        return self.store.keys()

    def __getitem__(self, key):
        if key not in self.store.keys():
            raise ValueError("%s not in global variable store" % key)
        else:
            return self.store[key]
    
    def get_total_log_prob(self, names=None):
        if names is None:
            names = self.store.keys()
        total_score = torch.tensor(0., dtype=torch.double)
        for key in set(names):
            value, dist = self.store[key]
            total_score += dist.log_prob(value.double()).double()
        return total_score

    def generate_yaml(self):
        out = {}
        for key in self.store.keys():
            out[key] = [float(x) for x in self.store[key][0].tolist()]
        return out
    
    def detach(self):
        # Create a copy of self with detached values, following Pytorch semantics
        new_gvs = GlobalVariableStore()
        for key in self.keys():
            new_gvs.store[key] = [self[key][0].detach(), self[key][1]]
        return new_gvs

class ProductionRule(object):
    ''' Abstract interface for a production rule.
    Callable to perform the production, but also
    queryable for what nodes this connects and able to
    provide scoring for whether a candidate production
    is a good idea at all. '''
    def __init__(self, product_types, name):
        self.product_types = product_types
        self.name = name
    def get_param_names(self):
        if not hasattr(self, "param_names"):
            return []
        return self.param_names
    def get_global_variable_names(self):
        if not hasattr(self, "global_variable_names"):
            return []
        return self.global_variable_names
    def sample_global_variables(self, global_variable_store):
        pass
    def sample_products(self, parent, obs_products=None):
        raise NotImplementedError()
    def score_products(self, parent, products):
        raise NotImplementedError()

class Node(object):
    def __init__(self, name):
        self.name = name
    def get_param_names(self):
        if not hasattr(self, "param_names"):
            return []
        return self.param_names
    def get_global_variable_names(self):
        if not hasattr(self, "global_variable_names"):
            return []
        return self.global_variable_names

class RootNode(Node):
    pass

class TerminalNode(Node):
    def __init__(self, name):
        Node.__init__(self, name=name)

class NonTerminalNode(Node):
    def sample_global_variables(self, global_variable_store):
        pass
    def sample_production_rules(self, parent, obs_production_rules=None):
        raise NotImplementedError()
    def score_products(self, parent, production_probs):
        raise NotImplementedError()

class OrNode(NonTerminalNode):
    def __init__(self, name, production_rules, production_weights):
        if len(production_weights) != len(production_rules):
            raise ValueError("# of production rules and weights must match.")
        if len(production_weights) == 0:
            raise ValueError("Must have nonzero number of production rules.")
        self.production_rules = production_rules
        self.production_weights = production_weights
        self.production_dist = dist.Categorical(production_weights)
        NonTerminalNode.__init__(self, name=name)

    def _recover_active_rule(self, production_rules):
        if production_rules[0] not in self.production_rules:
            print("Warning: rule not in OrNode production rules.")
        return torch.tensor(self.production_rules.index(production_rules[0]))

    def sample_production_rules(self, parent, obs_production_rules=None):
        if obs_production_rules is not None:
            active_rule = self._recover_active_rule(obs_production_rules)
        else:
            active_rule = None
        active_rule = pyro.sample(self.name + "_or_sample", self.production_dist, obs=active_rule)
        return [self.production_rules[active_rule]]

    def score_production_rules(self, parent, production_rules):
        if len(production_rules) != 1:
            return torch.tensor(-np.inf, dtype=torch.double)
        active_rule = self._recover_active_rule(production_rules)
        return self.production_dist.log_prob(active_rule).sum().double()


class AndNode(NonTerminalNode):
    def __init__(self, name, production_rules):
        if len(production_rules) == 0:
            raise ValueError("Must have nonzero # of production rules.")
        self.production_rules = production_rules
        Node.__init__(self, name=name)

    def sample_production_rules(self, parent, obs_production_rules=None):
        if obs_production_rules is not None:
            assert(obs_production_rules == self.production_rules)
        return self.production_rules

    def score_production_rules(self, parent, production_rules):
        if production_rules != self.production_rules:
            return torch.tensor(-np.inf, dtype=torch.double)
        else:
            return torch.tensor(-np.inf, dtype=torch.double)


class CovaryingSetNode(NonTerminalNode):
    @staticmethod
    def build_init_weights(num_production_rules, production_weights_hints = {},
                           remaining_weight = 1.):
        assert(remaining_weight >= 0.)
        num_combinations = 2**num_production_rules
        init_weights = torch.ones(num_combinations).double() * (remaining_weight + 1E-9)
        for hint in production_weights_hints.keys():
            val = production_weights_hints[hint]
            assert(val >= 0.)
            combination_index = 0
            for index in hint:
                assert(isinstance(index, int) and index >= 0 and
                       index < num_production_rules)
                combination_index += 2**index
            init_weights[combination_index] = val
        init_weights /= torch.sum(init_weights)
        return init_weights
        
    def __init__(self, name, production_rules, init_weights):
        ''' Make a categorical distribution over
           every possible combination of production rules
           that could be active, with a separate weight
           for each combination. (2^n weights!)

           Hints can be supplied in the form of a dictionary
           of (int tuple) : (float weight) pairs, and a float
           indicating the weight to distribute to the remaining
           pairs. These floats all indicate relative occurance
           weights. '''
        # Build the initial weights, taking the suggestion
        # weights into account.
        self.production_rules = production_rules
        self.exhaustive_set_weights = init_weights
        self.production_dist = dist.Categorical(
            logits=torch.log(self.exhaustive_set_weights / (1. - self.exhaustive_set_weights)))
        NonTerminalNode.__init__(self, name=name)

    def _recover_selected_rules(self, production_rules):
        selected_rules = torch.tensor(0)
        for rule in production_rules:
            if rule not in self.production_rules:
                print("Warning: rule not in CovaryingSetNode production rules: ", rule)
                return torch.tensor(-np.inf)
            k = self.production_rules.index(rule)
            selected_rules += 2**k
        assert(selected_rules >= 0 and selected_rules <= len(self.exhaustive_set_weights))
        return selected_rules

    def sample_production_rules(self, parent, obs_production_rules=None):
        if obs_production_rules is not None:
            selected_rules = self._recover_selected_rules(obs_production_rules)
            selected_rules = pyro.sample(
                self.name + "_exhaustive_set_sample",
                self.production_dist,
                obs=selected_rules)
            return obs_production_rules
        else:
            # Select out the appropriate rules
            selected_rules = pyro.sample(
                self.name + "_exhaustive_set_sample",
                self.production_dist)
            output = []
            for k, rule in enumerate(self.production_rules):
                if (selected_rules >> k) & 1:
                    output.append(rule)
            return output

    def score_production_rules(self, parent, production_rules):
        selected_rules = self._recover_selected_rules(production_rules)
        return self.production_dist.log_prob(selected_rules).sum().double()


class IndependentSetNode(NonTerminalNode):
    def __init__(self, name, production_rules,
                 production_probs):
        ''' Make a categorical distribution over production rules
            that could be active, where each rule occurs
            independently of the others. Each production weight
            is a probability of that rule being active. '''
        if len(production_probs) != len(production_rules):
            raise ValueError("Must have same number of production probs "
                             "as rules.")
        self.production_probs = production_probs
        self.production_dist = dist.Bernoulli(production_probs).to_event(1)
        self.production_rules = production_rules
        NonTerminalNode.__init__(self, name=name)

    def _recover_selected_rules(self, production_rules):
        selected_rules = torch.zeros(len(self.production_rules))
        for rule in production_rules:
            if rule not in self.production_rules:
                print("Warning: rule not in IndependentSetNode production rules: ", rule)
                return torch.tensor(-np.inf)
            selected_rules[self.production_rules.index(rule)] = 1
        return selected_rules

    def sample_production_rules(self, parent, obs_production_rules=None):
        if obs_production_rules is not None:
            selected_rules = self._recover_selected_rules(obs_production_rules)
            selected_rules = pyro.sample(
                self.name + "_independent_set_sample",
                self.production_dist, obs=selected_rules)
            return obs_production_rules
        else:
            selected_rules = pyro.sample(
                self.name + "_independent_set_sample",
                self.production_dist)
            # Select out the appropriate rules
            output = []
            for k, rule in enumerate(self.production_rules):
                if selected_rules[k] == 1:
                    output.append(rule)
            return output

    def score_production_rules(self, parent, production_rules):
        selected_rules = self._recover_selected_rules(production_rules)
        return self.production_dist.log_prob(selected_rules).sum().double()

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
from scene_generation.models.probabilistic_scene_grammar_nodes import *


def chain_pose_transforms(p_w1, p_12):
    ''' p_w1: xytheta Pose 1 in world frame
        p_12: xytheta Pose 2 in Pose 1's frame
        Returns: xytheta Pose 2 in world frame. '''
    out = torch.empty(3, dtype=p_w1.dtype)
    r = p_w1[2]
    out[0] = p_w1[0] + p_12[0]*torch.cos(r) - p_12[1]*torch.sin(r)
    out[1] = p_w1[1] + p_12[0]*torch.sin(r) + p_12[1]*torch.cos(r)
    out[2] = p_w1[2] + p_12[2]
    return out

def invert_pose(pose):
    # TF^-1 = [R^t  -R.' T]
    out = torch.empty(3, dtype=pose.dtype)
    r = pose[2]
    out[0] = -(pose[0]*torch.cos(-r) - pose[1]*torch.sin(-r))
    out[1] = -(pose[0]*torch.sin(-r) + pose[1]*torch.cos(-r))
    out[2] = -r
    return out


class PlaceSetting(CovaryingSetNode):

    class ObjectProductionRule(ProductionRule):
        def __init__(self, name, object_name, object_type, mean_prior_params, var_prior_params):
            self.object_name = object_name
            self.object_type = object_type
            self.mean_prior_params = mean_prior_params
            self.var_prior_params = var_prior_params
            self.global_variable_names = ["place_setting_%s_mean" % self.object_name,
                                         "place_setting_%s_var" % self.object_name]
            ProductionRule.__init__(self,
                name=name,
                product_types=[object_type])

        def _recover_rel_pose_from_abs_pose(self, parent, abs_pose):
            return chain_pose_transforms(invert_pose(parent.pose), abs_pose)

        def sample_global_variables(self, global_variable_store):
            # Handles class-general setup
            mean_prior_dist = dist.Normal(loc=self.mean_prior_params[0],
                                          scale=self.mean_prior_params[1]).to_event(1)
            var_prior_dist = dist.InverseGamma(concentration=self.var_prior_params[0],
                                               rate=self.var_prior_params[1]).to_event(1)
            mean = global_variable_store.sample_global_variable("place_setting_%s_mean" % self.object_name,
                                              mean_prior_dist).double()
            var = global_variable_store.sample_global_variable("place_setting_%s_var" % self.object_name,
                                             var_prior_dist).double()
            self.offset_dist = dist.Normal(loc=mean, scale=var).to_event(1)
            
        def sample_products(self, parent, obs_products=None):
            # Observation should be absolute position of the product
            if obs_products is not None:
                assert(len(obs_products) == 1 and isinstance(obs_products[0], self.object_type))
                obs_rel_pose = self._recover_rel_pose_from_abs_pose(parent, obs_products[0].pose)
                rel_pose = pyro.sample("%s_pose" % (self.name),
                                       self.offset_dist, obs=obs_rel_pose)
                return obs_products
            else:
                rel_pose = pyro.sample("%s_pose" % (self.name), self.offset_dist).detach()
                abs_pose = chain_pose_transforms(parent.pose, rel_pose)
                return [self.object_type(name="%s_%s" % (self.name, self.object_name), pose=abs_pose)]

        def score_products(self, parent, products):
            if len(products) != 1 or not isinstance(products[0], self.object_type):
                return torch.tensor(-np.inf)
            # Get relative offset of the PlaceSetting
            rel_pose = self._recover_rel_pose_from_abs_pose(parent, products[0].pose.detach())
            return self.offset_dist.log_prob(rel_pose.double()).double()

    def __init__(self, name, pose):
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
        # Key: Class name (from above)
        # Value: Nominal (Mean, Variance) used to set up prior distributions
        param_guesses_by_name = {
            "plate": ([0., 0.10, 0.], [0.05, 0.05, 3.]),
            "cup": ([0., 0.14 + 0.09, 0.], [0.05, 0.05, 3.]),
            "right_fork": ([0.10, 0.09, 0.], [0.05, 0.05, 0.05]),
            "left_fork": ([-0.10, 0.09, 0.], [0.05, 0.05, 0.05]),
            "left_spoon": ([-0.10, 0.09, 0.], [0.05, 0.05, 0.05]),
            "right_spoon": ([0.10, 0.09, 0.], [0.05, 0.05, 0.05]),
            "left_knife": ([-0.10, 0.09, 0.], [0.05, 0.05, 0.05]),
            "right_knife": ([0.10, 0.09, 0.], [0.05, 0.05, 0.05]),
        }
        self.distributions_by_name = {}
        production_rules = []
        name_to_ind = {}
        for k, object_name in enumerate(self.object_types_by_name.keys()):
            mean_init, var_init = param_guesses_by_name[object_name]
            # Reasonably broad prior on the mean
            mean_prior_variance = (torch.ones(3)*0.05).double()
            # Use an inverse gamma prior for variance. It has MEAN (rather than mode)
            # beta / (alpha - 1) = var
            # (beta / var) + 1 = alpha
            # Picking bigger beta/var ratio leads to tighter peak around the guessed variance.
            var_prior_width_fact = 1
            assert(var_prior_width_fact > 0.)
            beta = var_prior_width_fact*torch.tensor(var_init).double()
            alpha = var_prior_width_fact*torch.ones(len(var_init)).double() + 1
            production_rules.append(
                self.ObjectProductionRule(
                    name="%s_prod_%03d" % (name, k),
                    object_name=object_name,
                    object_type=self.object_types_by_name[object_name],
                    mean_prior_params=(torch.tensor(mean_init, dtype=torch.double), mean_prior_variance),
                    var_prior_params=(alpha, beta)))
            # Build name mapping for convenience of building the hint dictionary
            name_to_ind[object_name] = k

        # Initialize the production rules here. (They're parameters, so they don't have a prior.)
        production_weights_hints = {
            # tuple(): 1., # Nothing
            (name_to_ind["plate"],): 1.,
            (name_to_ind["cup"],): 1.0,
            (name_to_ind["plate"], name_to_ind["cup"]): 1.,
            (name_to_ind["plate"], name_to_ind["right_fork"]): 1.,
            (name_to_ind["plate"], name_to_ind["left_fork"]): 1.,
            (name_to_ind["plate"], name_to_ind["cup"], name_to_ind["left_fork"]): 1.,
            (name_to_ind["plate"], name_to_ind["cup"], name_to_ind["right_fork"]): 1.,
            (name_to_ind["plate"], name_to_ind["cup"], name_to_ind["right_spoon"]): 1.,
            (name_to_ind["plate"], name_to_ind["cup"], name_to_ind["left_fork"], name_to_ind["right_knife"], name_to_ind["right_spoon"]): 1.,
            (name_to_ind["plate"], name_to_ind["left_fork"], name_to_ind["right_knife"], name_to_ind["right_spoon"]): 1.,
            (name_to_ind["plate"], name_to_ind["cup"], name_to_ind["left_fork"], name_to_ind["right_knife"]): 1.,
            (name_to_ind["plate"], name_to_ind["cup"], name_to_ind["right_fork"], name_to_ind["right_knife"]): 1.,
            # Outlier condition adds these as well:
            #(name_to_ind["plate"], name_to_ind["left_spoon"]): 1.0,
            #(name_to_ind["plate"], name_to_ind["cup"], name_to_ind["left_spoon"]): 1.0,
        }

        init_weights = CovaryingSetNode.build_init_weights(
            num_production_rules=len(production_rules),
            #production_weights_hints=production_weights_hints,
            remaining_weight=0.001)
        init_weights = pyro.param("place_setting_production_weights", init_weights, constraint=constraints.simplex)
        self.param_names = ["place_setting_production_weights"]
        CovaryingSetNode.__init__(self, name=name, production_rules=production_rules, init_weights=init_weights)

class Table(CovaryingSetNode, RootNode):

    class PlaceSettingProductionRule(ProductionRule):
        def __init__(self, name, pose):
            # Relative offset from root pose is drawn from a diagonal
            # Normal. It's rotated into the root pose frame at sample time.
            #mean = pyro.param("table_place_setting_mean",
            #                  torch.tensor([0.0, 0., np.pi/2.]))
            #var = pyro.param("table_place_setting_var",
            #                  torch.tensor([0.01, 0.01, 0.1]),
            #                  constraint=constraints.positive)
            #self.param_names = ["table_place_setting_mean", "table_place_setting_var"]
            mean = torch.tensor([0.0, 0., np.pi/2.]).double()
            var = torch.tensor([0.01, 0.01, 0.01]).double()
            self.offset_dist = dist.Normal(mean, var).to_event(1)
            self.pose = pose
            ProductionRule.__init__(self,
                name=name,
                product_types=[PlaceSetting])
            
        def _recover_rel_offset_from_abs_offset(self, parent, abs_offset):
            pose_in_world = chain_pose_transforms(parent.pose, self.pose)
            return chain_pose_transforms(invert_pose(pose_in_world), abs_offset)

        def sample_products(self, parent, obs_products=None):
            if obs_products is not None:
                assert len(obs_products) == 1 and isinstance(obs_products[0], PlaceSetting)
                obs_rel_offset = self._recover_rel_offset_from_abs_offset(parent, obs_products[0].pose) 
                rel_offset = pyro.sample("%s_place_setting_offset" % self.name,
                                         self.offset_dist, obs=obs_rel_offset)
                return obs_products
            else:
                rel_offset = pyro.sample("%s_place_setting_offset" % self.name,
                                         self.offset_dist).detach()
                # Rotate offset
                pose_in_world = chain_pose_transforms(parent.pose, self.pose)
                abs_offset = chain_pose_transforms(pose_in_world, rel_offset)
                return [PlaceSetting(name=self.name + "_place_setting", pose=abs_offset)]

        def score_products(self, parent, products):
            if len(products) != 1 or not isinstance(products[0], PlaceSetting):
                return torch.tensor(-np.inf)
            # Get relative offset of the PlaceSetting
            rel_offset = self._recover_rel_offset_from_abs_offset(parent, products[0].pose)
            return self.offset_dist.log_prob(rel_offset).sum().double()

    def __init__(self, name="table", num_place_setting_locations=4):
        self.pose = torch.tensor([0.5, 0.5, 0.]).double()
        self.table_radius = 0.35 #pyro.param("%s_radius" % name, torch.tensor(0.45), constraint=constraints.positive)
        # Set-valued: a plate may appear at each location.
        production_rules = []
        for k in range(num_place_setting_locations):
            # TODO(gizatt) Root pose for each cluster could be a parameter.
            # This turns this into a GMM, sort of?
            r = torch.tensor((k / float(num_place_setting_locations))*np.pi*2.).double()
            pose = torch.empty(3).double()
            pose[0] = self.table_radius * torch.cos(r)
            pose[1] = self.table_radius * torch.sin(r)
            pose[2] = r
            production_rules.append(self.PlaceSettingProductionRule(
                name="%s_prod_%03d" % (name, k), pose=pose))
        #production_probs = pyro.param("%s_independent_set_production_probs" % name,
        #                              torch.ones(num_place_setting_locations)*0.5,
        #                              constraint=constraints.unit_interval)
        #self.param_names = [#"%s_radius" % name,
        #                    "%s_independent_set_production_probs" % name]
        init_weights = CovaryingSetNode.build_init_weights(
            num_production_rules=len(production_rules)) # Even weight on any possible combination to start with
        init_weights = pyro.param("%s_production_weights" % name, init_weights, constraint=constraints.simplex)
        self.param_names = ["%s_production_weights" % name]
        CovaryingSetNode.__init__(self, name=name, production_rules=production_rules, init_weights=init_weights)

class Plate(TerminalNode):
    def __init__(self, pose, params=[0.2], name="plate"):
        TerminalNode.__init__(self, name=name)
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
    def __init__(self, pose, params=[0.05], name="cup"):
        TerminalNode.__init__(self, name=name)
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
    def __init__(self, pose, params=[0.02, 0.14], name="fork"):
        TerminalNode.__init__(self, name)
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
    def __init__(self, pose, params=[0.015, 0.15], name="knife"):
        TerminalNode.__init__(self, name)
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
    def __init__(self, pose, params=[0.02, 0.12], name="spoon"):
        TerminalNode.__init__(self, name)
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

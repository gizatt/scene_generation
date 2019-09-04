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
from pydrake.math import (RollPitchYaw, RigidTransform)
import torch
import torch.distributions.constraints as constraints
import pyro
import pyro.distributions as dist
from pyro import poutine

from scene_generation.data.dataset_utils import (
    DrawYamlEnvironmentPlanar, ProjectEnvironmentToFeasibility)
from scene_generation.models.probabilistic_scene_grammar_nodes import *

# Simple form, for now:
# DishBin can independently produce each (up to maximum #) of the dishes or mugs.

def rotation_tensor(theta, phi, psi):
    rot_x = torch.eye(3, 3, dtype=theta.dtype)
    rot_x[1, 1] = theta.cos()
    rot_x[1, 2] = -theta.sin()
    rot_x[2, 1] = theta.sin()
    rot_x[2, 2] = theta.cos()

    rot_y = torch.eye(3, 3, dtype=theta.dtype)
    rot_y[0, 0] = phi.cos()
    rot_y[0, 2] = phi.sin()
    rot_y[2, 0] = -phi.sin()
    rot_y[2, 2] = phi.cos()
    
    rot_z = torch.eye(3, 3, dtype=theta.dtype)
    rot_z[0, 0] = psi.cos()
    rot_z[0, 1] = -psi.sin()
    rot_z[1, 0] = psi.sin()
    rot_z[1, 1] = psi.cos()
    return torch.mm(rot_z, torch.mm(rot_y, rot_x))

def pose_to_tf_matrix(pose):
    out = torch.empty(4, 4)
    out[3, :] = 0.
    out[3, 3] = 1.
    out[:3, :3] = rotation_tensor(pose[3], pose[4], pose[5])
    out[:3, 3] = pose[:3]
    return out

# Ref https://www.learnopencv.com/rotation-matrix-to-euler-angles/
def tf_matrix_to_pose(tf):
    out = torch.empty(6)
    R = tf[:3, :3]
    out[:3] = tf[:3, 3]
    sy = torch.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    if sy >= 1E-6: # not singular
        out[3] = torch.atan2(R[2, 1], R[2, 2])
        out[4] = torch.atan2(-R[2, 0], sy)
        out[5] = torch.atan2(R[1, 0], R[0, 0])
    else: # Singular
        out[3] = torch.atan2(-R[1, 2], R[1, 1])
        out[4] = torch.atan2(-R[2, 0], sy) 
        out[5] = 0.
    return out

class DishBin(IndependentSetNode, RootNode):
    class MugProductionRule(ProductionRule):
        def __init__(self, name, pose):
            self.pose = pose # rpy xyz
            ProductionRule.__init__(self,
                name=name,
                product_types=[Mug_1])
            
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
            return self.offset_dist.log_prob(rel_offset).sum()

    def __init__(self, name="dish_bin"):
        self.pose = torch.tensor([0.0, 0.0, 0., 0., 0., 0.])

        # Set-valued: total of 4 mugs and 4 plates can occur
        production_rules = []
        for k in range(4):
            production_rules.append(self.MugProductionRule(
                name="%s_prod_mug_%03d" % (name, k), pose=self.pose))
            production_rules.append(self.PlateProductionRule(
                name="%s_prod_plate_%03d" % (name, k), pose=self.pose))
        production_probs = pyro.param("%s_independent_set_production_probs" % name,
                                      torch.ones(8)*0.5,
                                      constraint=constraints.unit_interval)
        self.param_names = ["%s_independent_set_production_probs" % name]
        IndependentSetNode.__init__(self, name=name, production_rules=production_rules, production_probs=production_probs)

class Plate_11in(TerminalNode):
    def __init__(self, pose, params=[], name="plate_11in"):
        Plate_11in.__init__(self, name)
        self.pose = pose
        self.params = params
    
    def generate_yaml(self):
        return {
            "class": "plate_11in",
            "params": self.params,
            "params_names": [],
            "pose": self.pose.tolist()
        }

class Mug_1(TerminalNode):
    def __init__(self, pose, params=[], name="mug_1"):
        Mug_1.__init__(self, name)
        self.pose = pose
        self.params = params
    
    def generate_yaml(self):
        return {
            "class": "mug_1",
            "params": self.params,
            "params_names": [],
            "pose": self.pose.tolist()
        }


if __name__ == "__main__":
    # Test out the tf methods
    test_pose = np.array([0., 0., 0., 0.523, 0.235, 0.712])
    test_pose_tensor = torch.tensor(test_pose, requires_grad=True)

    print("Input pose: ", test_pose_tensor)
    tf = pose_to_tf_matrix(test_pose_tensor)
    tf_drake = RigidTransform(p=test_pose[:3], rpy=RollPitchYaw(test_pose[3:]))
    print("Tf: ", tf)
    print("Tf from drake: ", tf_drake.matrix())
    test_pose_tensor_out = tf_matrix_to_pose(tf)
    print("Output pose: ", test_pose_tensor_out)
    torch.sum(test_pose_tensor_out[5]).backward()
    print("Orig tensor grad: ", test_pose_tensor.grad)
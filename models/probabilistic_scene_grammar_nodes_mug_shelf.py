from __future__ import print_function
from copy import deepcopy
from functools import partial
import time
import networkx
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
    DrawYamlEnvironment, ProjectEnvironmentToFeasibility)
from scene_generation.models.probabilistic_scene_grammar_nodes import *
from scene_generation.models.probabilistic_scene_grammar_nodes_dish_bin import Mug_1

# General layout:
# MugShelf is rooted at world origin.
# It has 3 independent set production rules for a MugShelfLevel at the three levels.
# MugShelfLevel has two IndependentSetNodes: another MugShelfLevel at the same level, and Mug intermediate node.
# The Mug intermediate node is an OR rule over:
#  - Upright might
#  - Upside down mug
#  - Whatever mug

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

def invert_tf(tf):
    out = torch.eye(4, 4)
    # R <- R.'
    # T <- -R.' T
    out[:3, :3] = torch.t(tf[:3, :3])
    out[:3, 3] = torch.mm(-1. * out[:3, :3], tf[:3, 3].unsqueeze(-1)).squeeze()
    return out


class MugIntermediate(OrNode):
    class MugProductionRule(ProductionRule):
        def __init__(self, name, target_xyzrpy_offset, target_variance):
            self.product_type = Mug_1
            self.offset_dist = dist.Normal(loc=target_xyzrpy_offset, scale=target_variance)
            ProductionRule.__init__(self, name=name, product_types=[self.product_type])

        def _recover_rel_offset_from_abs_offset(self, parent, abs_offset):
            parent_pose_tf = pose_to_tf_matrix(parent.pose)
            rel_tf = torch.mm(invert_tf(parent_pose_tf), pose_to_tf_matrix(abs_offset))
            return tf_matrix_to_pose(rel_tf)

        def sample_products(self, parent, obs_products=None):
            if obs_products is not None:
                assert len(obs_products) == 1 and isinstance(obs_products[0], Mug_1)
                obs_rel_offset = self._recover_rel_offset_from_abs_offset(parent, obs_products[0].pose) 
                rel_offset = pyro.sample("%s_mug_offset" % (self.name),
                                         self.offset_dist, obs=obs_rel_offset)
                return obs_products
            else:
                rel_offset = pyro.sample("%s_mug_offset" % (self.name),
                                         self.offset_dist).detach()
                # Chain relative offset on top of current pose in world
                parent_pose_tf = pose_to_tf_matrix(parent.pose)
                offset_tf = pose_to_tf_matrix(rel_offset)
                abs_offset = tf_matrix_to_pose(torch.mm(parent_pose_tf, offset_tf))
                return [self.product_type(name=self.name + "_mug", pose=abs_offset)]

        def score_products(self, parent, products):
            if len(products) != 1 or not isinstance(products[0], self.product_type):
                return torch.tensor(-np.inf)
            # Get relative offset of the PlaceSetting
            rel_offset = self._recover_rel_offset_from_abs_offset(parent, products[0].pose)
            R = pose_to_tf_matrix(rel_offset)[:3, :3]
            option_1 = self.offset_dist.log_prob(rel_offset).sum()
            # Try alternative
            other_rel_offset = torch.empty(6).double()
            other_rel_offset[:3] = rel_offset[:3]
            # Flip to an equivalent RPY and check its log prob as well
            if rel_offset[3] > 0:
                other_rel_offset[3] = rel_offset[3] - 3.1415
            else:
                other_rel_offset[3] = rel_offset[3] + 3.1415
            if rel_offset[4] > 0:
                other_rel_offset[4] = rel_offset[4] - 3.1415
            else:
                other_rel_offset[4] = rel_offset[4] + 3.1415
            if rel_offset[5] > 0:
                other_rel_offset[5] = rel_offset[5] - 3.1415
            else:
                other_rel_offset[5] = rel_offset[5] + 3.1415
            R = pose_to_tf_matrix(other_rel_offset)[:3, :3]
            option_2 = self.offset_dist.log_prob(other_rel_offset).sum()
            score = torch.max(option_1, option_2)
            #print("%s got score %f for offset " % (self.name, score.item()), products[0].pose, rel_offset)
            return torch.max(option_1, option_2)

    def __init__(self, name, pose):
        self.pose = pose.clone()
        production_rules = [
            self.MugProductionRule("%s_up_prod" % name,
                                   target_xyzrpy_offset=torch.tensor([0., 0., 0., 3.1415/2., 0., 0.]),
                                   target_variance=torch.tensor([0.001, 0.001, 0.001, 0.05, 0.05, 3.1415])),
            self.MugProductionRule("%s_down_prod" % name,
                                   target_xyzrpy_offset=torch.tensor([0., 0., 0., -3.1415/2., 0., 0.]),
                                   target_variance=torch.tensor([0.001, 0.001, 0.001, 0.05, 0.05, 3.1415])),
            self.MugProductionRule("%s_random_prod" % name,
                                   target_xyzrpy_offset=torch.tensor([0., 0., 0., 0., 0., 0.]),
                                   target_variance=torch.tensor([0.001, 0.001, 0.001, 3.14, 3.14, 3.14]))
        ]

        production_probs = pyro.param("mug_orientation_production_weights", torch.tensor([0.45, 0.45, 0.1]), constraint=constraints.simplex)
        self.param_names = ["mug_orientation_production_weights"]
        OrNode.__init__(self, name=name, production_rules=production_rules, production_weights=production_probs)

    def seed_from_candidate_nodes(self, child_nodes):
        # Adopt the pose of the child.
        # (All possible candidate children will have pose in this model type.)
        if len(child_nodes) != 1 or not isinstance(child_nodes[0], Mug_1):
            return
        self.pose[:3] = child_nodes[0].pose.clone()[:3]
        self.pose[3] = 0.
        self.pose[4] = 0.
        self.pose[5] = 0.


class MugShelfLevel(IndependentSetNode):
    class MugProductionRule(ProductionRule):
        def __init__(self, name, shelf_name, xyz_mean_prior_params, xyz_var_prior_params):
            self.product_type = MugIntermediate
            self.shelf_name = shelf_name
            self.xyz_mean_prior_params = xyz_mean_prior_params
            self.xyz_var_prior_params = xyz_var_prior_params
            self.global_variable_names = ["%s_mug_xyz_mean" % shelf_name,
                                          "%s_mug_xyz_var" % shelf_name]
            ProductionRule.__init__(self,
                name=name,
                product_types=[self.product_type])
            
        def _recover_rel_offset_from_abs_offset(self, parent, abs_offset):
            parent_pose_tf = pose_to_tf_matrix(parent.pose)
            rel_tf = torch.mm(invert_tf(parent_pose_tf), pose_to_tf_matrix(abs_offset))
            return tf_matrix_to_pose(rel_tf)

        def sample_global_variables(self, global_variable_store):
            # Handles class-general setup
            offset_mean_prior_dist = dist.Normal(
                loc=self.xyz_mean_prior_params[0],
                scale=self.xyz_mean_prior_params[1]).to_event(1)
            offset_var_prior_dist = dist.InverseGamma(concentration=self.xyz_var_prior_params[0],
                                                      rate=self.xyz_var_prior_params[1]).to_event(1)
            offset_mean = global_variable_store.sample_global_variable(
                "%s_mug_xyz_mean" % self.shelf_name,
                offset_mean_prior_dist).double()
            offset_var = global_variable_store.sample_global_variable(
                "%s_mug_xyz_var" % self.shelf_name, offset_var_prior_dist).double()
            self.offset_mean = offset_mean
            self.offset_var = offset_var
            self.xyz_offset_dist = dist.Normal(loc=offset_mean, scale=offset_var).to_event(1)

        def sample_products(self, parent, obs_products=None):
            if obs_products is not None:
                assert len(obs_products) == 1 and isinstance(obs_products[0], Plate_11in)
                obs_rel_xyz_offset = self._recover_rel_offset_from_abs_offset(parent, obs_products[0].pose)[:3]
                rel_offset = pyro.sample("%s_mug_offset" % (self.name),
                                         self.xyz_offset_dist, obs=obs_rel_xyz_offset)
                return obs_products
            else:
                rel_xyz_offset = pyro.sample("%s_mug_offset" % (self.name),
                                             self.xyz_offset_dist).detach()
                rel_offset = torch.zeros(6).double()
                rel_offset[:3] = rel_xyz_offset[:3]
                offset_tf = pose_to_tf_matrix(rel_offset)
                # Chain relative offset on top of current pose in world
                parent_pose_tf = pose_to_tf_matrix(parent.pose)
                abs_offset = tf_matrix_to_pose(torch.mm(parent_pose_tf, offset_tf))
                return [self.product_type(name=self.name + "_mug", pose=abs_offset)]

        def score_products(self, parent, products):
            if len(products) != 1 or not isinstance(products[0], self.product_type):
                return torch.tensor(-np.inf)
            # Get relative offset of the PlaceSetting
            rel_offset = self._recover_rel_offset_from_abs_offset(parent, products[0].pose)
            return self.xyz_offset_dist.log_prob(rel_offset[:3]).sum()

    
    class MugShelfLevelSelfProductionRule(ProductionRule):
        def __init__(self, name):
            ProductionRule.__init__(self,
                name=name,
                product_types=[MugShelfLevel])

        def sample_products(self, parent, obs_products=None):
            # Observation should be absolute position of the product
            if obs_products is not None:
                assert(len(obs_products) == 1 and isinstance(obs_products[0], MugShelfLevel))
                return obs_products
            else:
                return [MugShelfLevel(name="%s_%s" % (self.name, "recurse"), pose=parent.pose.clone(), shelf_name=parent.shelf_name)]

        def score_products(self, parent, products):
            if len(products) != 1 or not isinstance(products[0], MugShelfLevel):
                return torch.tensor(-np.inf)
            return torch.tensor(0.).double()


    def __init__(self, name, shelf_name, pose):
        self.pose = pose.clone()
         # Shelf name is conserved for all recursive shelves generated from this shelf, so they can share some params
        self.shelf_name = shelf_name

        production_rules = []

        # Mug production
        mean_init = torch.tensor([0., 0., 0.])
        var_init = torch.tensor([0.1, 0.05, 0.005])
        
        # Pretty specific prior on mean and variance
        mean_prior_variance = torch.tensor([0.03, 0.03, 0.01])
        # Use an inverse gamma prior for variance. It has MEAN (rather than mode)
        # beta / (alpha - 1) = var
        # (beta / var) + 1 = alpha
        # Picking bigger beta/var ratio leads to tighter peak around the guessed variance.
        var_prior_width_fact = 1
        assert(var_prior_width_fact > 0.)
        beta = var_prior_width_fact*var_init.double()
        alpha = var_prior_width_fact*torch.ones(len(var_init)).double() + 1

        for k in range(6):
            production_rules.append(
                self.MugProductionRule(
                    name="%s_prod_mug_%d" % (name,  k),
                    shelf_name=self.shelf_name,
                    xyz_mean_prior_params=(mean_init, mean_prior_variance),
                    xyz_var_prior_params=(alpha, beta)))

        # Self-production for recursion
        #production_rules.append(self.MugShelfLevelSelfProductionRule("%s_prod_self" % name))

        # Even production probs to start out
        production_probs = torch.ones(len(production_rules))*0.25
        production_probs = pyro.param("mug_shelf_%s_production_weights" % self.shelf_name, production_probs, constraint=constraints.unit_interval)
        self.param_names = ["mug_shelf_%s_production_weights" % self.shelf_name]
        IndependentSetNode.__init__(self, name=name, production_rules=production_rules, production_probs=production_probs)


class MugShelf(IndependentSetNode, RootNode):
    class MugShelfLevelProductionRule(ProductionRule):
        def __init__(self, name, pose, shelf_name):
            self.product_type = MugShelfLevel
            self.shelf_name = shelf_name
            self.desired_pose = pose.clone()
            ProductionRule.__init__(self,
                name=name,
                product_types=[self.product_type])

        def sample_products(self, parent, obs_products=None):
            # Observation should be absolute position of the product
            if obs_products is not None:
                assert(len(obs_products) == 1 and isinstance(obs_products[0], self.product_type))
                return obs_products
            else:
                return [self.product_type(name="%s_shelf" % (self.name), pose=self.desired_pose.clone(), shelf_name=self.shelf_name)]

        def score_products(self, parent, products):
            if len(products) != 1 or not isinstance(products[0], self.product_type):
                return torch.tensor(-np.inf)
            return torch.tensor(0.).double()


    def __init__(self, name="mug_shelf"):
        self.pose = torch.tensor([0.0, 0.0, 0., 0., 0., 0.])

        # Independently produce a shelf at each level
        production_rules = []
        for k in range(3):
            production_rules.append(
                self.MugShelfLevelProductionRule("%s_prod_shelf_%d" % (name, k),
                    shelf_name="shelf_%d" % k,
                    pose=torch.tensor([0., 0., 0.15*k + 0.03, 0., 0., 0.])))

        init_weights = pyro.param("%s_shelf_production_weights" % name, torch.tensor([0.5, 0.5, 0.5]), constraint=constraints.unit_interval)
        self.param_names = ["%s_shelf_production_weights" % name]
        IndependentSetNode.__init__(self, name=name, production_rules=production_rules, production_probs=init_weights)


def convert_xyzrpy_to_quatxyz(pose):
    assert(len(pose) == 6)
    pose = np.array(pose)
    out = np.zeros(7)
    out[-3:] = pose[:3]
    out[:4] = RollPitchYaw(pose[3:]).ToQuaternion().wxyz()
    return out

if __name__ == "__main__":
    #seed = 52
    #torch.manual_seed(seed)
    #np.random.seed(seed)
    #random.seed(seed)
    torch.set_default_tensor_type(torch.DoubleTensor)
    pyro.enable_validation(True)

    root_node = MugShelf(name="shelf")

    plt.figure()
    from scene_generation.models.probabilistic_scene_grammar_model import *
    for k in range(100):
        # Draw + plot a few generated environments and their trees
        start = time.time()
        pyro.clear_param_store()
        trace = poutine.trace(generate_unconditioned_parse_tree).get_trace(root_node=root_node)
        parse_tree = trace.nodes["_RETURN"]["value"]
        end = time.time()

        print(bcolors.OKGREEN, "Generated data in %f seconds." % (end - start), bcolors.ENDC)
        print("Full trace values:" )
        for node_name in trace.nodes.keys():
            if node_name in ["_INPUT", "_RETURN"]:
                continue
            print(node_name, ": ", trace.nodes[node_name]["value"].detach().numpy())

        score, score_by_node = parse_tree.get_total_log_prob()
        print("Score by node: ", score_by_node)
        yaml_env = convert_tree_to_yaml_env(parse_tree)
        #yaml_env = ProjectEnvironmentToFeasibility(yaml_env, base_environment_type="dish_bin",
        #                                           make_nonpenetrating=True, make_static=False)[-1]
        DrawYamlEnvironment(yaml_env, base_environment_type="mug_shelf", alpha=0.5)
        draw_parse_tree_meshcat(parse_tree, color_by_score=True)
        print("Our score: %f" % score)
        print("Trace score: %f" % trace.log_prob_sum())

        plt.gca().clear()
        networkx.draw_networkx(parse_tree, labels={n:n.name for n in parse_tree})
        plt.pause(1E-3)
        #assert(abs(score - trace.log_prob_sum()) < 0.001)
        input("Press enter to continue...")
        #time.sleep(1)

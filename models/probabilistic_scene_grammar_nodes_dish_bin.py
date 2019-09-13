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
    DrawYamlEnvironment, ProjectEnvironmentToFeasibility)
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

def invert_tf(tf):
    out = torch.eye(4, 4)
    # R <- R.'
    # T <- -R.' T
    out[:3, :3] = torch.t(tf[:3, :3])
    out[:3, 3] = torch.mm(-1. * out[:3, :3], tf[:3, 3].unsqueeze(-1)).squeeze()
    return out


class DishStack(IndependentSetNode):
    class DishProductionRule(ProductionRule):
        def __init__(self, name, object_name, offset_mean_prior_params, offset_var_prior_params):
            self.product_type = Plate_11in
            self.object_name = object_name
            self.offset_mean_prior_params = offset_mean_prior_params
            self.offset_var_prior_params = offset_var_prior_params
            self.global_variable_names = ["dish_stack_pose_var",
                                          "dish_stack_%s_pose_mean" % object_name]
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
                loc=self.offset_mean_prior_params[0],
                scale=self.offset_mean_prior_params[1]).to_event(1)
            offset_var_prior_dist = dist.InverseGamma(concentration=self.offset_var_prior_params[0],
                                                      rate=self.offset_var_prior_params[1]).to_event(1)
            offset_mean = global_variable_store.sample_global_variable(
                "dish_stack_%s_pose_mean" % self.object_name,
                offset_mean_prior_dist).double()
            offset_var = global_variable_store.sample_global_variable(
                "dish_stack_pose_var", offset_var_prior_dist).double()
            self.offset_dist = dist.Normal(loc=offset_mean, scale=offset_var).to_event(1)

        def sample_products(self, parent, obs_products=None):
            if obs_products is not None:
                assert len(obs_products) == 1 and isinstance(obs_products[0], Plate_11in)
                obs_rel_offset = self._recover_rel_offset_from_abs_offset(parent, obs_products[0].pose) 
                rel_offset = pyro.sample("%s_%s_offset" % (self.name, self.object_name),
                                         self.offset_dist, obs=obs_rel_offset)
                return obs_products
            else:
                rel_offset = pyro.sample("%s_%s_offset" % (self.name, self.object_name),
                                         self.offset_dist).detach()
                # Chain relative offset on top of current pose in world
                parent_pose_tf = pose_to_tf_matrix(parent.pose)
                offset_tf = pose_to_tf_matrix(rel_offset)
                abs_offset = tf_matrix_to_pose(torch.mm(parent_pose_tf, offset_tf))
                return [self.product_type(name=self.name + "_" + self.object_name, pose=abs_offset)]

        def score_products(self, parent, products):
            if len(products) != 1 or not isinstance(products[0], self.product_type):
                return torch.tensor(-np.inf)
            # Get relative offset of the PlaceSetting
            rel_offset = self._recover_rel_offset_from_abs_offset(parent, products[0].pose)
            #print("Observed rel offset ", rel_offset)
            R = pose_to_tf_matrix(rel_offset)[:3, :3]
            #print("As a rotation matrix: ", )
            #print("As an angle: ", torch.acos( (torch.trace(R) - 1)/2.))
            #print("Rot axis: ", [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
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
            #print("Other option:")
            R = pose_to_tf_matrix(other_rel_offset)[:3, :3]
            #print("As a rotation matrix: ", )
            #print("As an angle: ", torch.acos( (torch.trace(R) - 1)/2.))
            #print("Rot axis: ", [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
            option_2 = self.offset_dist.log_prob(other_rel_offset).sum()
            return torch.max(option_1, option_2)

    def __init__(self, name, pose):
        self.pose = pose
        # Represent each dish's relative position to the
        # stack origin with a diagonal Normal distribution.
        
        # Key: Class name (from above)
        # Value: Nominal (Mean, Variance) used to set up prior distributions

        production_rules = []
        for k in range(4):
            mean_init = torch.tensor([0., 0., 0., 0., 0., 0.]).double()
            var_init = torch.tensor([0.025, 0.025, 0.025, 0.1, 2.0, 0.1]).double()
            
            # Pretty specific prior on mean and variance
            mean_prior_variance = (torch.ones(6)*0.01).double()
            # Use an inverse gamma prior for variance. It has MEAN (rather than mode)
            # beta / (alpha - 1) = var
            # (beta / var) + 1 = alpha
            # Picking bigger beta/var ratio leads to tighter peak around the guessed variance.
            var_prior_width_fact = 1
            assert(var_prior_width_fact > 0.)
            beta = var_prior_width_fact*var_init
            alpha = var_prior_width_fact*torch.ones(len(var_init)).double() + 1
            production_rules.append(
                self.DishProductionRule(
                    name="%s_prod_dish_%d" % (name, k),
                    object_name="stack_%d" % k,
                    offset_mean_prior_params=(mean_init, mean_prior_variance),
                    offset_var_prior_params=(alpha, beta)))

        # Even production probs to start out
        production_probs = torch.tensor([1., 0.8, 0.5, 0.3]).double()
        production_probs = pyro.param("dish_stack_production_weights", production_probs, constraint=constraints.unit_interval)
        self.param_names = ["dish_stack_production_weights"]
        IndependentSetNode.__init__(self, name=name, production_rules=production_rules, production_probs=production_probs)

    def seed_from_candidate_nodes(self, child_nodes):
        # Adopt the average pose of the child nodes.
        # (All possible candidate children will have pose in this model type.)
        avg_pose = torch.zeros(6).double()
        for child in child_nodes:
            if not isinstance(child, Plate_11in):
                return
            avg_pose = avg_pose + child.pose
        avg_pose = avg_pose / len(child_nodes)
        self.pose = avg_pose

class DishBin(IndependentSetNode, RootNode):
    class ObjectProductionRule(ProductionRule):
        def __init__(self, name, product_type, offset_mean_prior_params, offset_var_prior_params):
            self.product_type = product_type
            self.product_name = product_type.__name__
            self.offset_mean_prior_params = offset_mean_prior_params
            self.offset_var_prior_params = offset_var_prior_params
            self.global_variable_names = ["prod_%s_pose_var" % self.product_name,
                                          "prod_%s_pose_mean" % self.product_name]
            ProductionRule.__init__(self,
                name=name,
                product_types=[product_type])
            
        def sample_global_variables(self, global_variable_store):
            # Handles class-general setup
            offset_mean_prior_dist = dist.Normal(
                loc=self.offset_mean_prior_params[0],
                scale=self.offset_mean_prior_params[1]).to_event(1)
            offset_var_prior_dist = dist.InverseGamma(concentration=self.offset_var_prior_params[0],
                                                      rate=self.offset_var_prior_params[1]).to_event(1)
            offset_mean = global_variable_store.sample_global_variable(
                "prod_%s_pose_mean" % self.product_name,
                offset_mean_prior_dist).double()
            offset_var = global_variable_store.sample_global_variable(
                "prod_%s_pose_var" % self.product_name, offset_var_prior_dist).double()
            self.offset_dist = dist.Normal(loc=offset_mean, scale=offset_var).to_event(1)

        def _recover_rel_offset_from_abs_offset(self, parent, abs_offset):
            parent_pose_tf = pose_to_tf_matrix(parent.pose)
            rel_tf = torch.mm(invert_tf(parent_pose_tf), pose_to_tf_matrix(abs_offset))
            return tf_matrix_to_pose(rel_tf)

        def sample_products(self, parent, obs_products=None):
            if obs_products is not None:
                assert len(obs_products) == 1 and isinstance(obs_products[0], self.product_type)
                obs_rel_offset = self._recover_rel_offset_from_abs_offset(parent, obs_products[0].pose) 
                rel_offset = pyro.sample("%s_%s_offset" % (self.name, self.product_name),
                                         self.offset_dist, obs=obs_rel_offset)
                return obs_products
            else:
                rel_offset = pyro.sample("%s_%s_offset" % (self.name, self.product_name),
                                         self.offset_dist).detach()
                # Chain relative offset on top of current pose in world
                parent_pose_tf = pose_to_tf_matrix(parent.pose)
                offset_tf = pose_to_tf_matrix(rel_offset)
                abs_offset = tf_matrix_to_pose(torch.mm(parent_pose_tf, offset_tf))
                return [self.product_type(name=self.name + "_" + self.product_name, pose=abs_offset)]

        def score_products(self, parent, products):
            if len(products) != 1 or not isinstance(products[0], self.product_type):
                return torch.tensor(-np.inf)
            # Get relative offset of the PlaceSetting
            rel_offset = self._recover_rel_offset_from_abs_offset(parent, products[0].pose)
            #print("Observed rel offset ", rel_offset)
            R = pose_to_tf_matrix(rel_offset)[:3, :3]
            #print("As a rotation matrix: ", )
            #print("As an angle: ", torch.acos( (torch.trace(R) - 1)/2.))
            #print("Rot axis: ", [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
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
            #print("Other option:")
            R = pose_to_tf_matrix(other_rel_offset)[:3, :3]
            #print("As a rotation matrix: ", )
            #print("As an angle: ", torch.acos( (torch.trace(R) - 1)/2.))
            #print("Rot axis: ", [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
            option_2 = self.offset_dist.log_prob(other_rel_offset).sum()
            return torch.max(option_1, option_2)

    def __init__(self, name="dish_bin"):
        self.pose = torch.tensor([0.0, 0.0, 0., 0., 0., 0.])

        # Set-valued: 4 independent mugs, 4 independent plates, and a plate stack can all independently occur.
        production_rules = []

        mean_init = torch.tensor([0., 0., 0.1, 0., 0., 0.]).double()
        var_init = torch.tensor([0.1, 0.1, 0.1, 2., 2., 2.]).double()
        # Reasonably broad prior on the mean
        mean_prior_variance = (torch.ones(6)*0.05).double()
        # Use an inverse gamma prior for variance. It has MEAN (rather than mode)
        # beta / (alpha - 1) = var
        # (beta / var) + 1 = alpha
        # Picking bigger beta/var ratio leads to tighter peak around the guessed variance.
        var_prior_width_fact = 1.
        assert(var_prior_width_fact > 0.)
        beta = var_prior_width_fact*var_init
        alpha = var_prior_width_fact*torch.ones(len(var_init)).double() + 1
        
        # MUG
        rule_types = []
        for k in range(4):
            production_rules.append(self.ObjectProductionRule(
                name="%s_prod_mug_1_%03d" % (name, k), product_type=Mug_1,
                offset_mean_prior_params=(mean_init, mean_prior_variance),
                offset_var_prior_params=(alpha, beta)))
            rule_types.append("mug")
#
        # PLATE
        #for k in range(4):
        #    production_rules.append(self.ObjectProductionRule(
        #        name="%s_prod_plate_11in_%03d" % (name, k), product_type=Plate_11in,
        #        offset_mean_prior_params=(torch.tensor(mean_init, dtype=torch.double), mean_prior_variance),
        #        offset_var_prior_params=(alpha, beta)))
        #    rule_types.append("plate")
        
        # PLATE STACK
        for k in range(4):
            production_rules.append(self.ObjectProductionRule(
                name="%s_prod_dish_stack_%03d" % (name, k), product_type=DishStack,
                offset_mean_prior_params=(mean_init, mean_prior_variance),
                offset_var_prior_params=(alpha, beta)))
            rule_types.append("plate_stack")

        # STRONGLY prefer plate stacks over plates to bias towards using them
        #init_weights = CovaryingSetNode.build_init_weights(
        #    num_production_rules=len(production_rules))
        init_weights = torch.ones(len(rule_types))
        for k, rule in enumerate(rule_types):
            if rule == "mug":
                init_weights[k] = 0.5
            elif rule == "plate":
                init_weights[k] = 0.1
            elif rule == "plate_stack":
                init_weights[k] = 0.25
            else:
                raise NotImplementedError()
        init_weights = pyro.param("%s_production_weights" % name, init_weights, constraint=constraints.unit_interval)
        self.param_names = ["%s_production_weights" % name]
        IndependentSetNode.__init__(self, name=name, production_rules=production_rules, production_probs=init_weights)


def convert_xyzrpy_to_quatxyz(pose):
    assert(len(pose) == 6)
    pose = np.array(pose)
    out = np.zeros(7)
    out[-3:] = pose[:3]
    out[:4] = RollPitchYaw(pose[3:]).ToQuaternion().wxyz()
    return out

class Plate_11in(TerminalNode):
    def __init__(self, pose, params=[], name="plate_11in"):
        TerminalNode.__init__(self, name)
        self.pose = pose
        self.params = params
    
    def generate_yaml(self):
        return {
            "class": "plate_11in",
            "params": self.params,
            "params_names": [],
            "pose": convert_xyzrpy_to_quatxyz(self.pose).tolist()
        }

class Mug_1(TerminalNode):
    def __init__(self, pose, params=[], name="mug_1"):
        TerminalNode.__init__(self, name)
        self.pose = pose
        self.params = params
    
    def generate_yaml(self):
        return {
            "class": "mug_1",
            "params": self.params,
            "params_names": [],
            "pose": convert_xyzrpy_to_quatxyz(self.pose).tolist()
        }

if __name__ == "__main__":
    #seed = 52
    #torch.manual_seed(seed)
    #np.random.seed(seed)
    #random.seed(seed)
    torch.set_default_tensor_type(torch.DoubleTensor)
    pyro.enable_validation(True)

    from scene_generation.models.probabilistic_scene_grammar_model import *
    for k in range(10):
        # Draw + plot a few generated environments and their trees
        start = time.time()
        pyro.clear_param_store()
        trace = poutine.trace(generate_unconditioned_parse_tree).get_trace(root_node=DishBin())
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
        DrawYamlEnvironment(yaml_env, base_environment_type="dish_bin", alpha=0.5)
        draw_parse_tree_meshcat(parse_tree, color_by_score=True)
        print("Our score: %f" % score)
        print("Trace score: %f" % trace.log_prob_sum())
        #assert(abs(score - trace.log_prob_sum()) < 0.001)
        input("Press enter to continue...")
        #time.sleep(1)

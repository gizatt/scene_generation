from __future__ import print_function
from copy import deepcopy
from functools import partial
import time
import random
import sys
import yaml

import matplotlib.pyplot as plt
import meshcat
import meshcat.geometry as meshcat_geom
import meshcat.transformations as meshcat_tf
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
from scene_generation.models.probabilistic_scene_grammar_model import *

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


class DishStack(IndependentSetNode, RootNode):
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
            my_pose_tf = pose_to_tf_matrix(parent.pose)
            parent_pose_tf = pose_to_tf_matrix(parent.pose)
            my_pose_in_world_tf = torch.mm(parent_pose_tf, my_pose_tf)
            rel_tf = torch.mm(invert_tf(my_pose_tf), pose_to_tf_matrix(abs_offset))
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
                my_pose_tf = pose_to_tf_matrix(parent.pose)
                parent_pose_tf = pose_to_tf_matrix(parent.pose)
                my_pose_in_world_tf = torch.mm(parent_pose_tf, my_pose_tf)
                offset_tf = pose_to_tf_matrix(rel_offset)
                abs_offset = tf_matrix_to_pose(torch.mm(my_pose_in_world_tf, offset_tf))
                return [self.product_type(name=self.name + "_" + self.object_name, pose=abs_offset)]

        def score_products(self, parent, products):
            if len(products) != 1 or not isinstance(products[0], self.product_type):
                return torch.tensor(-np.inf)
            # Get relative offset of the PlaceSetting
            rel_offset = self._recover_rel_offset_from_abs_offset(parent, products[0].pose)
            print("Scoring dish stack element with rel offset ", rel_offset)
            return self.offset_dist.log_prob(rel_offset).sum()

    def __init__(self, name, pose):
        self.pose = pose
        # Represent each dish's relative position to the
        # stack origin with a diagonal Normal distribution.
        
        # Key: Class name (from above)
        # Value: Nominal (Mean, Variance) used to set up prior distributions

        production_rules = []
        vertical_spacing = 0.01
        for k in range(4):
            z_offset = k*vertical_spacing
            mean_init = torch.tensor([0., 0., z_offset, 0., 0., 0.])
            var_init = torch.tensor([0.02, 0.02, 0.02, 0.01, 0.01, 0.01])
            
            # Pretty specific prior on mean and variance
            mean_prior_variance = (torch.ones(6)*0.01).double()
            # Use an inverse gamma prior for variance. It has MEAN (rather than mode)
            # beta / (alpha - 1) = var
            # (beta / var) + 1 = alpha
            # Picking bigger beta/var ratio leads to tighter peak around the guessed variance.
            var_prior_width_fact = 1
            assert(var_prior_width_fact > 0.)
            beta = var_prior_width_fact*torch.tensor(var_init).double()
            alpha = var_prior_width_fact*torch.ones(len(var_init)).double() + 1
            production_rules.append(
                self.DishProductionRule(
                    name="%s_prod_dish_%d" % (name, k),
                    object_name="stack_%d" % k,
                    offset_mean_prior_params=(torch.tensor(mean_init, dtype=torch.double), mean_prior_variance),
                    offset_var_prior_params=(alpha, beta)))

        # Even production probs to start out
        production_probs = torch.ones(len(production_rules)).double() / len(production_rules)
        production_probs = pyro.param("dish_stack_production_weights", production_probs, constraint=constraints.simplex)
        self.param_names = ["dish_stack_production_weights"]
        IndependentSetNode.__init__(self, name=name, production_rules=production_rules, production_probs=production_probs)


class DishBin(CovaryingSetNode, RootNode):
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
            my_pose_tf = pose_to_tf_matrix(parent.pose)
            parent_pose_tf = pose_to_tf_matrix(parent.pose)
            my_pose_in_world_tf = torch.mm(parent_pose_tf, my_pose_tf)
            rel_tf = torch.mm(invert_tf(my_pose_tf), pose_to_tf_matrix(abs_offset))
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
                my_pose_tf = pose_to_tf_matrix(parent.pose)
                parent_pose_tf = pose_to_tf_matrix(parent.pose)
                my_pose_in_world_tf = torch.mm(parent_pose_tf, my_pose_tf)
                offset_tf = pose_to_tf_matrix(rel_offset)
                abs_offset = tf_matrix_to_pose(torch.mm(my_pose_in_world_tf, offset_tf))
                return [self.product_type(name=self.name + "_" + self.product_name, pose=abs_offset)]

        def score_products(self, parent, products):
            if len(products) != 1 or not isinstance(products[0], self.product_type):
                return torch.tensor(-np.inf)
            # Get relative offset of the PlaceSetting
            rel_offset = self._recover_rel_offset_from_abs_offset(parent, products[0].pose)
            print("In scoring for %s, I recovered rel offset of " % self.name, rel_offset)
            return self.offset_dist.log_prob(rel_offset).sum()

    def __init__(self, name="dish_bin"):
        self.pose = torch.tensor([0.0, 0.0, 0., 0., 0., 0.])

        # Set-valued: 4 independent mugs, 4 independent plates, and a plate stack can all independently occur.
        production_rules = []

        mean_init = torch.tensor([0., 0., 0.1, 0., 0., 0.]).double()
        var_init = torch.tensor([0.05, 0.05, 0.05, 2., 2., 2.]).double()
        # Reasonably broad prior on the mean
        mean_prior_variance = (torch.ones(6)*0.05).double()
        # Use an inverse gamma prior for variance. It has MEAN (rather than mode)
        # beta / (alpha - 1) = var
        # (beta / var) + 1 = alpha
        # Picking bigger beta/var ratio leads to tighter peak around the guessed variance.
        var_prior_width_fact = 10.
        assert(var_prior_width_fact > 0.)
        beta = var_prior_width_fact*torch.tensor(var_init).double()
        alpha = var_prior_width_fact*torch.ones(len(var_init)).double() + 1
        
        # MUG
        for k in range(4):
            production_rules.append(self.ObjectProductionRule(
                name="%s_prod_mug_1_%03d" % (name, k), product_type=Mug_1,
                offset_mean_prior_params=(torch.tensor(mean_init, dtype=torch.double), mean_prior_variance),
                offset_var_prior_params=(alpha, beta)))
#
        # PLATE
        for k in range(4):
            production_rules.append(self.ObjectProductionRule(
                name="%s_prod_plate_11in_%03d" % (name, k), product_type=Plate_11in,
                offset_mean_prior_params=(torch.tensor(mean_init, dtype=torch.double), mean_prior_variance),
                offset_var_prior_params=(alpha, beta)))
        
        # PLATE STACK
        production_rules.append(self.ObjectProductionRule(
            name="%s_prod_dish_stack" % (name), product_type=DishStack,
            offset_mean_prior_params=(torch.tensor(mean_init, dtype=torch.double), mean_prior_variance),
            offset_var_prior_params=(alpha, beta)))

        init_weights = CovaryingSetNode.build_init_weights(
            num_production_rules=len(production_rules)) # Even weight on any possible combination to start with
        init_weights = pyro.param("%s_production_weights" % name, init_weights, constraint=constraints.simplex)
        self.param_names = ["%s_production_weights" % name]
        CovaryingSetNode.__init__(self, name=name, production_rules=production_rules, init_weights=init_weights)


def convert_xyzrpy_to_quatxyz(pose):
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
        # Convert xyz rpy pose to qw qx qy qz x y z pose

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


class LineBasicMaterial(meshcat_geom.Material):
    def __init__(self, linewidth=1, color=0xffffff,
                 linecap="round", linejoin="round"):
        super(LineBasicMaterial, self).__init__()
        self.linewidth = linewidth
        self.color = color
        self.linecap = linecap
        self.linejoin = linejoin

    def lower(self, object_data):
        return {
            u"uuid": self.uuid,
            u"type": u"LineBasicMaterial",
            u"color": self.color,
            u"linewidth": self.linewidth,
            u"linecap": self.linecap,
            u"linejoin": self.linejoin
        }

def draw_parse_tree_meshcat(parse_tree, color_by_score=False, node_class_to_color_dict={}):
    pruned_tree = remove_production_rules_from_parse_tree(parse_tree)

    if color_by_score:
        score, scores_by_node = parse_tree.get_total_log_prob()
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

    # Do actual drawing in meshcat, starting from root of tree
    # So first find the root...
    root_node = list(pruned_tree.nodes)[0]
    while len(list(pruned_tree.predecessors(root_node))) > 0:
        root_node = pruned_tree.predecessors(root_node)[0]

    node_sphere_size = 0.01
    vis = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
    vis["parse_tree"].delete()
    node_queue = [root_node]
    def rgb_2_hex(rgb):
            # Turn a list of R,G,B elements (any indexable list
            # of >= 3 elements will work), where each element is
            # specified on range [0., 1.], into the equivalent
            # 24-bit value 0xRRGGBB.
            val = 0
            for i in range(3):
                val += (256**(2 - i)) * int(255 * rgb[i])
            return val
    if colors is not None and len(colors.shape) == 1:
        # Use cmap to get real colors
        assert(colors.shape[0] == len(pruned_tree.nodes))
        colors = plt.cm.get_cmap('jet')(colors)
    while len(node_queue) > 0:
        node = node_queue.pop(0)
        children = list(pruned_tree.successors(node))
        node_queue += children
        # Draw this node
        print(colors, colors.shape)
        if colors is not None:
            color = rgb_2_hex(colors[list(pruned_tree.nodes).index(node)])
        else:
            color = 0xff0000
        vis["parse_tree"][node.name].set_object(
            meshcat_geom.Sphere(node_sphere_size),
            meshcat_geom.MeshToonMaterial(color=color))

        # Get node global pose by going all the way up pose TF chain
        tf = pose_to_tf_matrix(node.pose).detach().numpy()
        vis["parse_tree"][node.name].set_transform(tf)

        # Draw connections to children
        verts = []
        for child in children:
            verts.append(node.pose[:3])
            verts.append(child.pose[:3])
        if len(verts) > 0:
            verts = np.vstack(verts).T
            vis["parse_tree"][node.name + "_child_connections"].set_object(
                meshcat_geom.Line(meshcat_geom.PointsGeometry(verts),
                                  LineBasicMaterial(linewidth=10, color=color)))


if __name__ == "__main__":
    #seed = 52
    #torch.manual_seed(seed)
    #np.random.seed(seed)
    #random.seed(seed)
    torch.set_default_tensor_type(torch.DoubleTensor)
    pyro.enable_validation(True)

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

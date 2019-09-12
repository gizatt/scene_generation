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

class DishBin(IndependentSetNode, RootNode):
    class ObjectProductionRule(ProductionRule):
        def __init__(self, name, pose, product_type, offset_dist):
            self.pose = pose # xyz rpy
            self.product_type = product_type
            self.product_name = product_type.__name__
            self.offset_dist = offset_dist
            ProductionRule.__init__(self,
                name=name,
                product_types=[product_type])
            
        def _recover_rel_offset_from_abs_offset(self, parent, abs_offset):
            my_pose_tf = pose_to_tf_matrix(self.pose)
            parent_pose_tf = pose_to_tf_matrix(parent.pose)
            my_pose_in_world_tf = torch.mm(parent_pose_tf, my_pose_tf)
            rel_tf = torch.mm(invert_tf(my_pose_tf), pose_to_tf_matrix(abs_offset))
            return tf_matrix_to_pose(rel_tf)

        def sample_products(self, parent, obs_products=None):
            if obs_products is not None:
                assert len(obs_products) == 1 and isinstance(obs_products[0], PlaceSetting)
                obs_rel_offset = self._recover_rel_offset_from_abs_offset(parent, obs_products[0].pose) 
                rel_offset = pyro.sample("%s_%s_offset" % (self.name, self.product_name),
                                         self.offset_dist, obs=obs_rel_offset)
                return obs_products
            else:
                rel_offset = pyro.sample("%s_%s_offset" % (self.name, self.product_name),
                                         self.offset_dist).detach()
                # Chain relative offset on top of current pose in world
                print("Sampled rel offset ", rel_offset)
                my_pose_tf = pose_to_tf_matrix(self.pose)
                print("My pose tf: ", my_pose_tf)
                parent_pose_tf = pose_to_tf_matrix(parent.pose)
                print("Parent pose tf: ", parent_pose_tf)
                my_pose_in_world_tf = torch.mm(parent_pose_tf, my_pose_tf)
                print("My pose in world: ", my_pose_in_world_tf)
                offset_tf = pose_to_tf_matrix(rel_offset)
                print("Offset tf: ", offset_tf)
                abs_offset = tf_matrix_to_pose(torch.mm(my_pose_in_world_tf, offset_tf))
                print("Abs offset sampled: ", abs_offset)
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

        # Set-valued: total of 4 mugs and 4 plates can occur
        production_rules = []
        for k in range(4):
            mug_mean = pyro.param("prod_mug_mean", torch.tensor([0.0, 0.0, 0.1, 0., 0., 0.]))
            mug_var = pyro.param("prod_mug_var", torch.tensor([0.05, 0.05, 0.05, 2.0, 2.0, 2.0]), constraint=constraints.positive)
            production_rules.append(self.ObjectProductionRule(
                name="%s_prod_mug_1_%03d" % (name, k), pose=self.pose, product_type=Mug_1,
                offset_dist=dist.Normal(mug_mean, mug_var)))

            plate_mean = pyro.param("prod_plate_mean", torch.tensor([0.0, 0.0, 0.1, 0., 0., 0.]))
            plate_var = pyro.param("prod_plate_var", torch.tensor([0.05, 0.05, 0.05, 2.0, 2.0, 2.0]), constraint=constraints.positive)
            production_rules.append(self.ObjectProductionRule(
                name="%s_prod_plate_11in_%03d" % (name, k), pose=self.pose, product_type=Plate_11in,
                offset_dist=dist.Normal(plate_mean, plate_var)))
        production_probs = pyro.param("%s_independent_set_production_probs" % name,
                                      torch.ones(8)*0.5,
                                      constraint=constraints.unit_interval)
        self.param_names = ["%s_independent_set_production_probs" % name, "prod_mug_mean", "prod_mug_var"]
        IndependentSetNode.__init__(self, name=name, production_rules=production_rules, production_probs=production_probs)

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
    pos_dict = {
        node: node.pose[:2].detach().numpy() for node in pruned_tree
    }
    if color_by_score:
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
        tf = RigidTransform(
            rpy=RollPitchYaw(node.pose[3:]),
            p=node.pose[:3]).GetAsMatrix4()
        vis["parse_tree"][node.name].set_transform(tf)

        # Draw connections to children
        verts = []
        for child in children:
            verts.append(node.pose[:3])
            verts.append(child.pose[:3])
        if len(verts) > 0:
            verts = np.vstack(verts).T
            vis["parse_tree"][node.name]["child_connections"].set_object(
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
        assert(abs(score - trace.log_prob_sum()) < 0.001)
        time.sleep(1)

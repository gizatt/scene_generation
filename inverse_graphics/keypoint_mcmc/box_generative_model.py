from collections import namedtuple
import datetime
from functools import partial, reduce
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import time
import yaml
import sys

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

import pydrake

import torch
import torch.distributions.constraints as constraints
import torch.nn.functional as F
import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist
from pyro.contrib.autoguide import AutoGuideList, AutoDelta, AutoDiagonalNormal, AutoDiscreteParallel
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO
from pyro.optim import Adam

from scene_generation.utils.torch_quaternion import (
    quat2mat,
    rotation_matrix_to_quaternion,
    transParamsToHomMatrix
)

import operator
from functools import reduce  # Required in Python 3
def prod(iterable):
    return reduce(operator.mul, iterable, 1)

class Box():
    def __init__(self, name, pose, dimensions):
        # Pose quatxyz in camera frame
        # Dimensions x y z
        # TODO: support more than 1 batching dimension? iff necessary?
        assert pose.shape[-1] == 7, pose.shape
        assert dimensions.shape[-1] == 3, dimensions.shape
        self.name = name
        self.batch_shape = tuple(pose.shape[:-1])
        self.pose = pose
        self.dimensions = dimensions

    def generate_keypoints(self):
        pts = torch.tensor(
            np.array([[-1., -1., -1., -1, 1., 1., 1., 1.],
                      [-1., -1., 1., 1., -1., -1., 1., 1.],
                      [-1., 1., -1., 1., -1., 1., -1., 1.]])).double()
        pts = pts.unsqueeze(0).repeat(prod(self.batch_shape), 1, 1)
        pts = pts * self.dimensions.reshape(-1, 3, 1).repeat(1, 1, 8) / 2.
        quat = self.pose[..., :4].reshape(-1, 4)
        xyz = self.pose[..., -3:].reshape(-1, 3)
        tf_mat = transParamsToHomMatrix(quat, xyz)
        pts_homog = torch.cat([pts, torch.ones(prod(self.batch_shape), 1, pts.shape[2]).double()], dim=1)
        pts = torch.bmm(tf_mat, pts_homog)[..., :3, :]
        return pts.reshape(self.batch_shape + (3, 8))

    def sample_keypoint_images(self, image_shape, camera_K, with_occlusions=False):
        assert(camera_K.shape == (3, 3))
        peak_width = 1.
        peak_height = 1.
        min_depth_var = 0.1
        max_depth_var = 10.

        if with_occlusions:
            raise NotImplementedError("Occlusions not done yet")

        pts = self.generate_keypoints().reshape(-1, 3, 8)
        grid_x, grid_y = torch.meshgrid(torch.arange(image_shape[0]), torch.arange(image_shape[1]))
        batched_grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1).reshape(1, -1, 2)
        batched_grid = batched_grid.repeat(prod(self.batch_shape), 1, 1)

        # Project each point into camera frame
        pts = torch.bmm(camera_K.repeat(prod(self.batch_shape), 1, 1), pts).permute(0, 2, 1)
        # Sample boolean occurance prob for each point
        visibility = pyro.sample(
                self.name + "_visibility",
                dist.Beta(torch.tensor(0.1).double(),
                          torch.tensor(0.1).double()).expand((8,)).to_event(1)
            ).reshape(-1, 8)
        # Collect output images across the different points, accounting for
        # visibility where we can
        pt_loc_heatmaps = []  # (self.batch_size, image_shape[0], image_shape[1]))
        depths_as_images = [] # (self.batch_size, image_shape[0], image_shape[1]))
        depth_vars = [] # (self.batch_size, image_shape[0], image_shape[1]))
        peak_heatmaps = []
        for k in range(8): # probably could batch this too...
            image_shaped_visibility = visibility[:, k].unsqueeze(-1).unsqueeze(-1).repeat(1, image_shape[0], image_shape[1])
            depths_as_images.append((pts[..., k, 2]*visibility[:, k]).unsqueeze(-1).unsqueeze(-1).repeat(1, image_shape[0], image_shape[1]))
            # Distances to the peak, for each pixel in the image
            # (Repeats pts to be the same size as the flattened batched grid so the
            # batching works out)
            distances_from_peak = torch.sum(
                    (batched_grid - pts[..., k, :2].unsqueeze(1).repeat(1, batched_grid.shape[1], 1))**2.,
                axis=-1)
            peak_heatmap = peak_height * torch.exp(-distances_from_peak / (2. * (peak_width ** 2.))) + 1E-6 # Avoid numerical issue when normalizing
            peak_heatmap = peak_heatmap.reshape(-1, image_shape[0], image_shape[1])
            peak_heatmaps.append(peak_heatmap)
            depth_vars.append(max_depth_var * (peak_height - peak_heatmap + min_depth_var)) # goes to low value at the observations
            pt_loc_heatmaps.append(image_shaped_visibility * peak_heatmap)
        # For point location occurance probs, just take the max
        total_pt_loc_heatmap = torch.stack(pt_loc_heatmaps, axis=-1).max(axis=-1)[0]
        # Likewise, take minimum depth observation variance at any given point
        total_depth_vars = torch.stack(depth_vars, axis=-1).min(axis=-1)[0]

        # The expected depth measurement for a pixel is a weighted average of the
        # point depths, weighted by the pixel heatmap value for each contributing
        # keypoint.
        total_heatmap_vals = torch.stack(peak_heatmaps, axis=-1).sum(axis=-1)
        # Get "padding" weight set up -- a pixel with no visible keypoints (low heatmap
        # values for all points) will instead get a default value of 0
        additional_padding = torch.clamp(1. - total_heatmap_vals, 0., 1.)
        peak_heatmaps.append(additional_padding)
        depths_as_images.append(torch.zeros(prod(self.batch_shape), image_shape[0], image_shape[1]).double())
        total_depth_means = (torch.stack(peak_heatmaps, axis=-1)*torch.stack(depths_as_images, axis=-1)).sum(axis=-1)

        # Finally, we can use those to sample a keypoint
        # occurance image and a depth image.
        # (The intent is for a consumer to condition the model
        # on these values.)
        keypoint_occurance_image = pyro.sample(
            self.name + "_keypoint_occurance",
            dist.Normal(total_pt_loc_heatmap.reshape(self.batch_shape + tuple(image_shape)), 0.1).to_event(2))
        depth_image = pyro.sample(
            self.name + "_depth",
            dist.Normal(total_depth_means.reshape(self.batch_shape + tuple(image_shape)),
                        total_depth_vars.reshape(self.batch_shape + tuple(image_shape))).to_event(2))

        return keypoint_occurance_image, depth_image


    @staticmethod
    def sample(name = "box", batch_size=1):
        quat = pyro.sample(
            name + "_quat",
            dist.Uniform(torch.ones(4).double()*-1.01,
                         torch.ones(4).double()*1.01).to_event(1)
        )
        if len(quat.shape) == 1:
            quat = quat.reshape(-1, 4)
        quat = F.normalize(quat, p=2, dim=-1)

        xyz_mean = torch.zeros(3).double()
        xyz_mean[2] = 1.
        xyz_scale = torch.ones(3).double() * 0.1
        xyz = pyro.sample(
            name + "_xyz",
            dist.Normal(xyz_mean, xyz_scale).to_event(1)
        )
        if len(xyz.shape) == 1:
            xyz = xyz.reshape(-1, 3)

        dimensions_alpha = pyro.param(
            "dimensions_alpha", torch.ones(3).double() * 3.,
            constraint=constraints.positive)
        dimensions_beta = pyro.param(
            "dimensions_beta", torch.ones(3).double() * 1.,
            constraint=constraints.positive)
        dimensions = pyro.sample(
            name + "_dimensions",
            dist.InverseGamma(dimensions_alpha, dimensions_beta).to_event(1)
        )
        if len(dimensions.shape) == 1:
            dimensions = dimensions.reshape(-1, 3)

        return Box(name=name,
                   pose = torch.cat([quat, xyz], dim=-1),
                   dimensions = dimensions)

# Simple generative model of keypoints: assume a number of observed keypoints are
# drawn from a GMM with components from each keypoint plus an "outlier" component.
# The number of observed keypoints is drawn from an (uninformative) discrete
# uniform distribution. (Maybe something with infinite support would be better?)
def model_keypoint_observations_gmm(keypoints, vals):
    max_num_keypoints = keypoints.shape[1] + 5
    num_keypoints = pyro.sample(
        "num_observed_keypoints_minus_one",
        dist.Categorical(probs=torch.ones(max_num_keypoints).double())) + 1

    keypoint_var = pyro.param(
        "keypoint_var",
        torch.tensor([0.01]).double(),
        constraint=constraints.positive)
    val_var = pyro.param(
        "val_var",
        torch.tensor([0.01]).double(),
        constraint=constraints.positive)
    outlier_var = pyro.param(
        "outlier_var",
        torch.tensor([1.0]).double(),
        constraint=constraints.positive)
    outlier_weight = pyro.param(
        "outlier_weight",
        torch.tensor([0.01]).double(),
        constraint=constraints.positive)

    # Make the model components
    locs = torch.cat([keypoints, vals], dim=0)
    scales = torch.cat([torch.ones(keypoints.shape).double()*keypoint_var,
                        torch.ones(vals.shape).double()*val_var])
    # Add the outlier component
    locs = torch.cat([locs, torch.zeros(locs.shape[0], 1).double()], dim=1)
    scales = torch.cat([scales, torch.ones(locs.shape[0], 1).double()*outlier_var], dim=1)

    # Component weights
    component_probs = torch.cat([
        torch.ones(keypoints.shape[1]).double(),
        outlier_weight], dim=0)
    component_probs = component_probs / torch.sum(component_probs)
    component_logits = torch.log(component_probs / (1. - component_probs))

    generation_dist = dist.MixtureOfDiagNormals(
        locs=torch.t(locs),
        coord_scale=torch.t(scales),
        component_logits=component_logits).expand(num_keypoints)

    observed_keypoints_and_vals = pyro.sample(
        "observed_keypoints_and_vals", generation_dist)

def full_model():
    box = box.sample()
    keypoints, vals = box.generate_keypoints()
    model_keypoint_observations_gmm(keypoints, vals)

def sample_correspondences(R, scaling, t, C, model_pts, model_vals,
                           observed_pts, observed_vals,
                           spatial_variance=0.01,
                           feature_variance=0.01,
                           num_mh_iters=5,
                           outlier_prob=0.01):
    # Given a fixed affine transform B, t of a set of model points,
    # a current correspondence set C, and the point sets,
    # sample from the set of correspondences of those point sets using MH
    # and a simple flipping proposal.
    # Inspired by  "EM, MCMC, and Chain Flipping for Structure from Motion
    # with Unknown Correspondence".

    N_s = observed_pts.shape[1]
    N_m = model_pts.shape[1]
    D_spatial = model_pts.shape[0]
    assert(len(model_vals.shape) == 2)
    D_feature = model_vals.shape[0]
    # Build the pairwise distance matrix in the complete feature space
    def compute_pairwise_distances(R, scaling, t):
        # Get affine transformation of the model points
        transformed_model_pts = torch.mm(R, torch.mm(scaling, model_pts)) + t
        # Concatenate both model and observed with their vals
        m_with_vals = torch.cat([transformed_model_pts, 100.*model_vals], dim=0)
        s_with_vals = torch.cat([observed_pts, 100.*observed_vals], dim=0)
        
        # Reshape each of these into D x N_m x N_s matrix so we can take distances
        m_expanded = m_with_vals.unsqueeze(-1).permute(0, 1, 2).repeat(1, 1, N_s)
        s_expanded = s_with_vals.unsqueeze(-1).permute(0, 2, 1).repeat(1, N_m, 1)
        distances = torch.norm(m_expanded - s_expanded, dim=0)
        return distances

    pairwise_distances = compute_pairwise_distances(R, scaling, t)
    # Translate into probabilities
    pairwise_distances[0:N_m, :] = (
        torch.exp(-pairwise_distances[0:N_m, :]/(2.*spatial_variance)) /
        math.sqrt(2*np.pi*spatial_variance))
    pairwise_distances[N_m:, :] = (
        torch.exp(-pairwise_distances[N_m:, :]/(2.*feature_variance)) /
        math.sqrt(2*np.pi*feature_variance))

    # Augment with a row of the outlier probabilities
    pairwise_distances = torch.cat([pairwise_distances,
                                    torch.ones(1, N_s).double()*outlier_prob],
                                    dim=0)
    assert(pairwise_distances.shape == C.shape)

    current_score = torch.sum(C * pairwise_distances)

    for k in range(num_mh_iters):
        # Sample a new correspondence set by randomly flipping some number of correspondences.
        C_new = C.clone().detach()
        num_to_flip = dist.Binomial(probs=0.5).sample() + 1
        for flip_k in range(int(num_to_flip)):
            to_add_ind_s = torch.randint(N_s, (1, 1))
            to_add_ind_m = dist.Categorical(
                probs=(pairwise_distances*(1. - C))[:, to_add_ind_s].squeeze()).sample()
            if to_add_ind_m != N_m:
                C_new[to_add_ind_m, :] = 0.
            C_new[:, to_add_ind_s] = 0.
            C_new[to_add_ind_m, to_add_ind_s] = 1.
        new_score = torch.sum(C_new * pairwise_distances)

        # MH acceptance
        if torch.rand(1).item() <= new_score / current_score:
            C = C_new

    return C

def sample_transform_given_correspondences(C, model_pts, observed_pts):
    # Given the fixed correspondences, find an optimal translation t
    # and affine transform B to align the corresponded model and observed
    # points.
    # In the case of these boxes, I'm going to use a restricted class of
    # affine transforms allowing scale but not skew: the full transform
    # is R * scaling * model_points + t = observed points
    # where scaling is a 3x3 diagonal matrix. As long as the model
    # points are 

    # Reorder into only the relevant point pairs
    keep_model_points = C.sum(dim=1) != 0
    keep_model_points[-1] = False
    C_reduced = C[keep_model_points, :]
    model_pts_reduced = model_pts[:, keep_model_points[:-1]]
    assert(model_pts_reduced.shape[1] <= observed_pts.shape[1])
    assert(all(torch.sum(C_reduced, dim=1) == 1.))
    assert(torch.sum(keep_model_points) > 0)
    corresp_inds = C_reduced.nonzero()[:, 1]
    observed_pts_reduced = observed_pts[:, corresp_inds]

    # Get R and s by alternations: see "Orthogonal, but not Orthonormal, Procrustes Problems."
    # Draw initial scaling from our prior
    scaling = torch.diag(dist.InverseGamma(pyro.param("dimensions_alpha"), pyro.param("dimensions_beta")).sample()).clone().detach()
    model_pts_aligned = torch.t(torch.t(model_pts_reduced) - torch.mean(model_pts_reduced, dim=1))
    observed_pts_aligned = torch.t(torch.t(observed_pts_reduced) - torch.mean(observed_pts_reduced, dim=1))

    for k in range(10):
        # Solve for rotation
        U, S, V = torch.svd(
            torch.mm(observed_pts_aligned,
                     torch.mm(torch.t(model_pts_aligned), scaling)))
        R = torch.mm(U, torch.t(V))
        # Check det and flip if necessary
        if(torch.det(R)) < 0:
            flip_eye = torch.eye(3).double()
            flip_eye[2, 2] = -1
            R = torch.mm(U, torch.mm(flip_eye, torch.t(V)))

        # Use a few steps of HMC for the remaining continuous parameters, to simulate
        # drawing from the posterior of these paramaters conditions.
        # (Parameters of interest: dimensions, label uv)

        # Closed form scaling update:
        for i in range(3):
            num = torch.sum(
                observed_pts_aligned *
                torch.mm(R[:, i].reshape(3, 1), model_pts_aligned[i, :].reshape(1, -1)))
            denom = torch.sum(torch.pow(model_pts_aligned[i, :], 2.))
            if torch.abs(num) > 0.01 and torch.abs(denom) > 0.01:
                scaling[i, i] = num / denom
        scaling = torch.clamp(scaling, 0.01, 1.0)

    t = (torch.mean(observed_pts_reduced, dim=1) - torch.mean(
        torch.mm(R, torch.mm(scaling, model_pts_reduced)), dim=1)).reshape(-1, 1)

    return R, scaling, t


def draw_pts_with_meshcat(vis, name, pts, val_channel, vals, size=0.01):
    colors = np.ones((3, pts.shape[1]))
    colors[val_channel, :] = vals[:]
    vis[name].set_object(
        g.PointCloud(position=pts, color=colors, size=size))

def draw_corresp_with_meshcat(vis, name, pts_A, pts_B, C, size=0.01):
    pts = np.zeros((3, 2*C.shape[1]))
    assert(C.shape[0] == pts_A.shape[1] + 1)
    assert(C.shape[1] == pts_B.shape[1])
    for k in range(C.shape[1]):
        nonzero = np.nonzero(C[:, k])
        if len(nonzero) > 0 and nonzero[0] < pts_A.shape[1]:
            pts[:, 2*k] = pts_A[:, nonzero[0]]
            pts[:, 2*k+1] = pts_B[:, k]
    vis[name].set_object(
        g.LineSegments(
            g.PointsGeometry(position=pts),
            g.PointsMaterial(size=size)))

def sample_box_from_observed_points(observed_keypoints_and_vals, n_samples=200,
                                    vis=None, verbose=False):
    observed_pts = torch.tensor(observed_keypoints_and_vals[:3, :]).double()
    observed_vals = torch.tensor(observed_keypoints_and_vals[3:, :]).double()

    # From the observed keypoints and values, sample correspondences and
    # a box pose.

    # Start by randomly initializing a box
    random_box = pyro.poutine.block(generate_single_box)()

    site_values = {
        "box_xyz": random_box.pose[-3:].clone().detach(),
        "box_quat": random_box.pose[:4].clone().detach(),
        "box_dimensions": random_box.dimensions.clone().detach(),
        "box_label_face": torch.tensor([0]).long(), # TODO(gizatt) Should this vary?
        "box_label_uv": random_box.label_uv.clone().detach(),
        "num_observed_keypoints_minus_one": torch.tensor([observed_pts.shape[1] - 1]).long(),
        "observed_keypoints_and_vals": torch.t(torch.cat([observed_pts, observed_vals], dim=0))
    }
    def sample_model_pts_and_vals():
        box = generate_single_box()
        label_origin, label_size = get_label_info_from_box_with_label(box)
        keypoints, vals = generate_keypoints_from_box_with_label(box)
        return keypoints, vals
    model_pts, model_vals = pyro.poutine.block(
        pyro.poutine.condition(
            sample_model_pts_and_vals,
            data=site_values
        ))()

    R = torch.eye(3).double()
    scaling = torch.eye(3).double()
    t = torch.zeros(3, 1).double()
    # Correspondence matrix: one in the corresponding column of a model point and its corresponded
    # model point, plus one spare row for correspondence with "outlier"
    C = torch.eye(n=(model_pts.shape[1] + 1), m=observed_pts.shape[1])

    best_score = -1000.
    all_scores = []
    best_params = None
    all_params = []
    for k in range(n_samples):
        if verbose:
            print("*******ITER %03d*******" % k)

        # Regenerate model-frame model points and values given current guesses
        # of some key latents
        model_pts, model_vals = pyro.poutine.block(
                    pyro.poutine.condition(
                        sample_model_pts_and_vals,
                        data={
                            "box_xyz": torch.zeros(3).double(),
                            "box_quat": torch.tensor([1., 0., 0., 0.]).double(),
                            "box_dimensions": torch.ones(3).double(),
                            "box_label_face": site_values["box_label_face"],
                            "box_label_uv": site_values["box_label_uv"]
                        }
                    ))()

        C = sample_correspondences(
            R, scaling, t, C, model_pts, model_vals,
            observed_pts, observed_vals)
        if torch.sum(C[:-1, :]) > 0: # Skip transform if there are no non-outlier corresps
            R, scaling, t = sample_transform_given_correspondences(
                C, model_pts, observed_pts)
        site_values["box_xyz"] = t.squeeze().detach()
        site_values["box_quat"] = rotation_matrix_to_quaternion(R).detach()
        #R = quaternionToRotMatrix(site_values["box_quat"].reshape(1, -1))[0, :, :]
        site_values["box_dimensions"] = torch.diag(scaling).detach()

        # Now set box_label_face and box_label_uv to the maximum likelihood setting
        # if there's an observed label keypoint.
        # TODO(gizatt) This is a huge cludge. The label keypoints appear
        # in the registration step as well as being just additional keypoints (with
        # a different value, so they only get registered to the model's label keypoint).
        # That and this seem like they might break / interact badly...
        # but I want to see if this works first. This whole system needs a rewrite anyway
        # once I find a config I'm OK with.
        model_val = torch.tensor([1.])
        observed_label_pt_index = (torch.norm(observed_vals - model_val, dim=0) < 0.01).nonzero()
        if len(observed_label_pt_index) > 0:
            observed_label_pt_index = observed_label_pt_index[0]
            observed_label_pt = observed_pts[:, observed_label_pt_index]
            # Reverse that point into body frame of the box
            inverse_rot = torch.t(quaternionToRotMatrix(site_values["box_quat"].unsqueeze(0))[0, ...])
            inverse_t = -torch.mm(inverse_rot, site_values["box_xyz"].reshape(3, 1))
            observed_label_pt_body = (torch.mm(inverse_rot, observed_label_pt) + inverse_t).squeeze()
            # Rescale into unit cube and assign it to a face based on the unit direction the ray most faces
            observed_label_pt_body /= site_values["box_dimensions"]
            face_ind = torch.argmax(torch.abs(observed_label_pt_body))
            face_sign = observed_label_pt_body[face_ind] > 0.
            face_string = "np"[face_sign] + "xyz"[face_ind]
            if face_ind == 0:
                uv = observed_label_pt_body[1:] + 0.5
            elif face_ind == 1:
                uv = observed_label_pt_body[[0, 2]] + 0.5
            else:
                uv = observed_label_pt_body[:2] + 0.5
            site_values["box_label_face"] = torch.tensor(index_to_label_map.index(face_string)).long()
            site_values["box_label_uv"] = uv

        site_values["box_label_uv"].data = torch.clamp(site_values["box_label_uv"].data, 0.01, 0.99)
        site_values["box_quat"].data = site_values["box_quat"].data / torch.norm(site_values["box_quat"].data)
        R = quaternionToRotMatrix(site_values["box_quat"].reshape(1, -1))[0, :, :]
        t = site_values["box_xyz"].reshape(-1, 1)
        scaling = torch.diag(site_values["box_dimensions"])

        # Compute score using the Pyro models
        conditioned_model = pyro.poutine.condition(
            full_model,
            data=site_values)
        trace = pyro.poutine.trace(conditioned_model).get_trace()
        lps = trace.log_prob_sum()
        lps.backward()

        if verbose:
            #for name, site in trace.nodes.items():
            #    if site["type"] is "sample":
            #        print("Name: ", name)
            #        print("\tValue: ", site["value"])
            #        print("\tlog prob sum: ", site["log_prob_sum"])
            print("\tTotal Log prob sum: ", lps.item())

        # TODO(gizatt): Accept/reject with something like MH?
        # Problem is that my "proposal distribution" is probably not
        # symmetric, so calculating a proper MH ratio won't be as simple
        # as the likelihood ratio... for now, I'll stick to just importance
        # sampling with this relatively arbitrary proposal.
        if len(all_scores) > 0:
            accept_threshold = dist.Uniform(
                torch.tensor([0.]).double(),
                torch.tensor([1.]).double()).sample()
            mh_ratio = torch.exp((lps - all_scores[-1])/10.)
            accept = mh_ratio >= accept_threshold
        else:
            accept = True
        if accept:
            all_scores.append(lps.detach().item())
            copied_dict = {}
            for key in site_values.keys():
                copied_dict[key] = site_values[key].clone().detach()
            all_params.append(copied_dict)
            if lps.item() > best_score:
                best_score = lps.item()
                best_params = (R, scaling, t, C)
        else:
            all_scores.append(all_scores[-1])
            all_params.append(all_params[-1])
            for key in site_values.keys():
                site_values[key] = all_params[-1][key].clone().detach()
            

        scaling = torch.diag(site_values["box_dimensions"])
        R = quaternionToRotMatrix(site_values["box_quat"].reshape(1, -1))[0, :, :]
        t = site_values["box_xyz"].reshape(-1, 1)
        model_pts_tf = torch.mm(R, torch.mm(scaling, model_pts)) + t

        if vis is not None:
            draw_pts_with_meshcat(vis, "fitting/observed", 
                                  observed_pts.detach().numpy(),
                                  val_channel=0, vals=observed_vals.detach().numpy())
            draw_pts_with_meshcat(vis, "fitting/fit", 
                                  model_pts_tf.detach().numpy(),
                                  val_channel=2, vals=model_vals.detach().numpy())
            draw_corresp_with_meshcat(vis, "fitting/corresp",
                model_pts_tf.detach().numpy(), observed_pts.detach().numpy(),
                C)
            input()

    model_pts_tf = torch.mm(best_params[0], torch.mm(best_params[1], model_pts)) + best_params[2]
    if vis is not None:
        draw_pts_with_meshcat(vis, "fitting/observed", 
                              observed_pts.detach().numpy(),
                              val_channel=0, vals=observed_vals.detach().numpy())
        draw_pts_with_meshcat(vis, "fitting/fit", 
                              model_pts_tf.detach().numpy(),
                              val_channel=2, vals=model_vals.detach().numpy())
        draw_corresp_with_meshcat(vis, "fitting/corresp",
            model_pts_tf.detach().numpy(), observed_pts.detach().numpy(),
            best_params[3])
        input()
    return all_scores, all_params, best_score, best_params


if __name__ == "__main__":
    with open("../data/box_observations.yaml", "r") as file:
        all_envs = yaml.load(file)

    vis = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")

    num_boxes_to_sample = 1
    #random.seed(42)
    shuffled_envs = list(all_envs.values())
    random.shuffle(shuffled_envs)

    all_scores = []
    all_params = []

    for env_k in range(num_boxes_to_sample):
        print("Starting env %d" % env_k)
        observed_box_info = shuffled_envs[env_k % len(shuffled_envs)]["obj_%04d" % 0]
        observed_keypoints = pickle.loads(observed_box_info["observed_keypoints"])
        observed_vals = pickle.loads(observed_box_info["observed_vals"])
        observed_keypoints_and_vals = np.vstack([observed_keypoints, observed_vals])

        print("Observed keypoints and vals: ", observed_keypoints_and_vals)
        if observed_keypoints_and_vals.shape[1] == 0:
            continue
        these_scores, these_params, _, _ = sample_box_from_observed_points(
            observed_keypoints_and_vals, n_samples=50, vis=vis)
        all_scores += these_scores[25:]
        all_params += these_params[25:]

    if os.path.exists('all_scores_and_params.pickle'):
        with open('all_scores_and_params.pickle', 'rb') as f:
            new_scores, new_params = pickle.load(f)
            all_scores += new_scores
            all_params += new_params
    with open('all_scores_and_params.pickle', 'wb') as f:
        pickle.dump((all_scores, all_params), f)

    plt.figure()
    plt.subplot(5, 1, 1)
    plt.title("Analysis of %d samples (%d runs)" % (len(all_scores), len(all_scores) / 25))
    plt.plot(all_scores)
    plt.xlabel("llog")
    plt.ylabel("count")

    def extract_data(param_name):
        # Returns as shape [<param_dims>, num_samples]
        return np.stack([params[param_name] for params in all_params], axis=-1)

    exp_vals = torch.exp(torch.tensor(all_scores).double())
    box_dims_data = extract_data("box_dimensions")

    prior_dists = [dist.InverseGamma(
        pyro.param("dimensions_alpha")[k],
        pyro.param("dimensions_beta")[k]) for k in range(3)]
    gt_dists = [dist.Uniform(0.1, 0.3),
                dist.Uniform(0.1, 0.5),
                dist.Uniform(0.1, 0.7)]
    hist_bins = np.arange(0., 2., 0.1)
    # Plot each separately -- due to rotational symmetry, this is hard to interpret.
    # (Could e.g. sort this by datapoint into a canonical increasing order, but that's
    # not perfect.)
    plt.subplot(5, 1, 2)
    n, bins, _ = plt.hist(np.sum(box_dims_data, axis=0), normed=True, label="Hist", bins=hist_bins, log=False)
    spacing = 0.01
    x = torch.arange(bins[0]+1E-3, bins[-1], spacing)
    prior_values = [torch.exp(prior_dists[k].log_prob(x)).detach().numpy() for k in range(3)]
    gt_values = [torch.exp(gt_dists[k].log_prob(x)).detach().numpy() for k in range(3)]
    def add_pdf(a, b):
        # I think this only works if the left boundary of the PDF is 0?
        return np.convolve(a, b, mode='full')[:x.shape[0]] * spacing
    prior_values = reduce(add_pdf, prior_values)
    gt_values = reduce(add_pdf, gt_values)
    plt.plot(x, prior_values, "--", color="red", label="Prior")
    plt.plot(x, gt_values, "--", color="lime", label="Ground truth")
    plt.xlabel("sum of box dims")
    plt.ylabel("weight")

    prior_dists = [dist.InverseGamma(
        pyro.param("dimensions_alpha")[k],
        pyro.param("dimensions_beta")[k]) for k in range(3)]
    gt_dists = [dist.Uniform(0.1, 0.3),
                dist.Uniform(0.1, 0.5),
                dist.Uniform(0.1, 0.7)]
    hist_bins = np.arange(0., 1., 0.1)
    for k in range(3):
        plt.subplot(5, 3, 7+k)
        n, bins, _ = plt.hist(box_dims_data[k, :], normed=True, label="Hist", bins=hist_bins, log=False)
        x = torch.linspace(bins[0], bins[-1], 100)
        prior_values = torch.exp(prior_dists[k].log_prob(x)).detach().numpy()
        gt_values = torch.exp(gt_dists[k].log_prob(x)).detach().numpy()
        plt.plot(x, prior_values, "--", color="red", label="Prior")
        plt.plot(x, gt_values, "--", color="lime", label="Ground Truth")
        plt.xlabel("box dim %d" % k)
        plt.ylabel("weight")

    label_uv_data = extract_data("box_label_uv")
    prior_dists = [dist.Uniform(0., 1.) for k in range(2)]
    gt_dists = [dist.Uniform(0.2, 0.8) for k in range(2)]
    hist_bins = np.arange(0., 1., 0.1)
    for k in range(2):
        plt.subplot(5, 2, 7+k)
        n, bins, _ = plt.hist(label_uv_data[k, :], normed=True, label="Hist", bins=hist_bins, log=False)
        x = torch.linspace(bins[0], bins[-1], 100)
        prior_values = torch.exp(prior_dists[k].log_prob(x)).detach().numpy()
        gt_values = torch.exp(gt_dists[k].log_prob(x)).detach().numpy()
        plt.plot(x, prior_values, "--", color="red", label="Prior")
        plt.plot(x, gt_values, "--", color="lime", label="Ground Truth")
        plt.xlabel("label uv %d" % k)
        plt.ylabel("count")

        plt.subplot(5, 2, 9+k)
        plt.plot(label_uv_data[k, :])
        plt.ylabel("label uv %d" % k)
        plt.xlabel("epoch")

    plt.tight_layout()

    plt.show()

import argparse
from collections import namedtuple
import datetime
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
import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist
from pyro.contrib.autoguide import AutoGuideList, AutoDelta, AutoDiagonalNormal, AutoDiscreteParallel
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO
from pyro.optim import Adam

from scene_generation.data.generate_boxes_with_keypoints import (
    BoxWithLabel
)


def generate_single_box(name="box"):
    quat = pyro.sample(
        name + "_quat",
        dist.Uniform(torch.ones(4).double()*-1.01,
                     torch.ones(4).double()*1.01).to_event(1)
    )
    quat = quat / torch.norm(quat)

    xyz_mean = torch.zeros(3).double()
    xyz_scale = torch.ones(3).double() * 0.1
    xyz = pyro.sample(
        name + "_xyz",
        dist.Normal(xyz_mean, xyz_scale).to_event(1)
    )

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

    face_weights = torch.ones(6).double() / 6.
    possible_labels = ['px', 'py', 'pz', 'nx', 'ny', 'nz']
    label_face_ind = pyro.sample(name + "_label_face",
        dist.Categorical(face_weights),
        infer={"enumerate":"sequential"})
    label_face = possible_labels[label_face_ind]

    # UV choices
    label_uv = pyro.sample(
        name + "_label_uv",
        dist.Uniform(torch.zeros(2).double(),
                     torch.ones(2).double()).to_event(1)
    )

    sampled_box = BoxWithLabel(
        pose = torch.cat([quat, xyz]),
        dimensions = dimensions,
        label_face = label_face,
        label_uv = label_uv)
    return sampled_box

def get_label_info_from_box_with_label(box):
    label_origin = torch.zeros(3, dtype=box.dimensions.dtype)
    s = 1
    if box.label_face[0] == 'n':
        s = -1
    box_label_dim = ord(box.label_face[1]) - ord('x')
    label_origin[box_label_dim] = s
    # In the other dims, offset by UV
    other_dims = [0, 1, 2]
    other_dims.remove(box_label_dim)
    label_origin[other_dims[0]] = (box.label_uv[0] - 0.5)*2.0
    label_origin[other_dims[1]] = (box.label_uv[1] - 0.5)*2.0
    label_origin *= box.dimensions/2.
    label_size = torch.ones(3, dtype=box.dimensions.dtype) * 0.05
    label_size[box_label_dim] = 0.001
    return label_origin, label_size

# https://github.com/lkhphuc/pytorch-3d-point-cloud-generation/blob/master/transform.py
def quaternionToRotMatrix(q):
    # q = [V, 4]
    qa, qb, qc, qd = torch.unbind(q, dim=1) # [V,]
    R = torch.stack(
        [torch.stack([1 - 2 * (qc**2 + qd**2),
                      2 * (qb * qc - qa * qd),
                      2 * (qa * qc + qb * qd)]),
         torch.stack([2 * (qb * qc + qa * qd),
                      1 - 2 * (qb**2 + qd**2),
                      2 * (qc * qd - qa * qb)]),
         torch.stack([2 * (qb * qd - qa * qc),
                      2 * (qa * qb + qc * qd),
                      1 - 2 * (qb**2 + qc**2)])]
    ).permute(2, 0, 1)
    return R.to(q.device)


def rotMatrixToQuaternion(R):
    """
    R = [3, 3]
    From https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201
    This code uses a modification of the algorithm described in:
    https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
    which is itself based on the method described here:
    http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    Altered to work with the column vector convention instead of row vectors
    """
    q = torch.empty(4, dtype=R.dtype)
    R = torch.t(R)
    if R[2, 2] < 0:
        if R[0, 0] > R[1, 1]:
            t = 1 + R[0, 0] - R[1, 1] - R[2, 2]
            q[:] = torch.tensor([R[1, 2]-R[2, 1],  t,  R[0, 1]+R[1, 0],  R[2, 0]+R[0, 2]])
        else:
            t = 1 - R[0, 0] + R[1, 1] - R[2, 2]
            q[:] = torch.tensor([R[2, 0]-R[0, 2],  R[0, 1]+R[1, 0],  t,  R[1, 2]+R[2, 1]])
    else:
        if R[0, 0] < -R[1, 1]:
            t = 1 - R[0, 0] - R[1, 1] + R[2, 2]
            q[:] = torch.tensor([R[0, 1]-R[1, 0],  R[2, 0]+R[0, 2],  R[1, 2]+R[2, 1],  t])
        else:
            t = 1 + R[0, 0] + R[1, 1] + R[2, 2]
            q[:] = torch.tensor([t,  R[1, 2]-R[2, 1],  R[2, 0]-R[0, 2],  R[0, 1]-R[1, 0]])

    q = q * 0.5 / torch.sqrt(t)
    return q


def transParamsToHomMatrix(q, t):
    """q = [V, 4], t = [V,3]"""
    N = q.size(0)
    R = quaternionToRotMatrix(q) # [V,3,3]
    Rt = torch.cat([R, t.unsqueeze(-1)], dim=2) # [V,3,4]
    hom_aug = torch.cat([torch.zeros([N, 1, 3]).double(),
                         torch.ones([N, 1, 1]).double()],
                        dim=2).to(Rt.device)
    RtHom = torch.cat([Rt, hom_aug], dim=1) # [V,4,4]
    return RtHom

def generate_keypoints_from_box_with_label(box):
    assert(isinstance(box, BoxWithLabel))
    pts = torch.tensor(
        np.array([[-1., -1., -1., -1, 1., 1., 1., 1.],
                  [-1., -1., 1., 1., -1., -1., 1., 1.],
                  [-1., 1., -1., 1., -1., 1., -1., 1.]])).double()
    pts = (pts.T * box.dimensions).T / 2.
    # At make a point for the label origin
    label_origin, _ = get_label_info_from_box_with_label(box)
    pts = torch.cat([pts, label_origin[:, np.newaxis]], dim=1)
    vals = torch.zeros(pts.shape[1]).double().reshape(1, -1)
    vals[0, -1] = 1.
    quat = box.pose[:4].reshape(1, -1)
    xyz = box.pose[-3:].reshape(1, -1)
    tf_mat = transParamsToHomMatrix(quat, xyz)[0]
    pts = torch.mm(tf_mat, torch.cat([pts, torch.ones(pts.shape[1]).double().reshape(1, -1)], dim=0))[:3, :]
    return pts, vals

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

    # Make the maodel components
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
    box = generate_single_box()
    label_origin, label_size = get_label_info_from_box_with_label(box)
    keypoints, vals = generate_keypoints_from_box_with_label(box)
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
        site_values["box_quat"] = rotMatrixToQuaternion(R).detach()
        #R = quaternionToRotMatrix(site_values["box_quat"].reshape(1, -1))[0, :, :]
        site_values["box_dimensions"] = torch.diag(scaling).detach()

        ## Additionally, do gradient descent on the remaining continuous params
        # TODO(gizatt) Maybe make this proper HMC?
        gd_site_names = ["box_label_uv"]
        for gd_k in range(5):
            for site_name in gd_site_names:
                site_values[site_name].requires_grad = True
                if site_values[site_name].grad is not None:
                    site_values[site_name].grad.data.zero_()
            # Compute score using the Pyro models
            conditioned_model = pyro.poutine.condition(
                full_model,
                data=site_values)
            trace = pyro.poutine.trace(conditioned_model).get_trace()
            lps = trace.log_prob_sum()
            lps.backward()
#
            #print("vals after backward: ", site_values)
            for site_name in gd_site_names:
                site_values[site_name].data += site_values[site_name].grad * 0.001
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
            mh_ratio = torch.exp((lps - all_scores[-1])/20.)
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

    return all_scores, all_params, best_score, best_params


if __name__ == "__main__":
    with open("../data/box_observations.yaml", "r") as file:
        all_envs = yaml.load(file)

    vis = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")

    num_boxes_to_sample = 50
    random.seed(42)
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
            observed_keypoints_and_vals, n_samples=15, vis=vis)
        all_scores += these_scores
        all_params += these_params

    plt.figure()
    plt.subplot(5, 1, 1)
    plt.plot(all_scores)
    plt.xlabel("llog")
    plt.ylabel("count")

    def extract_data(param_name):
        # Returns as shape [<param_dims>, num_samples]
        return np.stack([params[param_name] for params in all_params], axis=-1)

    box_dims_data = extract_data("box_dimensions")
    prior_dists = [dist.InverseGamma(
        pyro.param("dimensions_alpha")[k],
        pyro.param("dimensions_beta")[k]) for k in range(3)]
    for k in range(3):
        plt.subplot(5, 3, 4+k)
        n, bins, _ = plt.hist(box_dims_data[k, :], normed=True, label="Hist")
        x = torch.linspace(bins[0], bins[-1], 100)
        prior_values = torch.exp(prior_dists[k].log_prob(x)).detach().numpy()
        plt.plot(x, prior_values, "--", color="red", label="Prior")
        plt.xlabel("box dim %d" % k)
        plt.ylabel("weight")

        plt.subplot(5, 3, 7+k)
        plt.plot(box_dims_data[k, :])
        plt.ylabel("box dim %d" % k)
        plt.xlabel("epoch")

    label_uv_data = extract_data("box_label_uv")
    prior_dists = [dist.Uniform(0., 1.) for k in range(2)]
    for k in range(2):
        plt.subplot(5, 2, 7+k)
        n, bins, _ = plt.hist(label_uv_data[k, :], normed=True, label="Hist")
        x = torch.linspace(bins[0], bins[-1], 100)
        prior_values = torch.exp(prior_dists[k].log_prob(x)).detach().numpy()
        plt.plot(x, prior_values, "--", color="red", label="Prior")
        plt.xlabel("label uv %d" % k)
        plt.ylabel("count")

        plt.subplot(5, 2, 9+k)
        plt.plot(label_uv_data[k, :])
        plt.ylabel("label uv %d" % k)
        plt.xlabel("epoch")

    plt.tight_layout()
    plt.show()

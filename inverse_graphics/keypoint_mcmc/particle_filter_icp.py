from copy import deepcopy
from collections import namedtuple
import math
import os
import time
import sys

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.stats import gaussian_kde
from scipy.special import logsumexp
from skimage import measure
import trimesh

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

import pydrake

import torch
import torch.distributions.constraints as constraints
import torch.nn.functional as F
import torch.distributions as dist

from scene_generation.utils.torch_quaternion import (
    quat2mat,
    rotation_matrix_to_quaternion,
    transParamsToHomMatrix,
    expmap_to_quaternion
)

def make_unit_box_pts_and_normals(N):
    box = trimesh.creation.box()
    pts, faces = trimesh.sample.sample_surface_even(box, count=N)
    pts = torch.tensor(pts.T.copy(), dtype=torch.float32)
    normals = torch.stack([torch.tensor(box.face_normals[k].copy(), dtype=torch.float32) for k in faces]).T
    return pts, normals

class ParticleFilterIcp_Particles():
    ''' A single particle represents a complete object
    configuration guess -- a 3DOF translation t,
    a rotation R in SO(3), and x / y / z scaling
    in a 3-vector S.

    This collects them into a batch of particles,
    so S is Nx3, R is Nx3x3, and t is Nx3.'''
    def __init__(self, S, t, R):
        self.S = S
        self.t = t
        self.R = R

    def transform_points(self, model_pts):
        scale_matrices = torch.diag_embed(self.S)
        repeated_model_pts = model_pts.unsqueeze(0).repeat((self.S.shape[0], 1, 1))
        repeated_translations = self.t.unsqueeze(-1).repeat((1, 1, model_pts.shape[1]))
        pts = torch.bmm(torch.bmm(self.R, scale_matrices), repeated_model_pts) + repeated_translations
        return pts

    def transform_normals(self, model_normals):
        # https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/geometry/transforming-normals
        # Build our TF matrix, then invert + transpose it.
        # (If it was pure rotation, this would be unnecessary.)
        scale_matrices = torch.diag_embed(self.S)
        tf_mat = torch.bmm(self.R, scale_matrices)
        tf_mat = torch.inverse(tf_mat).transpose(-2, -1)
        repeated_model_normals = model_normals.unsqueeze(0).repeat((self.S.shape[0], 1, 1))
        normals = torch.bmm(tf_mat, repeated_model_normals)
        # Normalize
        normals = F.normalize(normals, p=2, dim=1)
        return normals

    def draw(self, vis, model_pts, model_normals=None, size=0.005):
        ''' Expects model_pts in 3xN format as a torch tensor. '''
        # Apply transform to the model pts
        pts = self.transform_points(model_pts)
        if model_normals is not None:
            normals = self.transform_normals(model_normals)
        cmap = mpl.cm.get_cmap("jet")
        for k, pt_set in enumerate(pts):
            color = cmap(k / float(pts.shape[0]))[:3]
            model_colors = np.tile(np.array(color).reshape(3, 1), (1, pt_set.shape[1]))
            vis["particle_%02d" % k].set_object(
                g.PointCloud(
                    position=pt_set.numpy(),
                    color=model_colors,
                    size=size))
            if model_normals is not None:
                verts = np.empty((3, pt_set.shape[1]*2))
                verts[:, 0::2] = pt_set.numpy()
                verts[:, 1::2] = pt_set.numpy() + normals[k, ...].numpy()*0.1
                vis["particle_%02d_normal" % k].set_object(
                    g.LineSegments(
                        g.PointsGeometry(
                            verts),
                        g.LineBasicMaterial(width=0.001)))

class ParticleFilterIcp():
    ''' Runs a particle filter that uses an ICP update
    (with random perturbations) as the process update,
    and a point-to-plane object (plus a prior over the
    object shape parameters) as the measurement. '''

    p2p_corresp_info_struct = namedtuple(
        'PointToPlaneScoreInfo',
        ['batched_model_pts',
         'batched_scene_pts',
         'batched_model_normals',
         'min_distances_per_scene_pt',
         'min_distances_per_model_pt',
         'closest_model_ind_per_scene_pt',
         'closest_scene_ind_per_model_pt']
    )
    def __init__(self, model_pts, model_normals,
                 model_descriptors=None,
                 descriptor_scaling=1.0,
                 keep_ratio=0.5,
                 num_particles = 10,
                 top_N=3,
                 min_corresp_distance=0.05,
                 random_walk_process_prob = 0.25,
                 shape_prior_mean = torch.tensor([0.25, 0.25, 0.25], dtype=torch.float32),
                 shape_prior_shape = torch.tensor([2., 2., 2.], dtype=torch.float32),# Bigger is wider
                 vis=None): 
        assert len(model_pts.shape) == 2 and model_pts.shape[0] == 3, model_pts.shape
        assert model_pts.shape == model_normals.shape
        self.model_pts = model_pts
        self.model_normals = model_normals
        if model_descriptors is not None:
            assert (len(model_descriptors.shape) == 2 and
                    model_descriptors.shape[1] == model_pts.shape[1]), model_descriptors.shape
        self.model_descriptors = model_descriptors
        self.descriptor_scaling = descriptor_scaling
        self.num_particles = num_particles
        self.random_walk_process_prob = random_walk_process_prob
        self.keep_ratio = keep_ratio
        self.min_corresp_distance = min_corresp_distance
        self.vis = vis
        self.top_N = top_N

        # TODO make these optional args?
        self.random_walk_shape_step_var = 0.005
        self.random_walk_trans_step_var = 0.005
        self.random_walk_rot_step_var = 0.025  # Radians
        self.num_icp_steps_per_update = 3

        assert(shape_prior_mean.shape == (3,))
        assert(shape_prior_shape.shape == (3,))
        self.shape_prior_dist = dist.Gamma(
            concentration=shape_prior_shape, rate=shape_prior_shape/shape_prior_mean)

        self.inlier_dist = dist.Normal(
            torch.tensor([0.]), torch.tensor([0.001]))
        self.outlier_dist = dist.Normal(
            torch.tensor([0.]), torch.tensor([1.0]))

        self.outlier_dist
        self.particles = None

    def _reset_particles(self, scene_pts=None):
        ''' Initialize a set of random particles, using the scene pts
        for the mean translation if they're provided. '''
        if scene_pts is None:
            average_translation = torch.zeros((3,))
        else:
            average_translation = torch.mean(scene_pts, dim=1)
        self.particles = []
        all_S = self.shape_prior_dist.sample(sample_shape=(self.num_particles,))
        all_S = torch.tensor([0.4, 0.15, 0.15]).repeat((self.num_particles, 1))
        all_t = dist.MultivariateNormal(
            loc=average_translation,
            covariance_matrix=torch.eye(3, 3)).sample(sample_shape=(self.num_particles,))
        # Make random unit quaternions
        all_q = dist.Normal(torch.tensor([0.]), torch.tensor([1.])).sample(sample_shape=(self.num_particles, 4))
        all_q = F.normalize(all_q, p=2, dim=1)
        # Convert to rotation matrices
        all_R = quat2mat(all_q)
        self.particles = ParticleFilterIcp_Particles(
            S=all_S, t=all_t, R=all_R)

    def _compute_correspondences(self, scene_pts, scene_descriptors=None, p2p=False):
        batched_model_pts = self.particles.transform_points(self.model_pts).cuda()
        batched_model_normals = self.particles.transform_normals(self.model_normals).cuda()

        # Batch up the scene pts as well
        batched_scene_pts = scene_pts.unsqueeze(0).repeat(self.num_particles, 1, 1).cuda()
        # Get all pairwise distances
        num_model_pts = batched_model_pts.shape[-1]
        num_scene_pts = batched_scene_pts.shape[-1]
        # Model pts in the rows, scene pts in the cols
        expanded_model_pts = batched_model_pts.unsqueeze(-1).repeat(1, 1, 1, num_scene_pts).cuda()
        expanded_model_normals = batched_model_normals.unsqueeze(-1).repeat(1, 1, 1, num_scene_pts).cuda()
        expanded_scene_pts = batched_scene_pts.unsqueeze(-2).repeat(1, 1, num_model_pts, 1).cuda()

        # Geometric scene point distances
        if p2p:
            # Point to plane            
            pairwise_distances = torch.abs(
                torch.sum((expanded_model_pts - expanded_scene_pts)*expanded_model_normals,
                dim=1))
        else:
            # Pure L2 distances
            pairwise_distances = torch.norm(expanded_model_pts - expanded_scene_pts, p=2, dim=1)

        # Plus descriptor distances, if they're being done
        if scene_descriptors is not None:
            assert scene_descriptors.shape[0] == self.model_descriptors.shape[0]
            batched_model_descriptors = self.model_descriptors.unsqueeze(0).repeat(self.num_particles, 1, 1).cuda()
            batched_scene_descriptors = scene_descriptors.unsqueeze(0).repeat(self.num_particles, 1, 1).cuda()
            expanded_model_descriptors = batched_model_descriptors.unsqueeze(-1).repeat(1, 1, 1, num_scene_pts).cuda()
            expanded_scene_descriptors = batched_scene_descriptors.unsqueeze(-2).repeat(1, 1, num_model_pts, 1).cuda()
            pairwise_distances += torch.norm(expanded_model_descriptors - expanded_scene_descriptors, p=2, dim=1)*self.descriptor_scaling

        # Get min per scene point
        if self.top_N == 1:
            # Faster, I'm guessing?
            min_distances_per_scene_pt, inds_per_scene_pt = torch.min(pairwise_distances, dim=-2)
            min_distances_per_scene_pt = min_distances_per_scene_pt.unsqueeze(1)
            inds_per_scene_pt = inds_per_scene_pt.unsqueeze(1)
            min_distances_per_model_pt, inds_per_model_pt = torch.min(pairwise_distances, dim=-1)
            min_distances_per_model_pt = min_distances_per_model_pt.unsqueeze(1)
            inds_per_model_pt = inds_per_model_pt.unsqueeze(1)
        else:
            min_distances_per_scene_pt, inds_per_scene_pt = torch.sort(pairwise_distances, dim=-2)
            min_distances_per_scene_pt = min_distances_per_scene_pt[:, :self.top_N, :]
            inds_per_scene_pt = inds_per_scene_pt[:, :self.top_N, :]
            min_distances_per_model_pt, inds_per_model_pt = torch.sort(pairwise_distances, dim=-1)
            min_distances_per_model_pt = min_distances_per_model_pt[:, :, :self.top_N].permute((0, 2, 1))
            inds_per_model_pt = inds_per_model_pt[:, :, :self.top_N].permute((0, 2, 1))

        return self.p2p_corresp_info_struct(
            batched_model_pts=batched_model_pts.cpu(),
            batched_scene_pts=batched_scene_pts.cpu(),
            batched_model_normals=batched_model_normals.cpu(),
            min_distances_per_scene_pt=min_distances_per_scene_pt.cpu(),
            min_distances_per_model_pt=min_distances_per_model_pt.cpu(),
            closest_model_ind_per_scene_pt=inds_per_scene_pt.cpu(),
            closest_scene_ind_per_model_pt=inds_per_model_pt.cpu()
        )

    def get_corresponded_model_pts_in_world(self, p2p_corresp_info):
        # Optional ICP-step-specific post-processing on the p2p corresp info
        # struct.
        all_corresponded_model_pts = []
        for k in range(self.num_particles):
            corresponded_model_pts = []
            for rank in range(self.top_N):
                corresponded_model_pts.append(
                    torch.index_select(
                        p2p_corresp_info.batched_model_pts[k, :],
                        dim=-1,
                        index=p2p_corresp_info.closest_model_ind_per_scene_pt[k, rank, :]))
            all_corresponded_model_pts.append(torch.stack(corresponded_model_pts, dim=0))

        return (p2p_corresp_info.batched_scene_pts.unsqueeze(1).repeat((1, self.top_N, 1, 1)),
                torch.stack(all_corresponded_model_pts, dim=0))

    def get_corresponded_scene_pts_in_world(self, p2p_corresp_info):
        # Optional ICP-step-specific post-processing on the p2p corresp info
        # struct.
        all_corresponded_scene_pts = []
        for k in range(self.num_particles):
            corresponded_scene_pts = []
            for rank in range(self.top_N):
                corresponded_scene_pts.append(
                    torch.index_select(
                        p2p_corresp_info.batched_scene_pts[k, :],
                        dim=-1,
                        index=p2p_corresp_info.closest_scene_ind_per_model_pt[k, rank, :]))
            all_corresponded_scene_pts.append(torch.stack(corresponded_scene_pts, dim=0))
        return (torch.stack(all_corresponded_scene_pts, dim=0),
                p2p_corresp_info.batched_model_pts.unsqueeze(1).repeat((1, self.top_N, 1, 1)))

    def get_full_correspondence_set(self, p2p_corresp_info):
        # [n_particles, N_closest_points, 3, N_scene_pts]
        corresponded_scene_pts_s2m, corresponded_model_pts_s2m = \
            self.get_corresponded_model_pts_in_world(p2p_corresp_info)
        # [n_particles, N_closest_points, 3, N_model_pts]
        corresponded_scene_pts_m2s, corresponded_model_pts_m2s = \
            self.get_corresponded_scene_pts_in_world(p2p_corresp_info)
        corresponded_scene_pts = torch.cat([corresponded_scene_pts_s2m, corresponded_scene_pts_m2s], dim=-1)
        corresponded_model_pts = torch.cat([corresponded_model_pts_s2m, corresponded_model_pts_m2s], dim=-1)

        # Use distances to discard points, taking mean over the top_N
        s2m_dists = torch.mean(p2p_corresp_info.min_distances_per_scene_pt, dim=1)
        m2s_dists = torch.mean(p2p_corresp_info.min_distances_per_model_pt, dim=1)

        # Discard those that are too far, to start with
        s2m_not_close_enough = s2m_dists >= self.min_corresp_distance
        m2s_not_close_enough = m2s_dists >= self.min_corresp_distance
        # But if too few, don't bother doing this step
        n_in_range = torch.sum(~s2m_not_close_enough) + torch.sum(~m2s_not_close_enough)
        print("In range: %d" % n_in_range)
        if n_in_range > 20:
            # Discard them
            not_close_enough = torch.cat([s2m_not_close_enough, m2s_not_close_enough], dim=-1)
            big = torch.max(s2m_dists) + torch.max(m2s_dists)
            s2m_dists[s2m_not_close_enough] += big
            m2s_dists[m2s_not_close_enough] += big
            for k in range(self.num_particles):
                corresponded_scene_pts[k, :, :, not_close_enough[k, :]] = 0.
                corresponded_model_pts[k, :, :, not_close_enough[k, :]] = 0.
        # Keep a ratio of points from both scene + model
        N_s2m = int(self.keep_ratio * s2m_dists.shape[-1])
        N_m2s = int(self.keep_ratio * m2s_dists.shape[-1])

        _, keep_s2m = torch.sort(s2m_dists, dim=-1, descending=False)
        _, keep_m2s = torch.sort(m2s_dists, dim=-1, descending=False)

        keep_inds = torch.cat([
            keep_s2m[:, :N_s2m], keep_m2s[:, :N_m2s] + s2m_dists.shape[-1]], dim=-1)

        downselected_scene_pts = []
        downselected_model_pts = []
        for k in range(self.num_particles):
            downselected_scene_pts.append(corresponded_scene_pts[k, :, :, keep_inds[k, :]])
            downselected_model_pts.append(corresponded_model_pts[k, :, :, keep_inds[k, :]])
        corresponded_scene_pts = torch.stack(downselected_scene_pts, dim=0)
        corresponded_model_pts = torch.stack(downselected_model_pts, dim=0)

        # Now we can stack them back into a more manage-able shape...
        corresponded_model_pts = torch.cat(
            [corresponded_model_pts[:, k, :, :] for k in range(self.top_N)], dim=-1)
        corresponded_scene_pts = torch.cat(
            [corresponded_scene_pts[:, k, :, :] for k in range(self.top_N)], dim=-1)
        return corresponded_scene_pts, corresponded_model_pts

    def _do_random_step(self):
        # Process update: lazy mode is just random steps
        size_dist = dist.Normal(
            torch.tensor([0.]),
            torch.tensor([self.random_walk_shape_step_var]))
        size_step = size_dist.sample(sample_shape=(self.num_particles, 3)).squeeze(-1)

        trans_dist = dist.Normal(
            torch.tensor([0.]),
            torch.tensor([self.random_walk_trans_step_var]))
        trans_step = trans_dist.sample(sample_shape=(self.num_particles, 3)).squeeze(-1)

        # TODO I'm being kind of ad-hoc randomly sampling rotations --
        # uniformly randomly sampling an axis, and folded-normally sampling
        # an angle, and then converting to rotation matrix. I'm not sure what
        # distribution this actually is, but it should produce relatively
        # random small rotations.
        axes = dist.Normal(
            torch.tensor([0.]),
            torch.tensor([1.])).sample(sample_shape=(self.num_particles, 3))
        axes = F.normalize(axes, p=2, dim=1)
        angles_dist = dist.Normal(
            torch.tensor([0.]),
            torch.tensor([self.random_walk_rot_step_var]))
        angles = torch.abs(angles_dist.sample(sample_shape=(self.num_particles, 1)))
        quats_step = expmap_to_quaternion(
            (axes*angles).squeeze(-1))
        Rs_step = quat2mat(quats_step)
        self.particles.S = torch.clamp(self.particles.S + size_step, 0.05)
        self.particles.t = self.particles.t + trans_step
        self.particles.R = torch.bmm(Rs_step, self.particles.R)

    def _do_icp_step(self, scene_pts, scene_descriptors=None):
        p2p_corresp_info = self._compute_correspondences(scene_pts, scene_descriptors=scene_descriptors, p2p=False)
        
        # Laziest, unfounded method I'll try first to get some
        # basic machinery online:
        # First filter into only inlier correspondences,
        # then align centroids to get translation update,
        # then solve for a combined rotation+scale matrix
        # to align the clouds.

        corresponded_scene_pts, corresponded_model_pts = self.get_full_correspondence_set(p2p_corresp_info)
        N_pts = corresponded_scene_pts.shape[-1]

        if self.vis is not None:
            # Draw transformed model points
            self.particles.draw(self.vis["model"],
                                self.model_pts,
                                model_normals=None,
                                size=0.01)

            # Draw correspondence lines
            for k in range(self.num_particles):
                verts = np.empty((3, N_pts*2))
                verts[:, 0::2] = corresponded_scene_pts[k, :, :].reshape(3, -1)
                verts[:, 1::2] = corresponded_model_pts[k, :, :].reshape(3, -1)
                self.vis['corresp_%02d' % (k)].set_object(
                    g.LineSegments(g.PointsGeometry(verts),
                                   g.LineBasicMaterial(width=0.001)))


        model_centroids = torch.mean(corresponded_model_pts, dim=-1)
        scene_centroids = torch.mean(corresponded_scene_pts, dim=-1)
        T_update = (scene_centroids - model_centroids)
        aligned_model_pts = corresponded_model_pts + T_update.unsqueeze(-1).repeat(1, 1, N_pts)
        self.particles.t = self.particles.t + T_update

        # Orthogonal-but-not-orthonormal Procrustes
        # via the tandem algorithm, see 2.1
        # http://empslocal.ex.ac.uk/people/staff/reverson/uploads/Site/procrustes.pdf
        M = torch.bmm(corresponded_scene_pts, aligned_model_pts.transpose(1, 2))
        # Pre-compute part of eq (8)
        denomerator = torch.sum(aligned_model_pts ** 2, dim=2)
        scale_estimate = torch.eye(3, 3).unsqueeze(0).repeat(self.num_particles, 1, 1) #torch.diag_embed(self.particles.S)
        for k in range(1):
            u, s, v = torch.svd(torch.bmm(M, scale_estimate))
            R_estimate = torch.bmm(u, v.transpose(1, 2))
            # Eq(8) update to scaling terms
            numerator = torch.sum(M * R_estimate, dim=1)
            scale_estimate = torch.diag_embed(numerator / denomerator)

        # Limit the amount the scale estimate can change per step
        # to prevent divergence in the first few steps.
        scale_estimate = torch.clamp(scale_estimate, 1.0, 1.0)

        self.particles.S = self.particles. S* torch.diagonal(scale_estimate, dim1=1, dim2=2)
        self.particles.S = torch.clamp(self.particles.S, 0.05, 1.0)
        self.particles.R = torch.bmm(self.particles.R, R_estimate)

    def _do_process_update(self, scene_pts, scene_descriptors=None):
        #if torch.rand(1) <= self.random_walk_process_prob:
        #self._do_random_step()
        #else:
        for k in range(self.num_icp_steps_per_update):
            self._do_icp_step(scene_pts, scene_descriptors=scene_descriptors)

    def _score_current_particles(self, scene_pts, scene_descriptors=None):
        # Get the distances to the nearest few points
        p2p_corresp_info = self._compute_correspondences(scene_pts, scene_descriptors=scene_descriptors, p2p=True)
        # Dims here are going to be [n_particles x self.top_N x n_scene_pts]
        min_distances_per_scene_pt = p2p_corresp_info.min_distances_per_scene_pt

        # Evaluate the actual score according to maximum likelihood under
        # a mixture-of-distribution thing. TODO(gizatt) This isn't precisely
        # formulated...
        log_inlier_scores = self.inlier_dist.log_prob(min_distances_per_scene_pt)
        log_outlier_scores = self.outlier_dist.log_prob(min_distances_per_scene_pt)
        scores = torch.mean(torch.max(log_inlier_scores, log_outlier_scores), dim=(1, 2))
        # Add to it the shape score vs the prior
        # TODO(gizatt) Isn't this going to be *totally overwhelmed* by the point score?
        shape_prior_log_score = torch.sum(self.shape_prior_dist.log_prob(self.particles.S), dim=1)
        scores = scores + shape_prior_log_score
        return scores

    def _do_resampling(self, scene_pts, scene_descriptors=None):
        scores = self._score_current_particles(scene_pts, scene_descriptors=scene_descriptors)

        # Resampling time!
        # Pure duplication resampling
        normalized_scores = scores - torch.logsumexp(scores, dim=0)
        resample_dist = dist.Categorical(logits=normalized_scores)
        new_inds = resample_dist.sample(sample_shape=(self.num_particles,))
        # Finally, repopulate our particles using those
        self.particles = ParticleFilterIcp_Particles(
            S=self.particles.S[new_inds, ...].clone(),
            R=self.particles.R[new_inds, ...].clone(),
            t=self.particles.t[new_inds, ...].clone())
        
    def _run(self, num_outer_steps, num_inner_steps, scene_pts, scene_descriptors=None):
        ''' Run the particle filter for the given number of steps. '''
        for step_j in range(num_outer_steps):
            for step_k in range(num_inner_steps):
                self._do_process_update(scene_pts)
            self._do_resampling(scene_pts)


    def infer(self, scene_pts, num_outer_steps, num_inner_steps, scene_descriptors=None):
        ''' Initialize and run the filter on the provided scene points,
        and return the resulting particle set. '''
        self._reset_particles(scene_pts)
        self._run(num_outer_steps, num_inner_steps, scene_pts, scene_descriptors=scene_descriptors)
        return self.particles

def collect_test_runs(save_dir="test_runs.pt"):
    vis = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
    vis.delete()

    model_pts, model_normals = make_unit_box_pts_and_normals(N=200)
    model_pts[2, :] *= 2.
    model_colors = np.zeros((3, model_pts.shape[1]))
    model_colors[0, :] = 1.


    scene_pts, _ = make_unit_box_pts_and_normals(N=200)
    scene_pts[1, :] *= 2.
    scene_pts = scene_pts[:, scene_pts[2, :] > 0.25]
    scene_colors = np.zeros((3, scene_pts.shape[1]))
    scene_colors[1, :] = 1.

    #vis["model"].set_object(
    #        g.PointCloud(
    #            position=model_pts.numpy(),
    #            color=model_colors,
    #            size=0.02))
    vis["scene"].set_object(
            g.PointCloud(
                position=scene_pts.numpy(),
                color=scene_colors,
                size=0.05))

    #vis["scene"].delete()

    icp = ParticleFilterIcp(
        model_pts=model_pts,
        model_normals=model_normals, num_particles = 50,
        vis=None) #vis["icp_internal"])
    icp._reset_particles(scene_pts)
    all_particle_history = []
    for k in range(10):
        for j in range(10):
            icp._do_process_update(scene_pts)
            icp.particles.draw(vis["model"], model_pts, size=0.02)
            scores = icp._score_current_particles(scene_pts)
            all_particle_history.append((scores, deepcopy(icp.particles)))
        icp._do_resampling(scene_pts)
        print("Resampling iter %02d" % k)

    # Collate the runs into some simple array formats
    all_Rs = []
    all_ts = []
    all_Ss = []
    all_scores = []
    for score, particle in all_particle_history:
        all_Rs.append(particle.R)
        all_ts.append(particle.t)
        all_Ss.append(particle.S)
        all_scores.append(score)
    all_Rs = torch.cat(all_Rs, dim=0)
    all_ts = torch.cat(all_ts, dim=0)
    all_Ss = torch.cat(all_Ss, dim=0)
    all_scores = torch.cat(all_scores, dim=0)

    torch.save({"all_Rs": all_Rs,
                "all_ts": all_ts,
                "all_Ss": all_Ss,
                "all_scores": all_scores},
                save_dir)

def plot_all_particles(R_history, R_vec_history, t_history, S_history, score_history):
    # Analysis of posterior shape distribution
    fig = plt.figure()
    ax = fig.add_subplot(211, projection='3d')
    colors = mpl.cm.get_cmap('jet')(score_history / np.max(score_history))
    ax.scatter(S_history[:, 0], S_history[:, 1], S_history[:, 2],
               s=25, c=colors, alpha=0.01)
    ax.set_title("All shape samples")
    ax.set_xlabel('x')
    ax.set_xlim(1.5, 2.5)
    ax.set_ylim(0.5, 1.5)
    ax.set_zlim(0, 1)
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax = fig.add_subplot(212, projection='3d')
    ax.scatter(R_vec_history[:, 0], R_vec_history[:, 1], R_vec_history[:, 2],
               c=colors, alpha=0.01)
    ax.set_title("Rotation directions of +x")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

def plot_kde(R_history, R_vec_history, t_history, S_history, score_history,
             mins, maxes, num_samples, additional_points_to_plot=None):
    print("Building KDE")
    kde = gaussian_kde(S_history.T,
                       weights=score_history)
    # 1D slice
    # xi = np.linspace(0, 5., 10000)
    # yi = np.zeros(xi.shape) + 1.
    # zi = np.zeros(xi.shape) + 1.
    # coords = np.vstack([item.ravel() for item in [xi, yi, zi]])
    # density = kde(coords)
    # plt.figure()
    # plt.plot(xi, density)
    # plt.show()
    spacing = (maxes - mins) / num_samples
    print((maxes-mins)/spacing)
    xi, yi, zi = np.mgrid[mins[0]:maxes[0]:spacing[0],
                          mins[1]:maxes[1]:spacing[1],
                          mins[2]:maxes[2]:spacing[2]]
    coords = np.vstack([item.ravel() for item in [xi, yi, zi]]) 
    density = kde(coords).reshape(xi.shape)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    levels = [100, 10, 2]
    cm = mpl.cm.get_cmap('jet')
    for k, levelset_divider in enumerate(levels):
        levelset = np.max(density)/levelset_divider
        print("Starting marching cubes at density %f / (out off max %f)"
                % (levelset, np.max(density)))
        verts, faces, normals, values = measure.marching_cubes_lewiner(
            density, levelset, spacing=spacing)
        verts += mins
        # Fancy indexing: `verts[faces]` to generate a collection of triangles
        color = cm(float(k) / len(levels))
        mesh = Poly3DCollection(
            verts[faces], alpha=0.2,
            edgecolors=None,
            facecolors=color)
        ax.add_collection3d(mesh)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.set_xlim(mins[0], maxes[0])
    ax.set_ylim(mins[1], maxes[1])
    ax.set_zlim(mins[2], maxes[2])

    if additional_points_to_plot is not None:
        ax.scatter(additional_points_to_plot[:, 0],
                   additional_points_to_plot[:, 1],
                   additional_points_to_plot[:, 2],
                   s=50,
                   c="red")


    plt.tight_layout()
    plt.show()

def do_analysis_of_saved_runs(save_dir="test_runs.pt"):
    keep_frac = 0.5

    data = torch.load(save_dir)
    score_history = data["all_scores"].numpy()
    keep_iters = int(len(score_history)*keep_frac)
    print("Keeping history of %d iters." % keep_iters)
    score_history = score_history[-keep_iters:]
    # Normalize scores
    score_history -= logsumexp(score_history)
    score_history = np.exp(score_history)
        
    R_history = data["all_Rs"][-keep_iters:]
    R_vec_history = torch.bmm(
        R_history,
        torch.tensor([1., 0., 0.]).unsqueeze(0).unsqueeze(-1).repeat(R_history.shape[0], 1, 1)
    ).squeeze().numpy()
    t_history = data["all_ts"].numpy()[-keep_iters:]
    S_history = data["all_Ss"].numpy()[-keep_iters:]
    # Sort shape into ascending order 
    S_history = np.sort(S_history, axis=1)[:, ::-1]
    plot_all_particles(R_history, R_vec_history,
                       t_history, S_history,
                       score_history)


    mins = np.array([1.5, 0.5, 0.])
    maxes = np.array([2.5, 1.5, 1.])
    num_samples = np.array([30, 30, 30])
    plot_kde(R_history, R_vec_history,
             t_history, S_history,
             score_history,
             mins, maxes, num_samples)

if __name__ == "__main__":
    torch.set_default_tensor_type(torch.FloatTensor)

    collect_test_runs("test_runs.pt")
    do_analysis_of_saved_runs("test_runs.pt")

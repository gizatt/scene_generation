import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import time

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
    pts = torch.tensor(pts.T)
    normals = torch.stack([torch.tensor(box.face_normals[k]) for k in faces]).T
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
        pts = torch.bmm(self.R, torch.bmm(scale_matrices, repeated_model_pts)) + repeated_translations
        return pts

    def transform_normals(self, model_normals):
        # https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/geometry/transforming-normals
        # Build our TF matrix, then invert + transpose it.
        # (If it was pure rotation, this would be unnecessary.)
        scale_matrices = torch.diag_embed(self.S)
        tf_mat = torch.bmm(self.R, scale_matrices)
        tf_mat = torch.inverse(tf_mat).transpose(-2, -1)
        repeated_model_pts = model_pts.unsqueeze(0).repeat((self.S.shape[0], 1, 1))
        pts = torch.bmm(tf_mat, repeated_model_pts)
        return pts

    def draw(self, vis, model_pts, size=0.005):
        ''' Expects model_pts in 3xN format as a torch tensor,
        and model colors as 3xN numpy array. '''
        # Apply transform to the model pts
        pts = self.transform_points(model_pts)
        cmap = mpl.cm.get_cmap("jet")
        for k, pt_set in enumerate(pts):
            color = cmap(k / float(pts.shape[0]))[:3]
            model_colors = np.tile(np.array(color).reshape(3, 1), (1, pt_set.shape[1]))
            vis["particle_%02d" % k].set_object(
                g.PointCloud(
                    position=pt_set.numpy(),
                    color=model_colors,
                    size=size))


class ParticleFilterIcp():
    ''' Runs a particle filter that uses an ICP update
    (with random perturbations) as the process update,
    and a point-to-plane object (plus a prior over the
    object shape parameters) as the measurement. '''
    def __init__(self, model_pts, model_normals,
                 num_particles = 10,
                 random_walk_process_prob = 0.1,
                 shape_prior_mean = torch.tensor([0.5, 0.5, 0.5]),
                 shape_prior_shape = torch.tensor([2., 2., 2.])): # Bigger is wider
        assert len(model_pts.shape) == 2 and model_pts.shape[0] == 3, model_pts.shape
        assert model_pts.shape == model_normals.shape
        self.model_pts = model_pts
        self.model_normals = model_normals
        self.num_particles = num_particles
        self.random_walk_process_prob = random_walk_process_prob

        # TODO make these optional args?
        self.random_walk_shape_step_var = 0.1
        self.random_walk_trans_step_var = 0.1
        self.random_walk_rot_step_var = 0.1  # Radians

        assert(shape_prior_mean.shape == (3,))
        assert(shape_prior_shape.shape == (3,))
        self.shape_prior_dist = dist.Gamma(
            concentration=shape_prior_shape, rate=shape_prior_shape/shape_prior_mean)

        self.point_to_plane_inlier_dist = dist.Normal(
            torch.tensor([0.]), torch.tensor([0.01]))
        self.point_to_plane_outlier_dist = dist.Normal(
            torch.tensor([0.]), torch.tensor([1.0]))

        self.point_to_plane_outlier_dist
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
        self.particles.S = torch.clamp(self.particles.S + size_step, 0.01)
        self.particles.t = self.particles.t + trans_step
        self.particles.R = torch.bmm(Rs_step, self.particles.R)

    def _do_process_update(self, scene_pts):
        self._do_random_step()
        # TODO ICP step

    def _do_resampling(self, scene_pts):
        batched_model_pts = self.particles.transform_points(self.model_pts)
        batched_model_normals = self.particles.transform_normals(self.model_normals)
        # Batch up the scene pts as well
        batched_scene_pts = scene_pts.unsqueeze(0).repeat(self.num_particles, 1, 1)
        # Get all pairwise distances
        num_model_pts = batched_model_pts.shape[-1]
        num_scene_pts = batched_scene_pts.shape[-1]
        # Model pts in the rows, scene pts in the cols
        expanded_model_pts = batched_model_pts.unsqueeze(-1).repeat(1, 1, 1, num_scene_pts)
        expanded_model_normals = batched_model_normals.unsqueeze(-1).repeat(1, 1, 1, num_scene_pts)
        expanded_scene_pts = batched_scene_pts.unsqueeze(-2).repeat(1, 1, num_model_pts, 1)

        # Pure L2 distances
        # pairwise_distances = torch.norm(batched_model_pts - batched_scene_pts, p=2, dim=1)
        # Point to plane distances
        pairwise_distances = torch.abs(
            torch.sum((expanded_model_pts - expanded_scene_pts)*expanded_model_normals,
                      dim=1))
        # Get min per scene point
        min_distances_per_scene_pt, inds = torch.min(pairwise_distances, dim=-2)

        # Finally, evaluate the actual score:
        # TODO(gizatt) OK, what am I doing here?
        log_inlier_scores = self.point_to_plane_inlier_dist.log_prob(min_distances_per_scene_pt)
        log_outlier_scores = self.point_to_plane_outlier_dist.log_prob(min_distances_per_scene_pt)
        scores = torch.sum(torch.max(log_inlier_scores, log_outlier_scores), dim=1)
        # Add to it the shape score vs the prior
        # TODO(gizatt) Isn't this going to be *totally overwhelmed* by the point score?
        shape_prior_log_score = torch.sum(self.shape_prior_dist.log_prob(self.particles.S), dim=1)
        print("Shape prior scores: ", shape_prior_log_score)
        scores = scores + shape_prior_log_score
        print("Scores: ", scores)

        # Only necessary for ICP update, I think
        # Pick out a vector of the corresponding scene pts
        # TODO(gizatt) This, too, could be done in a single vector op, but via
        # awkward flattening, I think?
        #corresponded_scene_pts = []
        #for k in range(self.num_particles):
        #    corresponded_scene_pts.append(torch.index_select(batched_model_pts[k, :], dim=-1, index=inds[k, :]))
        #corresponded_scene_pts = torch.stack(corresponded_scene_pts, dim=0)

        # Resampling time!
        # Pure duplication resampling
        normalized_scores = scores - torch.logsumexp(scores, dim=0)
        print("Normalized scores: ", normalized_scores)
        resample_dist = dist.Categorical(logits=normalized_scores)
        new_inds = resample_dist.sample(sample_shape=(self.num_particles,))
        # Finally, repopulate our particles using those
        self.particles = ParticleFilterIcp_Particles(
            S=self.particles.S[new_inds, ...],
            R=self.particles.R[new_inds, ...],
            t=self.particles.t[new_inds, ...])
        
    def _run(self, num_steps, scene_pts):
        ''' Run the particle filter for the given number of steps. '''
        for step_k in range(num_steps):
            self._do_process_update(scene_pts)
            self._do_resampling(scene_pts)


    def infer(self, scene_pts, num_steps=100):
        ''' Initialize and run the filter on the provided scene points,
        and return the resulting particle set. '''
        self._reset_particles(scene_pts)
        self._run(num_steps)
        return self.particles


if __name__ == "__main__":
    torch.set_default_tensor_type(torch.DoubleTensor)

    vis = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
    vis[""]

    model_pts, model_normals = make_unit_box_pts_and_normals(N=1000)
    model_colors = np.zeros((3, model_pts.shape[1]))
    model_colors[0, :] = 1.


    scene_pts, _ = make_unit_box_pts_and_normals(N=1000)
    scene_pts = scene_pts[:, scene_pts[2, :] > 0.25]
    scene_colors = np.zeros((3, scene_pts.shape[1]))
    scene_colors[1, :] = 1.

    vis["model"].set_object(
            g.PointCloud(
                position=model_pts.numpy(),
                color=model_colors,
                size=0.01))
    vis["scene"].set_object(
            g.PointCloud(
                position=scene_pts.numpy(),
                color=scene_colors,
                size=0.01))

    vis["scene"].delete()

    icp = ParticleFilterIcp(model_pts=model_pts, model_normals=model_normals, num_particles = 100)
    icp._reset_particles(scene_pts)
    for k in range(1000):
        icp._run(1, scene_pts)
        icp.particles.draw(vis["scene"], model_pts, size=0.01)
        time.sleep(0.1)
    

import functools
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import time

import torch
from torch.autograd.function import once_differentiable

import pyro
import pyro.distributions as dist

from pydrake.all import (AutoDiffXd,
                         GurobiSolver,
                         RigidBodyTree)
from pydrake.multibody.rigid_body import RigidBody
from pydrake.solvers import ik
from pydrake.multibody.joints import PrismaticJoint, RevoluteJoint
from pydrake.multibody.shapes import Box, VisualElement
from pydrake.multibody.collision import CollisionElement


def nullspace(A, atol=1e-13, rtol=0):
    """Compute an approximate basis for the nullspace of A.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length k will be treated
        as a 2-D with shape (1, k)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    ns : ndarray
        If `A` is an array with shape (m, k), then `ns` will be an array
        with shape (k, n), where n is the estimated dimension of the
        nullspace of `A`.  The columns of `ns` are a basis for the
        nullspace; each element in numpy.dot(A, ns) will be approximately
        zero.
    """
    A = np.atleast_2d(A)
    u, s, vh = np.linalg.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns


def object_origins_within_bounds_constraint_constructor_factory(
        x_min, x_max):
    def build_constraint(rbt, x_min, x_max):
        ik_constraints = []
        for body_i in range(rbt.get_num_bodies()-1):
            # Origin of body must be inside of the
            # bounds of the board
            points = np.zeros([3, 1])
            lb = np.array(x_min.reshape(3, 1))
            ub = np.array(x_max.reshape(3, 1))
            ik_constraints.append(ik.WorldPositionConstraint(
                rbt, body_i+1, points, lb, ub))
        return ik_constraints

    return functools.partial(build_constraint, x_min=x_min, x_max=x_max)


def objects_completely_within_bounds_constraint_constructor_factory(
        x_min, x_max):
    def build_constraint(rbt, x_min, x_max):
        ik_constraints = []
        for body_i in range(rbt.get_num_bodies()-1):
            # All corners on body must be inside of the
            # bounds of the board
            body = rbt.get_body(body_i+1)
            visual_elements = body.get_visual_elements()
            if len(visual_elements) > 0:
                points = visual_elements[0].getGeometry().getPoints()
                lb = np.tile(np.array(x_min), (points.shape[1], 1)).T
                ub = np.tile(np.array(x_max), (points.shape[1], 1)).T
                ik_constraints.append(ik.WorldPositionConstraint(
                    rbt, body_i+1, points, lb, ub))
        return ik_constraints

    return functools.partial(build_constraint, x_min=xmin, x_max=x_max)

def object_at_specified_pose_constraint_constructor_factory(
        body_i, lb_q, ub_q):
    if lb_q.shape[0] != 6 or ub_q.shape[0] != 6:
        raise ValueError("Expected q is 6x1")

    def build_constraint(rbt, body_i, lb_q, ub_q):
        ik_constraints = []
        body = rbt.get_body(body_i+1)
        # Abuse that everything is a floating body
        # the "proper" way to do this is a PostureConstraint,
        # but that RigidBodyConstraint type doesn't have an
        # eval method.
        points = np.zeros([3, 1])
        lb_pos = np.array(lb_q[0:3].reshape(3, 1))
        ub_pos = np.array(ub_q[0:3].reshape(3, 1))
        ik_constraints.append(ik.WorldPositionConstraint(
            rbt, body_i+1, points, lb_pos, ub_pos))
        lb_rot = np.array(lb_q[3:6].reshape(3, 1))
        ub_rot = np.array(ub_q[3:6].reshape(3, 1))
        ik_constraints.append(ik.WorldEulerConstraint(
            rbt, body_i+1, lb_rot, ub_rot))
        return ik_constraints

    return functools.partial(build_constraint, body_i=body_i, lb_q=lb_q, ub_q=ub_q)

def rbt_at_posture_constraint_constructor_factory(
        inds, lb_q, ub_q):
    def build_constraint(rbt, inds, lb_q, ub_q):
        posture_constraint = ik.PostureConstraint(rbt)
        posture_constraint.setJointLimits(inds, lb_q, ub_q)
        return [posture_constraint]

    return functools.partial(build_constraint, inds=inds, lb_q=lb_q, ub_q=ub_q)

def projectToFeasibilityWithIK(rbt, q0, extra_constraint_constructors=[],
                               verbose=False, max_num_retries=4):
    '''
    Given:
    - a Rigid Body Tree (rbt)
    - a pose (q0, with q0.shape == [rbt.get_num_positions(), 1])
    - a list of extra constraint constructors that each
        return a list of RigidBodyConstraints when given
        an RBT (see just above for some examples)
    Returns:
    - qf: The nearest pose that satisfies a MinDistanceConstraint plus
    all of the listed extra constraints, using RigidBodyTree IK
    (which is a nonlinear optimization under the hood, probably using
    SNOPT or IPOPT).
    - dqf_dq0: A local estimate of how the solution changes w.r.t.
    the initial configuration, by inspecting the null space
    of the optimization at the solution point.
    - constraint_violation_directions: A set of basis vectors that
    *positively* span the constrained space at the solution point.
    - constraint_allowance_directions: A set of basis vectors that
    *positively* span the null space at the solution point.
    '''

    ik_constraints = []
    ik_constraints.append(ik.MinDistanceConstraint(
        model=rbt, min_distance=0.001, active_bodies_idx=[],
        active_group_names=set()))
    for extra_constraint_constructor in extra_constraint_constructors:
        ik_constraints += extra_constraint_constructor(rbt)

    for k in range(max_num_retries):
        options = ik.IKoptions(rbt)
        options.setDebug(True)
        options.setMajorIterationsLimit(10000)
        options.setIterationsLimit(100000)
        # Each retry, add more random noise to the
        # seed, to try to break out of the broken initial seed.
        results = ik.InverseKin(
            rbt, q0 + np.random.normal(scale=k*0.1), q0, ik_constraints, options)
        if results.info[0] == 1:
            break

    qf = results.q_sol[0]
    info = results.info[0]
    dqf_dq0 = np.eye(qf.shape[0])
    constraint_violation_directions = []
    if info != 1:
        if verbose:
            print("Warning: returned info = %d != 1 after %d retries"  % (info, max_num_retries))
    if True or info == 1 or info == 100:
        # We've solved an NLP of the form:
        # qf = argmin_q || q - q_0 ||
        #        s.t. phi(q) >= 0
        #
        # which projects q_0 into the feasible set $phi(q) >= 0$.
        # We want to return the gradient of qf w.r.t. q_0.
        # We'll tackle an approximation of this (which isn't perfect,
        # but is a start):
        # We'll build a linear approximation of the active set at
        # the optimal value, and project the incremental dq_0 into
        # the null space of this active set.

        # These vectors all point in directions that would
        # bring q off of the constraint surfaces into
        # illegal space.
        # They *positively* span the infeasible space -- that is,
        # they all point into it, and if there's not a vector pointing
        # in a particular direction, that is feasible space.
        constraint_violation_directions = []

        cache = rbt.doKinematics(qf)
        for i, constraint in enumerate(ik_constraints):
            if not isinstance(constraint, ik.SingleTimeKinematicConstraint):
                continue
            c, dc = constraint.eval(0, cache)
            lb, ub = constraint.bounds(0)

            phi_lb = c - lb
            phi_ub = ub - c
            for k in range(c.shape[0]):
                if phi_lb[k] < -1E-4 or phi_ub[k] < -1E-4:
                    if verbose:
                        print("Bounds violation detected, "
                              "solution wasn't feasible")
                        print("%f <= %f <= %f" % (lb[k], c[k], ub[k]))
                        print("Constraint type ", type(constraint))
                        print("qf: ", qf)
                        print("q0: ", q0.reshape(1, -1))

                # If ub = lb and ub is active, then lb is also active,
                # and we'll have double-added this vector. But this is
                # OK since we care about positive span.
                if phi_lb[k] < 1E-6:
                    # Not allowed to go down
                    constraint_violation_directions.append(-dc[k, :])
                if phi_ub[k] < 1E-6:
                    # Not allowed to go down
                    constraint_violation_directions.append(dc[k, :])

        # Build a full matrix C(q0_new - qstar) = 0
        # that approximates the feasible set.
        if len(constraint_violation_directions) > 0:
            constraint_violation_directions = np.vstack(
                constraint_violation_directions)
            ns = nullspace(constraint_violation_directions)
            dqf_dq0 = np.eye(qf.shape[0])
            dqf_dq0 = np.dot(np.dot(dqf_dq0, ns), ns.T)
        else:
            # No null space so movements
            dqf_dq0 = np.eye(qf.shape[0])

    if len(constraint_violation_directions) == 0:
        constraint_violation_directions = np.zeros([0, qf.shape[0]])

    return qf, info, dqf_dq0, constraint_violation_directions


def buildRegularizedGradient(dqf_dq0, viol_dirs, gamma):
    return (torch.eye(dqf_dq0.shape[0], dtype=dqf_dq0.dtype)*gamma
            + (1. - gamma)*dqf_dq0)


class projectToFeasibilityWithIKTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q0, rbt, ik_constraints, gamma=0.01):
        # q0 being a tensor, rbt and constraints being inputs
        # to projectToFeasibilityWithIK
        qf, info, dqf_dq0, viol_dirs = projectToFeasibilityWithIK(
            rbt, q0.cpu().detach().numpy().copy(), ik_constraints)
        qf = qf.reshape(-1, 1)

        ctx.save_for_backward(
            torch.tensor(qf, dtype=q0.dtype),
            torch.tensor(dqf_dq0, dtype=q0.dtype),
            torch.tensor(viol_dirs, dtype=q0.dtype),
            torch.tensor(gamma))
        return torch.tensor(qf, dtype=q0.dtype, requires_grad=q0.requires_grad)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        qf, dqf_dq0, viol_dirs, gamma = ctx.saved_tensors

        # Better way to do this:
        # only regularize within the null space, rather than
        # uniformly across the matrix.
        # Should only make a meaningful difference when the
        # dqf/dq_0 is not identity in the feasible set, I think,
        # and I think gradient will still be in at least approximately
        # the right direction in that case.

        # Basic gradient descent:
        # "Regularize" the gradient a bit by suggesting
        # a little off-axis movement
        regularized_dqf_dq0 = buildRegularizedGradient(
            dqf_dq0, viol_dirs, gamma)
        return (torch.mm(
                    regularized_dqf_dq0,
                    grad.view(-1, 1)),
                None, None, None)

        '''
        This is dumb, wasn't a good idea.
        This assumes that the gradient is being asked for in the
        direction that descent is going to happen.
        This kind of logic is needed at the gradient descent implementation
        level.
        # Do projection into the violation directions to figure out
        # what gradient directions are "illegal", and remove only
        # those directions.
        # viol dirs is N x nq
        grad_out = grad.clone()
        viol_dirs_t = torch.t(viol_dirs)
        # I think this has to be done iteratively to not
        # overcount / doublecount violations that are are in the
        # same direction -- viol dirs are not necessarily
        # all pointing in different dirs.
        for k in range(viol_dirs.shape[0]):
            grad_projection = torch.dot(viol_dirs_t[:, k].flatten(), grad_out.flatten())
            if grad_projection > 0:
                grad_out -= grad_projection*viol_dirs_t[:, k].view(-1, 1)
        return (grad_out, None, None)
        '''


class PassthroughWithGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, dy_dx):
        ctx.save_for_backward(dy_dx.clone())
        return y.clone()

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        dy_dx,  = ctx.saved_tensors
        nq = dy_dx.shape[1]
        n_batch = dy_dx.shape[0]
        return (torch.bmm(dy_dx,
                          torch.unsqueeze(grad, -1)).squeeze(),
                torch.bmm(torch.eye(nq, nq).expand(n_batch, nq, nq),
                          torch.unsqueeze(grad, -1)).squeeze(), None)


class ProjectToFeasibilityWithIKAsDistribution(dist.TorchDistribution):
    """
        Given a point q0 to project to feasibility (as well as
        supporting rbt + constraint list, for input to
        projectToFeasibilityWithIK), and weights representing
        uncertainty about points varying within the feasible
        set + outside of it, enable forward sampling of
        projected qf as well as log-likelihood calculation.

        At construction time, the actual projection (qf) is computed
        by a call to projectToFeasibilityWithIK, and this result
        is stored for any time a call to "sample" is made. Sampling
        is deterministic.

        However, to enable SVI, the log prob calculation is done by
        representing the projection as a draw from the composition
        of two multivariate normals:
        - If the value is in the cone of local infeasible space
        (as represented by the vector of infeasible directions returned
        by the original projection call), the log likelihood is the same
        as a multivariate normal centered at qf with variance
        outside_feasible_set_variance.
        - Otherwise, the log likelihood is the same as a multivariate
        normal centered at qf with variance within_feasible_set_variance.

        q0_fixed is prepended to q0 to build the full configuration vector
        for the RBT, and qf is constrained to equal q0_fixed for those
        first indices.

        Gamma is a regularization to soften the derivative of the
        sample w.r.t. the input -- see the Torch autograd function above.
    """
    has_rsample = True
    arg_constraints = {"q0": torch.distributions.constraints.real,
                       "within_feasible_set_variance": torch.distributions.constraints.positive,
                       "outside_feasible_set_variance": torch.distributions.constraints.positive}
    def __init__(self, rbt, q0,
                 ik_constraints,
                 within_feasible_set_variance,
                 outside_feasible_set_variance,
                 q0_fixed=None,
                 gamma=0.01,
                 noisy_projection=False,
                 validate_args=False,
                 event_select_inds=None):
        batch_shape = q0.shape[:-1]
        if event_select_inds is not None:
            self.event_select_inds = event_select_inds
        else:
            self.event_select_inds = torch.tensor(range(q0.shape[-1]))
        event_shape = (self.event_select_inds.shape[0],)

        if isinstance(within_feasible_set_variance, float):
            within_feasible_set_variance = torch.tensor(
                within_feasible_set_variance, dtype=q0.dtype)
        if isinstance(outside_feasible_set_variance, float):
            outside_feasible_set_variance = torch.tensor(
                outside_feasible_set_variance, dtype=q0.dtype)

        if not isinstance(rbt, list):
            rbt = [rbt]

        # Basically repeat what the Torch autograd implementation does,
        # but we need to extract some intermediate state to build
        # the log prob multivariate distribs.
        nq = q0.shape[-1]
        if q0_fixed is not None:
            nq += q0_fixed.shape[-1]
        assert(len(batch_shape) == 1)
        all_qfs = []
        all_regularized_dqf_dq0s = []
        all_viol_dirs = []
        for k in range(batch_shape[0]):
            if rbt[min(k, len(rbt)-1)] is not None:
                q0_variable_part = q0[k, :].cpu().detach().numpy().copy().reshape(-1, 1)
                extra_ik_constraints = []
                if q0_fixed is None:
                    q0_full = q0_variable_part
                    extract_inds = range(nq)
                else:
                    q0_fixed_part = q0_fixed[k, :].cpu().detach().numpy().copy().reshape(-1, 1)
                    q0_full = np.vstack([q0_fixed_part, q0_variable_part])
                    extract_inds = range(q0_fixed_part.shape[0], nq)
                    # It's OK to use a PostureConstraint here
                    # as this should *always* be satisfied or we have
                    # series numerical problems, and this is only a constraint
                    # on non-variable elements anyway and won't appear in gradients.
                    extra_ik_constraints.append(rbt_at_posture_constraint_constructor_factory(
                        range(q0_fixed_part.shape[0]), q0_fixed_part, q0_fixed_part))
                qf, info, dqf_dq0, viol_dirs = projectToFeasibilityWithIK(
                    rbt[min(k, len(rbt)-1)], q0_full, ik_constraints + extra_ik_constraints)
                qf = torch.tensor(qf[extract_inds], dtype=q0.dtype)
                dqf_dq0 = torch.tensor(dqf_dq0[extract_inds, :][:, extract_inds], dtype=q0.dtype)
                viol_dirs = torch.tensor(viol_dirs[:, extract_inds], dtype=q0.dtype)
                regularized_dqf_dq0 = buildRegularizedGradient(
                    dqf_dq0, viol_dirs, gamma)
            else:
                # Direct passthrough. Being passed "None" for RBT will happen
                # during batched analysis of variable-length scenes when scenes
                # that are "done" generating are still being processed.
                # Their log prob should be frozen, so whatever we produce is irrelevant.
                qf = q0[k, :].clone().flatten()
                regularized_dqf_dq0 = torch.eye(qf.shape[-1])
                viol_dirs = torch.empty(0, qf.shape[-1])

            all_qfs.append(qf)
            all_regularized_dqf_dq0s.append(regularized_dqf_dq0)
            all_viol_dirs.append(viol_dirs)

        all_qfs_tensor = torch.stack(all_qfs)
        all_regularized_dqf_dq0s_tensor = torch.stack(all_regularized_dqf_dq0s)

        self._nq_variable = qf.shape[0]
        self._noisy_projection = noisy_projection
        self._rsample = PassthroughWithGradient.apply(
            q0, all_qfs_tensor, all_regularized_dqf_dq0s_tensor).index_select(
                -1, self.event_select_inds)
        self._viol_dirs = all_viol_dirs
        self._within_feasible_set_distrib = dist.Normal(
            self._rsample,
            within_feasible_set_variance.expand(event_shape)).to_event(1)
        self._outside_feasible_set_distrib = dist.Normal(
            self._rsample,
            outside_feasible_set_variance.expand(event_shape)).to_event(1)

        super(ProjectToFeasibilityWithIKAsDistribution, self).__init__(
            batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(
            ProjectToFeasibilityWithIKAsDistribution, _instance)
        batch_shape = torch.Size(batch_shape)
        new.event_select_inds = self.event_select_inds
        new._nq_variable = self._nq_variable
        new._noisy_projection = self._noisy_projection
        new._rsample = self._rsample.expand(batch_shape + self.event_shape)
        new._viol_dirs = self._viol_dirs

        if batch_shape != self.batch_shape:
            raise NotImplementedError("Not handling viol_dirs properly")

        new._within_feasible_set_distrib = self._within_feasible_set_distrib.expand(batch_shape)
        new._outside_feasible_set_distrib = self._outside_feasible_set_distrib.expand(batch_shape)

        super(ProjectToFeasibilityWithIKAsDistribution, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @torch.distributions.constraints.dependent_property
    def support(self):
        return torch.distributions.constraints.real

    def log_prob(self, value):
        assert value.shape[-1] == self.event_shape[0]
        if value.dim() > 1:
            assert value.shape[0] == self.batch_shape[0]

        # Difference of each new value from the projected point
        diff_values = value - self._rsample
        # Project that into the infeasible cone -- we're moving out
        # towards local infeasible space if any of these inner products
        # is positive.
        # I'll use a "large" threshold of violation for now...
        # TODO(gizatt) What's a good val for eps?
        eps = 1E-5
        use_outside_feasible_set_distrib = torch.zeros(self.batch_shape[0])
        for k in range(self.batch_shape[0]):
            if self._viol_dirs[k].shape[0] > 0:
                use_outside_feasible_set_distrib[k] = torch.any(
                    torch.mm(self._viol_dirs[k].index_select(-1, self.event_select_inds),
                             diff_values[k, :].view(-1, 1)) >= eps)
        # Get log probs from both distribs, and choose across
        # the batch based on presence in the infeasible cone
        return ((1. - use_outside_feasible_set_distrib) * self._within_feasible_set_distrib.log_prob(value) +
                (use_outside_feasible_set_distrib) * self._outside_feasible_set_distrib.log_prob(value))

    def rsample(self, sample_shape=torch.Size()):
        if self._noisy_projection:
            return self._within_feasible_set_distrib.sample(sample_shape)
        else:
            return self._rsample.expand(sample_shape + self.batch_shape + self.event_shape)
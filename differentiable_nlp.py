import functools
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import sys

import torch
from torch.autograd.function import once_differentiable
from torch.distributions import constraints

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
        constraints = []
        for body_i in range(rbt.get_num_bodies()-1):
            # Origin of body must be inside of the
            # bounds of the board
            points = np.zeros([3, 1])
            lb = np.array(x_min.reshape(3, 1))
            ub = np.array(x_max.reshape(3, 1))
            constraints.append(ik.WorldPositionConstraint(
                rbt, body_i+1, points, lb, ub))
        return constraints

    return functools.partial(build_constraint, x_min=x_min, x_max=x_max)


def objects_completely_within_bounds_constraint_constructor_factory(
        x_min, x_max):
    def build_constraint(rbt, x_min, x_max):
        constraints = []
        for body_i in range(rbt.get_num_bodies()-1):
            # All corners on body must be inside of the
            # bounds of the board
            body = rbt.get_body(body_i+1)
            visual_elements = body.get_visual_elements()
            if len(visual_elements) > 0:
                points = visual_elements[0].getGeometry().getPoints()
                lb = np.tile(np.array(x_min), (points.shape[1], 1)).T
                ub = np.tile(np.array(x_max), (points.shape[1], 1)).T
                constraints.append(ik.WorldPositionConstraint(
                    rbt, body_i+1, points, lb, ub))
        return constraints

    return functools.partial(build_constraint, x_min=xmin, x_max=x_max)


def projectToFeasibilityWithIK(rbt, q0, extra_constraint_constructors=[],
                               verbose=False):
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

    constraints = []
    constraints.append(ik.MinDistanceConstraint(
        model=rbt, min_distance=0.01, active_bodies_idx=[],
        active_group_names=set()))
    for extra_constraint_constructor in extra_constraint_constructors:
        constraints += extra_constraint_constructor(rbt)

    options = ik.IKoptions(rbt)
    options.setDebug(True)
    options.setMajorIterationsLimit(10000)
    options.setIterationsLimit(100000)
    results = ik.InverseKin(
        rbt, q0, q0, constraints, options)

    qf = results.q_sol[0]
    info = results.info[0]
    dqf_dq0 = np.eye(qf.shape[0])
    constraint_violation_directions = []
    # if info != 1:
    #    print("Warning: returned info = %d != 1" % info)
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
        for i, constraint in enumerate(constraints):
            c, dc = constraint.eval(0, cache)
            lb, ub = constraint.bounds(0)

            phi_lb = c - lb
            phi_ub = ub - c
            for k in range(c.shape[0]):
                if phi_lb[k] < -1E-6 or phi_ub[k] < -1E-6:
                    if verbose:
                        print("Bounds violation detected, "
                              "solution wasn't feasible")
                        print("%f <= %f <= %f" % (lb[k], c[k], ub[k]))
                        print("Constraint type ", type(constraint))
                    return qf, info, dqf_dq0, constraint_violation_directions

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
    def forward(ctx, q0, rbt, constraints, gamma=0.01):
        # q0 being a tensor, rbt and constraints being inputs
        # to projectToFeasibilityWithIK
        qf, info, dqf_dq0, viol_dirs = projectToFeasibilityWithIK(
            rbt, q0.cpu().detach().numpy().copy(), constraints)
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
    def forward(ctx, x, dx):
        ctx.save_for_backward(dx.clone())
        return x.clone()

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        dx = ctx.saved_tensors
        return (dx, None)


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

        Gamma is a regularization to soften the derivative of the
        sample w.r.t. the input -- see the Torch autograd function above.
    """
    has_rsample = False
    arg_constraints = {"q0": constraints.real,
                       "within_feasible_set_variance": constraints.positive,
                       "outside_feasible_set_variance": constraints.positive}

    def __init__(self, rbt, q0, constraints,
                 within_feasible_set_variance,
                 outside_feasible_set_variance,
                 gamma=0.01,
                 validate_args=False):
        self.batch_mode = q0.dim > 1
        print "constructor q0: ", q0
        batch_shape = q0.shape[:-1]
        event_shape = (q0.shape[-1],)
        print "batch shape: ", batch_shape
        print "event shape: ", event_shape

        if isinstance(within_feasible_set_variance, float):
            within_feasible_set_variance = torch.tensor(
                within_feasible_set_variance, dtype=q0.dtype)
        if isinstance(outside_feasible_set_variance, float):
            outside_feasible_set_variance = torch.tensor(
                outside_feasible_set_variance, dtype=q0.dtype)

        # Basically repeat what the Torch autograd implementation does,
        # but we need to extract some intermediate state to build
        # the log prob multivariate distribs.
        nq = rbt.get_num_positions()
        assert(len(batch_shape) == 1)
        all_qfs = []
        all_regularized_dqf_dq0s = []
        all_viol_dirs = []
        for k in range(batch_shape[0]):
            qf, info, dqf_dq0, viol_dirs = projectToFeasibilityWithIK(
                rbt, q0[k, :].cpu().detach().numpy().copy().reshape(nq, 1), constraints)
            qf = torch.tensor(qf.reshape(1, -1), dtype=q0.dtype)
            dqf_dq0 = torch.tensor(dqf_dq0, dtype=q0.dtype)
            viol_dirs = torch.tensor(viol_dirs, dtype=q0.dtype)
            regularized_dqf_dq0 = buildRegularizedGradient(
                dqf_dq0, viol_dirs, gamma)
            all_qfs.append(qf)
            all_regularized_dqf_dq0s.append(regularized_dqf_dq0)
            all_viol_dirs.append(viol_dirs)

        all_qfs = torch.stack(all_qfs)
        all_regularized_dqf_dq0s = torch.stack(all_regularized_dqf_dq0s)

        self._nq = qf.shape[0]
        self._rsample = PassthroughWithGradient.apply(qf, regularized_dqf_dq0)
        self._viol_dirs = all_viol_dirs
        self._within_feasible_set_distrib = dist.Normal(
            qf,
            within_feasible_set_variance.expand(qf.shape[0])).to_event(1)
        self._outside_feasible_set_distrib = dist.Normal(
            qf,
            outside_feasible_set_variance.expand(qf.shape[0])).to_event(1)

        super(ProjectToFeasibilityWithIKAsDistribution, self).__init__(
            batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        print "Expanding to ", batch_shape
        new = self._get_checked_instance(
            ProjectToFeasibilityWithIKAsDistribution, _instance)
        new.batch_mode = True
        batch_shape = torch.Size(batch_shape)
        new._nq = self._nq
        new._rsample = self._rsample.expand(batch_shape + self.event_shape)
        new._viol_dirs = self._viol_dirs

        if batch_shape != self.batch_shape:
            raise NotImplementedError("Not handling viol_dirs properly")

        new._within_feasible_set_distrib = self._within_feasible_set_distrib.expand(batch_shape)
        new._outside_feasible_set_distrib = self._outside_feasible_set_distrib.expand(batch_shape)

        super(ProjectToFeasibilityWithIKAsDistribution, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def log_prob(self, value):
        assert value.dim() == 1 and value.size(0) == self._nq
        # TODO: project onto viol dirs, figure out if we're in the
        # infeasible cone
        return self._within_feasible_set_distrib.log_prob(value)

    def sample(self, sample_shape=torch.Size()):
        print "Sample shape: ", sample_shape
        print "Self.event_shape: ", self.event_shape
        print "Self.batch_shape: ", self.batch_shape
        return self._rsample.expand(sample_shape + self.batch_shape + self.event_shape)
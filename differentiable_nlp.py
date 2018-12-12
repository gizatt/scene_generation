import functools
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import sys

import torch
from torch.autograd.function import once_differentiable

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


def object_origins_within_bounds_constraint_constructor_factory(x_min, x_max):
    
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


def objects_completely_within_bounds_constraint_constructor_factory(x_min, x_max):
    
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


def projectToFeasibilityWithIK(rbt, q0, extra_constraint_constructors=[]):
    # Interface for extra_constraint_constructors:
    #  constraints += extra_constraint_constructor(rbt)
    # (should return constraint list and accept rbt)
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
    #if info != 1:
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
        # bring q off of the constraint surfaces.
        constraint_violation_directions = []

        cache = rbt.doKinematics(qf)
        for i, constraint in enumerate(constraints):
            c, dc = constraint.eval(0, cache)
            lb, ub = constraint.bounds(0)

            phi_lb = c - lb
            phi_ub = ub - c
            for k in range(c.shape[0]):
                if phi_lb[k] < -1E-6 or phi_ub[k] < -1E-6:
                    print("Bounds violation detected, "
                          "solution wasn't feasible")
                    print("%f <= %f <= %f" % (lb[k], c[k], ub[k]))
                    print("Constraint type ", type(constraint))
                    return qf, info, dqf_dq0

                if phi_lb[k] < 1E-6:
                    # Not allowed to go down
                    constraint_violation_directions.append(-dc[k, :])

                # If ub = lb and ub is active, then lb is also active,
                # and we don't need to double-add this vector.
                if phi_ub[k] < 1E-6 and phi_ub[k] != phi_lb[k]:
                    # Not allowed to go up
                    constraint_violation_directions.append(dc[k, :])

        # Build a full matrix C(q0_new - qstar) = 0
        # that approximates the feasible set.
        if len(constraint_violation_directions) > 0:
            C = np.vstack(constraint_violation_directions)
            ns = nullspace(C)
            dqf_dq0 = np.eye(qf.shape[0])
            dqf_dq0 = np.dot(np.dot(dqf_dq0, ns), ns.T)
        else:
            # No null space so movements
            dqf_dq0 = np.eye(qf.shape[0])
    return qf, info, dqf_dq0


class projectToFeasibilityWithIKTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q0, rbt, constraints):
        # q0 being a tensor, rbt and constraints being inputs
        # to projectToFeasibilityWithIK
        qf, info, dqf_dq0 = projectToFeasibilityWithIK(
            rbt, q0.cpu().detach().numpy(), constraints)

        ctx.save_for_backward(qf, info, dqf_dq0)
        return torch.Tensor(qf)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        print "I didn't check this yet"
        return torch.Tensor(dqf_dq0)


def approximate_differentiable_manifold_projection_as_multivariate_normal(
        projection_operator, q0, within_manifold_variance,
        null_space_variance):
    """
    Given a 1-differentiable manifold projection operator that projects
    from `$q \\in R^N$` to another point `$\\hat{q} \\in $R^N$`,
    and given an initial point `$q_0$`, and given weights
    representing downstream uncertainty about points being both
    off the manifold and along the manifold chart, models
    the manifold projection operation as a multivariate gaussian
    with mean `$\\hat{q}$` and variance based on the local chart
    and the given scaling parameters.


    TODO: establish an interface for the
    feasibility projection operator + the things it needs to tell me.
    (For example, just knowing the 1st derivative of the projection,
    dqhat_dq, might be enough, but I'd have to work a little to
    back out the nullspace -- which I already calculate in the
    middle of the feasibility projection method itself. SVD is
    not *that* expensive... but it's still awkward?)

    :param torch.autograd.Function projection_operator:
        Function that performs projection. Must be once-differentiable --
        the local chart (as viewed as a mapping between movements in the
        ambient space and changes in the projected point -- dqhat/dq)
        is encoded as its first derivative.
    :param torch.tensor q0: N-dimensional input to projection_operator that
        represents a point before projection.
    :param torch.tensor within_manifold_variance: variance within the
        chart directions.
    :param torch.tensor null_space_variance: variance in the null space
        of the projection (off-manifold).
    :return pyro.distributions.MultivariateNormal
    """
    return dist.MultivariateNormal(aaah, aaAAAH)


def projectToFeasibilityWithNLP(rbt, q0, board_width, board_height):
    # More generic than above... instead of using IK to quickly
    # assembly the nlp solve that goes to snopt, build it ourselves.
    # (Gives us lower-level control at the cost of complexity.)

    print("TODO, I think this requires a few new drake bindings"
          " for generic nonlinear constraints")
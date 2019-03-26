from __future__ import print_function

from collections import namedtuple
import functools
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import time

import pydrake
from pydrake.autodiffutils import AutoDiffXd
from pydrake.common.eigen_geometry import Quaternion, AngleAxis, Isometry3
from pydrake.forwarddiff import gradient, jacobian
from pydrake.geometry import (
    Box,
    Sphere
)
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.multibody.plant import (
    AddMultibodyPlantSceneGraph,
    CoulombFriction,
    MultibodyPlant
)
from pydrake.multibody.tree import (
    PrismaticJoint,
    SpatialInertia,
    UniformGravityFieldElement,
    UnitInertia
)
from pydrake.solvers.mathematicalprogram import MathematicalProgram, Solve
from pydrake.systems.framework import DiagramBuilder

import torch
from torch.autograd.function import once_differentiable

import pyro
import pyro.distributions as dist


def SetArguments(f, **kwargs):
    return functools.partial(f, **kwargs)


def AddMinimumDistanceConstraint(ik, minimum_distance=0.01):
    ik.AddMinimumDistanceConstraint(minimum_distance)


def GetValAndJacobianOfAutodiffArray(autodiff_ndarray):
    val = np.array([v.value() for v in autodiff_ndarray]).reshape(
        autodiff_ndarray.shape)
    grad = np.stack([v.derivatives() for v in autodiff_ndarray]).reshape(
        autodiff_ndarray.shape + (-1,))
    return val, grad


ProjectMBPToFeasibilityOutput = namedtuple(
    'ProjectMBPToFeasibilityOutput',
    ['qf', 'success', 'dqf_dq0', 'constraint_violation_directions'])


def ProjectMBPToFeasibility(q0, mbp, mbp_context, constraint_adders=[],
                            compute_gradients_at_solution=False,
                            verbose=False):
    '''
        Inputs:
            - q0: Initial guess configuration for the projection.
            - mbp: A MultiBodyPlant. Needs to have a registered
                   and connected SceneGraph for collision query-related
                   constraints to work.
            - constraint_adders: A list of functions f(ik) that mutate
                   a passed IK program to add additional constraints.
            - compute_grads and verbose flags

        Outputs: a namedtuple with fields:
            - "qf": same size as q0, np array
            - "success": True/False
            - "dqf_dq0": if compute_grads flag is False, None.
               Otherwise, nq x nq array.
            - "constraint_violation_directions": If compute_grads flag is
                False, None. Otherwise, a set of vectors that positively
                span the infeasible cone from the solution point.
    '''
    nq = q0.shape[0]
    print("MBP in final: ", mbp)
    ik = InverseKinematics(mbp)
    print("setup done")
    q_dec = ik.q()
    prog = ik.prog()

    # It's always a projection, so we always have this
    # Euclidean norm error between the optimized q and
    # q0.
    prog.AddQuadraticErrorCost(np.eye(nq), q0, q_dec)

    for constraint_adder in constraint_adders:
        constraint_adder(ik)

    # TODO(gizatt) On retries, we may want to mutate
    # this initial seed by a random amount to try to
    # get things to converge.
    prog.SetInitialGuess(q_dec, q0)

    if verbose:
        print("Initial guess: ", q0)
    result = Solve(prog)
    qf = result.GetSolution(q_dec)

    if verbose:
        print("Used solver: ", result.get_solver_id().name())
        print("Success? ", result.is_success())
        print("qf: ", qf)

    # Trickiness to untangle here:
    # We want dqf_dq0. But q0 may only be a subset of the
    # decision variables.
    if compute_gradients_at_solution:
        all_decision_vars = result.GetSolution()

        # Initialize Autodiff version of the decision vars.
        all_decision_vars_autodiff = np.empty(all_decision_vars.shape,
                                              dtype=np.object)
        q_dec_indices = prog.FindDecisionVariableIndices(q_dec)
        for i in range(all_decision_vars.size):
            der = np.zeros(nq)
            if i in q_dec_indices:
                der[q_dec_indices[i]] = 1
            all_decision_vars_autodiff.flat[i] = AutoDiffXd(x.flat[i], der)

        constraints = prog.GetAllConstraints()
        total_constraint_gradient = np.zeros(nq)
        for constraint_i, constraint in enumerate(constraints):
            val_autodiff = prog.EvalBinding(
                constraint, all_decision_vars_autodif)
            # Add only for violations / near-boundaries.
            # TODO(gizatt) verify behavior for equality constraints.
            val_full, jac_full = GetValAndJacobianOfAutodiffArray(val_autodiff)
            val = val_full[q_dec_indices]
            jac = jac_full[q_dec_indices, :]
            if verbose:
                print("Constraint %d:", constraint_i)
                print("Val ad: ", val_autodiff)
                print("Val full: ", val_full)
                print("Val: ", val)
                print("Jac full: ", jac_full)
                print("Jac: ", jac)
            total_constraint_gradient -= (
                val <= constraint.evaluator().lower_bound() + 1E-6).dot(jac)
            total_constraint_gradient += (
                val >= constraint.evaluator().upper_bound() - 1E-6).dot(jac)

        constraint_violation_directions = []
        return ProjectMBPToFeasibilityOutput(
            qf=qf.copy(), success=result.is_success(),
            dqf_dq0=total_constraint_gradient,
            constraint_violation_directions=constraint_violation_directions)
    else:
        return ProjectMBPToFeasibilityOutput(
            qf=qf.copy(), success=result.is_success(),
            dqf_dq0=None, constraint_violation_directions=None)


class PassthroughWithGradient(torch.autograd.Function):
    '''
    Utility for inserting a pre-computed function
    and its gradient into the computation graph.
    Maybe useful if you don't feel like rolling an
    autograd.Function subclass to wrap your differentiable
    function, I guess?

    Inputs:
       - ctx: Pytorch context.
       - x: The input value to this function. Not used.
       - y: The output value of this function. Cloned +
            returned from the forward pass.
       - dy_dx: Precomputed dy/dx. Saved and applied via
            chain rule for backward gradient propagation.
    '''
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


class ProjectToFeasibilityTorch(torch.autograd.Function):
    '''
    Inputs:
        q0:  Initial configuration of the robot that we want to project,
             as a pytorch tensor.
        mbp: A Drake MultiBodyPlant for which q0 is a valid configuration,
            which will be used to build an IK / nonlinear program to
            do the projection.
        gamma: A regularization, multiplied by identity and added to the
            gradient. TODO(gizatt) Justification or better handling.
    '''
    @staticmethod
    def forward(ctx, q0, mbp, gamma=0.01):
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


def testGetValAndJacobianOfAutodiffarray():
    # Test GetValAndJacobianOfAutodiffArray
    def get_y(x):
        return 2*x
    # Manually
    x = np.array([1., 2., 3.])
    y = get_y(x)
    dy_dx = jacobian(get_y, x)

    print("Non-AD:")
    print("x: ", x)
    print("y: ", y)
    print("dy_dx: ", dy_dx)

    # With Autodiff
    x_ad = np.empty(x.shape, dtype=np.object)
    for i in range(x_ad.size):
        der = np.zeros(x_ad.size)
        der[i] = 1.
        x_ad.flat[i] = AutoDiffXd(x.flat[i], der)
    y_ad = get_y(x_ad)

    print("AD:")
    print("x: ", x_ad)
    print("y_ad: ", y_ad)
    print("GetValAndJacobianOfAutodiffArray: ",
          GetValAndJacobianOfAutodiffArray(y_ad))


def setupMBPForProjection():
    builder = DiagramBuilder()
    mbp, _ = AddMultibodyPlantSceneGraph(
        builder, MultibodyPlant(time_step=0.01))

    world_body = mbp.world_body()
    ground_shape = Box(10., 10., 10.)
    ground_body = mbp.AddRigidBody("ground", SpatialInertia(
        mass=10.0, p_PScm_E=np.array([0., 0., 0.]),
        G_SP_E=UnitInertia(1.0, 1.0, 1.0)))
    mbp.WeldFrames(world_body.body_frame(), ground_body.body_frame(),
                   Isometry3(rotation=np.eye(3), translation=[0, 0, -5]))
    mbp.RegisterVisualGeometry(
        ground_body, Isometry3(), ground_shape, "ground_vis",
        np.array([0.5, 0.5, 0.5, 1.]))
    mbp.RegisterCollisionGeometry(
        ground_body, Isometry3(), ground_shape, "ground_col",
        CoulombFriction(0.9, 0.8))

    n_bodies = 2
    for k in range(n_bodies):
        body = mbp.AddRigidBody("body_{}".format(k), SpatialInertia(
            mass=1.0, p_PScm_E=np.array([0., 0., 0.]),
            G_SP_E=UnitInertia(0.1, 0.1, 0.1)))

        body_box = Box(1.0, 1.0, 1.0)
        mbp.RegisterVisualGeometry(
            body, Isometry3(), body_box, "body_{}_vis".format(k),
            np.array([1., 0.5, 0., 1.]))
        mbp.RegisterCollisionGeometry(
            body, Isometry3(), body_box, "body_{}_box".format(k),
            CoulombFriction(0.9, 0.8))

    mbp.AddForceElement(UniformGravityFieldElement())
    mbp.Finalize()

    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    mbp_context = diagram.GetMutableSubsystemContext(
        mbp, diagram_context)
    q0 = mbp.GetPositions(mbp_context).copy()

    return q0, mbp, mbp_context


def testProjection(q0, mbp, mbp_context):
    ProjectMBPToFeasibility(q0, mbp, mbp_context, verbose=True)


if __name__ == "__main__":
    # testGetValAndJacobianOfAutodiffarray()
    q0, mbp, mbp_context = setupMBPForProjection()
    testProjection(q0, mbp, mbp_context)
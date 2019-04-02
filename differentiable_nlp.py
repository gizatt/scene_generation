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
    JointIndex,
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


def AddMBPQuaternionConstraints(ik, mbp):
    # TODO(gizatt) This is a hack, as I can't figure out a way
    # to actually ask which bodies are floating bases.
    # Need resolution of Drake issue #10736.
    # Right now, I assume *all* bodies are floating bases
    # in this function.
    q_dec = ik.q()
    prog = ik.prog()

    def squaredNorm(x):
        return x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2
    for k in range(mbp.num_bodies() - 2):  # Ignore world body + ground body.
        # Quaternion norm
        prog.AddConstraint(
            squaredNorm(q_dec[(k*7):(k*7+4)]) == 1.)
        # Trivial quaternion bounds
        prog.AddBoundingBoxConstraint(
            -np.ones(4), np.ones(4), q_dec[(k*7):(k*7+4)])
        # Conservative bounds on on XYZ
        prog.AddBoundingBoxConstraint(
            np.array([-10., -10., -10.]), np.array([10., 10., 10.]),
            q_dec[(k*7+4):(k*7+7)])


def GetValAndJacobianOfAutodiffArray(autodiff_ndarray):
    val = np.array([v.value() for v in autodiff_ndarray]).reshape(
        autodiff_ndarray.shape)
    grad = np.stack([v.derivatives() for v in autodiff_ndarray]).reshape(
        autodiff_ndarray.shape + (-1,))
    return val, grad


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


ProjectMBPToFeasibilityOutput = namedtuple(
    'ProjectMBPToFeasibilityOutput',
    ['qf', 'success', 'dqf_dq0', 'constraint_violation_directions'])


def EvaluateProjectionDerivativeInfo(
        prog, x, q_dec_indices=None, verbose=None):
    ''' Given a MathematicalProgram prog and the values of all decision
        variables, returns a list of vectors (in decision variable space)
        that positively span the infeasible region. Can optionally only
        consider movement of the decision variables if their indices are
        supplied. '''
    if q_dec_indices is None:
        q_dec_indices = range(x.size)

    # Initialize Autodiff version of the decision vars.
    nq = len(q_dec_indices)
    x_autodiff = np.empty(x.shape, dtype=np.object)
    for i in range(x.size):
        der = np.zeros(nq)
        if i in q_dec_indices:
            der[q_dec_indices[i]] = 1
        x_autodiff.flat[i] = AutoDiffXd(
            x.flat[i], der)

    constraints = prog.GetAllConstraints()
    total_constraint_gradient = np.zeros(nq)
    constraint_violation_directions = []
    for constraint_i, constraint in enumerate(constraints):
        val_autodiff = prog.EvalBinding(
            constraint, x_autodiff)
        # Add only for violations / near-boundaries.
        # TODO(gizatt) verify behavior for equality constraints.
        val, jac_full = GetValAndJacobianOfAutodiffArray(val_autodiff)
        jac = jac_full[:, q_dec_indices]

        if verbose >= 3:
            print("Constraint #", constraint_i)
            print("Val: %s (range [%s, %s])" % (
                val, constraint.evaluator().lower_bound(),
                constraint.evaluator().upper_bound()))
            print("Jac: ", jac)

        lb = constraint.evaluator().lower_bound()
        ub = constraint.evaluator().upper_bound()
        for k in range(jac.shape[0]):
            # Be liberal about not stepping into constraints
            if val[k] < (lb[k] + 1E-6):
                constraint_violation_directions.append(-jac[k, :])
            if val[k] > (ub[k] - 1E-6):
                constraint_violation_directions.append(jac[k, :])

    if len(constraint_violation_directions) == 0:
        return np.empty((0, nq))
    else:
        return np.stack(constraint_violation_directions)


def ProjectMBPToFeasibility(q0, mbp, mbp_context, constraint_adders=[],
                            compute_gradients_at_solution=False,
                            verbose=1):
    '''
        Inputs:
            - q0: Initial guess configuration for the projection.
            - mbp: A MultiBodyPlant. Needs to have a registered
                   and connected SceneGraph for collision query-related
                   constraints to work.
            - constraint_adders: A list of functions f(ik) that mutate
                   a passed IK program to add additional constraints.
            - compute_grads and verbose flags:
                verbose = 0 or False: No printing.
                verbose = 1: Warn on failed projection only.
                verbose = 2: Print initial + resulting positions and
                             solver info.
                verbose = 3: The above + also dqf_dq0 and constraint viol dirs.

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
    ik = InverseKinematics(mbp, mbp_context)
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

    if verbose >= 2:
        print("Initial guess: ", q0)
    result = Solve(prog)
    qf = result.GetSolution(q_dec)

    if verbose >= 2 or (verbose >= 1 and not result.is_success()):
        print("Used solver: ", result.get_solver_id().name())
        print("Success? ", result.is_success())
        print("qf: ", qf)

    if compute_gradients_at_solution:
        constraint_violation_directions = EvaluateProjectionDerivativeInfo(
            prog, result.GetSolution(),
            prog.FindDecisionVariableIndices(q_dec),
            verbose=verbose)

        if constraint_violation_directions.shape[0] > 0:
            ns = nullspace(constraint_violation_directions)
            dqf_dq0 = np.eye(nq)  # Unconstrained version of dqf_dq0
            dqf_dq0 = np.dot(np.dot(dqf_dq0, ns), ns.T)  # Projection step
        else:
            # No null space so movements
            dqf_dq0 = np.eye(nq)

        if verbose >= 3:
            print("Constraint viol dirs: ", constraint_violation_directions)
            print("dqf_dq0: ", dqf_dq0)

        return ProjectMBPToFeasibilityOutput(
            qf=qf.copy(), success=result.is_success(),
            dqf_dq0=dqf_dq0,
            constraint_violation_directions=constraint_violation_directions)
    else:
        return ProjectMBPToFeasibilityOutput(
            qf=qf.copy(), success=result.is_success(),
            dqf_dq0=None, constraint_violation_directions=None)


def buildRegularizedGradient(dqf_dq0, viol_dirs, gamma):
    return (torch.eye(dqf_dq0.shape[0], dtype=dqf_dq0.dtype)*gamma
            + (1. - gamma)*dqf_dq0)


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
        mbp_context: MBP context from a diagram containing the MBP (and a SG
            if you want collision checking to work).
        gamma: A regularization, multiplied by identity and added to the
            gradient. TODO(gizatt) Justification or better handling.
        verbose: Passed through to ProjectMBPToFeasibility.
    '''
    @staticmethod
    def forward(ctx, q0, mbp, mbp_context, constraint_adders=[],
                gamma=0.01, verbose=1):
        output = ProjectMBPToFeasibility(
            q0.cpu().detach().numpy().copy(), mbp, mbp_context,
            constraint_adders, compute_gradients_at_solution=True,
            verbose=verbose)

        if not output.success and verbose > 0:
            print("Warning: projection didn't not succeed.")

        qf = output.qf.reshape(q0.shape)
        dqf_dq0 = output.dqf_dq0
        viol_dirs = output.constraint_violation_directions

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
                    grad.view(-1, 1)).view(qf.shape),
                None, None, None, None, None)

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
        return (grad_out, None, None, None, None, None)
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
    print("\n**** NO CONSTRAINTS *****")
    ProjectMBPToFeasibility(
        q0, mbp, mbp_context,
        compute_gradients_at_solution=True,
        verbose=3)

    print("\n**** +QUATERNION CONSTRAINT *****")
    ProjectMBPToFeasibility(
        q0, mbp, mbp_context,
        [SetArguments(AddMBPQuaternionConstraints, mbp=mbp)],
        compute_gradients_at_solution=True,
        verbose=3)

    print("\n**** +MIN DISTANCE CONSTRAINT *****")
    ProjectMBPToFeasibility(
        q0, mbp, mbp_context,
        [SetArguments(AddMinimumDistanceConstraint, minimum_distance=0.01),
         SetArguments(AddMBPQuaternionConstraints, mbp=mbp)],
        compute_gradients_at_solution=True,
        verbose=3)


def testProjectionTorch(q0, mbp, mbp_context):
    print("\n**** NO CONSTRAINTS *****")
    q0_tensor = torch.tensor(q0, requires_grad=True)
    qf_tensor = ProjectToFeasibilityTorch.apply(
        q0_tensor, mbp, mbp_context, [], 2)
    print("qf_tensor: ", qf_tensor)
    loss = qf_tensor.sum()
    loss.backward()
    print("q0_tensor backward: ", q0_tensor.grad)

    print("\n**** +QUATERNION CONSTRAINT *****")
    qf_tensor = ProjectToFeasibilityTorch.apply(
        q0_tensor, mbp, mbp_context,
        [SetArguments(AddMBPQuaternionConstraints, mbp=mbp)], 2)
    print("qf_tensor: ", qf_tensor)
    loss = qf_tensor.sum()
    loss.backward()
    print("q0_tensor backward: ", q0_tensor.grad)

    print("\n**** +MIN DISTANCE CONSTRAINT *****")
    qf_tensor = ProjectToFeasibilityTorch.apply(
        q0_tensor, mbp, mbp_context,
        [SetArguments(AddMinimumDistanceConstraint, minimum_distance=0.01),
         SetArguments(AddMBPQuaternionConstraints, mbp=mbp)], 2)
    print("qf_tensor: ", qf_tensor)
    loss = qf_tensor.sum()
    loss.backward()
    print("q0_tensor backward: ", q0_tensor.grad)


if __name__ == "__main__":
    # testGetValAndJacobianOfAutodiffarray()
    q0, mbp, mbp_context = setupMBPForProjection()
    # testProjection(q0, mbp, mbp_context)
    testProjectionTorch(q0, mbp, mbp_context)

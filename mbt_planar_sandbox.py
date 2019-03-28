import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
import yaml
import sys

import pydrake
from pydrake.common import FindResourceOrThrow
from pydrake.common.eigen_geometry import Quaternion, AngleAxis, Isometry3
from pydrake.geometry import (
    Box,
    HalfSpace,
    SceneGraph,
    Sphere
)
from pydrake.multibody.tree import (
    PrismaticJoint,
    SpatialInertia,
    UniformGravityFieldElement,
    UnitInertia
)
from pydrake.multibody.plant import (
    AddMultibodyPlantSceneGraph,
    CoulombFriction,
    MultibodyPlant
)

from pydrake.forwarddiff import gradient
from pydrake.multibody.parsing import Parser
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.solvers.ipopt import (IpoptSolver)
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.meshcat_visualizer import MeshcatVisualizer
from pydrake.systems.rendering import PoseBundle

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    builder = DiagramBuilder()
    mbp, scene_graph = AddMultibodyPlantSceneGraph(builder, MultibodyPlant(time_step=0.001))

    # Add ground
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

    n_bodies = 20
    for k in range(n_bodies):
        body = mbp.AddRigidBody("body_{}".format(k), SpatialInertia(
            mass=1.0, p_PScm_E=np.array([0., 0., 0.]),
            G_SP_E=UnitInertia(0.1, 0.1, 0.1)))

        #body_joint_x = PrismaticJoint(
        #    name="body_pris_{}".format(k),
        #    frame_on_parent=world_body.body_frame(),
        #    frame_on_child=body.body_frame(),
        #    axis=[0, 0, 1]
        #    damping=0.)
        #mbp.AddJoint(body_joint_x)

        body_box = Box(1.0, 1.0, 1.0)
        mbp.RegisterVisualGeometry(
            body, Isometry3(), body_box, "body_{}_vis".format(k),
            np.array([1., 0.5, 0., 1.]))
        mbp.RegisterCollisionGeometry(
            body, Isometry3(), body_box, "body_{}_box".format(k),
            CoulombFriction(0.9, 0.8))

    mbp.AddForceElement(UniformGravityFieldElement())
    mbp.Finalize()

    visualizer = builder.AddSystem(MeshcatVisualizer(
        scene_graph,
        zmq_url="tcp://127.0.0.1:6000"))
    builder.Connect(scene_graph.get_pose_bundle_output_port(),
                    visualizer.get_input_port(0))

    diagram = builder.Build()

    diagram_context = diagram.CreateDefaultContext()
    mbp_context = diagram.GetMutableSubsystemContext(
        mbp, diagram_context)
    sg_context = diagram.GetMutableSubsystemContext(
        scene_graph, diagram_context)

    q0 = mbp.GetPositions(mbp_context).copy()
    for k in range(n_bodies):
        offset = k*7
        q0[(offset):(offset+4)] = np.random.random(4)*2.0 - 1.0
        q0[(offset):(offset+4)] /= np.linalg.norm(q0[(offset):(offset+4)])
        q0[(offset+4):(offset+7)] = np.random.randn(3)*2

    ik = InverseKinematics(mbp, mbp_context)
    q_dec = ik.q()
    prog = ik.prog()

    def squaredNorm(x):
        return x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2
    for k in range(n_bodies):
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

    constraint = ik.AddMinimumDistanceConstraint(0.01)
    prog.AddQuadraticErrorCost(np.eye(q0.shape[0])*1.0, q0, q_dec)

    mbp.SetPositions(mbp_context, q0)
    query_object = scene_graph.get_query_output_port().Eval(sg_context)
    print query_object.ComputeSignedDistancePairwiseClosestPoints()[0]

    def draw_at_config(q):
        poses = scene_graph.get_pose_bundle_output_port().Eval(
            diagram.GetMutableSubsystemContext(scene_graph, diagram_context))
        mbp.SetPositions(mbp_context, q)
        visualizer._DoPublish(mbp_context, [])

    q_now = q0.copy()
    draw_at_config(q_now)
    evaluator = constraint.evaluator()
    for k in range(100):
        score = evaluator.Eval(q_now)
        grad = gradient(evaluator.Eval, q_now)
        print "score: ", score
        print "qnow: ", q_now
        print "grad: ", grad
        print "\n\n"
        if np.allclose(grad, grad*0.):
            break
        q_now = q_now - 0.001*grad
        q_now[0:4] /= np.linalg.norm(q_now[0:4])
        time.sleep(0.01)
        draw_at_config(q_now)
    q0_proj = q_now
#
    prog.SetInitialGuess(q_dec, q0)
    print("Solving")
    print "Initial guess: ", q0
    #print IpoptSolver().Solve(prog)
    print prog.Solve()
    #print prog.Solve()
    print prog.GetSolverId().name()
    q0_proj = prog.GetSolution(q_dec)
    print "Final: ", q0_proj
#
    mbp.SetPositions(mbp_context, q0_proj)
    #print mbp.GetPositions(mbp_context)

    #mbp_context.FixInputPort(
    #    mbp.get_actuation_input_port().get_index(), np.zeros(
    #        mbp.get_actuation_input_port().size()))

    simulator = Simulator(diagram, diagram_context)
    simulator.set_target_realtime_rate(1.0)
    simulator.set_publish_every_time_step(False)
    simulator.StepTo(10.0)

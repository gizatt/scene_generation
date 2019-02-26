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
from pydrake.math import (RollPitchYaw)
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
    for scene_iter in range(10000):
        try:
            #np.random.seed(42)
            #random.seed(42)
            builder = DiagramBuilder()
            mbp, scene_graph = AddMultibodyPlantSceneGraph(
                builder, MultibodyPlant(time_step=0.0001))

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

            n_levels = np.random.randint(2, 6)
            level_width = 3
            accept_rate = 0.6
            block_width = 0.25
            block_height = 0.3
            block_margin = 0.01
            poses = []  # [quat, pos]
            for k in range(n_levels):
                for l in range(level_width):
                    if np.random.random() > 1. - accept_rate:
                        body = mbp.AddRigidBody("body_{}-{}".format(k, l), SpatialInertia(
                            mass=0.1, p_PScm_E=np.array([0., 0., 0.]),
                            G_SP_E=UnitInertia(0.01, 0.01, 0.01)))
                        body_box = Box(block_width-block_margin,
                                       block_width*level_width-block_margin,
                                       block_height-block_margin)
                        mbp.RegisterVisualGeometry(
                            body, Isometry3(), body_box, "body_{}_vis".format(k),
                            np.array([np.random.uniform(0.75, 0.95), np.random.uniform(0.45, 0.55), np.random.uniform(0.1, 0.2), 1.]))
                        mbp.RegisterCollisionGeometry(
                            body, Isometry3(), body_box, "body_{}_box".format(k),
                            CoulombFriction(0.9, 0.8))
                        if (k % 2) == 0:
                            poses.append(
                                [RollPitchYaw(0, 0, 0).ToQuaternion().wxyz(),
                                 [l*block_width+block_width/2.,
                                  block_width*level_width/2.,
                                  k*block_height+block_height/2.]])
                        else:
                            poses.append([
                                RollPitchYaw(0, 0, np.pi/2.).ToQuaternion()
                                 .wxyz(),
                                [block_width*level_width/2.,
                                 l*block_width+block_width/2.,
                                 k*block_height+block_height/2.]])

            for pose in poses:
                pose[0] = np.array(pose[0]) + np.random.randn()*0.01
                pose[0] = pose[0] / np.linalg.norm(pose[0])
                pose[1] = np.array(pose[1]) + np.random.randn()*0.01

            mbp.AddForceElement(UniformGravityFieldElement())
            mbp.Finalize()

            visualizer = builder.AddSystem(MeshcatVisualizer(
                scene_graph,
                zmq_url="tcp://127.0.0.1:6000",
                draw_period=0.001))
            builder.Connect(scene_graph.get_pose_bundle_output_port(),
                            visualizer.get_input_port(0))

            diagram = builder.Build()

            diagram_context = diagram.CreateDefaultContext()
            mbp_context = diagram.GetMutableSubsystemContext(
                mbp, diagram_context)
            sg_context = diagram.GetMutableSubsystemContext(
                scene_graph, diagram_context)

            q0 = mbp.GetPositions(mbp_context).copy()
            for k in range(len(poses)):
                offset = k*7
                q0[(offset):(offset+4)] = poses[k][0]
                q0[(offset+4):(offset+7)] = poses[k][1]

            #ik = InverseKinematics(mbp, mbp_context)
            #q_dec = ik.q()
            #prog = ik.prog()
        #
            #def squaredNorm(x):
            #    return np.array([x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2])
            #for k in range(len(poses)):
            #    # Quaternion norm
            #    prog.AddConstraint(
            #        squaredNorm, [1], [1], q_dec[(k*7):(k*7+4)])
            #    # Trivial quaternion bounds
            #    prog.AddBoundingBoxConstraint(
            #        -np.ones(4), np.ones(4), q_dec[(k*7):(k*7+4)])
            #    # Conservative bounds on on XYZ
            #    prog.AddBoundingBoxConstraint(
            #        np.array([-10., -10., -10.]), np.array([10., 10., 10.]),
            #        q_dec[(k*7+4):(k*7+7)])
        #
            #constraint = ik.AddMinimumDistanceConstraint(0.01)
            #prog.AddQuadraticErrorCost(np.eye(q0.shape[0])*1.0, q0, q_dec)
        #
            #mbp.SetPositions(mbp_context, q0)
            #query_object = scene_graph.get_query_output_port().Eval(sg_context)
        #
            #prog.SetInitialGuess(q_dec, q0)
            #print("Solving")
            #print "Initial guess: ", q0
            #print prog.Solve()
            #print prog.GetSolverId().name()
            #q0_proj = prog.GetSolution(q_dec)
            #print "Final: ", q0_proj
            #mbp.SetPositions(mbp_context, q0_proj)

            q0_initial = q0.copy()
            mbp.SetPositions(mbp_context, q0)
            simulator = Simulator(diagram, diagram_context)
            simulator.set_target_realtime_rate(1.0)
            simulator.set_publish_every_time_step(False)
            simulator.StepTo(0.5)
            q0_final = mbp.GetPositions(mbp_context).copy()

            diff = np.linalg.norm(q0_final - q0_initial)
            print "DIFF: ", diff
            output_dict = {"n_objects": len(poses)}
            for k in range(len(poses)):
                offset = k*7
                pose = q0[(offset):(offset+7)]
                output_dict["obj_%04d" % k] = {
                    "class": "jenga_block",
                    "block_width": block_width,
                    "block_height": block_height,
                    "pose": pose.tolist()
                }
            if diff < 0.2:
                # Stable config
                with open("jenga_stable_arrangements.yaml", "a") as file:
                    yaml.dump({"env_%d" % int(round(time.time() * 1000)):
                               output_dict},
                              file)
            else:
                # Unstable config
                with open("jenga_unstable_arrangements.yaml", "a") as file:
                    yaml.dump({"env_%d" % int(round(time.time() * 1000)):
                               output_dict},
                              file)
        except Exception as e:
            print "Unhandled exception ", e

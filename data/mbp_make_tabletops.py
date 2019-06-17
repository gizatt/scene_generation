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
from pydrake.solvers.mathematicalprogram import (SolverOptions)
from pydrake.solvers.ipopt import (IpoptSolver)
from pydrake.solvers.nlopt import (NloptSolver)
from pydrake.solvers.snopt import (SnoptSolver)
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import AbstractValue, DiagramBuilder
from pydrake.systems.meshcat_visualizer import MeshcatVisualizer
from pydrake.systems.rendering import PoseBundle

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    for scene_iter in range(1000):
        try:
            builder = DiagramBuilder()
            mbp, scene_graph = AddMultibodyPlantSceneGraph(
                builder, MultibodyPlant(time_step=0.001))

            # Add ground
            world_body = mbp.world_body()
            ground_shape = Box(2., 2., 2.)
            ground_body = mbp.AddRigidBody("ground", SpatialInertia(
                mass=10.0, p_PScm_E=np.array([0., 0., 0.]),
                G_SP_E=UnitInertia(1.0, 1.0, 1.0)))
            mbp.WeldFrames(world_body.body_frame(), ground_body.body_frame(),
                           Isometry3(rotation=np.eye(3), translation=[0, 0, -1]))
            mbp.RegisterVisualGeometry(
                ground_body, Isometry3(), ground_shape, "ground_vis",
                np.array([0.5, 0.5, 0.5, 1.]))
            mbp.RegisterCollisionGeometry(
                ground_body, Isometry3(), ground_shape, "ground_col",
                CoulombFriction(0.9, 0.8))

            parser = Parser(mbp, scene_graph)

            candidate_model_files = [
                "/home/gizatt/drake/manipulation/models/mug/mug.urdf",
                "/home/gizatt/drake/manipulation/models/mug_big/mug_big.urdf",
                "/home/gizatt/drake/manipulation/models/dish_models/bowl_6p25in_decomp/bowl_6p25in_decomp.urdf",
                "/home/gizatt/drake/manipulation/models/dish_models/plate_11in_decomp/plate_11in_decomp.urdf",
                "/home/gizatt/drake/manipulation/models/dish_models/plate_8p5in_decomp/plate_8p5in_decomp.urdf",
            ]

            n_objects = np.random.randint(3, 7)
            poses = []  # [quat, pos]
            classes = []
            for k in range(n_objects):
                model_name = "model_%d" % k
                model_ind = np.random.randint(0, len(candidate_model_files))
                class_path = candidate_model_files[model_ind]
                classes.append(class_path)
                parser.AddModelFromFile(class_path, model_name=model_name)
                poses.append([
                    RollPitchYaw(np.random.uniform(0., 2.*np.pi, size=3)).ToQuaternion().wxyz(),
                    [np.random.uniform(-0.1, 0.1),
                     np.random.uniform(-0.1, 0.1),
                     np.random.uniform(0.1, 0.2)]])

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

            simulator = Simulator(diagram, diagram_context)
            simulator.set_target_realtime_rate(1.0)
            simulator.set_publish_every_time_step(False)
            simulator.Initialize()

            ik = InverseKinematics(mbp, mbp_context)
            q_dec = ik.q()
            prog = ik.prog()

            def squaredNorm(x):
                return np.array([x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2])
            for k in range(len(poses)):
                # Quaternion norm
                prog.AddConstraint(
                    squaredNorm, [1], [1], q_dec[(k*7):(k*7+4)])
                # Trivial quaternion bounds
                prog.AddBoundingBoxConstraint(
                    -np.ones(4), np.ones(4), q_dec[(k*7):(k*7+4)])
                # Conservative bounds on on XYZ
                prog.AddBoundingBoxConstraint(
                    np.array([-2., -2., -2.]), np.array([2., 2., 2.]),
                    q_dec[(k*7+4):(k*7+7)])

            def vis_callback(x):
                mbp.SetPositions(mbp_context, x)
                pose_bundle = scene_graph.get_pose_bundle_output_port().Eval(sg_context)
                context = visualizer.CreateDefaultContext()
                context.FixInputPort(0, AbstractValue.Make(pose_bundle))
                #print(pose_bundle.get_pose(0))
                visualizer.Publish(context)
                #print("Here")

            prog.AddVisualizationCallback(vis_callback, q_dec)
            prog.AddQuadraticErrorCost(np.eye(q0.shape[0])*1.0, q0, q_dec)

            ik.AddMinimumDistanceConstraint(0.001, threshold_distance=1.0)

            prog.SetInitialGuess(q_dec, q0)
            print("Solving")
#            print "Initial guess: ", q0
            start_time = time.time()
            solver = SnoptSolver()
            #solver = NloptSolver()
            sid = solver.solver_type()
            # SNOPT
            prog.SetSolverOption(sid, "Print file", "test.snopt")
            prog.SetSolverOption(sid, "Major feasibility tolerance", 1e-3)
            prog.SetSolverOption(sid, "Major optimality tolerance", 1e-2)
            prog.SetSolverOption(sid, "Minor feasibility tolerance", 1e-3)
            prog.SetSolverOption(sid, "Scale option", 0)
            #prog.SetSolverOption(sid, "Elastic weight", 1e1)
            #prog.SetSolverOption(sid, "Elastic mode", "Yes")
            # NLOPT
            #prog.SetSolverOption(sid, "initial_step", 0.1)
            #prog.SetSolverOption(sid, "xtol_rel", 1E-2)
            #prog.SetSolverOption(sid, "xtol_abs", 1E-2)

            #prog.SetSolverOption(sid, "Major step limit", 2)

            print("Solver opts: ", prog.GetSolverOptions(solver.solver_type()))
            print solver.Solve(prog)
            print "Solved in %f seconds" % (time.time() - start_time)
            #print IpoptSolver().Solve(prog)
            print prog.GetSolverId().name()
            q0_proj = prog.GetSolution(q_dec)
#            print "Final: ", q0_proj
            mbp.SetPositions(mbp_context, q0_proj)
            q0_initial = q0_proj.copy()
            simulator.StepTo(0.0001)
            q0_final = mbp.GetPositions(mbp_context).copy()

            output_dict = {"n_objects": len(poses)}
            for k in range(len(poses)):
                offset = k*7
                pose = q0[(offset):(offset+7)]
                output_dict["obj_%04d" % k] = {
                    "class": classes[k],
                    "pose": pose.tolist()
                }
            with open("tabletop_arrangements.yaml", "a") as file:
                yaml.dump({"env_%d" % int(round(time.time() * 1000)):
                           output_dict},
                          file)

            time.sleep(1.0)

        except Exception as e:
            print "Unhandled exception ", e
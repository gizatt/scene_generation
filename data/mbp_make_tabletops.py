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
from pydrake.math import (RollPitchYaw, RigidTransform)
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
import pydrake.solvers.mathematicalprogram as mp
from pydrake.solvers.mathematicalprogram import (SolverOptions)
from pydrake.solvers.ipopt import (IpoptSolver)
from pydrake.solvers.nlopt import (NloptSolver)
from pydrake.solvers.snopt import (SnoptSolver)
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import AbstractValue, DiagramBuilder, LeafSystem, PortDataType
from pydrake.systems.meshcat_visualizer import MeshcatVisualizer
from pydrake.systems.rendering import PoseBundle
from pydrake.systems.sensors import RgbdSensor, Image, PixelType, PixelFormat
from pydrake.geometry.render import DepthCameraProperties, MakeRenderEngineVtk, RenderEngineVtkParams

import matplotlib.pyplot as plt


class RgbAndLabelImageVisualizer(LeafSystem):
    def __init__(self,
                 draw_timestep=0.033333):
        LeafSystem.__init__(self)
        self.set_name('image viz')
        self.timestep = draw_timestep
        self._DeclarePeriodicPublish(draw_timestep, 0.0)
        
        self.rgb_image_input_port = \
            self._DeclareAbstractInputPort("rgb_image_input_port",
                                   AbstractValue.Make(Image[PixelType.kRgba8U](640, 480, 3)))
        self.label_image_input_port = \
            self._DeclareAbstractInputPort("label_image_input_port",
                                   AbstractValue.Make(Image[PixelType.kLabel16I](640, 480, 1)))
        self.fig = plt.figure()
        self.ax = plt.gca()
        plt.draw()

    def _DoPublish(self, context, event):
        rgb_image = self.EvalAbstractInput(context, 0).get_value()
        label_image = self.EvalAbstractInput(context, 1).get_value()
        print(rgb_image)
        plt.pause(0.1)


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    for scene_iter in range(1000):
        try:
            builder = DiagramBuilder()
            mbp, scene_graph = AddMultibodyPlantSceneGraph(
                builder, MultibodyPlant(time_step=0.001))
            renderer_params = RenderEngineVtkParams()
            scene_graph.AddRenderer("renderer", MakeRenderEngineVtk(renderer_params))
            # Add ground
            world_body = mbp.world_body()
            ground_shape = Box(2., 2., 2.)
            ground_body = mbp.AddRigidBody("ground", SpatialInertia(
                mass=10.0, p_PScm_E=np.array([0., 0., 0.]),
                G_SP_E=UnitInertia(1.0, 1.0, 1.0)))
            mbp.WeldFrames(world_body.body_frame(), ground_body.body_frame(),
                           RigidTransform(Isometry3(rotation=np.eye(3), translation=[0, 0, -1])))
            mbp.RegisterVisualGeometry(
                ground_body, RigidTransform.Identity(), ground_shape, "ground_vis",
                np.array([0.5, 0.5, 0.5, 1.]))
            mbp.RegisterCollisionGeometry(
                ground_body, RigidTransform.Identity(), ground_shape, "ground_col",
                CoulombFriction(0.9, 0.8))

            parser = Parser(mbp, scene_graph)

            candidate_model_files = [
                "/home/gizatt/drake/manipulation/models/mug_clean/mug.urdf",
                #"/home/gizatt/drake/manipulation/models/mug_big/mug_big.urdf",
                #"/home/gizatt/drake/manipulation/models/dish_models/bowl_6p25in_decomp/bowl_6p25in_decomp.urdf",
                #"/home/gizatt/drake/manipulation/models/dish_models/plate_11in_decomp/plate_11in_decomp.urdf",
                #"/home/gizatt/drake/manipulation/models/dish_models/plate_8p5in_decomp/plate_8p5in_decomp.urdf",
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

            # Add camera
            depth_camera_properties = DepthCameraProperties(
                width=640, height=480, fov_y=np.pi/2, renderer_name="renderer", z_near=0.1, z_far=2.0)
            parent_frame_id = scene_graph.world_frame_id()
            # TODO: read the drake docs about RgbdSensor and figure out what to set this transform to
            # to get the camera upright and facing towards the origin.
            camera_tf = RigidTransform(p=[1., 0., 0.5], rpy=RollPitchYaw([np.pi/2., 0., 0.]))
            camera = builder.AddSystem(RgbdSensor(parent_frame_id, camera_tf, depth_camera_properties, show_window=True))
            builder.Connect(scene_graph.get_query_output_port(),
                            camera.query_object_input_port())

            camera_viz = builder.AddSystem(RgbAndLabelImageVisualizer(draw_timestep=0.1))
            builder.Connect(camera.color_image_output_port(),
                            camera_viz.get_input_port(0))
            builder.Connect(camera.label_image_output_port(),
                            camera_viz.get_input_port(1))

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
            prog = ik.get_mutable_prog()

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
            print(type(prog))
            result = mp.Solve(prog)
            print("Solve info: ", result)
            print("Solved in %f seconds" % (time.time() - start_time))
            #print(IpoptSolver().Solve(prog))
            print(result.get_solver_id().name())
            q0_proj = result.GetSolution(q_dec)
#            print "Final: ", q0_proj
            mbp.SetPositions(mbp_context, q0_proj)
            q0_initial = q0_proj.copy()
            simulator.StepTo(10.0)
            q0_final = mbp.GetPositions(mbp_context).copy()

            #output_dict = {"n_objects": len(poses)}
            #for k in range(len(poses)):
            #    offset = k*7
            #    pose = q0[(offset):(offset+7)]
            #    output_dict["obj_%04d" % k] = {
            #        "class": classes[k],
            #        "pose": pose.tolist()
            #    }
            #with open("tabletop_arrangements.yaml", "a") as file:
            #    yaml.dump({"env_%d" % int(round(time.time() * 1000)):
            #               output_dict},
            #              file)
#
            time.sleep(1.0)

        except Exception as e:
            print("Unhandled exception ", e)

        except:
            print("Unhandled unnamed exception, probably sim error")
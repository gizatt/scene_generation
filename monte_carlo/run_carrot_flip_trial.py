# -*- coding: utf8 -*-

'''
Uses an open-loop trajectory (in world frame)
to attempt to flip a carrot.
'''

import pydrake

import argparse
import os
import random
import time
import sys

import matplotlib
matplotlib.use("Qt5agg") # or "Qt5agg" depending on you version of Qt
import numpy as np

from pydrake.common.eigen_geometry import Quaternion, AngleAxis, Isometry3
from pydrake.examples.manipulation_station import (
    ManipulationStation)
from pydrake.geometry import (
    Box,
)
from pydrake.multibody.tree import (
    PrismaticJoint,
    SpatialInertia,
    RevoluteJoint,
    UniformGravityFieldElement,
    UnitInertia,
    FixedOffsetFrame
)
from pydrake.multibody.plant import MultibodyPlant, CoulombFriction
from pydrake.manipulation.planner import (
    DifferentialInverseKinematicsParameters)
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.parsing import Parser
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import (BasicVector, DiagramBuilder,
                                       LeafSystem)
from pydrake.systems.meshcat_visualizer import MeshcatVisualizer
from pydrake.systems.primitives import FirstOrderLowPassFilter, TrajectorySource, ConstantVectorSource
from pydrake.trajectories import PiecewisePolynomial

from differential_ik import DifferentialIK
from mesh_creation import create_cut_cylinder, export_sdf, create_cylinder

import matplotlib.pyplot as plt
import numpy as np
from underactuated.planar_scenegraph_visualizer import PlanarSceneGraphVisualizer


try:
    import pygame
    from pygame.locals import *
except ImportError:
    print("ERROR: missing pygame.  Please install pygame to use this example.")
    # Fail silently (until pygame is supported in python 3 on all platforms)
    sys.exit(0)


def print_instructions():
    print("")
    print("END EFFECTOR CONTROL")
    print("mouse left/right   - move in the manipulation station's y/z plane")
    print("mouse buttons      - roll left/right")
    print("w / s              - move forward/back this y/z plane")
    print("q / e              - yaw left/right \
                                (also can use mouse side buttons)")
    print("a / d              - pitch up/down")
    print("")
    print("GRIPPER CONTROL")
    print("mouse wheel        - open/close gripper")
    print("")
    print("space              - switch out of teleop mode")
    print("enter              - return to teleop mode (be sure you've")
    print("                     returned focus to the pygame app)")
    print("escape             - quit")


class TeleopMouseKeyboardManager():

    def __init__(self, grab_focus=True):
        pygame.init()
        # We don't actually want a screen, but
        # I can't get this to work without a tiny screen.
        # Setting it to 1 pixel.
        screen_size = 1
        self.screen = pygame.display.set_mode((screen_size, screen_size))

        self.side_button_back_DOWN = False
        self.side_button_fwd_DOWN = False
        if grab_focus:
            self.grab_mouse_focus()

    def grab_mouse_focus(self):
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)

    def release_mouse_focus(self):
        pygame.event.set_grab(False)
        pygame.mouse.set_visible(True)

    def get_events(self):
        mouse_wheel_up = mouse_wheel_down = False

        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit(0)
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                sys.exit(0)
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:
                    mouse_wheel_up = True
                if event.button == 5:
                    mouse_wheel_down = True
                if event.button == 8:
                    self.side_button_back_DOWN = True
                if event.button == 9:
                    self.side_button_fwd_DOWN = True
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 8:
                    self.side_button_back_DOWN = False
                if event.button == 9:
                    self.side_button_fwd_DOWN = False

        keys = pygame.key.get_pressed()
        delta_x, delta_y = pygame.mouse.get_rel()
        left_mouse_button, _, right_mouse_button = pygame.mouse.get_pressed()

        if keys[K_RETURN]:
            self.grab_mouse_focus()
        if keys[K_SPACE]:
            self.release_mouse_focus()

        events = dict()
        events["delta_x"] = delta_x
        events["delta_y"] = delta_y
        events["w"] = keys[K_w]
        events["a"] = keys[K_a]
        events["s"] = keys[K_s]
        events["d"] = keys[K_d]
        events["r"] = keys[K_r]
        events["q"] = keys[K_q]
        events["e"] = keys[K_e]
        events["p"] = keys[K_p]
        events["mouse_wheel_up"] = mouse_wheel_up
        events["mouse_wheel_down"] = mouse_wheel_down
        events["left_mouse_button"] = left_mouse_button
        events["right_mouse_button"] = right_mouse_button
        events["side_button_back"] = self.side_button_back_DOWN
        events["side_button_forward"] = self.side_button_fwd_DOWN
        return events


class MouseKeyboardTeleop(LeafSystem):
    def __init__(self, grab_focus=True):
        LeafSystem.__init__(self)
        self.DeclareVectorOutputPort("rpy_xyz", BasicVector(6),
                                     self.DoCalcOutput)
        self.DeclareVectorOutputPort("position", BasicVector(1),
                                     self.CalcPositionOutput)
        self.DeclareVectorOutputPort("force_limit", BasicVector(1),
                                     self.CalcForceLimitOutput)

        # Note: This timing affects the keyboard teleop performance. A larger
        #       time step causes more lag in the response.
        self.DeclarePeriodicPublish(0.01, 0.0)

        self.teleop_manager = TeleopMouseKeyboardManager(grab_focus=grab_focus)
        self.roll = self.pitch = self.yaw = 0
        self.x = self.y = self.z = 0
        self.gripper_max = 0.107
        self.gripper_min = 0.01
        self.gripper_goal = self.gripper_max
        self.p_down = False

    def SetPose(self, pose):
        """
        @param pose is an Isometry3.
        """
        tf = RigidTransform(pose)
        self.SetRPY(RollPitchYaw(tf.rotation()))
        self.SetXYZ(pose.translation())

    def SetRPY(self, rpy):
        """
        @param rpy is a RollPitchYaw object
        """
        self.roll = rpy.roll_angle()
        self.pitch = rpy.pitch_angle()
        self.yaw = rpy.yaw_angle()

    def SetXYZ(self, xyz):
        """
        @param xyz is a 3 element vector of x, y, z.
        """
        self.x = xyz[0]
        self.y = xyz[1]
        self.z = xyz[2]

    def SetXyzFromEvents(self, events):
        scale_down = 0.0001
        delta_x = events["delta_x"]*-scale_down
        delta_y = events["delta_y"]*-scale_down

        forward_scale = 0.00005
        delta_forward = 0.0
        if events["w"]:
            delta_forward += forward_scale
        if events["s"]:
            delta_forward -= forward_scale

        self.x += -delta_x
        self.y += delta_forward
        self.z += delta_y

    def SetRpyFromEvents(self, events):
        roll_scale = 0.0003
        if events["left_mouse_button"]:
            self.roll += roll_scale
        if events["right_mouse_button"]:
            self.roll -= roll_scale
        self.roll = np.clip(self.roll, a_min=-2*np.pi, a_max=2*np.pi)

        yaw_scale = 0.0003
        if events["side_button_back"] or events["q"]:
            self.yaw += yaw_scale
        if events["side_button_forward"] or events["e"]:
            self.yaw -= yaw_scale
        self.yaw = np.clip(self.yaw, a_min=-2*np.pi, a_max=2*np.pi)

        pitch_scale = 0.0003
        if events["d"]:
            self.pitch += pitch_scale
        if events["a"]:
            self.pitch -= pitch_scale
        self.pitch = np.clip(self.pitch, a_min=-2*np.pi, a_max=2*np.pi)

    def SetGripperFromEvents(self, events):
        gripper_scale = 0.01
        if events["mouse_wheel_up"]:
            self.gripper_goal += gripper_scale
        if events["mouse_wheel_down"]:
            self.gripper_goal -= gripper_scale
        self.gripper_goal = np.clip(self.gripper_goal,
                                    a_min=self.gripper_min,
                                    a_max=self.gripper_max)

    def CalcPositionOutput(self, context, output):
        output.SetAtIndex(0, self.gripper_goal)

    def CalcForceLimitOutput(self, context, output):
        self._force_limit = 40
        output.SetAtIndex(0, self._force_limit)

    def DoCalcOutput(self, context, output):
        events = self.teleop_manager.get_events()
        self.SetXyzFromEvents(events)
        self.SetRpyFromEvents(events)
        self.SetGripperFromEvents(events)
        output.SetAtIndex(0, self.roll)
        output.SetAtIndex(1, self.pitch)
        output.SetAtIndex(2, self.yaw)
        output.SetAtIndex(3, self.x)
        output.SetAtIndex(4, self.y)
        output.SetAtIndex(5, self.z)
        if (not self.p_down and events["p"]):
            print("Pose: ", output.CopyToVector())
            self.p_down = True
        elif (not events["p"]):
            self.p_down = False

if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("-T", "--duration",
                        type=float,
                        help="Duration of task.",
                        default=3.0)
    parser.add_argument("--seed",
                        type=float, default=time.time(),
                        help="RNG seed")
    parser.add_argument("--planar_viz",
                        action='store_true',
                        help="Show planar viz?")
    parser.add_argument("--teleop",
                        action='store_true',
                        help="Control with mouse keyboard mode?")
    MeshcatVisualizer.add_argparse_argument(parser)
    args = parser.parse_args()
    if args.teleop:
        args.duration = 100000.0
    int_seed = int(args.seed*1000. % 2**32)
    random.seed(int_seed)
    np.random.seed(int_seed)

    builder = DiagramBuilder()
    
    station = builder.AddSystem(ManipulationStation(time_step=0.001))

    # Initializes the chosen station type.
    station.SetupDefaultStation()

    # Add random carrot.
    height = np.random.random() * 0.05 + 0.05
    radius = np.random.random() * 0.02 + 0.02
    cut_dirs = [np.array([1., 0., 0.])]
    cut_points = [np.array([0.0, 0, 0])]
    cutting_planes = zip(cut_points, cut_dirs)
    # Create a mesh programmatically for that cylinder
    cyl = create_cut_cylinder(
        radius, height, cutting_planes, sections=20)
    cyl.density = 1000.  # Same as water
    export_sdf(cyl, "carrot", "/home/gizatt/drake/build/install/share/drake/manipulation/models/", color=[0.75, 0.2, 0.2, 1.])

    #carrot_pose = RigidTransform(RollPitchYaw([0., 0., 0.]), [0.6, 0., 0.1])
    #station.AddManipulandFromFile("drake/manipulation/models/carrot.sdf", carrot_pose);

    mbp = station.get_mutable_multibody_plant()

    mbp_parser = Parser(mbp)
    model_instance = mbp_parser.AddModelFromFile("/home/gizatt/drake/build/install/share/drake/manipulation/models/carrot.sdf", "carrot");
    body = mbp.GetBodyByName("carrot")

    no_mass_no_inertia = SpatialInertia(
        mass=0.0, p_PScm_E=np.array([0., 0., 0.]),
        G_SP_E=UnitInertia(0., 0., 0.))
    body_pre_z = mbp.AddRigidBody("carrot_pre_z", model_instance,
                                  no_mass_no_inertia)
    body_pre_theta = mbp.AddRigidBody("carrot_pre_theta", model_instance,
                                      no_mass_no_inertia)

    world_carrot_origin = mbp.AddFrame(frame=FixedOffsetFrame(
            name="world_carrot_origin", P=mbp.world_frame(),
            X_PF=RigidTransform(
                RollPitchYaw([0., 0., 0.]),
                [0.5, 0., 0.0])))
    carrot_rotated_origin = mbp.AddFrame(frame=FixedOffsetFrame(
            name="carrot_rotated_origin", P=body.body_frame(),
            X_PF=RigidTransform(
                RollPitchYaw([np.pi/2., 0., 0.]),
                [0.0, 0., 0.0])))
    body_joint_x = PrismaticJoint(
        name="carrot_x",
        frame_on_parent=world_carrot_origin,
        frame_on_child=body_pre_z.body_frame(),
        axis=[1, 0, 0],
        damping=0.)
    mbp.AddJoint(body_joint_x)

    body_joint_z = PrismaticJoint(
        name="carrot_z",
        frame_on_parent=body_pre_z.body_frame(),
        frame_on_child=body_pre_theta.body_frame(),
        axis=[0, 0, 1],
        damping=0.)
    mbp.AddJoint(body_joint_z)

    body_joint_theta = RevoluteJoint(
        name="carrot_theta",
        frame_on_parent=body_pre_theta.body_frame(),
        frame_on_child=carrot_rotated_origin,
        axis=[0, 1, 0],
        damping=0.)
    mbp.AddJoint(body_joint_theta)

    print("here")
    station.Finalize()
    
    body_joint_x.set_default_translation(0.0)
    body_joint_z.set_default_translation(0.05)
    body_joint_theta.set_default_angle(np.pi/2.)
    
    if args.meshcat:
        meshcat = builder.AddSystem(MeshcatVisualizer(
            station.get_scene_graph(), zmq_url=args.meshcat))
        builder.Connect(station.GetOutputPort("pose_bundle"),
                        meshcat.get_input_port(0))

    if args.planar_viz:
        viz = builder.AddSystem(PlanarSceneGraphVisualizer(
            station.get_scene_graph(),
            xlim=[0.25, 0.8], ylim=[-0.1, 0.5],
            ax=plt.gca()))
        builder.Connect(station.GetOutputPort("pose_bundle"),
                        viz.get_input_port(0))

    robot = station.get_controller_plant()
    params = DifferentialInverseKinematicsParameters(robot.num_positions(),
                                                     robot.num_velocities())

    time_step = 0.005
    params.set_timestep(time_step)
    # True velocity limits for the IIWA14 (in rad, rounded down to the first
    # decimal)
    iiwa14_velocity_limits = np.array([1.4, 1.4, 1.7, 1.3, 2.2, 2.3, 2.3])
    # Stay within a small fraction of those limits for this teleop demo.
    factor = 1.0
    params.set_joint_velocity_limits((-factor*iiwa14_velocity_limits,
                                      factor*iiwa14_velocity_limits))

    differential_ik = builder.AddSystem(DifferentialIK(
        robot, robot.GetFrameByName("iiwa_link_7"), params, time_step))

    builder.Connect(differential_ik.GetOutputPort("joint_position_desired"),
                    station.GetInputPort("iiwa_position"))

    if (args.teleop):
        teleop = builder.AddSystem(MouseKeyboardTeleop(grab_focus=True))
        filter_ = builder.AddSystem(
            FirstOrderLowPassFilter(time_constant=0.005, size=6))

        builder.Connect(teleop.get_output_port(0), filter_.get_input_port(0))
        builder.Connect(filter_.get_output_port(0),
                        differential_ik.GetInputPort("rpy_xyz_desired"))

        builder.Connect(teleop.GetOutputPort("position"), station.GetInputPort(
            "wsg_position"))
        builder.Connect(teleop.GetOutputPort("force_limit"),
                        station.GetInputPort("wsg_force_limit"))
    else:
        # Playback open-loop trajectory
        knots = np.array([
            [-3.32249, -0.05747,  4.70519,  0.59113, -0.     ,  0.2915 ],
            [-3.46739, -0.05747,  4.70519,  0.59763, -0.     ,  0.272  ],
            [-3.46739, -0.05747,  4.70519,  0.58493, -0.     ,  0.2403 ],
            [-3.46739, -0.05747,  4.70519,  0.57473, -0.     ,  0.2226 ],
            [-3.46739, -0.05747,  4.70519,  0.55583, -0.     ,  0.2256 ],
            [-3.46739, -0.05747,  4.70519,  0.54173, -0.     ,  0.229  ],
            [-3.46739, -0.05747,  4.70519,  0.52563, -0.     ,  0.2351 ],
            [-3.46739, -0.05747,  4.70519,  0.51193, -0.     ,  0.2448 ],
            [-3.46739, -0.05747,  4.70519,  0.48803, -0.     ,  0.2522 ],
            [-3.46739, -0.05747,  4.70519,  0.46983, -0.     ,  0.2587 ],
            [-3.46739, -0.05747,  4.70519,  0.45433, -0.     ,  0.273  ],
            [-3.46739, -0.05747,  4.70519,  0.42623, -0.     ,  0.2933 ],
            [-3.46739, -0.05747,  4.70519,  0.40653, -0.     ,  0.3128 ],
        ]).T
        ts = np.linspace(0., args.duration, knots.shape[1])
        print(ts.shape)
        print(knots.shape)

        ee_traj = PiecewisePolynomial.Pchip(
            ts, knots, True)
        setpoints = builder.AddSystem(TrajectorySource(ee_traj))
        builder.Connect(setpoints.get_output_port(0),
                        differential_ik.GetInputPort("rpy_xyz_desired"))

        wsg_force_limit_source = builder.AddSystem(
            ConstantVectorSource(np.array([40])))
        wsg_position_source = builder.AddSystem(
            ConstantVectorSource(np.array([0.107])))
        builder.Connect(wsg_position_source.get_output_port(0),
                        station.GetInputPort("wsg_position"))
        builder.Connect(wsg_force_limit_source.get_output_port(0),
                        station.GetInputPort("wsg_force_limit"))




    diagram = builder.Build()
    simulator = Simulator(diagram)

    # This is important to avoid duplicate publishes to the hardware interface:
    simulator.set_publish_every_time_step(False)

    station_context = diagram.GetMutableSubsystemContext(
        station, simulator.get_mutable_context())

    station.GetInputPort("iiwa_feedforward_torque").FixValue(
        station_context, np.zeros(7))

    simulator.AdvanceTo(1e-6)
    q0 = station.GetOutputPort("iiwa_position_measured").Eval(station_context)
    differential_ik.parameters.set_nominal_joint_position(q0)

    if (args.teleop):
        teleop.SetPose(differential_ik.ForwardKinematics(q0))
        filter_.set_initial_output_value(
            diagram.GetMutableSubsystemContext(
                filter_, simulator.get_mutable_context()),
            teleop.get_output_port(0).Eval(diagram.GetMutableSubsystemContext(
                teleop, simulator.get_mutable_context())))

    differential_ik.SetPositions(diagram.GetMutableSubsystemContext(
        differential_ik, simulator.get_mutable_context()), q0)

    simulator.set_target_realtime_rate(1.0)

    print_instructions()
    simulator.AdvanceTo(args.duration)
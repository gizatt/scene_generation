# -*- coding: utf8 -*-

'''
Demonstrates uses an open-loop trajectory (in world frame) 
to attempt to flip a carrot. Simplified frun `run_carrot_flip_trial.py`
to give Sadra a starting point.
'''

import pydrake

import argparse
import os
import random
import time
import sys
import yaml

import matplotlib
matplotlib.use("Qt5agg") # or "Qt5agg" depending on you version of Qt
import matplotlib.animation as animation
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

class IiwaEeTrajectorySpoolingSystem(LeafSystem):
    ''' Example system that plays out a rpy_xyz goal trajectory to be
    consumed by a DifferentialIK controller hooked up to the
    ManipulationStation robot. '''
    def __init__(self):
        LeafSystem.__init__(self)
        # These are the inputs the DifferentialIK system needs:
        #   rpy_xyz_desired of the EE frame
        self.DeclareVectorOutputPort("rpy_xyz_desired", BasicVector(6),
                                     self.CalcRpyXyzOutput)
        # Gripper position + force limit.
        self.DeclareVectorOutputPort("wsg_position", BasicVector(1),
                                     self.CalcGripperPositionOutput)
        self.DeclareVectorOutputPort("wsg_force_limit", BasicVector(1),
                                     self.CalcGripperForceLimitOutput)

        # Prepare the trajectory that we'll be spooling out.
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
        ts = np.linspace(0., 3.0, knots.shape[1])
        self.ee_traj = PiecewisePolynomial.Pchip(ts, knots, True)
        self.wsg_position = 0.1 # Fully open
        self.wsg_force_limit = 40 # Newtons

    def CalcGripperPositionOutput(self, context, output):
        output.SetAtIndex(0, self.wsg_position)

    def CalcGripperForceLimitOutput(self, context, output):
        output.SetAtIndex(0, self.wsg_force_limit)

    def CalcRpyXyzOutput(self, context, output):
        t = context.get_time()
        output.SetFromVector(self.ee_traj.value(t))


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
    parser.add_argument("--no_planar_viz",
                        action='store_true',
                        help="Hide planar viz?")
    parser.add_argument("--planar_record",
                        action='store_true',
                        help="Record with planar viz?")
    parser.add_argument("--realtime_rate",
                        type=float,
                        default=1.0)
    MeshcatVisualizer.add_argparse_argument(parser)
    args = parser.parse_args()
    
    int_seed = int(args.seed*1000. % 2**32)
    random.seed(int_seed)
    np.random.seed(int_seed)

    builder = DiagramBuilder()
    
    station = builder.AddSystem(ManipulationStation(time_step=0.002))
    station.SetupManipulationClassStation()
    mbp = station.get_mutable_multibody_plant()
    scene_graph = station.get_scene_graph()

    # Generate a random carrot model.
    # Carrot parameters and bounds:
    #   Carrot radius: [0.01, 0.05], meters
    #   X init: -0.05, 0.05, meters
    radius = 0.025
    x_init = 0.0
    # "Height" is out-of-plane cylinder length. It shouldn't matter
    # but if it's too small collision gets weird.
    height = 0.05
    # Carrot is created by cutting up a cylinder with
    # a plane described by this point and normal. 
    cut_dirs = [np.array([1., 0., 0.])]
    cut_points = [np.array([0.0, 0, 0])]
    cutting_planes = zip(cut_points, cut_dirs)
    # Create a mesh programmatically for that cylinder
    cyl = create_cut_cylinder(
        radius, height, cutting_planes, sections=15)
    cyl.density = 1000.  # Same as water
    carrot_dir = "/tmp/carrot_%2.5f_%2.5f/" % (radius, x_init)
    export_sdf(cyl, "carrot", carrot_dir, color=[0.75, 0.2, 0.2, 1.])

    # Finally import that generated model into the MBP.
    mbp_parser = Parser(mbp)
    model_instance = mbp_parser.AddModelFromFile(carrot_dir + "carrot.sdf", "carrot");
    body = mbp.GetBodyByName("carrot")

    # Give it a planar floating base by manually specifying x-z-theta joints.
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

    # Done adding the carrot.
    station.Finalize()
    
    # Set the default state of the carrot
    body_joint_x.set_default_translation(x_init)
    body_joint_z.set_default_translation(0.05)
    body_joint_theta.set_default_angle(np.pi/2.)
    
    # Visualization options -- both are hooked up in a similar
    # way to the SceneGraph PoseBundle output.
    if args.meshcat:
        meshcat = builder.AddSystem(MeshcatVisualizer(
            scene_graph, zmq_url=args.meshcat))
        builder.Connect(station.GetOutputPort("pose_bundle"),
                        meshcat.get_input_port(0))

    if not args.no_planar_viz or args.planar_record:
        plt.gca().clear()
        viz = builder.AddSystem(PlanarSceneGraphVisualizer(
            scene_graph,
            xlim=[0.25, 0.8], ylim=[-0.1, 0.5],
            ax=plt.gca(),
            draw_period=0.1))
        builder.Connect(station.GetOutputPort("pose_bundle"),
                        viz.get_input_port(0))
        if args.planar_record:
            viz.start_recording(show=not args.no_planar_viz)

    # Set up a task-space position tracking controller.
    # The "controller plant" of the ManipulationStation is just the IIWA,
    # I think. It's distinct from the full MBP of the station, which has
    # all of the world geometry present oto.
    mbp_for_control = station.get_controller_plant()
    params = DifferentialInverseKinematicsParameters(mbp_for_control.num_positions(),
                                                     mbp_for_control.num_velocities())
    time_step = 0.005  # i.e. control tick rate
    params.set_timestep(time_step)
    # True velocity limits for the IIWA14 (in rad, rounded down to the first
    # decimal)
    iiwa14_velocity_limits = np.array([1.4, 1.4, 1.7, 1.3, 2.2, 2.3, 2.3])
    # Stay within a small fraction of those limits for this teleop demo.
    factor = 1.0
    params.set_joint_velocity_limits((-factor*iiwa14_velocity_limits,
                                      factor*iiwa14_velocity_limits))
    # It'll be end-effector tracking of the very final link of the iiwa.
    differential_ik = builder.AddSystem(DifferentialIK(
        mbp_for_control, mbp_for_control.GetFrameByName("iiwa_link_7"), params, time_step))
    # The controller produces position goals for the robot. (We'll later set the
    # force tracking goal of the robot to always be zero, functionally putting
    # the robot in position-only mode with some small compliance based on whatever
    # position/force gains the robot was configured with from the pendant.)
    builder.Connect(differential_ik.GetOutputPort("joint_position_desired"),
                    station.GetInputPort("iiwa_position"))

    # Make a trajectory source system and rig it up.
    trajectory_source_system = builder.AddSystem(IiwaEeTrajectorySpoolingSystem())
    builder.Connect(trajectory_source_system.GetOutputPort("rpy_xyz_desired"),
                    differential_ik.GetInputPort("rpy_xyz_desired"))
    builder.Connect(trajectory_source_system.GetOutputPort("wsg_position"),
                    station.GetInputPort("wsg_position"))
    builder.Connect(trajectory_source_system.GetOutputPort("wsg_force_limit"),
                    station.GetInputPort("wsg_force_limit"))

    iiwa_feedforward_torque_fixed = builder.AddSystem(
        ConstantVectorSource(np.zeros(7)))
    builder.Connect(iiwa_feedforward_torque_fixed.get_output_port(0),
                    station.GetInputPort("iiwa_feedforward_torque"))

    diagram = builder.Build()
    simulator = Simulator(diagram)

    # This is important to avoid duplicate publishes to the hardware interface:
    simulator.set_publish_every_time_step(False)

    # Hacky way of getting the initial state of the robot, and setting
    # the diffik controller to use that as its nominal.
    # This will hopefully be cleaned up in the future...
    station_context = diagram.GetMutableSubsystemContext(
        station, simulator.get_mutable_context())
    simulator.AdvanceTo(1e-6)
    q0 = station.GetOutputPort("iiwa_position_measured").Eval(station_context)
    differential_ik.parameters.set_nominal_joint_position(q0)
    differential_ik.SetPositions(diagram.GetMutableSubsystemContext(
        differential_ik, simulator.get_mutable_context()), q0)

    simulator.set_target_realtime_rate(args.realtime_rate)

    try:
        simulator.AdvanceTo(trajectory_source_system.ee_traj.end_time() + 1.)
    except:
        print("Unhandled error in simulate. Not saving.")
        sys.exit(-1)

    # Calculate score + success
    final_mbp_context = diagram.GetMutableSubsystemContext(mbp, simulator.get_mutable_context())
    final_carrot_tf = mbp.CalcRelativeTransform(
        final_mbp_context, mbp.world_frame(), body.body_frame())
    print("Final carrot tf: ", final_carrot_tf.matrix())
    
    # How much does Carrot +x face in +z?
    # > 0 means it flipped.
    if final_carrot_tf.matrix()[2, 0] > 0:
        print("Carrot was flipped.")
        score = 1.
    else:
        score = 0.
        print("Carrot was not flipped.")

    if args.planar_record:
        os.system("mkdir -p results/videos/")
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=1./viz.timestep, metadata=dict(artist='Me'), bitrate=1800)
        viz.get_recording().save(
            "results/videos/" + 'radius_%2.5f_xinit_%2.5f.mp4' % (radius, x_init),
            writer=writer)

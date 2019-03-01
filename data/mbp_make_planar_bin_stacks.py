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
    RevoluteJoint,
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

from underactuated.planar_multibody_visualizer import PlanarMultibodyVisualizer

def RegisterVisualAndCollisionGeometry(
        mbp, body, pose, shape, name, color, friction):
    mbp.RegisterVisualGeometry(body, pose, shape, name + "_vis", color)
    mbp.RegisterCollisionGeometry(body, pose, shape, name + "_col",
                                  friction)


def generate_example():
    builder = DiagramBuilder()
    mbp, scene_graph = AddMultibodyPlantSceneGraph(
        builder, MultibodyPlant(time_step=0.005))

    # Add ground
    world_body = mbp.world_body()
    ground_shape = Box(2., 2., 1.)
    wall_shape = Box(0.1, 2., 1.1)
    ground_body = mbp.AddRigidBody("ground", SpatialInertia(
        mass=10.0, p_PScm_E=np.array([0., 0., 0.]),
        G_SP_E=UnitInertia(1.0, 1.0, 1.0)))
    mbp.WeldFrames(world_body.body_frame(), ground_body.body_frame(),
                   Isometry3())
    RegisterVisualAndCollisionGeometry(
        mbp, ground_body,
        Isometry3(rotation=np.eye(3), translation=[0, 0, -0.5]),
        ground_shape, "ground", np.array([0.5, 0.5, 0.5, 1.]),
        CoulombFriction(0.9, 0.8))
    # Short table walls
    RegisterVisualAndCollisionGeometry(
        mbp, ground_body,
        Isometry3(rotation=np.eye(3), translation=[-1, 0, 0]),
        wall_shape, "wall_nx",
        np.array([0.5, 0.5, 0.5, 1.]), CoulombFriction(0.9, 0.8))
    RegisterVisualAndCollisionGeometry(
        mbp, ground_body,
        Isometry3(rotation=np.eye(3), translation=[1, 0, 0]),
        wall_shape, "wall_px",
        np.array([0.5, 0.5, 0.5, 1.]), CoulombFriction(0.9, 0.8))

    n_stacks = np.random.randint(1, 5)
    n_bodies_per_stack = np.random.randint(2, 5, size=n_stacks)
    n_bodies = np.sum(n_bodies_per_stack)
    output_dict = {"n_objects": n_bodies}

    k = 0
    for stack_i in range(n_stacks):
        for obj_i in range(n_bodies_per_stack[stack_i]):
            no_mass_no_inertia = SpatialInertia(
                mass=1.0, p_PScm_E=np.array([0., 0., 0.]),
                G_SP_E=UnitInertia(0., 0., 0.))
            body_pre_z = mbp.AddRigidBody("body_{}_pre_z".format(k),
                                          no_mass_no_inertia)
            body_pre_theta = mbp.AddRigidBody("body_{}_pre_theta".format(k),
                                              no_mass_no_inertia)
            body = mbp.AddRigidBody("body_{}".format(k), SpatialInertia(
                mass=1.0, p_PScm_E=np.array([0., 0., 0.]),
                G_SP_E=UnitInertia(0.1, 0.1, 0.1)))

            body_joint_x = PrismaticJoint(
                name="body_{}_x".format(k),
                frame_on_parent=world_body.body_frame(),
                frame_on_child=body_pre_z.body_frame(),
                axis=[1, 0, 0],
                damping=0.)
            mbp.AddJoint(body_joint_x)

            body_joint_z = PrismaticJoint(
                name="body_{}_z".format(k),
                frame_on_parent=body_pre_z.body_frame(),
                frame_on_child=body_pre_theta.body_frame(),
                axis=[0, 0, 1],
                damping=0.)
            mbp.AddJoint(body_joint_z)

            body_joint_theta = RevoluteJoint(
                name="body_{}_theta".format(k),
                frame_on_parent=body_pre_theta.body_frame(),
                frame_on_child=body.body_frame(),
                axis=[0, 1, 0],
                damping=0.)
            mbp.AddJoint(body_joint_theta)

            if (obj_i == n_bodies_per_stack[stack_i] - 1) and np.random.random() > 0.5:
                radius = np.random.uniform(0.05, 0.15)
                color = np.array([0.25, 0.5, np.random.uniform(0.5, 0.8), 1.0])
                body_shape = Sphere(radius)
                output_dict["obj_%04d" % k] = {
                    "class": "2d_sphere",
                    "params": [radius],
                    "params_names": ["radius"],
                    "color": color.tolist(),
                    "pose": [0, 0, 0]
                }
            else:
                length = np.random.uniform(0.1, 0.3)
                height = np.random.uniform(0.1, 0.3)
                body_shape = Box(length, 0.25, height)
                color = np.array([0.5, 0.25, np.random.uniform(0.5, 0.8), 1.0])
                output_dict["obj_%04d" % k] = {
                    "class": "2d_box",
                    "params": [height, length],
                    "params_names": ["height", "length"],
                    "color": color.tolist(),
                    "pose": [0, 0, 0]
                }

            RegisterVisualAndCollisionGeometry(
                mbp, body, Isometry3(), body_shape, "body_{}".format(k),
                color, CoulombFriction(0.9, 0.8))
            k += 1

    mbp.AddForceElement(UniformGravityFieldElement())
    mbp.Finalize()

    visualizer = builder.AddSystem(MeshcatVisualizer(
        scene_graph,
        zmq_url="tcp://127.0.0.1:6000"))
    builder.Connect(scene_graph.get_pose_bundle_output_port(),
                    visualizer.get_input_port(0))

    plt.gca().clear()
    visualizer = builder.AddSystem(PlanarMultibodyVisualizer(scene_graph, ylim=[-0.5, 1.0], ax=plt.gca()))
    builder.Connect(scene_graph.get_pose_bundle_output_port(),
                    visualizer.get_input_port(0))
    diagram = builder.Build()

    diagram_context = diagram.CreateDefaultContext()
    mbp_context = diagram.GetMutableSubsystemContext(
        mbp, diagram_context)
    sg_context = diagram.GetMutableSubsystemContext(
        scene_graph, diagram_context)

    q0 = mbp.GetPositions(mbp_context).copy()

    body_i = 0
    for stack_i in range(n_stacks):
        stack_base_location = np.random.uniform(-0.85, 0.85)
        for obj_i in range(n_bodies_per_stack[stack_i]):
            body_x_index = mbp.GetJointByName("body_{}_x".format(obj_i)).position_start()
            q0[body_x_index] = stack_base_location
            body_z_index = mbp.GetJointByName("body_{}_z".format(obj_i)).position_start()
            q0[body_z_index] = 0.3 * obj_i + 0.3
            body_i += 1

    ik = InverseKinematics(mbp, mbp_context)
    q_dec = ik.q()
    prog = ik.prog()

    constraint = ik.AddMinimumDistanceConstraint(0.01)
    prog.AddQuadraticErrorCost(np.eye(q0.shape[0])*1.0, q0, q_dec)
    for i in range(n_bodies):
        prog.AddBoundingBoxConstraint(-1, 1, q_dec[i*3])
        prog.AddBoundingBoxConstraint(0, 1, q_dec[i*3+1])

    mbp.SetPositions(mbp_context, q0)

    prog.SetInitialGuess(q_dec, q0)
    print("Solving")
    print "Initial guess: ", q0
    print prog.Solve()
    print prog.GetSolverId().name()
    q0_proj = prog.GetSolution(q_dec)
    print "Final: ", q0_proj

    mbp.SetPositions(mbp_context, q0_proj)

    # mbp_context.FixInputPort(
    #    mbp.get_actuation_input_port().get_index(), np.zeros(
    #        mbp.get_actuation_input_port().size()))

    simulator = Simulator(diagram, diagram_context)
    simulator.set_target_realtime_rate(1.0)
    simulator.set_publish_every_time_step(False)
    simulator.StepTo(5.0)

    # Update poses in output dict
    qf = mbp.GetPositions(mbp_context).copy().tolist()
    for k in range(n_bodies):
        x_index = mbp.GetJointByName("body_{}_x".format(k)).position_start()
        z_index = mbp.GetJointByName("body_{}_z".format(k)).position_start()
        t_index = mbp.GetJointByName("body_{}_theta".format(k)).position_start()

        pose = [qf[x_index], qf[z_index], qf[t_index]]
        output_dict["obj_%04d" % k]["pose"] = pose
    return output_dict


if __name__ == "__main__":
    #np.random.seed(42)
    #random.seed(42)

    # Somewhere in the n=1000 range, I hit a
    # "Unhandled exception: Too many open files" error somewhere
    # between Meshcat setup and the print "Solving" line.
    import matplotlib.pyplot as plt
    plt.plot(10, 10)
    for example_num in range(100):
        try:
            output_dict = generate_example()
            with open("planar_bin_static_scenes_stacks.yaml", "a") as file:
                yaml.dump({"env_%d" % int(round(time.time() * 1000)):
                          output_dict},
                          file)
        except Exception as e:
            print "Unhandled exception: ", e

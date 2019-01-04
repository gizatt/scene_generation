import argparse
import curses
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
import yaml

import pydrake
from pydrake.solvers import ik
from pydrake.all import (
    AddFlatTerrainToWorld,
    AddModelInstancesFromSdfString,
    AddModelInstanceFromUrdfFile,
    Box,
    CollisionElement,
    FloatingBaseType,
    RigidBodyFrame,
    RigidBodyTree,
    RollPitchYawFloatingJoint,
    VisualElement
)
from pydrake.multibody.rigid_body import RigidBody
from underactuated import PlanarRigidBodyVisualizer

'''

Rejection-samples feasible configurations of non-penetrating
shapes. Generates a uniform distribution over
feasible configurations.

'''


def add_cube(rbt, name, size, frame, color):
    link = RigidBody()
    link.set_name(name)
    link.add_joint(
        rbt.world(), RollPitchYawFloatingJoint(
            name + "_base", frame.get_transform_to_body()))
    box_element = Box(size)
    visual_element = VisualElement(box_element, np.eye(4), color)
    link.AddVisualElement(visual_element)
    link.set_spatial_inertia(np.eye(6))
    rbt.add_rigid_body(link)

    collision_element = CollisionElement(box_element, np.eye(4))
    collision_element.set_body(link)
    rbt.addCollisionElement(collision_element, link, "default")


# Factories for each object
object_adders = {
    "small_box": lambda rbt, name, frame: add_cube(
        rbt, name=name, size=[0.1, 0.1, 0.1], frame=frame,
        color=[1., 0., 0., 1.]),
    "small_box_blue": lambda rbt, name, frame: add_cube(
        rbt, name=name, size=[0.1, 0.1, 0.1], frame=frame,
        color=[0.3, 0.5, 0.9, 1.]),
    "long_box": lambda rbt, name, frame: add_cube(
        rbt, name=name, size=[0.5, 0.1, 0.1], frame=frame,
        color=[0.9, 0.5, 0.1, 1.]),
    "long_box_blue": lambda rbt, name, frame: add_cube(
        rbt, name=name, size=[0.5, 0.1, 0.1], frame=frame,
        color=[0.3, 0.5, 0.9, 1.])
}


def sample_scene_prefer_sorted_grid(p=0.5, num_objects=None):
    rbt = RigidBodyTree()
    AddFlatTerrainToWorld(rbt)
    rbt_summary = {}

    if num_objects is None:
        num_objects = np.random.geometric(p)
    num_objects = np.random.geometric(p)
    rbt_summary["n_objects"] = num_objects
    for i in range(num_objects):
        class_ind = np.random.randint(len(object_adders.keys()))
        class_name = object_adders.keys()[class_ind]

        theta = (np.random.random()-0.5)*0.1 + np.random.randint(5)*np.pi/2.
        if class_name == "small_box":
            # Small box on -x half
            pose = [np.random.random()*0.5, np.random.random(),
                    theta]
        elif class_name == "long_box":
            pose = [np.random.random()*0.5+0.5, np.random.random(),
                    theta]

        full_name = "%s_%03d" % (class_name, i)
        object_init_frame = RigidBodyFrame(
            "%s_init_frame" % full_name, rbt.world(),
            [pose[0], pose[1], 0.5],
            [0., 0., pose[2]])
        object_adders[class_name](rbt, full_name, object_init_frame)
        rbt_summary["obj_%04d" % i] = {
            "class": class_name,
            "pose":  pose
        }

    rbt.compile()

    q0 = np.zeros(rbt.get_num_positions())

    return rbt, q0, rbt_summary


def sample_scene_uniform_random(p=0.5, num_objects=None):
    rbt = RigidBodyTree()
    AddFlatTerrainToWorld(rbt)
    rbt_summary = {}

    if num_objects is None:
        num_objects = np.random.geometric(p)
    rbt_summary["n_objects"] = num_objects
    for i in range(num_objects):
        class_ind = np.random.randint(len(object_adders.keys()))
        class_name = object_adders.keys()[class_ind]
        full_name = "%s_%03d" % (class_name, i)
        # Planar pose random on [0:1, 0:1, 0:2pi]
        if class_name == "long_box":
            pose = [np.random.random()*0.5, np.random.random(),
                    np.random.random()*np.pi*2.]
        else:
            pose = [np.random.random()*0.5 + 0.25, np.random.random(),
                    np.random.random()*np.pi*2.]
        object_init_frame = RigidBodyFrame(
            "%s_init_frame" % full_name, rbt.world(),
            [pose[0], pose[1], 0.5],
            [0., 0., pose[2]])
        object_adders[class_name](rbt, full_name, object_init_frame)
        rbt_summary["obj_%04d" % i] = {
            "class": class_name,
            "pose":  pose
        }

    rbt.compile()

    q0 = np.zeros(rbt.get_num_positions())

    return rbt, q0, rbt_summary


def build_rbt_from_summary(rbt_summary):
    rbt = RigidBodyTree()
    AddFlatTerrainToWorld(rbt)

    num_objects = int(rbt_summary["n_objects"])
    q0 = np.zeros(6*num_objects)
    for i in range(num_objects):
        obj = rbt_summary["obj_%04d" % i]
        class_name = obj["class"]
        full_name = "%s_%03d" % (class_name, i)
        pose = obj["pose"]
        object_init_frame = RigidBodyFrame(
            "%s_init_frame" % full_name, rbt.world(),
            [0., 0., 0.5],
            [0., 0., 0.0])
        q0[(i*6):(i*6+6)] = np.array([
            pose[0], pose[1], 0.0, 0., 0., pose[2]])
        object_adders[class_name](rbt, full_name, object_init_frame)

    rbt.compile()

    return rbt, q0


def draw_board_state(ax, rbt, q):
    Tview = np.array([[1., 0., 0., 0.],
                      [0., 1., 0., 0.],
                      [0., 0., 0., 0.],
                      [0., 0., 0., 1.]])

    viz = PlanarRigidBodyVisualizer(
        rbt, Tview, xlim=[-0.25, 1.25], ylim=[-0.25, 1.25], ax=ax)
    viz.draw(q)


def main(stdscr, args):
    # Clear screen
    stdscr.clear()
    # Make cursor invisible.
    curses.curs_set(0)

    num_rejected = 0
    num_accepted = 0

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.append:
        file = open(args.output_file, 'a')
    else:
        file = open(args.output_file, 'w')

    try:

        def update_print_str(a, r):
            stdscr.addstr(0, 0, "Rejected %d, Accepted %d, Rate %f" %
                          (r, a, float(a) / max(1, (a + r))))
            stdscr.move(20, 0)
            stdscr.refresh()

        if args.draw:
            fig, ax = plt.subplots(1, 1)
            plt.show(block=False)

        for k in range(args.n_arrangements):
            has_no_collision = False
            while has_no_collision is False:
                rbt, q0, rbt_summary = sample_scene_uniform_random(
                    args.geometric_p, args.num_objects)

                # Check collision distances
                kinsol = rbt.doKinematics(q0)
                ptpairs = rbt.ComputeMaximumDepthCollisionPoints(kinsol)
                if args.draw:
                    draw_board_state(ax, rbt, q0)
                    plt.draw()
                    plt.pause(1e-6)

                update_print_str(num_accepted, num_rejected)

                if len(ptpairs) == 0:
                    has_no_collision = True
                    num_accepted += 1
                else:
                    num_rejected += 1

            yaml.dump({"env_%04d" % k: rbt_summary}, file)

        update_print_str(num_accepted, num_rejected)
    except Exception as e:
        stdscr.addstr(3, 0, "Exception: " + str(e))
        file.close()
        stdscr.getkey()

    stdscr.erase()
    stdscr.refresh()


if __name__ == "__main__":
    default_output_file = "data/%s_uniform_feasible.yaml" % (
        datetime.datetime.now().strftime("%Y%m%d_%H%M"))

    np.set_printoptions(precision=5, suppress=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_arrangements",
                        type=int,
                        default=1000,
                        help="Number of arrangements to generate.")
    parser.add_argument("--geometric_p",
                        type=int,
                        default=0.5,
                        help="p for the geometric distrib for spawning a new object.")
    parser.add_argument("--num_objects",
                        type=int,
                        default=None,
                        help="Number of objects to spawn in a scene. Overrides geometric distrib.")
    parser.add_argument("-o", "--output_file",
                        type=str,
                        default=default_output_file,
                        help="Output file.")
    parser.add_argument("--append",
                        action="store_true",
                        help="Append to output file?")
    parser.add_argument("--seed",
                        type=int,
                        default=int(time.time()*1000) % (2**32 - 1),
                        help="Random seed for rng, "
                             "including scene generation.")
    parser.add_argument("--draw",
                        action="store_true",
                        help="Draw as we go? (Slow)")
    args = parser.parse_args()

    curses.wrapper(main, args)

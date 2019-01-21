import argparse
import curses
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

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


def load_environments(base_dir, verbose=True):
    keys = ["train", "valid", "test"]
    all_environments = {}
    for key in keys:
        file = os.path.join(base_dir, "%s.yaml" % key)
        with open(file, "r") as f:
            new_env = yaml.load(f, Loader=Loader)
            # Vectorize
            all_environments[key] = [new_env[k] for k in new_env.keys()]
            if verbose:
                print("Loaded %d %s environments from file %s" % (
                      len(all_environments[key]), key, file))
    return all_environments


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

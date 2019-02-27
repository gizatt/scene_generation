import os
# Pydrake must be imported before torch to avoid a weird segfault?
import pydrake
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from pydrake.common.eigen_geometry import Quaternion, AngleAxis, Isometry3
from pydrake.systems.framework import DiagramBuilder
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


class ScenesDataset(Dataset):
    def __init__(self, yaml_file):
        with open(yaml_file, "r") as f:
            raw_yaml_environments = yaml.load(f, Loader=Loader)

        # Get them into a list format for more efficient extraction.
        self.yaml_environments, self.yaml_environments_names = zip(*[
            (raw_yaml_environments[k], k) for k in raw_yaml_environments.keys()
            ])

    def __len__(self):
        return len(self.yaml_environments)

    def __getitem__(self, idx):
        return self.yaml_environments[idx]


def RegisterVisualAndCollisionGeometry(
        mbp, body, pose, shape, name, color, friction):
    mbp.RegisterVisualGeometry(body, pose, shape, name + "_vis", color)
    mbp.RegisterCollisionGeometry(body, pose, shape, name + "_col",
                                  friction)


def BuildMbpAndSgFromYamlEnvironment(
        yaml_environment,
        base_environment_type,
        timestep=0.01):
    builder = DiagramBuilder()
    mbp, scene_graph = AddMultibodyPlantSceneGraph(
        builder, MultibodyPlant(time_step=timestep))

    if base_environment_type == "planar_bin":
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
        mbp.AddForceElement(UniformGravityFieldElement())
    else:
        raise ValueError("Unknown base environment type.")

    for k in range(yaml_environment["n_objects"]):
        obj_yaml = yaml_environment["obj_%04d" % k]

        # Planar joints
        if len(obj_yaml["pose"]) == 3:
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

            if obj_yaml["class"] == "2d_sphere":
                radius = obj_yaml["radius"]
                body_shape = Sphere(radius)
            elif obj_yaml["class"] == "2d_box":
                length = obj_yaml["length"]
                height = obj_yaml["height"]
                body_shape = Box(length, 0.25, height)
            else:
                raise NotImplementedError(
                    "Can't handle planar object of type %s yet." %
                    obj_yaml["class"])
        else:
            raise NotImplementedError("Haven't done 6DOF floating bases yet.")

        color = [1., 0., 0.]
        if "color" in obj_yaml.keys():
            color = obj_yaml["color"]
        RegisterVisualAndCollisionGeometry(
            mbp, body, Isometry3(), body_shape, "body_{}".format(k),
            color, CoulombFriction(0.9, 0.8))
    mbp.Finalize()

    # TODO(gizatt) When default position setting for all relevant
    # joint types is done, replace this mess.
    q0 = np.zeros(mbp.num_positions())
    for k in range(yaml_environment["n_objects"]):
        obj_yaml = yaml_environment["obj_%04d" % k]
        if len(obj_yaml["pose"]) == 3:
            x_index = mbp.GetJointByName(
                "body_{}_x".format(k)).position_start()
            z_index = mbp.GetJointByName(
                "body_{}_z".format(k)).position_start()
            t_index = mbp.GetJointByName(
                "body_{}_theta".format(k)).position_start()
            q0[x_index] = obj_yaml["pose"][0]
            q0[z_index] = obj_yaml["pose"][1]
            q0[t_index] = obj_yaml["pose"][2]
        else:
            raise NotImplementedError(
                "Haven't done position setting for 6DOF floating bases yet.")
    return builder, mbp, scene_graph, q0


if __name__ == "__main__":
    dataset = ScenesDataset("../planar_bin_static_scenes.yaml")
    print dataset[10]
    print BuildMbpAndSgFromYamlEnvironment(dataset[10], "planar_bin")
    print("Done")

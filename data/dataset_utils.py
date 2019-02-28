from collections import namedtuple
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
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.meshcat_visualizer import MeshcatVisualizer
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
    ''' Each entry in the dataset is an environment dictionary entry
    from the scene yaml file without further processing. '''
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


VectorizedEnvironments = namedtuple(
    "VectorizedEnvironments",
    ["batch_size", "keep_going", "classes", "params_by_class"],
    verbose=False)


def SubsampleVectorizedEnvironments(data, subsample_inds):
    return VectorizedEnvironments(
        batch_size=len(subsample_inds),
        keep_going=data.keep_going[subsample_inds, ...],
        classes=data.classes[subsample_inds, ...],
        params_by_class=[p[subsample_inds, ...] for p in data.params_by_class])


class ScenesDatasetVectorized(Dataset):
    '''
        Each entry is a set of the following:
        - a keep_going binary vector 1 x max_num_objects in size
        - a class integer vector, 1 x max_num_objects in size,
            with mapping to object names queriable from this class.
        - a list of n_classes float vectors of size
                1 x max_num_objects x pose_size + param_size
            representing generated object poses, with zeros for those
            classes not generated.
    '''

    def __init__(self, yaml_file, max_num_objects=None):
        with open(yaml_file, "r") as f:
            raw_yaml_environments = yaml.load(f, Loader=Loader)

        # Get them into a list format for more efficient extraction.
        self.yaml_environments, self.yaml_environments_names = zip(*[
            (raw_yaml_environments[k], k) for k in raw_yaml_environments.keys()
            ])

        self.class_name_to_id = {}
        self.class_id_to_name = []

        # If we don't know the max # of objs, figure it out
        if max_num_objects is None:
            self.max_num_objects = max([
                env["n_objects"] for env in self.yaml_environments])
        else:
            self.max_num_objects = max_num_objects

        # Vectorize
        self.n_envs = len(self.yaml_environments)
        self.keep_going = torch.zeros(self.n_envs, self.max_num_objects)
        self.classes = torch.zeros(self.n_envs, self.max_num_objects,
                                   dtype=torch.long)
        self.params_by_class = []
        self.params_names_by_class = []

        for env_i, env in enumerate(self.yaml_environments):
            self.keep_going[env_i, 0:env["n_objects"]] = 1
            for k in range(env["n_objects"]):
                obj_yaml = env["obj_%04d" % k]
                # New object, initialize its generated params
                pose = obj_yaml["pose"]
                params = obj_yaml["params"]
                if obj_yaml["class"] not in self.class_name_to_id.keys():
                    class_id = len(self.class_name_to_id)
                    self.class_id_to_name.append(obj_yaml["class"])
                    self.class_name_to_id[obj_yaml["class"]] = class_id
                    self.params_by_class.append(
                        torch.zeros(self.n_envs, self.max_num_objects,
                                    len(pose) + len(params)))
                    self.params_names_by_class.append(obj_yaml["params_names"])
                else:
                    class_id = self.class_name_to_id[obj_yaml["class"]]
                self.classes[env_i, k] = class_id
                self.params_by_class[class_id][env_i, k, :] = torch.tensor(
                    pose + params)

    def get_num_params_by_class(self):
        # Returns TOTAL # of params, including pose params.
        return [
            self.params_by_class[i].shape[-1]
            for i in range(self.get_num_classes())
        ]

    def get_max_num_objects(self):
        return self.max_num_objects

    def get_num_classes(self):
        return len(self.class_id_to_name)

    def get_full_dataset(self):
        return VectorizedEnvironments(
            batch_size=self.n_envs,
            keep_going=self.keep_going,
            classes=self.classes,
            params_by_class=self.params_by_class)

    def get_class_name_from_id(self, i):
        if i == -1:
            return 'none'
        return self.class_id_to_name[i]

    def convert_vectorized_environment_to_yaml(self, data):
        assert(isinstance(data, VectorizedEnvironments))
        yaml_environments = []
        for env_i in range(data.batch_size):
            env = {}
            for obj_i in range(self.max_num_objects):
                if data.keep_going[env_i, obj_i] != 0:
                    class_i = data.classes[env_i, obj_i]
                    params_for_this_class = len(
                        self.params_names_by_class[class_i])
                    params = data.params_by_class[class_i][env_i, obj_i, :]
                    pose_split = params[:-params_for_this_class]
                    params_split = params[-params_for_this_class:]
                    # Decode those, splitting off the last params as
                    # params, and the first few as poses.
                    # TODO(gizatt) Maybe I should collapse pose into params
                    # in my datasets too...
                    obj_entry = {"class": self.class_id_to_name[class_i],
                                 "color": [np.random.uniform(0.5, 0.8), 1., 1., 0.5],
                                 "pose": pose_split.tolist(),
                                 "params": params_split.tolist(),
                                 "params_names": self.params_names_by_class[
                                    class_i]}
                    env["obj_%04d" % obj_i] = obj_entry
                else:
                    break
            env["n_objects"] = obj_i + 1
            yaml_environments.append(env)
        return yaml_environments

    def __len__(self):
        return self.n_envs

    # This might not / should not get used if we're using Pyro,
    # since Pyro handles its own subsampling / batching.
    def __getitem__(self, idx):
        return (self.keep_going[idx, :],
                self.classes[idx, :],
                [p[idx, :] for p in self.params_by_class])


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
                radius = obj_yaml["params"][0]
                body_shape = Sphere(radius)
            elif obj_yaml["class"] == "2d_box":
                height = obj_yaml["params"][0]
                length = obj_yaml["params"][1]
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


def DrawYamlEnvironment(yaml_environment, base_environment_type):
    builder, mbp, scene_graph, q0 = BuildMbpAndSgFromYamlEnvironment(
        yaml_environment, base_environment_type)
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

    sim = Simulator(diagram, diagram_context)
    sim.Initialize()

    poses = scene_graph.get_pose_bundle_output_port().Eval(
        diagram.GetMutableSubsystemContext(scene_graph, diagram_context))
    mbp.SetPositions(mbp_context, q0)
    visualizer._DoPublish(mbp_context, [])
    visualizer._DoPublish(mbp_context, [])


if __name__ == "__main__":
    dataset = ScenesDataset("planar_bin/planar_bin_static_scenes.yaml")
    print dataset[10]
    print BuildMbpAndSgFromYamlEnvironment(dataset[10], "planar_bin")

    dataset_vectorized = ScenesDatasetVectorized(
        "planar_bin/planar_bin_static_scenes.yaml")
    print len(dataset_vectorized), dataset_vectorized[10]
    print dataset_vectorized.get_full_dataset()

    print("Done")

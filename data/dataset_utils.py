from __future__ import print_function

from collections import namedtuple
import os
from copy import deepcopy
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

from pydrake.common.eigen_geometry import Quaternion, AngleAxis
from pydrake.math import (RollPitchYaw, RotationMatrix, RigidTransform)
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.meshcat_visualizer import MeshcatVisualizer
from pydrake.solvers.mathematicalprogram import MathematicalProgram, Solve
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.multibody.parsing import Parser
from pydrake.systems.analysis import Simulator
from pydrake.geometry import (
    Box,
    Cylinder,
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

import scene_generation.differentiable_nlp as diff_nlp


class ScenesDataset(Dataset):
    ''' Each entry in the dataset is an environment dictionary entry
    from the scene yaml file without further processing. '''
    def __init__(self, file_or_folder):
        if os.path.isdir(file_or_folder):
            # Load all yaml files in folder.
            candidate_files = [
                os.path.join(file_or_folder, file)
                for file in os.listdir(file_or_folder)
                if os.path.splitext(file)[-1] == ".yaml"]
        else:
            candidate_files = [file_or_folder]

        self.yaml_environments = []
        self.yaml_environments_names = []
        for yaml_file in candidate_files:
            with open(yaml_file, "r") as f:
                raw_yaml_environments = yaml.load(f, Loader=Loader)
            # Get them into a list format for more efficient extraction.
            new_yaml_environments, new_yaml_environments_names = zip(*[
                (raw_yaml_environments[k], k)
                for k in raw_yaml_environments.keys()])
            self.yaml_environments += new_yaml_environments
            self.yaml_environments_names += new_yaml_environments_names

    def get_environment_index_by_name(self, env_name):
        return self.yaml_environments_names.index(env_name)

    def __len__(self):
        return len(self.yaml_environments)

    def __getitem__(self, idx):
        return self.yaml_environments[idx]


class VectorizedEnvironments:
    def __init__(self, batch_size, keep_going, classes,
                 params_by_class, dataset=None):
        self.batch_size = batch_size
        self.keep_going = keep_going
        self.classes = classes
        self.params_by_class = params_by_class
        self.dataset = dataset

    def convert_to_yaml(self):
        assert(self.dataset)
        return self.dataset.convert_vectorized_environment_to_yaml(self)

    def subsample(self, subsample_inds):
        return VectorizedEnvironments(
            batch_size=len(subsample_inds),
            keep_going=self.keep_going[subsample_inds, ...],
            classes=self.classes[subsample_inds, ...],
            params_by_class=[p[subsample_inds, ...] for p in self.params_by_class],
            dataset=self.dataset)


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

    def __init__(self, file_or_folder, base_environment_type, memorize_params=False, max_num_objects=None):
        temp_dataset = ScenesDataset(file_or_folder)
        self.yaml_environments = temp_dataset.yaml_environments
        self.yaml_environments_names = temp_dataset.yaml_environments_names
        self.base_environment_type = base_environment_type

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
        self.memorized_params_by_class = []

        for env_i, env in enumerate(self.yaml_environments):
            # Keep-going is shifted by one -- it's whether to add an
            # object *after* the ith.
            self.keep_going[env_i, 0:env["n_objects"]-1] = 1
            for k in range(min(env["n_objects"], self.max_num_objects)):
                obj_yaml = env["obj_%04d" % k]
                # New object, initialize its generated params
                pose = obj_yaml["pose"]
                params = obj_yaml["params"]
                params_names = obj_yaml["params_names"]

                if obj_yaml["class"] not in self.class_name_to_id.keys():                    
                    if memorize_params:
                        self.memorized_params_by_class.append(dict(zip(params_names, params)))
                        params = []
                        params_names = []
                    else:
                        self.memorized_params_by_class.append([])

                    class_id = len(self.class_name_to_id)
                    self.class_id_to_name.append(obj_yaml["class"])
                    self.class_name_to_id[obj_yaml["class"]] = class_id
                    self.params_by_class.append(
                        torch.zeros(self.n_envs, self.max_num_objects,
                                    len(pose) + len(params)))
                    self.params_names_by_class.append(params_names)
                else:
                    class_id = self.class_name_to_id[obj_yaml["class"]]
                    if memorize_params:
                        params = []
                        params_names = []
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
            params_by_class=self.params_by_class,
            dataset=self)

    def get_subsample_dataset(self, subsample_inds):
        return VectorizedEnvironments(
            batch_size=len(subsample_inds),
            keep_going=self.keep_going[subsample_inds, :],
            classes=self.classes[subsample_inds, :],
            params_by_class=[p[subsample_inds, :] for p in self.params_by_class],
            dataset=self)

    def get_class_id_from_name(self, name):
        return self.class_name_to_id[name]

    def get_class_name_from_id(self, i):
        if i == -1:
            return 'none'
        return self.class_id_to_name[i]

    def convert_vectorized_environment_to_yaml(self, data):
        assert(isinstance(data, VectorizedEnvironments))
        yaml_environments = []
        for env_i in range(data.batch_size):
            env = {}
            max_obj_i = -1
            for obj_i in range(self.max_num_objects):
                max_obj_i = obj_i
                class_i = data.classes[env_i, obj_i]
                params_for_this_class = len(
                    self.params_names_by_class[class_i])
                params = data.params_by_class[class_i][env_i, obj_i, :]
                if params_for_this_class > 0:
                    pose_split = params[:-params_for_this_class]
                    params_split = params[-params_for_this_class:]
                else:
                    pose_split = params[:]
                    params_split = torch.tensor([])
                # Decode those, splitting off the last params as
                # params, and the first few as poses.
                # TODO(gizatt) Maybe I should collapse pose into params
                # in my datasets too...
                memorized_params_dict = self.memorized_params_by_class[class_i]
                memorized_params_names = memorized_params_dict.keys()
                memorized_params_values = memorized_params_dict.values()
                obj_entry = {"class": self.class_id_to_name[class_i],
                             "color": [np.random.uniform(0.5, 0.8), 0., 1., 1.0],
                             "pose": pose_split.tolist(),
                             "params": params_split.tolist() + memorized_params_values,
                             "params_names": self.params_names_by_class[class_i] + 
                                memorized_params_names}
                env["obj_%04d" % obj_i] = obj_entry
                if data.keep_going[env_i, obj_i] == 0:
                    break
            env["n_objects"] = max_obj_i + 1
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
    # TODO(gizatt): Split this into one function per environment type rather than
    # interleaving them all. Jesus.
    builder = DiagramBuilder()
    mbp, scene_graph = AddMultibodyPlantSceneGraph(
        builder, MultibodyPlant(time_step=timestep))
    world_body = mbp.world_body()
    parser = Parser(mbp, scene_graph)

    if base_environment_type == "planar_bin":
        # Add ground
        ground_shape = Box(2., 2., 1.)
        wall_shape = Box(0.1, 2., 1.1)
        ground_body = mbp.AddRigidBody("ground", SpatialInertia(
            mass=10.0, p_PScm_E=np.array([0., 0., 0.]),
            G_SP_E=UnitInertia(1.0, 1.0, 1.0)))
        mbp.WeldFrames(world_body.body_frame(), ground_body.body_frame(),
                       RigidTransform())
        RegisterVisualAndCollisionGeometry(
            mbp, ground_body,
            RigidTransform(p=[0, 0, -0.5]),
            ground_shape, "ground", np.array([0.5, 0.5, 0.5, 1.]),
            CoulombFriction(0.9, 0.8))
        # Short table walls
        RegisterVisualAndCollisionGeometry(
            mbp, ground_body,
            RigidTransform(p=[-1, 0, 0]),
            wall_shape, "wall_nx",
            np.array([0.5, 0.5, 0.5, 1.]), CoulombFriction(0.9, 0.8))
        RegisterVisualAndCollisionGeometry(
            mbp, ground_body,
            RigidTransform(p=[1, 0, 0]),
            wall_shape, "wall_px",
            np.array([0.5, 0.5, 0.5, 1.]), CoulombFriction(0.9, 0.8))
        mbp.AddForceElement(UniformGravityFieldElement([0., 0., -9.81]))
    elif base_environment_type == "planar_tabletop":
        pass
    elif base_environment_type == "table_setting":
        # Add table
        table_shape = Cylinder(radius=0.9/2, length=0.2)
        table_body = mbp.AddRigidBody("ground", SpatialInertia(
            mass=10.0, p_PScm_E=np.array([0., 0., 0.]),
            G_SP_E=UnitInertia(1.0, 1.0, 1.0)))
        mbp.WeldFrames(world_body.body_frame(), table_body.body_frame(),
                       RigidTransform(p=np.array([0., 0., -0.1])))
        RegisterVisualAndCollisionGeometry(
            mbp, table_body,
            RigidTransform(p=[0.5, 0.5, -0.1]),
            table_shape, "table", np.array([0.8, 0.8, 0.8, 0.01]),
            CoulombFriction(0.9, 0.8))
    elif base_environment_type == "dish_bin":
        ground_shape = Box(2., 2., 2.)
        ground_body = mbp.AddRigidBody("ground", SpatialInertia(
            mass=10.0, p_PScm_E=np.array([0., 0., 0.]),
            G_SP_E=UnitInertia(1.0, 1.0, 1.0)))
        mbp.WeldFrames(world_body.body_frame(), ground_body.body_frame(),
                       RigidTransform(p=[0, 0, -1]))
        mbp.RegisterVisualGeometry(
            ground_body, RigidTransform.Identity(), ground_shape, "ground_vis",
            np.array([0.5, 0.5, 0.5, 1.]))
        mbp.RegisterCollisionGeometry(
            ground_body, RigidTransform.Identity(), ground_shape, "ground_col",
            CoulombFriction(0.9, 0.8))
        # Add the dish bin itself
        dish_bin_model = "/home/gizatt/projects/scene_generation/models/dish_models/bus_tub_01_decomp/bus_tub_01_decomp.urdf"
        parser.AddModelFromFile(dish_bin_model)
        mbp.WeldFrames(world_body.body_frame(), mbp.GetBodyByName("bus_tub_01_decomp_body_link").body_frame(),
                       RigidTransform(p=[0.0, 0., 0.], rpy=RollPitchYaw(np.pi/2., 0., 0.)))
        mbp.AddForceElement(UniformGravityFieldElement())
    else:
        raise ValueError("Unknown base environment type.")

    for k in range(yaml_environment["n_objects"]):
        obj_yaml = yaml_environment["obj_%04d" % k]

        p_offset = np.zeros(3)
        if obj_yaml["class"] == "table":
            p_offset[2] = -0.02

        # Planar joints
        if len(obj_yaml["pose"]) == 3:
            no_mass_no_inertia = SpatialInertia(
                mass=0.0, p_PScm_E=np.array([0., 0., 0.]),
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

            if base_environment_type == "planar_bin":
                z_axis = [0, 0, 1]
                theta_axis = [0, 1, 0]
            else:
                z_axis = [0, 1, 0]
                theta_axis = [0, 0, 1]
            body_joint_z = PrismaticJoint(
                name="body_{}_z".format(k),
                frame_on_parent=body_pre_z.body_frame(),
                frame_on_child=body_pre_theta.body_frame(),
                axis=z_axis,
                damping=0.)
            mbp.AddJoint(body_joint_z)

            body_joint_theta = RevoluteJoint(
                name="body_{}_theta".format(k),
                frame_on_parent=body_pre_theta.body_frame(),
                frame_on_child=body.body_frame(),
                axis=theta_axis,
                damping=0.)
            mbp.AddJoint(body_joint_theta)

            if obj_yaml["class"] == "2d_sphere":
                radius = max(obj_yaml["params"][0], 0.02)
                body_shape = Sphere(radius)
            elif obj_yaml["class"] == "2d_box":
                height = max(obj_yaml["params"][0], 0.02)
                length = max(obj_yaml["params"][1], 0.02)
                if base_environment_type == "planar_bin":
                    body_shape = Box(length, 0.25, height)
                else:
                    body_shape = Box(length, height, 0.25)
            elif obj_yaml["class"] in ["table", "plate", "cup"]:
                assert(base_environment_type == "table_setting")
                radius = max(obj_yaml["params"][0], 0.01)
                length = 0.05
                body_shape = Cylinder(radius=radius/2, length=0.25)
            elif obj_yaml["class"] in ["fork", "knife", "spoon"]:
                assert(base_environment_type == "table_setting")
                width = max(obj_yaml["params"][0], 0.01)
                height = max(obj_yaml["params"][1], 0.01)
                body_shape = Box(width, height, 0.25)
            else:
                raise NotImplementedError(
                    "Can't handle planar object of type %s yet." %
                    obj_yaml["class"])

            color = [1., 0., 0., 1]
            if "color" in obj_yaml.keys():
                if obj_yaml["class"] == "table":
                    color = np.ones(4)*0.1
                elif obj_yaml["class"] == "fork":
                    color = [1., 0.5, 0.5, 1.]
                elif obj_yaml["class"] == "knife":
                    color = [0.5, 1., 0.5, 1.]
                elif obj_yaml["class"] == "spoon":
                    color = [0.5, 0.5, 1., 1.]
                elif obj_yaml["color"] is not None:
                    color = obj_yaml["color"]
            RegisterVisualAndCollisionGeometry(
                mbp, body, RigidTransform(p=p_offset), body_shape, "body_{}".format(k),
                color, CoulombFriction(0.9, 0.8))

        else:
            assert(base_environment_type is "dish_bin")
            candidate_model_files = {
                "mug_1": "/home/gizatt/projects/scene_generation/models/dish_models/mug_1_decomp/mug_1_decomp.urdf",
                "plate_11in": "/home/gizatt/drake/manipulation/models/dish_models/plate_11in_decomp/plate_11in_decomp.urdf",
            }
            assert(obj_yaml["class"] in candidate_model_files.keys())
            model_id = parser.AddModelFromFile(candidate_model_files[obj_yaml["class"]], model_name="model_{}".format(k))

    mbp.Finalize()

    # TODO(gizatt) Eventually, we'll be able to do this default
    # setup stuff before Finalize()... yuck...
    if base_environment_type is "dish_bin":
        q0 = []
        for k in range(yaml_environment["n_objects"]):
            pose = yaml_environment["obj_%04d" % k]["pose"]
            assert(len(pose) == 7) 
            q0 += pose
        q0 = np.array(q0)
    else:
        for k in range(yaml_environment["n_objects"]):
            obj_yaml = yaml_environment["obj_%04d" % k]
            assert(obj_yaml["pose"] == 3)
            mbp.GetMutableJointByName("body_{}_theta".format(k)).set_default_angle(obj_yaml["pose"][2])
            mbp.GetMutableJointByName("body_{}_x".format(k)).set_default_translation(obj_yaml["pose"][0])
            mbp.GetMutableJointByName("body_{}_z".format(k)).set_default_translation(obj_yaml["pose"][1])
        q0 = mbp.GetPositions(mbp.CreateDefaultContext())

    return builder, mbp, scene_graph, q0


def ProjectEnvironmentToFeasibility(yaml_environment, base_environment_type,
                                    make_nonpenetrating=True,
                                    make_static=True):
    builder, mbp, scene_graph, q0 = BuildMbpAndSgFromYamlEnvironment(
        yaml_environment, base_environment_type)
    diagram = builder.Build()

    diagram_context = diagram.CreateDefaultContext()
    mbp_context = diagram.GetMutableSubsystemContext(
        mbp, diagram_context)
    sg_context = diagram.GetMutableSubsystemContext(
        scene_graph, diagram_context)

    outputs = []

    if make_nonpenetrating:
        ik = InverseKinematics(mbp, mbp_context)
        q_dec = ik.q()
        prog = ik.prog()

        constraint = ik.AddMinimumDistanceConstraint(0.001)
        prog.AddQuadraticErrorCost(np.eye(q0.shape[0])*1.0, q0, q_dec)

        if base_environment_type in ["planar_tabletop", "planar_bin"]:
            for i in range(yaml_environment["n_objects"]):
                body_x_index = mbp.GetJointByName("body_{}_x".format(i)).position_start()
                body_z_index = mbp.GetJointByName("body_{}_z".format(i)).position_start()
                body_theta_index = mbp.GetJointByName("body_{}_theta".format(i)).position_start()
                prog.AddBoundingBoxConstraint(-0.9, 0.9, q_dec[body_x_index])
                prog.AddBoundingBoxConstraint(0, 2, q_dec[body_z_index])
        elif base_environment_type in ["table_setting"]:
            for i in range(yaml_environment["n_objects"]):
                body_x_index = mbp.GetJointByName("body_{}_x".format(i)).position_start()
                body_z_index = mbp.GetJointByName("body_{}_z".format(i)).position_start()
                body_theta_index = mbp.GetJointByName("body_{}_theta".format(i)).position_start()
                prog.AddBoundingBoxConstraint(0., 1., q_dec[body_x_index])
                prog.AddBoundingBoxConstraint(0., 1, q_dec[body_z_index])
        else:
            raise NotImplementedError()
        mbp.SetPositions(mbp_context, q0)

        prog.SetInitialGuess(q_dec, q0)
        print("Initial guess: ", q0)
        result = Solve(prog)
        qf = result.GetSolution(q_dec)
        print("Used solver: ", result.get_solver_id().name())
        print("Success? ", result.is_success())
        print("qf: ", qf)

        outputs.append(qf.copy().tolist())
    else:
        qf = q0

    if make_static:
        mbp.SetPositions(mbp_context, qf)

        simulator = Simulator(diagram, diagram_context)
        simulator.set_target_realtime_rate(1.0)
        simulator.set_publish_every_time_step(False)
        simulator.StepTo(5.0)
        qf = mbp.GetPositions(mbp_context).copy()
        outputs.append(qf.copy().tolist())

    # Update poses in output dict
    output_dicts = []
    for output_qf in outputs:
        output_dict = deepcopy(yaml_environment)
        if base_environment_type in ["planar_tabletop", "planar_bin", "table_setting"]:
            for k in range(yaml_environment["n_objects"]):
                x_index = mbp.GetJointByName(
                    "body_{}_x".format(k)).position_start()
                z_index = mbp.GetJointByName(
                    "body_{}_z".format(k)).position_start()
                t_index = mbp.GetJointByName(
                    "body_{}_theta".format(k)).position_start()

                pose = [output_qf[x_index],
                        output_qf[z_index],
                        output_qf[t_index]]
                output_dict["obj_%04d" % k]["pose"] = pose
        else:
            raise NotImplementedError()
        output_dicts.append(output_dict)
    return output_dicts


def DrawYamlEnvironment(yaml_environment, base_environment_type,
                        zmq_url="tcp://127.0.0.1:6000"):
    builder, mbp, scene_graph, q0 = BuildMbpAndSgFromYamlEnvironment(
        yaml_environment, base_environment_type)
    visualizer = builder.AddSystem(MeshcatVisualizer(
                scene_graph,
                zmq_url=zmq_url,
                draw_period=0.0))
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


def DrawYamlEnvironmentPlanar(yaml_environment, base_environment_type,
                              **kwargs):
    builder, mbp, scene_graph, q0 = BuildMbpAndSgFromYamlEnvironment(
        yaml_environment, base_environment_type)

    from underactuated.planar_scenegraph_visualizer import (
        PlanarSceneGraphVisualizer)

    if base_environment_type == "planar_bin":
        Tview = np.array([[1., 0., 0., 0.],
                          [0., 0., 1., 0.],
                          [0., 0., 0., 1.]])
    elif base_environment_type == "planar_tabletop":
        Tview = np.array([[1., 0., 0., 0.],
                          [0., 1., 0., 0.],
                          [0., 0., 0., 1.]])
    elif base_environment_type == "table_setting":
        Tview = np.array([[1., 0., 0., 0.],
                          [0., 1., 0., 0.],
                          [0., 0., 0., 1.]])
    else:
        raise NotImplementedError()

    visualizer = builder.AddSystem(
        PlanarSceneGraphVisualizer(scene_graph, Tview=Tview, **kwargs))
    builder.Connect(scene_graph.get_pose_bundle_output_port(),
                    visualizer.get_input_port(0))
    diagram = builder.Build()

    diagram_context = diagram.CreateDefaultContext()
    mbp_context = diagram.GetMutableSubsystemContext(
        mbp, diagram_context)
    poses = scene_graph.get_pose_bundle_output_port().Eval(
        diagram.GetMutableSubsystemContext(scene_graph, diagram_context))
    mbp.SetPositions(mbp_context, q0)
    visualizer.draw(diagram.GetMutableSubsystemContext(
        visualizer, diagram_context))
    plt.xlim([-0.2, 1.2])
    plt.ylim([-0.2, 1.2])

def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf

if __name__ == "__main__":
    dataset = ScenesDataset("planar_bin/planar_bin_static_scenes.yaml")
    print(dataset[10])
    print(BuildMbpAndSgFromYamlEnvironment(dataset[10], "planar_bin"))

    dataset_vectorized = ScenesDatasetVectorized(
        "planar_bin/planar_bin_static_scenes.yaml")
    print(len(dataset_vectorized), dataset_vectorized[10])
    print(dataset_vectorized.get_full_dataset())

    print("Done")

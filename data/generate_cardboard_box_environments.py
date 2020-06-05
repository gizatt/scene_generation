import argparse
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import PIL, PIL.ImageDraw
import os
import random
import time
import traceback
import yaml
import sys

import pydrake
from pydrake.common import FindResourceOrThrow
from pydrake.common.eigen_geometry import Quaternion, AngleAxis, Isometry3
from pydrake.geometry import (
    Box,
    HalfSpace,
    SceneGraph
)
from pydrake.math import (RollPitchYaw, RigidTransform, RotationMatrix)
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
from pydrake.geometry.render import (
    DepthCameraProperties,
    RenderLabel,
    MakeRenderEngineVtk,
    RenderEngineVtkParams,
)
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

import matplotlib.pyplot as plt

from blender_server.drake_blender_visualizer.blender_visualizer import (
    BlenderColorCamera,
    BlenderLabelCamera
)

from scene_generation.utils.type_convert import matrix_to_dict, dict_to_matrix

'''
scene_info.yaml formatting:

objects:
  object_name:
      class: class_name
      sdf_path: relative path to the object SDF/URDF from the dataset root
      label_index: int
      keypoints: 4xN matrix dict of keypoints in body frame

cameras:
    camera_name:
        calibration: dict of matrix dicts of ROS-format camera intrinsics

data:
    -   camera_frames:
            camera_name: 
                pose: quat xyz format pose vector
                rgb_image_filename: path
                depth_image_filename: path
                label_image_filename: path
        object_poses:
            object_name: quat xyz format pose vector

'''

reserved_labels = [
    RenderLabel.kDoNotRender,
    RenderLabel.kDontCare,
    RenderLabel.kEmpty,
    RenderLabel.kUnspecified,
]

def colorize_labels(image):
    """Colorizes labels."""
    # TODO(eric.cousineau): Revive and use Kuni's palette.
    cc = mpl.colors.ColorConverter()
    color_cycle = plt.rcParams["axes.prop_cycle"]
    colors = np.array([cc.to_rgb(c["color"]) for c in color_cycle])
    bg_color = [0, 0, 0]
    image = np.squeeze(image)
    background = np.zeros(image.shape[:2], dtype=bool)
    for label in reserved_labels:
        background |= image == int(label)
    color_image = colors[image % len(colors)]
    color_image[background] = bg_color
    return color_image


def make_camera_calibration_dict(color_camera_info):
    K = color_camera_info.intrinsic_matrix()
    return {
        "camera_matrix": matrix_to_dict(K),
        "projection_matrix": matrix_to_dict(np.hstack([K, np.zeros((3, 1))])),
        "rectification_matrix": matrix_to_dict(np.eye(3)),
        "distortion_model": "plump_bob",
        "distortion_coefficients": matrix_to_dict(np.zeros((1, 5))),
        "image_width": color_camera_info.width(),
        "image_height": color_camera_info.height(),
    }

def generate_keypoints_for_box(sticker_info_dict):
    # First generate the 3D keypoints in body frame
    scale = np.array(sticker_info_dict["scale"])*2.
    pts = np.array([[-1., -1., -1., -1, 1., 1., 1., 1.],
                    [-1., -1., 1., 1., -1., -1., 1., 1.],
                    [-1., 1., -1., 1., -1., 1., -1., 1.]])
    pts = (pts.T * scale).T / 2.
    values = np.zeros((1, 8))

    for applied_sticker in sticker_info_dict["all_applied_stickers"]:
        if applied_sticker["type"] == "bar_code_sticker":
            pts = np.vstack([pts.T, np.array(applied_sticker["center_xyz"])]).T
            values = np.hstack([values, np.ones((1, 1))])

    return np.vstack([pts, values])

class RgbAndDepthAndLabelImageVisualizer(LeafSystem):
    def __init__(self,
                 depth_camera_properties,
                 draw_timestep=0.1,
                 out_prefix=None):
        LeafSystem.__init__(self)
        self.set_name('image viz')
        self.timestep = draw_timestep
        self.DeclarePeriodicPublish(draw_timestep, 0.)
        
        self.rgb_image_input_port = \
            self.DeclareAbstractInputPort("rgb_image_input_port",
                                   AbstractValue.Make(Image[PixelType.kRgba8U](640, 480, 3)))
        self.depth_image_input_port = \
            self.DeclareAbstractInputPort("depth_image_input_port",
                                   AbstractValue.Make(Image[PixelType.kDepth16U](640, 480, 3)))
        self.label_image_input_port = \
            self.DeclareAbstractInputPort("label_image_input_port",
                                   AbstractValue.Make(Image[PixelType.kLabel16I](640, 480, 1)))
        self.out_prefix = out_prefix
        self.iter_count = 0
        self.depth_near = depth_camera_properties.z_near
        self.depth_far = depth_camera_properties.z_far
        
    def DoPublish(self, context, event):
        LeafSystem.DoPublish(self, context, event)
        if context.get_time() <= 1E-3:
            return

        rgb_image = self.EvalAbstractInput(context, 0).get_value().data
        depth_image = self.EvalAbstractInput(context, 1).get_value().data
        label_image = self.EvalAbstractInput(context, 2).get_value().data

        print("Shapes: ", rgb_image.shape, depth_image.shape, label_image.shape)
        #rgb_image = np.frombuffer(rgb_image.data, dtype=np.uint8).reshape(rgb_image.shape, order='C')
        PIL.Image.fromarray(rgb_image, mode='RGBA').save(self.out_prefix + "%08d_drake_col.png" % self.iter_count)

        #depth_image = np.frombuffer(depth_image.data, dtype=np.uint16).reshape(depth_image.shape, order='C').astype(np.int32)
        #depth_image = ((depth_image.astype(np.int32) / 2**16) * (self.depth_far - self.depth_near) + self.depth_near)*1000
        depth_image = depth_image.astype(np.int32)
        PIL.Image.fromarray(depth_image[:, :, 0], mode='I').save(self.out_prefix + "%08d_drake_depth.png" % self.iter_count)

        #label_image = np.frombuffer(label_image.data, dtype=np.int16).reshape(label_image.shape, order='C').astype(np.int32)
        PIL.Image.fromarray(label_image[:, :, 0].astype(np.int32), mode='I').save(self.out_prefix + "%08d_drake_label.png" % self.iter_count)
        colored_label_image = (colorize_labels(label_image)*255).astype(np.int8)
        PIL.Image.fromarray(colored_label_image, mode='RGB').save(self.out_prefix + "%08d_drake_label_colored.png" % self.iter_count)

        print("Saved to ", self.out_prefix + "%08d_drake_label_colored.png" % self.iter_count)
        self.iter_count += 1


def sanity_check_keypoint_visibility(scene_info_file, show_duration=0.):
    ''' For each camera listed in the scene info file, draws
    the visible keypoints for each object over the color image.

    If show_duration is 0 (default), no images will be plotted for review
    (though they'll still be saved). If it's negative, a plot will open
    and block until it is closed. If it's positive, it'll be shown for that
    duration before the script moves on.'''
    this_dir = os.path.split(scene_info_file)[0]
    with open(scene_info_file, "r") as f:
        scene_info_dict = yaml.load(f, Loader=yaml.FullLoader)
    for data_frame_dict in scene_info_dict["data"]:
        for camera_name in list(data_frame_dict["camera_frames"].keys()):
            this_camera_data = data_frame_dict["camera_frames"][camera_name]
            K = dict_to_matrix(scene_info_dict["cameras"][camera_name]["calibration"]["camera_matrix"])
            pose = np.array(this_camera_data["pose"])
            tf = RigidTransform(p=pose[-3:], quaternion=Quaternion(pose[:4]))
            depth_pil = PIL.Image.open(os.path.join(this_dir, this_camera_data["depth_image_filename"]))
            depth_image = np.asarray(depth_pil)
            color_pil = PIL.Image.open(os.path.join(this_dir, this_camera_data["rgb_image_filename"]))
            drawer = PIL.ImageDraw.Draw(color_pil)
            for object_name in list(data_frame_dict["object_poses"].keys()):
                obj_info_dict = scene_info_dict["objects"][object_name]
                obj_pose = np.array(data_frame_dict["object_poses"][object_name])
                quat_part = obj_pose[:4]
                quat_part /= np.linalg.norm(quat_part)
                keypoints = dict_to_matrix(obj_info_dict["keypoints"])
                # Transform keypoints to camera frame, and then camera image coordinates.
                keypoints_world = RigidTransform(p=obj_pose[-3:], quaternion=Quaternion(quat_part)).multiply(keypoints[:3, :])
                keypoints_cam = tf.inverse().multiply(keypoints_world[:, :])
                keypoints_im = K.dot(keypoints_cam)
                keypoints_im = keypoints_im / keypoints_im[2, :]
                # Get the depth value at each point from the depth image,
                # and use it to decide the visibility of each keypoint from
                # the camera.
                visibility = np.zeros(keypoints.shape[1])
                for keypoint_k in range(keypoints.shape[1]):
                    u, v = keypoints_im[:2, keypoint_k].astype(int)
                    z = keypoints_cam[2, keypoint_k]
                    if u >= 0 and u < depth_image.shape[1] and \
                        v >= 0 and v < depth_image.shape[0]:
                        if z < float(depth_image[v, u])/1000. + 1E-2:
                            visibility[keypoint_k] = 1.
                            color = (255, int(255*keypoints[3, keypoint_k]), 0)
                            drawer.ellipse([u-5, v-5, u+5, v+5], outline=color, fill=None)
                        else:
                            drawer.ellipse([u-2, v-2, u+2, v+2], outline='blue', fill=None)
            color_pil.save(os.path.join(
                this_dir,
                this_camera_data["rgb_image_filename"][:-4] + "_with_keypoints.png"))
            if show_duration != 0:
                plt.imshow(color_pil)
                if show_duration < 0:
                    plt.show()
                elif show_duration > 0:
                    plt.pause(0.1)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("num_scenes", type=int, help="Number of scenes to generate.")
    parser.add_argument("-i", "--data_dir", type=str,
                        help="Directory containing a subdir containing box model folders.",
                        default="~/data/generated_cardboard_envs")
    parser.add_argument("--box_folder_name", type=str,
                        help="Name of subdir with box model folders.",
                        default="cardboard_boxes")
    parser.add_argument("-n", "--num_cameras", type=int,
                        help="Number of simultaneous camera views.",
                        default=3)
    parser.add_argument("-o", "--output_dir", type=str,
                        help="Directory to output scenes into.",
                        default="generated_scenes")
    parser.add_argument("-t", "--sim_time", type=float,
                        help="Duration to simulate.",
                        default=1.0)
    args = parser.parse_args()

    out_dir = args.output_dir
    os.system("mkdir -p %s" % out_dir)
    assert(os.path.exists(args.data_dir) and os.path.exists(os.path.join(args.data_dir, args.box_folder_name)))
    d = os.path.join(args.data_dir, args.box_folder_name)
    candidate_model_files = [
        os.path.abspath(os.path.join(d, o, "box.sdf")) for o in os.listdir(d) 
        if os.path.isdir(os.path.join(d ,o))
    ]

    for scene_iter in range(args.num_scenes):
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

            n_objects = np.random.randint(3, 10)
            poses = []  # [quat, pos]
            object_info_dicts = []
            classes = []
            label_indices = []
            material_overrides = []
            for k in range(n_objects):
                model_name = "model_%d" % k
                model_ind = np.random.randint(0, len(candidate_model_files))
                class_path = candidate_model_files[model_ind]
                classes.append(os.path.relpath(class_path, args.data_dir))
                # Load the info yaml file for this box, which will be right next to it.
                with open(os.path.join(os.path.split(class_path)[0], "info.yaml"), "r") as f:
                    object_info_dicts.append(yaml.load(f, Loader=yaml.FullLoader))
                model_index = parser.AddModelFromFile(class_path, model_name=model_name)
                poses.append([
                    RollPitchYaw(np.random.uniform(0., 2.*np.pi, size=3)).ToQuaternion().wxyz(),
                    [np.random.uniform(-0.1, 0.1),
                     np.random.uniform(-0.1, 0.1),
                     np.random.uniform(0.1, 0.2)]])
                # Get body indices for this model, which drive the labels that'll be
                # assigned to this model
                possible_label_ids = mbp.GetBodyIndices(model_index)
                if len(possible_label_ids) != 1:
                    raise NotImplementedError("Parsed model had more than one body -- this'll"
                                              " mess up the label image.")
                label_indices.append(int(possible_label_ids[0]))
                material_overrides.append(
                    (".*model_%d.*" % k,
                     {"material_type": "CC0_texture",
                      "path": candidate_model_files[model_ind][:-4]}))
            mbp.Finalize()

            #visualizer = builder.AddSystem(MeshcatVisualizer(
            #    scene_graph,
            #    zmq_url="tcp://127.0.0.1:6000",
            #    draw_period=0.01))
            #builder.Connect(scene_graph.get_pose_bundle_output_port(),
            #                visualizer.get_input_port(0))

            # Simulate to stability and get statically stable pose.
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

            #prog.AddVisualizationCallback(vis_callback, q_dec)
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
            result = mp.Solve(prog)
            print("Solve info: ", result)
            print("Solved in %f seconds with %s" % (time.time() - start_time, result.get_solver_id().name()))
            q0_proj = result.GetSolution(q_dec)
            mbp.SetPositions(mbp_context, q0_proj)
            q0_initial = q0_proj.copy()
            simulator.AdvanceTo(args.sim_time + 0.1)
            qf = mbp.GetPositions(mbp_context).copy()

            ## RENDERING STUFF
            rendering_builder = DiagramBuilder()
            cam_tfs = []
            for camera_k in range(args.num_cameras):
                cam_trans_base = np.array([0., 0., np.random.uniform(1.0, 1.5)])
                # Rotate randomly around z axis
                cam_quat_base = RollPitchYaw(
                    0., 0., np.random.uniform(0., np.pi*2.)).ToQuaternion()
                cam_tf_base = Isometry3(quaternion=cam_quat_base,
                                        translation=cam_trans_base)
                # Rotate that camera away from vertical by up to pi/2
                cam_tf_base = Isometry3(quaternion=RollPitchYaw(np.random.uniform(-np.pi/2, np.pi/2), 0., 0.).ToQuaternion(),
                                        translation=[0, 0, 0]).multiply(cam_tf_base)
                # And then rotate again around z axis
                cam_tf_base = Isometry3(quaternion=RollPitchYaw(0., 0., np.random.uniform(-np.pi/2, np.pi/2)).ToQuaternion(),
                                        translation=[0, 0, 0]).multiply(cam_tf_base)
                cam_tfs.append(cam_tf_base)

            offset_quat_base = RollPitchYaw(0., 0., 0.).ToQuaternion().wxyz()
            out_dir_iter = os.path.abspath(os.path.join(out_dir, "scene_%03d" % scene_iter)) + '/'
            os.system("mkdir -p %s" % out_dir_iter)

            # Add materials for ground
            material_overrides.append(
                (".*ground.*",
                 {"material_type": "CC0_texture",
                  "path": 
                    random.choice(["/home/gizatt/tools/blender_server/data/test_pbr_mats/Wood15/Wood15",
                                   "/home/gizatt/tools/blender_server/data/test_pbr_mats/Wood08/Wood08",
                                   "/home/gizatt/tools/blender_server/data/test_pbr_mats/Metal09/Metal09",
                                   "/home/gizatt/tools/blender_server/data/test_pbr_mats/Metal26/Metal26"])}))
            blender_color_cam = BlenderColorCamera(
                scene_graph,
                save_scene=True,
                draw_period=args.sim_time,
                camera_tfs=cam_tfs,
                zmq_url="tcp://127.0.0.1:5556",
                env_map_path=random.choice(
                    ["/home/gizatt/tools/blender_server/data/env_maps/aerodynamics_workshop_4k.hdr",
                     "/home/gizatt/tools/blender_server/data/env_maps/cave_wall_4k.hdr",
                     "/home/gizatt/tools/blender_server/data/env_maps/lab_from_phone.jpg",
                     "/home/gizatt/tools/blender_server/data/env_maps/small_hangar_01_4k.hdr"]),
                material_overrides=material_overrides,
                global_transform=Isometry3(translation=[0, 0, 0],
                                           quaternion=Quaternion(offset_quat_base)),
                out_prefix=out_dir_iter
            )
            cam_context = blender_color_cam.CreateDefaultContext()
            cam_context.SetTime(args.sim_time)
            cam_context.FixInputPort(
                blender_color_cam.get_input_port(0).get_index(),
                AbstractValue.Make(
                    scene_graph.get_pose_bundle_output_port().Eval(sg_context)))
            blender_color_cam.load() # How do I do this through drake systems?
            blender_color_cam.Publish(cam_context)

            # Add and render equivalent cameras on Drake side
            color_camera_info_list = []
            for camera_k in range(args.num_cameras):
                rendering_builder = DiagramBuilder()
                # Rotate cam to get it from blender +y up, +x right, -z forward
                # to Drake X-right, Y-Down, Z-Forward
                cam_tf_base = cam_tfs[camera_k]
                new_rot = cam_tf_base.matrix()[:3, :3].dot(RollPitchYaw([np.pi, 0., 0.]).ToRotationMatrix().matrix())
                new_cam_tf = RigidTransform(R=RotationMatrix(new_rot), p=cam_tf_base.translation())
                cam_tfs[camera_k] =  new_cam_tf

                # Blender is using 90* FOV along the long dimension (width)
                # Find equivalent aspect ratio along vertical dim (y)
                hfov = np.pi/2.
                width = 640
                height = 480
                f = 0.5 * width / np.tan(hfov/2)
                vfov = 2 * np.arctan2( .5 * height, f)
                depth_camera_properties = DepthCameraProperties(
                    width=640, height=480, fov_y=vfov, renderer_name="renderer", z_near=0.1, z_far=3.0)
                parent_frame_id = scene_graph.world_frame_id()
                # Above origin facing straight down
                camera = rendering_builder.AddSystem(
                    RgbdSensor(parent_frame_id, new_cam_tf, depth_camera_properties, show_window=False))
                color_camera_info_list.append(camera.color_camera_info())
                rendering_builder.ExportInput(camera.query_object_input_port(), "camera_query_object_input")

                camera_viz = rendering_builder.AddSystem(RgbAndDepthAndLabelImageVisualizer(
                    depth_camera_properties=depth_camera_properties, 
                    draw_timestep=args.sim_time, out_prefix=out_dir_iter + "/%02d_" % camera_k))
                rendering_builder.Connect(camera.color_image_output_port(),
                                          camera_viz.get_input_port(0))
                rendering_builder.Connect(camera.depth_image_16U_output_port(),
                                          camera_viz.get_input_port(1))
                rendering_builder.Connect(camera.label_image_output_port(),
                                          camera_viz.get_input_port(2))

                rendering_diagram = rendering_builder.Build()
                rendering_diagram_context = rendering_diagram.CreateDefaultContext()
                rendering_diagram_context.SetTime(args.sim_time)
                rendering_diagram_context.FixInputPort(
                    camera.query_object_input_port().get_index(),
                    AbstractValue.Make(
                        scene_graph.get_query_output_port().Eval(sg_context))
                )
                rendering_diagram.Publish(rendering_diagram_context)

            # Finally, format the output scene_info yaml for this scene.
            objects_info_dict = {}
            for k in range(len(poses)):
                keypoints = generate_keypoints_for_box(object_info_dicts[k])
                objects_info_dict["obj_%04d" % k] = {
                    "class": "prime_box",
                    "sdf": classes[k],
                    "label_index": label_indices[k],
                    "keypoints": matrix_to_dict(keypoints),
                    "parameters": object_info_dicts[k]["scale"],
                    "parameter_names": ["scale_x", "scale_y", "scale_z"]
                }

            cameras_info_dict = {}
            for camera_k in range(args.num_cameras):
                cameras_info_dict["cam_%02d" % camera_k] = {
                    "calibration": make_camera_calibration_dict(color_camera_info_list[camera_k])
                }

            # and the per-frame information (just one frame at the moment)
            scene_camera_frames = {}
            for camera_k in range(args.num_cameras):
                pose = np.hstack([cam_tfs[camera_k].rotation().ToQuaternion().wxyz(),
                                  cam_tfs[camera_k].translation()]).tolist()
                scene_camera_frames["cam_%02d" % camera_k] = {
                    "pose": pose,
                    "rgb_image_filename": "%02d_%08d.jpg" % (camera_k, 0),
                    "label_image_filename": "%02d_%08d_drake_label.png" % (camera_k, 0),
                    "depth_image_filename": "%02d_%08d_drake_depth.png" % (camera_k, 0),
                }
            scene_object_poses = {}
            for k in range(len(poses)):
                offset = k*7
                pose = qf[(offset):(offset+7)]
                scene_object_poses["obj_%04d" % k] = pose.tolist()

            data_info_dict = {
                "camera_frames": scene_camera_frames,
                "object_poses": scene_object_poses
            }
            output_dict = {
                "objects": objects_info_dict,
                "cameras": cameras_info_dict,
                "data": [data_info_dict]
            }

            with open(os.path.join(out_dir_iter, "scene_info.yaml"), "w") as file:
                yaml.dump(output_dict, file)
            sanity_check_keypoint_visibility(os.path.join(out_dir_iter, "scene_info.yaml"),
                                             show_duration=0.)

        except Exception as e:
            print("Unhandled exception ", e)
            print("Traceback: ")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback, limit=4, file=sys.stdout)
        except:
            print("Unhandled unnamed exception, probably sim error")

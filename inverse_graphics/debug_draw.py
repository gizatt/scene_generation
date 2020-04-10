import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import trimesh
from io import BytesIO

from pydrake.common.eigen_geometry import Quaternion, Isometry3
from pydrake.math import RigidTransform, RollPitchYaw

import torch
from detectron2.utils.visualizer import Visualizer

from scene_generation.utils.type_convert import dict_to_matrix

def draw_shape_and_pose_predictions(input, pred, test_metadata):
    with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
        # Draw the prediction image itself
        im = input["image"].detach().cpu().numpy().transpose([1, 2, 0])
        v = Visualizer(
                cv2.cvtColor(im, cv2.COLOR_BGR2RGB),
                metadata=test_metadata, 
                scale=1)
        v = v.draw_instance_predictions(pred["instances"].to("cpu"))
        im_rgb = v.get_image()[:, :, ::-1]
        # For each prediction in the image, create a box of the relevant size and pose
        # in the trimesh scene.
        # Need to copy camera calibration, and flip to -z forward.
        shape = [im.shape[1], im.shape[0]]
        K = dict_to_matrix(input["camera_calibration"]["camera_matrix"])
        scene = trimesh.scene.Scene(
           camera = trimesh.scene.cameras.Camera(
               name="cam",
               resolution=shape,
               focal=(K[0, 0], K[1, 1])),
           camera_transform=RigidTransform(rpy=RollPitchYaw(0, np.pi, np.pi), p=np.zeros(3)).matrix())

        if len(pred["instances"]) > 0:
            pred_shapes = pred["instances"].get("pred_shape_params").cpu().detach().numpy()
            pred_poses = pred["instances"].get("pred_pose").cpu().detach().numpy()
            for obj_k in range(pred_shapes.shape[0]):
                pose = pred_poses[obj_k, :]
                pose[:4] = pose[:4] / np.linalg.norm(pose[:4])
                tf_mat = RigidTransform(quaternion=Quaternion(pose[:4]), p=pose[-3:]).matrix()
                scene.add_geometry(
                    trimesh.creation.box(extents=pred_shapes[obj_k, :]*2.,
                                         transform=tf_mat,
                                         face_colors=np.random.random([3])*255))
            png = scene.save_image(background=[0, 0, 0], resolution=shape)
            im_boxes = np.asarray(Image.open(BytesIO(png)))[:, :, -2::-1]
            # Get black and white original image
            bw_im = np.tile(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[:, :, None], [1, 1, 3])
            im_boxes = (im_boxes*0.8 + bw_im*0.2).astype(np.uint8)
        else:
            im_boxes = im*0.
    return im_rgb, im_boxes

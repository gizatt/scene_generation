import attr
from copy import deepcopy
import datetime
import json
import numpy as np
import os
import PIL.Image
import random
from typing import List, Dict, Any, Tuple
import yaml

from pycocotools import mask
from skimage import measure

from scene_generation.utils import (
    imgproc,
    transformations,
    type_convert,
)

'''
Builds a flattened COCO-style dataset description JSON
from a raw synthetic scene database.

Database structure based significantly on Wei's code
as it appears in `github.com/weigao95/mankey_ros`.
'''

@attr.s
class RawSyntheticSceneDatabaseConfig:
    # Global path to root of raw generated scene database.
    data_root: str = ''

    # Global path to a file that lists which scene group dirs will be
    # used for this dataset.
    scene_group_list_filepath: str = ''

    # The name of the yaml file with pose annotation
    scene_config_yaml: str = 'scene_info.yaml'

    # Want it to print lots of crap?
    verbose: bool = True


class RawSyntheticSceneDatabase():

    def __init__(self, config):
        self._config = config

        # Build a list of scenes that will be used as the dataset
        self._scene_path_list = self._get_scene_list_from_config(config)

        # Build up complete list of scene entries, starting at the
        # scene group level.
        self._scene_info_list = []  # type: List[SyntheticSceneDatabaseScene]
        for scene_root in self._scene_path_list:
            # Load in the scene subdir, which may contain multiple
            # scene entries.
            scene_info_yaml_path = os.path.join(scene_root, self._config.scene_config_yaml)
            self._scene_info_list.append((scene_info_yaml_path, scene_root))

        # Simple info
        if config.verbose:
            print('The number of scenes is %d' % len(self._scene_info_list))


    @staticmethod
    def _get_scene_list_from_config(config):
        assert os.path.exists(config.data_root)
        assert os.path.exists(config.scene_group_list_filepath)

        # Simple checker
        def is_scene_valid(scene_root_path, scene_config_yaml):  # type: (str, str) -> bool
            # The path must be valid
            if not os.path.exists(scene_root_path):
                return False

            # Must contains keypoint annotation
            scene_yaml_path = os.path.join(scene_root_path, scene_config_yaml)
            if not os.path.exists(scene_yaml_path):
                return False

            # OK
            return True

        # Read the config file
        scene_root_list = []
        with open(config.scene_group_list_filepath, 'r') as config_file:
            lines = config_file.read().split('\n')
            for line in lines:
                if len(line) == 0:
                        continue    
                scene_group_root = os.path.join(config.data_root, line)
                scene_paths = [f.path for f in os.scandir(scene_group_root)
                               if f.is_dir()]
                for scene_root_candidate in scene_paths:
                    if is_scene_valid(scene_root_candidate, config.scene_config_yaml):
                        scene_root_list.append(scene_root_candidate)

        # OK
        return scene_root_list

    def _add_image_set_to_db(
            self, rgb_img_path, depth_img_path, label_img_path,
            camera_pose_dict, calibration, output_db):
        rgb_id = len(output_db["images"])
        depth_id = rgb_id + 1
        label_id = rgb_id + 2
        output_db["images"].append(
            {
                "id": rgb_id,
                "type": "RGB",
                "corresponding_depth_id": depth_id,
                "corresponding_label_id": label_id,
                "file_name": os.path.relpath(rgb_img_path,
                                             self._config.data_root),
                "pose": camera_pose_dict,
                "calibration": calibration,
                "height": calibration["image_height"],
                "width": calibration["image_width"]
            })
        output_db["images"].append(
            {
                "id": depth_id,
                "type": "depth",
                "corresponding_rgb_id": rgb_id,
                "corresponding_label_id": label_id,
                "file_name": os.path.relpath(depth_img_path,
                                             self._config.data_root),
                "pose": camera_pose_dict,
                "calibration": calibration,
                "height": calibration["image_height"],
                "width": calibration["image_width"]
            })
        output_db["images"].append(
            {
                "id": label_id,
                "type": "label",
                "corresponding_rgb_id": rgb_id,
                "corresponding_depth_id": depth_id,
                "file_name": os.path.relpath(label_img_path,
                                             self._config.data_root),
                "pose": camera_pose_dict,
                "calibration": calibration,
                "height": calibration["image_height"],
                "width": calibration["image_width"]
            })
        return rgb_id, depth_id, label_id

    @staticmethod
    def _get_encoded_mask(image_mask):
        return mask.encode(np.asfortranarray(image_mask))

    @staticmethod
    def _get_area_of_encoded_mask(encoded_mask):
        # return the area of the mask (by counting the nonzero pixels)
        return mask.area(encoded_mask)

    @staticmethod
    def _get_bounding_box(encoded_mask):
        # returns x, y (top left), width, height
        bounding_box = mask.toBbox(encoded_mask)
        return bounding_box.astype(int)

    @staticmethod
    def _get_polygons(image_mask, tolerance=0):
        """
        code from https://github.com/waspinator/pycococreator/blob/master/pycococreatortools/pycococreatortools.py
        Args:
            image_mask: a 2D binary numpy array where '1's represent the object
            tolerance: Maximum distance from original points of polygon to approximated
                polygonal chain. If tolerance is 0, the original coordinate array is returned.
        """
        polygons = []
        # pad mask to close contours of shapes which start and end at an edge
        padded_binary_mask = np.pad(image_mask, pad_width=1, mode='constant', constant_values=0)
        contours = measure.find_contours(padded_binary_mask, 0.5)
        contours = np.subtract(contours, 1)
        for contour in contours:
            if not np.array_equal(contour[0], contour[-1]):
                contour = np.vstack((contour, contour[0]))
            contour = measure.approximate_polygon(contour, tolerance)
            if len(contour) < 3:
                continue
            contour = np.flip(contour, axis=1)
            segmentation = contour.ravel().tolist()
            # after padding and subtracting 1 we may get -0.5 points in our segmentation
            segmentation = [0 if i < 0 else i for i in segmentation]
            polygons.append(segmentation)

        # OK
        return polygons

    def _check_annotation_valid(self, ann):
        if ann['area'] < self._config.AREA_THRESHOLD:
            return False

        [x, y, width, height] = ann['bbox']
        if (width < self._config.BBOX_MIN_WIDTH) or (height < self._config.BBOX_MIN_HEIGHT):
            return False

        # Everything is OK here
        return True

    @staticmethod
    def _compile_keypoint_info(
            keypoint_object, camera_info_map, camera2world, object2world,
            depth_image, output_annotation):

        K = type_convert.dict_to_matrix(camera_info_map["calibration"]["camera_matrix"])
        n_keypoints = keypoint_object.shape[1]
        assert(keypoint_object.shape[0] == 4) # Assume 3D keypoint + scalar value        

        keypoint_values = keypoint_object[-1, :].copy()
        # Convert to homogenous coords and do frame transformations
        keypoint_object[-1, :] = 1.
        keypoint_world = object2world.dot(keypoint_object)
        keypoint_cam = np.linalg.inv(camera2world).dot(keypoint_world)
        keypoint_pixelxy_depth = K.dot(keypoint_cam[:3, :])
        keypoint_pixelxy_depth /= keypoint_pixelxy_depth[2, :]
        keypoint_pixelxy_depth[2, :] = keypoint_cam[2, :]*1000.

        # Calculate validity by checking if it's in the image,
        # in the bounding box, and if it's not occluded (according to
        # the depth image)
        keypoint_validity_weight = np.ones((1, n_keypoints))
        for i in range(n_keypoints):
            pixel = imgproc.PixelCoord()
            pixel.x, pixel.y, depth_mm = keypoint_pixelxy_depth[:, i]
            valid = True

            # The pixel must be in bounding box
            bbx, bby, bbw, bbh = output_annotation["bbox"]
            if not imgproc.pixel_in_bbox(pixel,
                                         imgproc.PixelCoord(x=bbx, y=bby),
                                         imgproc.PixelCoord(x=bbx+bbw, y=bby+bbh)):
                valid = False
            elif depth_mm < 0:
                valid = False
            elif depth_mm > (float(depth_image[int(pixel.y), int(pixel.x)]) + 10.):
                # The depth cannot be behind the depth image (with 10mm margin):
                valid = False
            # Invalid all the dimension
            if not valid:
                keypoint_validity_weight[0, i] = 0

        output_annotation["keypoint_vals"] = keypoint_values.tolist()
        output_annotation["keypoint_pixelxy_depth"] = type_convert.matrix_to_dict(keypoint_pixelxy_depth)
        output_annotation["keypoint_validity"] = keypoint_validity_weight.tolist()


    def _compile_entries_from_scene(
            self, yaml_path, scene_root, category_name2id, output_db):
        ''' Adds entries from the given yaml_path and scene_root to the
        output_db in-place. '''

        with open(yaml_path, 'r') as f:
            datamap = yaml.load(f, Loader=yaml.FullLoader)

        # The info about camera
        camera_name_list = list(datamap["cameras"].keys())
        num_cameras = len(camera_name_list)
        assert num_cameras > 0

        # For each time step in observing this scene...
        for entry_map in datamap["data"]:
            # For each camera observing this scene...
            assert(len(entry_map["camera_frames"].keys()) == num_cameras)
            for camera_name in camera_name_list:
                camera_entry_info_map = entry_map["camera_frames"][camera_name]
                camera_info_map = datamap["cameras"][camera_name]
                camera_pose_dict = camera_entry_info_map["pose"]
                camera2world = type_convert.transform_from_pose_vector(camera_pose_dict)
                        
                rgb_img_path = os.path.join(scene_root,
                    camera_entry_info_map['rgb_image_filename'])
                depth_img_path = os.path.join(scene_root,
                    camera_entry_info_map['depth_image_filename'])
                label_img_path = os.path.join(scene_root,
                    camera_entry_info_map['label_image_filename'])
                rgb_image_id, depth_image_id, label_image_id = self._add_image_set_to_db(
                    rgb_img_path, depth_img_path, label_img_path,
                    camera_pose_dict, camera_info_map["calibration"], output_db)

                # Pre-load the label and depth images here so they don't have to be
                # loaded for each annotation.
                label_image = np.asarray(PIL.Image.open(label_img_path))
                depth_image = np.asarray(PIL.Image.open(depth_img_path))

                # Build the annotation entry for each object.
                for object_name in list(entry_map["object_poses"].keys()):
                    assert(object_name in datamap["objects"].keys())
                    object_info_map = datamap["objects"][object_name]
                    label_index = object_info_map["label_index"]
                    object_type = object_info_map["class"]
                    assert(object_type in category_name2id)
                    category_id = category_name2id[object_type]
                    object_pose = entry_map["object_poses"][object_name]
                    object2world = type_convert.transform_from_pose_vector(
                        object_pose)
                    object_params = object_info_map["parameters"]
                    # Get the binary mask image from the label image
                    mask = label_image == label_index
                    if np.sum(mask) == 0.:
                        continue

                    encoded_mask = self._get_encoded_mask(mask)
                    x, y, width, height = self._get_bounding_box(encoded_mask)
                    area = self._get_area_of_encoded_mask(encoded_mask)
                    segmentation = self._get_polygons(mask)
                    new_annotation_id = len(output_db["annotations"])
                    new_annotation = {
                        "id": new_annotation_id,
                        "image_id": rgb_image_id,
                        "depth_image_id": depth_image_id,
                        "label_image_id": label_image_id,
                        "label_id_in_label_image": label_index,
                        "category_id": category_id,
                        "iscrowd": 0,
                        "score": 1.,
                        "segmentation": segmentation,
                        "area": float(area),
                        "bbox": [int(x), int(y), int(width), int(height)],
                        "object_sdf": object_info_map["sdf"],
                        "pose": object_pose,
                        "parameters": object_params
                    }
                    keypoint_object = type_convert.dict_to_matrix(object_info_map["keypoints"])
                    self._compile_keypoint_info(keypoint_object, camera_info_map, camera2world, object2world,
                                                depth_image, new_annotation)
                    output_db["annotations"].append(new_annotation)

    def compile_generated_scenes_to_dataset(
            self, output_filename, description="XenCOCO", human_readable=False):
        # Make the base dictionary
        output_db = {
            "info": {
                "description": description,
                "year": int(datetime.datetime.now().year),
                "date_created": datetime.datetime.now().strftime('%Y/%m/%d')
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [
                {"supercategory": "prime_box",
                 "id": 1,
                 "name": "prime_box",
                 "parameters": ["scale_x", "scale_y", "scale_z"]}
            ]
        }

        # Build name2id
        category_name2id = {}
        for category_info in output_db["categories"]:
            assert 'name' in category_info
            assert 'id' in category_info
            category_name2id[category_info['name']] = category_info['id']

        # Process each sub-scene.
        for yaml_path, scene_root in self._scene_info_list:
            if self._config.verbose:
                print('Processing: ', scene_root)
            self._compile_entries_from_scene(yaml_path, scene_root, category_name2id, output_db)

        # Save it out!
        with open(output_filename, 'w') as output_json_file:
            if human_readable:
                json.dump(output_db, output_json_file, indent=2)
            else:
                json.dump(output_db, output_json_file)


if __name__ == '__main__':
    config = RawSyntheticSceneDatabaseConfig()
    config.data_root = "/home/gizatt/data/generated_cardboard_envs"
    config.scene_group_list_filepath = os.path.join(
        config.data_root, "scene_groups_mini.txt")
    db = RawSyntheticSceneDatabase(config)
    db.compile_generated_scenes_to_dataset(
        output_filename=os.path.join(config.data_root, "train.json"))

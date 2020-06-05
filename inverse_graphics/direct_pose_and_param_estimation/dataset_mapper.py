import copy
import logging
import numpy as np

import torch
from detectron2.data.transforms import RandomFlip
from detectron2.data import detection_utils as utils
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data import transforms as T
from detectron2.structures import (
    BitMasks, Boxes, BoxMode, Instances,
    PolygonMasks, polygons_to_bitmask
)
from fvcore.common.file_io import PathManager, file_lock

from scene_generation.utils.type_convert import (
    dict_to_matrix,
    rigidtransform_from_pose_vector,
    pose_vector_from_rigidtransform
)
from scene_generation.utils.torch_quaternion import qeuler


"""
This file contains functions to parse XenCOCO-format annotations into dicts in "Detectron2 format",

This is heavily based on https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/datasets/coco.py.
and
https://github.com/facebookresearch/meshrcnn/blob/master/meshrcnn/data/meshrcnn_transforms.py
"""


logger = logging.getLogger(__name__)

def annotations_to_instances(annos, image_size, camera_pose=None):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.
    Args:
        annos (list[dict]): a list of annotations, one per instance.
        image_size (tuple): height, width
        camera_pose: quat xyz camera pose, np array or list
    Returns:
        Instances: It will contains fields "gt_boxes", "gt_classes",
            "gt_masks", "gt_keypoints", if they can be obtained from `annos`.
    """
    boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
    target = Instances(image_size)
    boxes = target.gt_boxes = Boxes(boxes)
    boxes.clip(image_size)

    classes = [obj["category_id"] for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    if len(annos) and "segmentation" in annos[0]:
        masks = [obj["segmentation"] for obj in annos]
        target.gt_masks = BitMasks(torch.stack(masks))

    # camera calibration
    #if len(annos) and "camera_calibration" in annos[0]:
    #     K = [torch.tensor(
    #             dict_to_matrix(
    #                 obj["camera_calibration"]["camera_matrix"]))
    #          for obj in annos]
    #     target.gt_K = torch.stack(K, dim=0)

    if len(annos) and "parameters" in annos[0]:
        shape_params = [obj["parameters"] for obj in annos]
        target.gt_shape_params = torch.stack(shape_params, dim=0)
    
    if len(annos) and "pose" in annos[0] and camera_pose is not None:
        pose = []
        cam_tf = rigidtransform_from_pose_vector(camera_pose)
        for obj in annos:
            obj_tf = rigidtransform_from_pose_vector(obj["pose"])
            obj_in_cam = cam_tf.inverse().multiply(obj_tf)
            pose.append(
                torch.tensor(pose_vector_from_rigidtransform(obj_in_cam).astype("float32")))

        target.gt_pose_quatxyz = torch.stack(pose, dim=0)

        # Convert to RPY, should be range [-pi, pi]
        target.gt_pose_rpy = qeuler(target.gt_pose_quatxyz[:, :4], order='zyx')

    return target

class XenRCNNMapper:
    """
    A callable which takes a dict produced by the synthetic dataset, and applies transformations,
    including image resizing and flipping. The transformation parameters are parsed from cfg file
    and depending on the is_train condition.
    Note that for our existing models, mean/std normalization is done by the model instead of here.
    """

    def __init__(self, cfg, is_train=True):
        self.tfm_gens = utils.build_transform_gen(cfg, is_train)
        # Force it to not use random flip
        for gen in self.tfm_gens:
            if isinstance(gen, RandomFlip):
                self.tfm_gens.remove(gen)
        # fmt: off
        self.img_format     = cfg.INPUT.FORMAT

        # fmt: on
        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        Transform the dataset_dict according to the configured transformations.
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a new dict that's going to be processed by the model.
                It currently does the following:
                1. Read the image from "file_name"
                2. Transform the image and annotations
                3. Prepare the annotations to :class:`Instances`
        """
        #dataset_dict = {key: value for key, value in dataset_dict.items()}
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        depth_image = utils.read_image(dataset_dict["depth_file_name"], format='I')
        utils.check_image_size(dataset_dict, image)

        orig_image_shape = image.shape[:2]
        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        depth_image, transforms = T.apply_transform_gens(self.tfm_gens, depth_image)

        image_shape = image.shape[:2]  # h, w
        camera_pose = dataset_dict["camera_pose"]

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
        # Convert depth image to H W C (c = 1), and convert to meters
        dataset_dict["depth_image"] = torch.as_tensor(np.expand_dims(depth_image, 0).astype("float32")) / 1000.
        # Can use uint8 if it turns out to be slow some day

        dataset_dict["K"] = torch.as_tensor(dict_to_matrix(
            dataset_dict["camera_calibration"]["camera_matrix"]).astype("float32"))

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            annos = [
                self.transform_annotations(obj, transforms, image_shape, orig_image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            # Should not be empty during training
            instances = annotations_to_instances(annos, image_shape, camera_pose)
            dataset_dict["instances"] = instances[instances.gt_boxes.nonempty()]

        return dataset_dict

    def transform_annotations(self, annotation, transforms, image_size, orig_image_size):
        """
        Apply image transformations to the annotations.
        After this method, the box mode will be set to XYXY_ABS.
        """
        bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
        # Note that bbox is 1d (per-instance bounding box)
        annotation["bbox"] = transforms.apply_box([bbox])[0]
        annotation["bbox_mode"] = BoxMode.XYXY_ABS

        # each instance contains 1 mask
        annotation["segmentation"] = self._process_mask(annotation["segmentation"], transforms, orig_image_size)

        # camera
        h, w = image_size
        #annotation["K"] = [annotation["K"][0], w / 2.0, h / 2.0]
        annotation["pose"] = torch.tensor(annotation["pose"])
        annotation["parameters"] = torch.tensor(annotation["parameters"])

        return annotation

    def _process_mask(self, mask, transforms, image_size):
        # applies image transformations to mask
        mask = np.array(polygons_to_bitmask(mask, image_size[0], image_size[1]), dtype=np.uint8)
        mask = transforms.apply_image(mask)
        mask = torch.as_tensor(np.ascontiguousarray(mask), dtype=torch.float32) / 255.0
        return mask

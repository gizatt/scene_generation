import copy
import io
import logging
import contextlib
import os
from PIL import Image
import datetime
import json
import numpy as np

import torch
from torch.distributions import MultivariateNormal
from fvcore.common.timer import Timer
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
from scene_generation.utils.torch_misc_math import (
    conv_output_shape
)
from scene_generation.utils.torch_quaternion import qeuler


"""
This file contains functions to parse XenCOCO-format annotations into dicts in "Detectron2 format",

This is heavily based on https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/datasets/coco.py.
and
https://github.com/facebookresearch/meshrcnn/blob/master/meshrcnn/data/meshrcnn_transforms.py
"""


logger = logging.getLogger(__name__)

def load_xencoco_json(json_file, data_root, dataset_name=None):
    """
    Load a json file with XenCOCO's instances annotation format.
    It differs from detectron2's load_coco_json in that it returns one
        dict per RGB/Depth/Label image triplet, and it includes all of the
        supporting information at the image (camera pose, camera intrinsics)
        and annotation (object pose and parameters) levels.
    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        data_root (str or path-like): the directory from which paths in the json originate.
        dataset_name (str): the name of the dataset (e.g., coco_2017_train).
            If provided, this function will also put "thing_classes" into
            the metadata associated with this dataset.
    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        meta.thing_classes = thing_classes

        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning(
                    """
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
"""
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = coco_api.loadImgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'iscrowd': 0,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]

    # Collect annotations for the RGB images
    rgb_img_ids = [img_id for img_id in img_ids if imgs[img_id]["type"] == "RGB"]
    rgb_imgs = [imgs[img_id] for img_id in rgb_img_ids]
    anns = [coco_api.imgToAnns[img_id] for img_id in rgb_img_ids]

    if "minival" not in json_file:
        # The popular valminusminival & minival annotations for COCO2014 contain this bug.
        # However the ratio of buggy annotations there is tiny and does not affect accuracy.
        # Therefore we explicitly white-list them.
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(
            json_file
        )

    rgb_imgs_anns = list(zip(rgb_imgs, anns))

    logger.info("Loaded {} image groups in XenCOCO format from {}".format(len(rgb_imgs_anns), json_file))

    dataset_dicts = []

    ann_keys = ["iscrowd",
                "bbox",
                "keypoint_vals",
                "keypoint_pixelxy_depth",
                "keypoint_validity",
                "category_id",
                "object_sdf",
                "pose",
                "parameters",
                "score"]

    num_instances_without_valid_segmentation = 0

    for (img_dict, anno_dict_list) in rgb_imgs_anns:
        record = {}
        record["file_name"] = os.path.join(data_root, img_dict["file_name"])
        record["depth_file_name"] = os.path.join(data_root, imgs[img_dict["corresponding_depth_id"]]["file_name"])
        record["label_file_name"] = os.path.join(data_root, imgs[img_dict["corresponding_label_id"]]["file_name"])
        record["camera_pose"] = img_dict["pose"]
        record["camera_calibration"] = img_dict["calibration"]
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            assert anno["image_id"] == image_id

            assert anno.get("ignore", 0) == 0

            obj = {key: anno[key] for key in ann_keys if key in anno}

            segm = anno.get("segmentation", None)
            if segm:  # either list[list[float]] or dict(RLE)
                if not isinstance(segm, dict):
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

            keypts = anno.get("keypoints", None)
            if keypts:  # list[int]
                for idx, v in enumerate(keypts):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # Therefore we assume the coordinates are "pixel indices" and
                        # add 0.5 to convert to floating point coordinates.
                        keypts[idx] = v + 0.5
                obj["keypoints"] = keypts

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if id_map:
                obj["category_id"] = id_map[obj["category_id"]]
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. "
            "There might be issues in your dataset generation process.".format(
                num_instances_without_valid_segmentation
            )
        )
    return dataset_dicts


def annotations_to_instances(annos, image_size, camera_pose=None,
                             target_heatmap_size=None):
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

    if len(annos) and "keypoint_vals" in annos[0] and target_heatmap_size is not None:
        # Build bbox-space keypoint heatmaps for the
        # valid (visible) keypoints of each type
        keypoint_types = np.array([0., 1.]) # Forcing this into just two manual keypoint types for now
        target.gt_heatmaps = torch.zeros((len(annos), len(keypoint_types), target_heatmap_size, target_heatmap_size))
        grid_x, grid_y = torch.meshgrid(torch.arange(target_heatmap_size), torch.arange(target_heatmap_size))
        batched_grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1).view(-1, 2)

        # Figure out x and y scaling from orig image to these coords
        xy_scaling = torch.tensor([target_heatmap_size/image_size[0],
                                   target_heatmap_size/image_size[1]])
        for k, obj in enumerate(annos):
            covar = torch.diag(torch.tensor([1., 1.]))
            vals = obj["keypoint_vals"]
            # unused
            xyzs = torch.tensor(dict_to_matrix(obj["keypoint_pixelxy_depth"]).astype("float32"))
            good_keypoints_mask = obj["keypoint_validity"][0]
            #print(vals, xyz, good_keypoints_mask)
            for v, xyz, m in zip(vals, xyzs.T, good_keypoints_mask):
                if not m:
                    continue
                dist = MultivariateNormal(xyz[[1, 0]] * xy_scaling, covar)
                logprobs = dist.log_prob(batched_grid)
                ind = np.argmin(np.abs(v - keypoint_types))
                target.gt_heatmaps[k, ind, ...] += torch.exp(logprobs.view(target_heatmap_size, target_heatmap_size))
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
        self.img_format     = cfg.INPUT.FORMAT
        self.is_train       = is_train

        self.target_heatmap_size = None
        if cfg.MODEL.ROI_HEATMAP_HEAD is not None:
            res = [cfg.MODEL.ROI_HEATMAP_HEAD.POOLER_RESOLUTION,
                   cfg.MODEL.ROI_HEATMAP_HEAD.POOLER_RESOLUTION]
            for k in range(cfg.MODEL.ROI_HEATMAP_HEAD.NUM_CONV):
                # Awkward -- have to synchronize these stride / pad / dilation
                # configs with the ones in heatmap_head.py
                res = conv_output_shape(
                    res,
                    kernel_size=cfg.MODEL.ROI_HEATMAP_HEAD.CONV_SIZES[k],
                    stride=1,
                    pad=1,
                    dilation=1)
            assert(res[0] == res[1])
            self.target_heatmap_size = res[0] 


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
            instances = annotations_to_instances(
                annos, image_shape, camera_pose,
                target_heatmap_size=self.target_heatmap_size)
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

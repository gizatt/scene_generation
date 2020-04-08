from copy import deepcopy
import logging
import numpy as np
from typing import Any, List, Sequence, Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F

from detectron2.layers import ShapeSpec
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.structures import ImageList
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

class ImageListWithDepthAndCalibration(ImageList):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    Extends ImageList by also storing a corresponding
    list of depth images, as well as a list of camera
    calibrations for each image.

    Attributes:
        image_sizes (list[tuple[int, int]]): each tuple is (h, w)
        calibrations (list[3x3 tensors]): each entry is a K matrix
    """

    def __init__(self, rgb_tensor: torch.Tensor,
                       depth_tensor: torch.Tensor,
                       calibrations: List[torch.Tensor],
                       image_sizes: List[Tuple[int, int]]):
        """
        Arguments:
            tensor (Tensor): of shape (N, H, W) or (N, C_1, ..., C_K, H, W) where K >= 1
            image_sizes (list[tuple[int, int]]): Each tuple is (h, w). It can
                be smaller than (H, W) due to padding.
        """
        self.tensor = torch.cat([rgb_tensor, depth_tensor], dim=-3)
        self.rgb_tensor = rgb_tensor
        self.depth_tensor = depth_tensor
        self.calibrations = calibrations
        self.image_sizes = image_sizes

    @staticmethod
    def from_tensors(
        tensors: Sequence[torch.Tensor],
        depth_tensors: Sequence[torch.Tensor],
        calibrations: Sequence[torch.Tensor],
        size_divisibility: int = 0,
        pad_value: float = 0.0
    ) -> "ImageList":
        """
        Args:
            tensors: a tuple or list of `torch.Tensors`, each of shape (Hi, Wi) or
                (C_1, ..., C_K, Hi, Wi) where K >= 1. The Tensors will be padded
                to the same shape with `pad_value`.
            depth_tensors: a tuple or list of `torch.Tensors`, each of shape (Hi, Wi) or
                (C_1, ..., C_K, Hi, Wi) where K >= 1. The Tensors will be padded the
                same way as the corresponding tensor in the tensors list.
            size_divisibility (int): If `size_divisibility > 0`, add padding to ensure
                the common height and width is divisible by `size_divisibility`.
                This depends on the model and many models need a divisibility of 32.
            pad_value (float): value to pad
        Returns:
            an `ImageList`.
        """
        assert len(tensors) > 0
        assert len(tensors) == len(depth_tensors)
        assert len(tensors) == len(calibrations)
        assert isinstance(tensors, (tuple, list))
        for t, d in zip(tensors, depth_tensors):
            assert isinstance(t, torch.Tensor), type(t)
            assert isinstance(d, torch.Tensor), type(d)
            assert t.shape[1:-2] == tensors[0].shape[1:-2], t.shape
            assert d.shape[1:-2] == depth_tensors[0].shape[1:-2], d.shape
        # per dimension maximum (H, W) or (C_1, ..., C_K, H, W) where K >= 1 among all tensors
        max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))

        if size_divisibility > 0:
            import math

            stride = size_divisibility
            max_size = list(max_size)  # type: ignore
            max_size[-2] = int(math.ceil(max_size[-2] / stride) * stride)  # type: ignore
            max_size[-1] = int(math.ceil(max_size[-1] / stride) * stride)  # type: ignore
            max_size = tuple(max_size)

        depth_max_size = list(max_size)
        depth_max_size[-3] = 1 # depth should be single-channel
        depth_max_size = tuple(depth_max_size)

        image_sizes = [tuple(im.shape[-2:]) for im in tensors]
        depth_image_sizes = [tuple(im.shape[-2:]) for im in depth_tensors]
        assert(image_sizes == depth_image_sizes)

        if len(tensors) == 1:
            # This seems slightly (2%) faster.
            # TODO: check whether it's faster for multiple images as well
            image_size = image_sizes[0]
            padding_size = [0, max_size[-1] - image_size[1], 0, max_size[-2] - image_size[0]]
            if all(x == 0 for x in padding_size):  # https://github.com/pytorch/pytorch/issues/31734
                batched_imgs = tensors[0].unsqueeze(0)
                batched_depth_imgs = depth_tensors[0].unsqueeze(0)
            else:
                padded = F.pad(tensors[0], padding_size, value=pad_value)
                padded_depth = F.pad(depth_tensors[0], padding_size, value=pad_value)
                batched_imgs = padded.unsqueeze_(0)
                batched_depth_imgs = padded_depth.unsqueeze_(0)
        else:
            batch_shape = (len(tensors),) + max_size
            batched_imgs = tensors[0].new_full(batch_shape, pad_value)
            depth_batch_shape = (len(tensors),) + depth_max_size
            batched_depth_imgs = depth_tensors[0].new_full(depth_batch_shape, pad_value)
            for img, pad_img in zip(tensors, batched_imgs):
                pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)
            for depth_img, pad_depth_img in zip(depth_tensors, batched_depth_imgs):
                pad_depth_img[..., : depth_img.shape[-2], : depth_img.shape[-1]].copy_(depth_img)

        return ImageListWithDepthAndCalibration(
                    batched_imgs.contiguous(),
                    batched_depth_imgs.contiguous(),
                    calibrations, image_sizes)


@META_ARCH_REGISTRY.register()
class GeneralizedRCNNWithDepthAndCalibration(GeneralizedRCNN):
    """
     Only difference from GeneralizedRCNN is that it packs the
     depth and calibration info into a specialized ImageList during
     image preprocessing.
    """
    def __init__(self, cfg):
        nn.Module.__init__(self)

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg, input_shape=ShapeSpec(channels=4))
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())
        self.vis_period = cfg.VIS_PERIOD
        self.input_format = cfg.INPUT.FORMAT

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.depth_normalizer = lambda x: (x - cfg.MODEL.DEPTH_PIXEL_MEAN) / cfg.MODEL.DEPTH_PIXEL_STD
        self.to(self.device)

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images, and pack in depth
        and calibration info.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        depth_images = [x["depth_image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        depth_images = [self.depth_normalizer(x) for x in depth_images]
        calibrations = [x["K"].to(self.device) for x in batched_inputs]
        images = ImageListWithDepthAndCalibration.from_tensors(
            images, depth_images, calibrations, self.backbone.size_divisibility)
        return images
import fvcore.nn.weight_init as weight_init
import numpy as np
import torch
from detectron2.layers import Conv2d, ConvTranspose2d, cat, get_norm
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from torch import nn
from torch.nn import functional as F
from scene_generation.utils.torch_misc_math import (
    conv_output_shape
)

ROI_HEATMAP_SHAPE_REGISTRY = Registry("ROI_HEATMAP_HEAD")

@ROI_HEATMAP_SHAPE_REGISTRY.register()
class RCNNHeatmapHead(nn.Module):
    """
    Takes an ROI an spits out, per pixel of the pooled ROI,
        and per type of keypoint, a heatmap of whether a
        key point is at that location.

    Scored by pre-generating target heatmaps
    my taking the max over narrow gaussian peaks for each gt
    keypoint type.


    Architecture: series of convs.
    """

    def __init__(self, cfg, input_shape):
        super().__init__()

        # fmt: off
        num_conv         = cfg.MODEL.ROI_HEATMAP_HEAD.NUM_CONV
        conv_channels    = cfg.MODEL.ROI_HEATMAP_HEAD.CONV_CHANNELS
        conv_sizes       = cfg.MODEL.ROI_HEATMAP_HEAD.CONV_SIZES
        norm             = cfg.MODEL.ROI_HEATMAP_HEAD.NORM

        # fmt: on
        assert num_conv > 0

        self._output_size = [input_shape.channels, input_shape.height, input_shape.width]

        self.conv_norm_relus = []
        for k in range(num_conv):
            conv = Conv2d(
                self._output_size[0],
                conv_channels[k],
                kernel_size=conv_sizes[k],
                padding=1,
                bias=not norm,
                norm=get_norm(norm, conv_channels[k]),
                activation=F.relu,
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size[0] = conv_channels[k]
            self._output_size[1:] = conv_output_shape(
                self._output_size[1:],
                kernel_size=conv_sizes[k],
                stride=1,
                pad=1,
                dilation=1)
        print("Heatmap head has final output shape ", self._output_size)

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)

    def forward(self, x):
        # Pass through the conv layers
        for layer in self.conv_norm_relus:
            x = layer(x)
        return x

    def heatmap_loss(self, heatmap_estimate,
                     instances, loss_weight=1.0):
        """
        Compute the heatmap prediction loss.
        Args:
            heatmap_estimate (Tensor): A tensor of shape (B, W*, H*) for batch size B
                and predicted heatmap shape W*, H*.
            instances (list[Instances]): A list of N Instances, where N is the number of images
                in the batch. These instances are in 1:1
                correspondence with the heatmap_estimate. The ground-truth labels (class, box, mask,
                ...) associated with each instance are stored in fields.
            loss_weight (float): A float to multiply the loss with.
        Returns:
            heatmap_loss (Tensor): A scalar tensor containing the loss.
        """
        batch_size = heatmap_estimate.size(0)

        # Build GT heatmaps for each image.
        all_gt_heatmaps = []
        for instances_per_image in instances:
            if len(instances_per_image) == 0:
                continue
            all_gt_heatmaps.append(instances_per_image.gt_heatmaps.to(device=heatmap_estimate.device))

        if len(all_gt_heatmaps) == 0:
            return 0.

        gt_heatmaps = cat(all_gt_heatmaps, dim=0)
        assert gt_heatmaps.numel() > 0, gt_heatmaps.shape

        heatmap_loss = F.mse_loss(
            heatmap_estimate, gt_heatmaps, reduction='mean')
        return heatmap_loss * loss_weight

def build_heatmap_head(cfg, input_shape):
    name = cfg.MODEL.ROI_HEATMAP_HEAD.NAME
    return ROI_HEATMAP_SHAPE_REGISTRY.get(name)(cfg, input_shape)
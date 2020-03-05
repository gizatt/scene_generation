import fvcore.nn.weight_init as weight_init
import numpy as np
import torch
from detectron2.layers import Conv2d, ConvTranspose2d, cat, get_norm
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from torch import nn
from torch.nn import functional as F


ROI_SHAPE_HEAD_REGISTRY = Registry("ROI_SHAPE_HEAD")


def shape_rcnn_loss(shape_estimate, instances, loss_weight=1.0):
    """
    Compute the voxel prediction loss defined in the Mesh R-CNN paper.
    Args:
        shape_estimate (Tensor): A tensor of shape (B, D) for batch size B
            and # of shape parameters D.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the shape_estimate. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
        loss_weight (float): A float to multiply the loss with.
    Returns:
        shape_loss (Tensor): A scalar tensor containing the loss.
    """
    total_num_shape_estimates = shape_estimate.size(0)
    num_shape_params = shape_estimate.size(1)
    
    # Gather up shape params from the list of Instances objects
    all_gt_shape_params = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        all_gt_shape_params.append(instances_per_image.gt_shape_params.to(device=shape_estimate.device))

    if len(all_gt_shape_params) == 0:
        return 0.

    gt_shape_params = cat(all_gt_shape_params, dim=0)
    assert gt_shape_params.numel() > 0, gt_shape_params.shape

    shape_loss = F.mse_loss(
        shape_estimate, gt_shape_params, reduction="mean"
    )
    shape_loss = shape_loss * loss_weight
    return shape_loss


@ROI_SHAPE_HEAD_REGISTRY.register()
class RCNNShapeHead(nn.Module):
    """
    Takes an ROI an spits out estimates of the object shape parameters.
    """

    def __init__(self, cfg, input_shape):
        super().__init__()

        # fmt: off
        num_conv         = cfg.MODEL.ROI_SHAPE_HEAD.NUM_CONV
        conv_dim         = cfg.MODEL.ROI_SHAPE_HEAD.CONV_DIM
        num_fc           = cfg.MODEL.ROI_SHAPE_HEAD.NUM_FC
        fc_dim           = cfg.MODEL.ROI_SHAPE_HEAD.FC_DIM
        norm             = cfg.MODEL.ROI_SHAPE_HEAD.NORM
        num_shape_params = cfg.MODEL.ROI_SHAPE_HEAD.NUM_SHAPE_PARAMS

        # fmt: on
        assert num_conv + num_fc > 0

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)

        self.conv_norm_relus = []
        for k in range(num_conv):
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not norm,
                norm=get_norm(norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

        self.fcs = []
        for k in range(num_fc):
            if k < num_fc - 1:
                fc = nn.Linear(np.prod(self._output_size), fc_dim)
                self._output_size = fc_dim
            else:
                fc = nn.Linear(np.prod(self._output_size), num_shape_params)
                self._output_size = num_shape_params
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        for layer in self.conv_norm_relus:
            x = layer(x)
        if len(self.fcs):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            for layer in self.fcs:
                x = F.relu(layer(x))
        return x

    @property
    def output_size(self):
        return self._output_size

def build_shape_head(cfg, input_shape):
    name = cfg.MODEL.ROI_SHAPE_HEAD.NAME
    return ROI_SHAPE_HEAD_REGISTRY.get(name)(cfg, input_shape)
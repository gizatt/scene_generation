import fvcore.nn.weight_init as weight_init
import numpy as np
import torch
from detectron2.layers import Conv2d, ConvTranspose2d, cat, get_norm
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from torch import nn
from torch.nn import functional as F


ROI_SHAPE_HEAD_REGISTRY = Registry("ROI_SHAPE_HEAD")

def batched_index_select(input, dim, index):
    # https://discuss.pytorch.org/t/batched-index-select/9115/7
    views = [1 if i != dim else -1 for i in range(len(input.shape))]
    expanse = list(input.shape)
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index)

@ROI_SHAPE_HEAD_REGISTRY.register()
class RCNNShapeHead(nn.Module):
    """
    Takes an ROI an spits out estimates of the object shape parameters.

    Operates by applying a number of convolutional + FC layers with
    a final soft classification output over a discretization of the
    shape parameters.
    """

    def __init__(self, cfg, input_shape):
        super().__init__()

        # fmt: off
        num_conv         = cfg.MODEL.ROI_SHAPE_HEAD.NUM_CONV
        conv_dim         = cfg.MODEL.ROI_SHAPE_HEAD.CONV_DIM
        num_fc           = cfg.MODEL.ROI_SHAPE_HEAD.NUM_FC
        fc_dim           = cfg.MODEL.ROI_SHAPE_HEAD.FC_DIM
        norm             = cfg.MODEL.ROI_SHAPE_HEAD.NORM
        self.num_shape_params = cfg.MODEL.ROI_SHAPE_HEAD.NUM_SHAPE_PARAMS
        self.num_shape_bins   = cfg.MODEL.ROI_SHAPE_HEAD.NUM_SHAPE_BINS
        self.shape_bin_ranges = cfg.MODEL.ROI_SHAPE_HEAD.SHAPE_BIN_RANGES

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
                fc = nn.Linear(np.prod(self._output_size), self.num_shape_params * self.num_shape_bins)
                self._output_size = self.num_shape_params * self.num_shape_bins
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        # Temperature for softmax (gets exponentiated in the
        # forward pass to ensure it's always positive).
        self.log_T = torch.nn.Parameter(torch.log(torch.tensor([0.5])))
        self.log_T.requires_grad = True

        # Pre-compute shape bin centers
        shape_bin_corners = []
        shape_bin_widths = []
        for k in range(self.num_shape_params):
            bottom, top = self.shape_bin_ranges[k]
            shape_bin_widths.append( (top - bottom) / (self.num_shape_bins - 1) )
            shape_bin_corners.append(
                torch.linspace(bottom, top, steps=self.num_shape_bins))
        self.register_buffer("shape_bin_corners", torch.stack(shape_bin_corners))
        self.register_buffer("shape_bin_widths", torch.tensor(shape_bin_widths))

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        # Pass through the conv and FC layers to get the bins
        for layer in self.conv_norm_relus:
            x = layer(x)
        if len(self.fcs):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            for layer in self.fcs:
                x = F.relu(layer(x))

        x = x.reshape(x.shape[0], self.num_shape_params, self.num_shape_bins)
        P = F.softmax(x / torch.exp(self.log_T), dim=-1)
        shape_estimate = torch.sum(P * self.shape_bin_corners, dim=2)
        return shape_estimate, P

    def shape_rcnn_loss(self, shape_estimate, P, instances,
                        loss_weight=1.0, loss_type="l1"):
        """
        Compute the shape prediction loss.
        Args:
            shape_estimate (Tensor): A tensor of shape (B, D) for batch size B
                and # of shape parameters D.
            P (Tensor): A tensor of shape (B, D, N_bins) for batch size B,
                # of shape params D, and # of shape bins N_bins.
            instances (list[Instances]): A list of N Instances, where N is the number of images
                in the batch. These instances are in 1:1
                correspondence with the shape_estimate. The ground-truth labels (class, box, mask,
                ...) associated with each instance are stored in fields.
            loss_weight (float): A float to multiply the loss with.
        Returns:
            shape_loss (Tensor): A scalar tensor containing the loss.
        """
        total_num_shape_estimates = shape_estimate.size(0)
        assert(shape_estimate.size(1) == self.num_shape_params)
        assert(P.size(0) == total_num_shape_estimates)
        assert(P.size(1) == self.num_shape_params)
        assert(P.size(2) == self.num_shape_bins)
        
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

        # Compute the bin index in which the ground truth shapes fall
        # by subtracting off the bin left boundaries and dividing by the bin widths
        distance_into_bins = gt_shape_params.detach() - self.shape_bin_corners[:, 0]
        bin_indices = (distance_into_bins / self.shape_bin_widths).floor()
        bin_indices = torch.clamp(bin_indices, 0, self.num_shape_bins).long()

        active_probs = torch.stack(
            [P[k, range(3), bin_indices[k, :]]
             for k in range(total_num_shape_estimates)])
        shape_loss = torch.mean(-torch.log(active_probs))

        if loss_type == "l1":
            shape_loss = shape_loss + F.l1_loss(
                shape_estimate, gt_shape_params, reduction="mean"
            )
        elif loss_type == "l2":
            shape_loss = shape_loss + F.mse_loss(
                shape_estimate, gt_shape_params, reduction="mean"
            )
        else:
            raise NotImplementedError("Unrecognized loss type: ", loss_type)

        shape_loss = shape_loss * loss_weight
        return shape_loss

def build_shape_head(cfg, input_shape):
    name = cfg.MODEL.ROI_SHAPE_HEAD.NAME
    return ROI_SHAPE_HEAD_REGISTRY.get(name)(cfg, input_shape)
import fvcore.nn.weight_init as weight_init
import numpy as np
import torch
from detectron2.layers import Conv2d, ConvTranspose2d, cat, get_norm
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from torch import nn
from torch.nn import functional as F

from scene_generation.utils.torch_quaternion import (
    quat2mat
)

ROI_POSE_HEAD_REGISTRY = Registry("ROI_POSE_HEAD")


@ROI_POSE_HEAD_REGISTRY.register()
class RCNNPoseXyzHead(nn.Module):
    """
    Takes an ROI an spits out estimates of the object XYZ pose.

    Operates by applying a number of convolutional + FC layers with
    a final soft classification output over a discretization of the
    pose xyz components.

    Layout is:
        
        conv layers --> FC layers --> N pose estimate bins --> final regression
                                                |                   |
                                                v                   v
                                              cross-entropy       L1 loss
                                                 loss

    """

    def __init__(self, cfg, input_shape):
        super().__init__()

        # fmt: off
        num_conv         = cfg.MODEL.ROI_POSE_XYZ_HEAD.NUM_CONV
        conv_dim         = cfg.MODEL.ROI_POSE_XYZ_HEAD.CONV_DIM
        num_fc           = cfg.MODEL.ROI_POSE_XYZ_HEAD.NUM_FC
        fc_dim           = cfg.MODEL.ROI_POSE_XYZ_HEAD.FC_DIM
        norm             = cfg.MODEL.ROI_POSE_XYZ_HEAD.NORM

        self.num_bins       = cfg.MODEL.ROI_POSE_XYZ_HEAD.NUM_BINS
        self.xyz_bin_ranges = cfg.MODEL.ROI_POSE_XYZ_HEAD.XYZ_BIN_RANGES


        # fmt: on
        assert num_conv + num_fc > 0

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)

        num_output_params = self.num_bins * 3

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
            if k == 0:   
                # Takes 3x3 calibrations, rotations, and Hinfs as input as well
                fc = nn.Linear(np.prod(self._output_size) + 27, fc_dim)
                self._output_size = fc_dim
            elif k < num_fc - 1:
                fc = nn.Linear(np.prod(self._output_size), fc_dim)
                self._output_size = fc_dim
            else:
                fc = nn.Linear(np.prod(self._output_size), num_output_params)
                self._output_size = num_output_params
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        # Temperature for softmax (gets exponentiated in the
        # forward pass to ensure it's always positive).
        self.log_T = torch.nn.Parameter(torch.log(torch.tensor([0.5])))
        self.log_T.requires_grad = True

        # Pre-compute xyz bin centers
        xyz_bin_corners = []
        xyz_bin_widths = []
        for k in range(3):
            bottom, top = self.xyz_bin_ranges[k]
            xyz_bin_widths.append( (top - bottom) / (self.num_bins - 1) )
            xyz_bin_corners.append(
                torch.linspace(bottom, top, steps=self.num_bins))
        self.register_buffer("xyz_bin_corners", torch.stack(xyz_bin_corners))
        self.register_buffer("xyz_bin_widths", torch.tensor(xyz_bin_widths))

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x,  Kcs, rotations, Hinfs):
        for layer in self.conv_norm_relus:
            x = layer(x)
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        Kcs = torch.flatten(Kcs, start_dim=1)
        rotations = torch.flatten(rotations, start_dim=1)
        Hinfs = torch.flatten(Hinfs, start_dim=1)
        x = torch.cat([x, Kcs, rotations, Hinfs], dim=-1)
        if len(self.fcs):
            for layer in self.fcs:
                x = F.relu(layer(x))
        x = x.reshape(x.shape[0], 3, self.num_bins)
        log_P = F.log_softmax(x / torch.exp(self.log_T), dim=-1)
        xyz_estimate = torch.sum(torch.exp(log_P) * self.xyz_bin_corners, dim=2)

        if self.training:
            get_event_storage().put_scalar("pose_xyz_log_T", self.log_T)

        return xyz_estimate, log_P

    def pose_xyz_rcnn_loss(self, pose_xyz_estimate, log_P,
                           instances, loss_weight=1.0, loss_type="l1"):
        """
        Compute the error between the estimated and actual pose.
        Args:
            pose_xyz_estimate (Tensor): A tensor of shape (B, 3) for batch size B.
            log_P (Tensor): A tensor of shape (B, 3, N_bins) for batch size B,
                and # of xyz bins N_bins.
            instances (list[Instances]): A list of N Instances, where N is the number of images
                in the batch. These instances are in 1:1
                correspondence with the pose estimates. The ground-truth labels (class, box, mask,
                ...) associated with each instance are stored in fields.
            loss_weight (float): A float to multiply the loss with.
            loss_type (string): 'l1' or 'l2'
        Returns:
            xyz_pose_loss (Tensor): A scalar tensor containing the loss.
        """
        total_num_pose_estimates = pose_xyz_estimate.size(0)
        assert(pose_xyz_estimate.size(1) == 3)
        assert(log_P.size(0) == total_num_pose_estimates)
        assert(log_P.size(1) == 3)
        assert(log_P.size(2) == self.num_bins)

        # Gather up gt xyz poses from the list of Instances objects
        all_gt_pose_xyz = []
        for instances_per_image in instances:
            if len(instances_per_image) == 0:
                continue
            all_gt_pose_xyz.append(instances_per_image.gt_pose_quatxyz[:, -3:].to(device=pose_xyz_estimate.device))

        if len(all_gt_pose_xyz) == 0:
            return 0.

        all_gt_pose_xyz = cat(all_gt_pose_xyz, dim=0)
        assert all_gt_pose_xyz.numel() > 0, all_gt_pose_xyz.shape

        # Compute the bin index in which the ground truth xyz poses fall
        # by subtracting off the bin left boundaries and dividing by the bin widths
        distance_into_bins = all_gt_pose_xyz.detach() - self.xyz_bin_corners[:, 0]
        bin_indices = (distance_into_bins / self.xyz_bin_widths).floor()
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins).long()

        active_log_probs = torch.stack(
            [log_P[k, range(3), bin_indices[k, :]]
             for k in range(total_num_pose_estimates)])
        pose_loss = torch.mean(-active_log_probs)

        if loss_type == "l1":
            pose_loss = pose_loss + F.l1_loss(
                pose_xyz_estimate, all_gt_pose_xyz, reduction="mean"
            )
        elif loss_type == "l2":
            pose_loss = pose_loss + F.mse_loss(
                pose_xyz_estimate, all_gt_pose_xyz, reduction="mean"
            )
        else:
            raise NotImplementedError("Unrecognized loss type: ", loss_type)

        pose_loss = pose_loss * loss_weight
        return pose_loss


@ROI_POSE_HEAD_REGISTRY.register()
class RCNNPoseRpyHead(nn.Module):
    """
    Takes an ROI an spits out estimates of the object pose RPY components.

    Operates by applying a number of convolutional + FC layers with
    a final soft classification output over a discretization of the
    pose rpy components.

    Layout is:
        
        conv layers --> FC layers --> N pose estimate bins --> final regression
                                                |                   |
                                                v                   v
                                              cross-entropy       L1 loss
                                                 loss

    RPY is treated differently than XYZ, as in the 3dRCNN paper (Kundu et al):
    XYZ is discretized over some range with loss taken directly, whereas RPY is
    discretized over the  entire range [0, 2pi] with a complex loss that wraps
    around from 2pi to 0.
    """

    def __init__(self, cfg, input_shape):
        super().__init__()

        # fmt: off
        num_conv         = cfg.MODEL.ROI_POSE_RPY_HEAD.NUM_CONV
        conv_dim         = cfg.MODEL.ROI_POSE_RPY_HEAD.CONV_DIM
        num_fc           = cfg.MODEL.ROI_POSE_RPY_HEAD.NUM_FC
        fc_dim           = cfg.MODEL.ROI_POSE_RPY_HEAD.FC_DIM
        norm             = cfg.MODEL.ROI_POSE_RPY_HEAD.NORM

        self.num_bins       = cfg.MODEL.ROI_POSE_RPY_HEAD.NUM_BINS

        # fmt: on
        assert num_conv + num_fc > 0

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)

        num_output_params = self.num_bins * 3

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
            if k == 0:   
                # Takes 3x3 calibrations + Hinfs + rotmats as input as well
                fc = nn.Linear(np.prod(self._output_size) + 27, fc_dim)
                self._output_size = fc_dim
            elif k < num_fc - 1:
                fc = nn.Linear(np.prod(self._output_size), fc_dim)
                self._output_size = fc_dim
            else:
                fc = nn.Linear(np.prod(self._output_size), num_output_params)
                self._output_size = num_output_params
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        # Temperature for softmax (gets exponentiated in the
        # forward pass to ensure it's always positive).
        self.log_T = torch.nn.Parameter(torch.log(torch.tensor([0.5])))
        self.log_T.requires_grad = True

        # Pre-compute rpy bin centers -- to make computing the complex
        # expectation easier, prepare the real + imaginary component of
        # the complex exponential of each bin center.
        rpy_bin_corners = []
        rpy_bin_widths = []
        for k in range(3):
            bottom, top = -np.pi, np.pi
            rpy_bin_widths.append( (top - bottom) / (self.num_bins - 1) )
            rpy_bin_corners.append(
                torch.linspace(bottom, top, steps=self.num_bins))
        rpy_bin_corners = torch.stack(rpy_bin_corners)
        self.register_buffer("rpy_bin_corners",
                             rpy_bin_corners)
        self.register_buffer("rpy_bin_corners_real",
                             torch.cos(rpy_bin_corners))
        self.register_buffer("rpy_bin_corners_imag",
                             torch.sin(rpy_bin_corners))
        self.register_buffer("rpy_bin_widths", torch.tensor(rpy_bin_widths))

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x, Kcs, rotations, Hinfs):
        for layer in self.conv_norm_relus:
            x = layer(x)
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        Kcs = torch.flatten(Kcs, start_dim=1)
        rotations = torch.flatten(rotations, start_dim=1)
        Hinfs = torch.flatten(Hinfs, start_dim=1)
        x = torch.cat([x, Kcs, rotations, Hinfs], dim=-1)
        if len(self.fcs):
            for layer in self.fcs:
                x = F.relu(layer(x))
        x = x.reshape(x.shape[0], 3, self.num_bins)
        log_P = F.log_softmax(x / torch.exp(self.log_T), dim=-1)
        # To get the estimate, take the *complex* expectation -- see
        # eq. (2) in the 3dRCNN paper.
        P = torch.exp(log_P)
        real_total = torch.sum(P * self.rpy_bin_corners_real, dim=2)
        imag_total = torch.sum(P * self.rpy_bin_corners_imag, dim=2)
        rpy_estimate = torch.atan2(imag_total, real_total)

        if self.training:
            get_event_storage().put_scalar("pose_rpy_log_T", self.log_T)


        return rpy_estimate, log_P

    def pose_rpy_rcnn_loss(self, pose_rpy_estimate, log_P,
                           instances, loss_weight=1.0, loss_type="l1"):
        """
        Compute the error between the estimated and actual pose.
        Args:
            pose_rpy_estimate (Tensor): A tensor of shape (B, 3) for batch size B.
            P (Tensor): A tensor of shape (B, 3, N_bins) for batch size B,
                and # of rpy bins N_bins.
            instances (list[Instances]): A list of N Instances, where N is the number of images
                in the batch. These instances are in 1:1
                correspondence with the pose estimates. The ground-truth labels (class, box, mask,
                ...) associated with each instance are stored in fields.
            loss_weight (float): A float to multiply the loss with.
            loss_type (string): 'l1' or 'l2'
        Returns:
            rpy_pose_loss (Tensor): A scalar tensor containing the loss.
        """
        total_num_pose_estimates = pose_rpy_estimate.size(0)
        assert(pose_rpy_estimate.size(1) == 3)
        assert(log_P.size(0) == total_num_pose_estimates)
        assert(log_P.size(1) == 3)
        assert(log_P.size(2) == self.num_bins)

        # Gather up gt rpy poses from the list of Instances objects
        all_gt_pose_rpy = []
        for instances_per_image in instances:
            if len(instances_per_image) == 0:
                continue
            all_gt_pose_rpy.append(instances_per_image.gt_pose_rpy.to(device=pose_rpy_estimate.device))

        if len(all_gt_pose_rpy) == 0:
            return 0.

        all_gt_pose_rpy = cat(all_gt_pose_rpy, dim=0)
        assert all_gt_pose_rpy.numel() > 0, all_gt_pose_rpy.shape

        # Compute the bin index in which the ground truth rpy poses fall
        # by subtracting off the bin left boundaries and dividing by the bin widths
        distance_into_bins = all_gt_pose_rpy.detach() - self.rpy_bin_corners[:, 0]
        bin_indices = (distance_into_bins / self.rpy_bin_widths).floor()
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins).long()

        active_log_probs = torch.stack(
            [log_P[k, range(3), bin_indices[k, :]]
             for k in range(total_num_pose_estimates)])
        pose_loss = torch.mean(-active_log_probs)

        # In either loss case, collapse among the minimum (elementwise) loss among
        # the original angle estimate, as well as the angle estimate rotated left
        # and right by 2pi.
        pose_loss_0 = torch.abs(pose_rpy_estimate - all_gt_pose_rpy)
        pose_loss_1 = torch.abs(pose_rpy_estimate + np.pi*2. - all_gt_pose_rpy)
        pose_loss_2 = torch.abs(pose_rpy_estimate - np.pi*2. - all_gt_pose_rpy)
        pose_loss_min, _ = torch.min(torch.stack([pose_loss_0, pose_loss_1, pose_loss_2], dim=0), dim=0)
        if loss_type == "l1":
            pose_loss = pose_loss + torch.mean(pose_loss_min)
        elif loss_type == "l2":
            pose_loss = pose_loss + torch.mean(pose_loss_min**2.)
        else:
            raise NotImplementedError("Unrecognized loss type: ", loss_type)

        pose_loss = pose_loss * loss_weight
        return pose_loss


@ROI_POSE_HEAD_REGISTRY.register()
class RCNNPose6DOFRotHead(nn.Module):
    """
    Takes an ROI an spits out estimates of the object pose rotation estimate
    (in rotation matrix form).

    Operates by applying a number of convolutional + FC layers with
    that spit out a 6DOF overparameterized rotation format (following
    https://arxiv.org/pdf/1812.07035.pdf).

    Layout is:
        
        conv layers --> FC layers --> 6DOF representation -> postprocess into rotmat
    """

    def __init__(self, cfg, input_shape):
        super().__init__()

        # fmt: off
        num_conv         = cfg.MODEL.ROI_POSE_6DOF_ROT_HEAD.NUM_CONV
        conv_dim         = cfg.MODEL.ROI_POSE_6DOF_ROT_HEAD.CONV_DIM
        num_fc           = cfg.MODEL.ROI_POSE_6DOF_ROT_HEAD.NUM_FC
        fc_dim           = cfg.MODEL.ROI_POSE_6DOF_ROT_HEAD.FC_DIM
        norm             = cfg.MODEL.ROI_POSE_6DOF_ROT_HEAD.NORM

        # fmt: on
        assert num_conv + num_fc > 0

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)

        num_output_params = 6

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
            if k == 0:   
                # Takes 3x3 calibrations + Hinfs + rotmats as input as well
                fc = nn.Linear(np.prod(self._output_size) + 27, fc_dim)
                self._output_size = fc_dim
            elif k < num_fc - 1:
                fc = nn.Linear(np.prod(self._output_size), fc_dim)
                self._output_size = fc_dim
            else:
                fc = nn.Linear(np.prod(self._output_size), num_output_params)
                self._output_size = num_output_params
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x, Kcs, rotations, Hinfs):
        for layer in self.conv_norm_relus:
            x = layer(x)
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        Kcs = torch.flatten(Kcs, start_dim=1)
        rotations = torch.flatten(rotations, start_dim=1)
        Hinfs = torch.flatten(Hinfs, start_dim=1)
        x = torch.cat([x, Kcs, rotations, Hinfs], dim=-1)
        if len(self.fcs):
            for layer in self.fcs:
                x = F.relu(layer(x))
        x = x.reshape(x.shape[0], 2, 3)
        # We've predicted two 3-vectors -- normalize them
        # and form a rotation matrix from them referencing Appendix B
        # in https://arxiv.org/pdf/1812.07035.pdf
        a1 = x[:, 0, :]
        a2 = x[:, 1, :]
        b1 = F.normalize(a1, p=2, dim=1)
        # Sum is repeated out to [batch x 3] from [batch] so it
        # broadcast-multiplies with [batch x 3] b1 happily
        b2 = F.normalize(a2 - (torch.sum(b1*a2, dim=-1).view(-1, 1).expand(-1, 3)*b1), dim=1)
        b3 = torch.cross(b1, b2)
        R = torch.stack([b1, b2, b3], dim=-1)
        return R

    def pose_6DOF_rot_rcnn_loss(self, R_estimate,
                                instances, loss_weight=1.0, loss_type="l1"):
        """
        Compute the error between the estimated and actual pose.
        Args:
            R_estimate (Tensor): A tensor of shape (B, 3, 3) for batch size B.
            instances (list[Instances]): A list of N Instances, where N is the number of images
                in the batch. These instances are in 1:1
                correspondence with the pose estimates. The ground-truth labels (class, box, mask,
                ...) associated with each instance are stored in fields.
            loss_weight (float): A float to multiply the loss with.
            loss_type (string): 'l1' or 'l2'
        Returns:
            loss (Tensor): A scalar tensor containing the loss.
        """
        total_num_pose_estimates = R_estimate.size(0)
        assert(R_estimate.shape[-2:] == (3, 3))

        # Gather up gt rotation matrices from the list of Instances objects
        all_gt_pose_quat = []
        for instances_per_image in instances:
            if len(instances_per_image) == 0:
                continue
            all_gt_pose_quat.append(instances_per_image.gt_pose_quatxyz[:, :4].detach().to(device=R_estimate.device))

        if len(all_gt_pose_quat) == 0:
            return 0.

        all_gt_pose_quat = cat(all_gt_pose_quat, dim=0)
        assert all_gt_pose_quat.numel() > 0, all_gt_pose_quat.shape

        all_gt_pose_rotmat = quat2mat(all_gt_pose_quat)
        # Get rotation difference between predicted and target
        diff_rotations = torch.matmul(all_gt_pose_rotmat, torch.transpose(R_estimate, 1, 2)).contiguous()

        # Batch trace implementation from torch_quaternion.py        
        rotation_matrix_vec = diff_rotations.reshape(*diff_rotations.shape[:-2], 9)
        m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.chunk(
            rotation_matrix_vec, chunks=9, dim=-1)
        trace = m00 + m11 + m22
        angle_errors = torch.acos(torch.clamp((trace - 1.)/2., -0.9999, 0.9999))
        if loss_type == "l1":
            pose_loss = torch.mean(angle_errors)
        elif loss_type == "l2":
            pose_loss = torch.mean(angle_errors**2.)
        else:
            raise NotImplementedError("Unrecognized loss type: ", loss_type)
        pose_loss = pose_loss * loss_weight
        return pose_loss

def build_pose_xyz_head(cfg, input_shape):
    name = cfg.MODEL.ROI_POSE_XYZ_HEAD.NAME
    return ROI_POSE_HEAD_REGISTRY.get(name)(cfg, input_shape)


def build_pose_rpy_head(cfg, input_shape):
    name = cfg.MODEL.ROI_POSE_RPY_HEAD.NAME
    return ROI_POSE_HEAD_REGISTRY.get(name)(cfg, input_shape)

def build_pose_6DOF_rot_head(cfg, input_shape):
    name = cfg.MODEL.ROI_POSE_6DOF_ROT_HEAD.NAME
    return ROI_POSE_HEAD_REGISTRY.get(name)(cfg, input_shape)
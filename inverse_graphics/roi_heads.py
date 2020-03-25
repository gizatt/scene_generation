from typing import Dict
import torch
from detectron2.layers import ShapeSpec, cat
from detectron2.modeling import ROI_HEADS_REGISTRY
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs
from detectron2.modeling.roi_heads.roi_heads import StandardROIHeads, select_foreground_proposals

from scene_generation.inverse_graphics.shape_head import build_shape_head
from scene_generation.inverse_graphics.pose_head import build_pose_xyz_head, build_pose_rpy_head
from scene_generation.utils.torch_quaternion import euler_to_quaternion

'''
Implements shape and pose regression from a cropped instance proposal region.
Based, roughly, on https://github.com/facebookresearch/meshrcnn/blob/master/meshrcnn/modeling/roi_heads/roi_heads.py,
but implementing some architecture similar to 3d-RCNN (Kundu, Li, Rehg).
'''


@ROI_HEADS_REGISTRY.register()
class XenRCNNROIHeads(StandardROIHeads):
    """
    The ROI specific heads for my simplification of 3D-RCNN.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__(cfg, input_shape)
        assert(cfg.MODEL.ROI_HEADS.NUM_CLASSES == 1)
        self.with_mask = cfg.MODEL.MASK_ON
        self.with_shape = cfg.MODEL.SHAPE_ON
        self.with_pose = cfg.MODEL.POSE_ON
        self.shape_loss_weight = cfg.MODEL.ROI_HEADS.SHAPE_LOSS_WEIGHT
        self.shape_loss_norm = cfg.MODEL.ROI_HEADS.SHAPE_LOSS_NORM
        self.pose_loss_weight = cfg.MODEL.ROI_HEADS.POSE_LOSS_WEIGHT
        self.pose_loss_norm = cfg.MODEL.ROI_HEADS.POSE_LOSS_NORM
        self.shared_pooler_shape = self._init_pooler(cfg, input_shape)
        if self.with_shape:
            self.shape_head = build_shape_head(cfg, self.shared_pooler_shape)
        if self.with_pose:
            self.pose_xyz_head = build_pose_xyz_head(cfg, self.shared_pooler_shape)
            self.pose_rpy_head = build_pose_rpy_head(cfg, self.shared_pooler_shape)
        # If MODEL.VIS_MINIBATCH is True we store minibatch targets
        # for visualization purposes
        self._vis = None #cfg.MODEL.VIS_MINIBATCH
        self._misc = {}
        self._vis_dir = cfg.OUTPUT_DIR

    def _init_pooler(self, cfg, input_shape):
        # Shared pooler between the shape and pose heads.
        shared_pooler_resolution = cfg.MODEL.ROI_SHARED_HEAD.POOLER_RESOLUTION # Default 14x14
        shared_pooler_scales     = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        shared_sampling_ratio    = cfg.MODEL.ROI_SHARED_HEAD.POOLER_SAMPLING_RATIO
        shared_pooler_type       = cfg.MODEL.ROI_SHARED_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in self.in_features][0]

        self.shared_pooler = ROIPooler(
            output_size=shared_pooler_resolution,
            scales=shared_pooler_scales,
            sampling_ratio=shared_sampling_ratio,
            pooler_type=shared_pooler_type,
        )
        return ShapeSpec(
            channels=in_channels, width=shared_pooler_resolution, height=shared_pooler_resolution
        )

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        if self._vis:
            self._misc["images"] = images
        del images

        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self._vis:
            self._misc["proposals"] = proposals


        if self.training:
            losses = self._forward_box(features, proposals)

            if self.with_mask:
                losses.update(self._forward_mask(features, proposals))

            # During training the (fully labeled) proposals used by the box head are
            # used by the shape and pose heads.

            # Compute shared features + proposal boxes
            features = [features[f] for f in self.in_features]
            proposals, _ = select_foreground_proposals(proposals, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            shared_features = self.shared_pooler(features, proposal_boxes)
            if self.with_shape:
                losses.update(self._forward_shape(shared_features, proposals))
            if self.with_pose:
                losses.update(self._forward_pose(shared_features, proposals))

            # print minibatch examples
            if self._vis:
                raise NotImplementedError("vis")
                vis_utils.visualize_minibatch(self._misc["images"], self._misc, self._vis_dir, True)
            return [], losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(self, features, instances):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.
        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.
        Returns:
            instances (Instances): the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_voxels`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") # TODO(gizatt) and anything else?

        if self.with_mask:
            instances = self._forward_mask(features, instances)

        # Compute shared features.
        features = [features[f] for f in self.in_features]
        pred_boxes = [x.pred_boxes for x in instances]
        shared_features = self.shared_pooler(features, pred_boxes)

        if self.with_shape:
            instances = self._forward_shape(shared_features, instances)
        if self.with_pose:
            instances = self._forward_pose(shared_features, instances)
        return instances

    def _forward_pose(self, features, instances):
        """
        Forward logic for the shape estimation branch.
        Args:
            features (list[Tensor]): #level input features for xyz prediction
            instances (list[Instances]): the per-image instances to train/predict meshes.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.
        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new field "pred_pose" and return it.
        """
        if self.training:
            losses = {}
            pose_xyz_estimate, P_xyz = self.pose_xyz_head(features)
            pose_xyz = self.pose_xyz_head.pose_xyz_rcnn_loss(
                pose_xyz_estimate, P_xyz, instances,
                loss_weight=self.pose_loss_weight,
                loss_type=self.pose_loss_norm
            )
            losses.update({"pose_xyz": pose_xyz})

            pose_rpy_estimate, P_rpy = self.pose_rpy_head(features)
            pose_rpy = self.pose_rpy_head.pose_rpy_rcnn_loss(
                pose_rpy_estimate, P_rpy, instances,
                loss_weight=self.pose_loss_weight,
                loss_type=self.pose_loss_norm
            )
            losses.update({"pose_rpy": pose_rpy})

            return losses

        else:
            pose_xyz_estimates, _ = self.pose_xyz_head(features)
            pose_rpy_estimates, _ = self.pose_rpy_head(features)

            pose_estimates = torch.cat(
                [euler_to_quaternion(pose_rpy_estimates, order='zyx'),
                 pose_xyz_estimates], dim=1)
            num_instances_per_image = [len(i) for i in instances]
            pose_by_instance_group = pose_estimates.split(num_instances_per_image)
            for pose_estimate_k, instances_k in zip(pose_by_instance_group, instances):
                instances_k.pred_pose = pose_estimate_k
            return instances

    def _forward_shape(self, features, instances):
        """
        Forward logic for the shape estimation branch.
        Args:
            features (list[Tensor]): #level input features for shape prediction
            instances (list[Instances]): the per-image instances to train/predict meshes.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.
        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new field "pred_shape_params" and return it.
        """
        
        if self.training:
            losses = {}
            shape_estimate, P = self.shape_head(features)
            loss_shape = self.shape_head.shape_rcnn_loss(
                shape_estimate, P, instances,
                loss_weight=self.shape_loss_weight,
                loss_type=self.shape_loss_norm
            )
            losses.update({"loss_shape": loss_shape})
            return losses

        else:
            shape_estimate, P = self.shape_head(features)
            num_instances_per_image = [len(i) for i in instances]
            pred_shapes_by_instance_group = shape_estimate.split(num_instances_per_image)
            for shape_estimate_k, instances_k in zip(pred_shapes_by_instance_group, instances):
                instances_k.pred_shape_params = shape_estimate_k
            return instances

if __name__ == "__main__":
    # Test out the XenRCNN head and XenCOCO data loaders
    pass
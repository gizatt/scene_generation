from typing import Dict
import torch
from detectron2.layers import ShapeSpec, cat
from detectron2.modeling import ROI_HEADS_REGISTRY
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs
from detectron2.modeling.roi_heads.roi_heads import StandardROIHeads, select_foreground_proposals

from scene_generation.inverse_graphics.keypoint_mcmc.heatmap_head import build_heatmap_head

'''
Wraps the keypoint estimation head.
'''


@ROI_HEADS_REGISTRY.register()
class KeypointMCMCROIHeads(StandardROIHeads):
    """
    The ROI specific heads for this keypoint-mcmc experiment.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__(cfg, input_shape)
        assert(cfg.MODEL.ROI_HEADS.NUM_CLASSES == 1)
        self.with_mask = cfg.MODEL.MASK_ON
        self.heatmap_loss_weight = cfg.MODEL.ROI_HEADS.HEATMAP_LOSS_WEIGHT
        self.heatmap_pooler_shape = self._init_pooler(cfg, input_shape)
        self.heatmap_head = build_heatmap_head(cfg, self.heatmap_pooler_shape)

    def _init_pooler(self, cfg, input_shape):
        # Pooler for the heatmap head.
        heatmap_pooler_resolution = cfg.MODEL.ROI_HEATMAP_HEAD.POOLER_RESOLUTION # Default 14x14
        heatmap_pooler_scales     = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        heatmap_pooler_sampling_ratio    = cfg.MODEL.ROI_HEATMAP_HEAD.POOLER_SAMPLING_RATIO
        heatmap_pooler_type       = cfg.MODEL.ROI_HEATMAP_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in self.in_features][0]

        self.heatmap_pooler = ROIPooler(
            output_size=heatmap_pooler_resolution,
            scales=heatmap_pooler_scales,
            sampling_ratio=heatmap_pooler_sampling_ratio,
            pooler_type=heatmap_pooler_type,
        )
        return ShapeSpec(
            channels=in_channels, width=heatmap_pooler_resolution, height=heatmap_pooler_resolution
        )

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals)

            if self.with_mask:
                losses.update(self._forward_mask(features, proposals))

            # During training the (fully labeled) proposals used by the box head are
            # used by the heatmap estimation head.

            # Compute heatmap features + proposal boxes
            features = [features[f] for f in self.in_features]
            proposals, _ = select_foreground_proposals(proposals, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            heatmap_features = self.heatmap_pooler(features, proposal_boxes)
            losses.update(self._forward_heatmap(heatmap_features, proposals))
            return [], losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the heatmap heads are only
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
            calibrations (list[Tensor[3x3]]): K matrix of each image
        Returns:
            instances (Instances): the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_voxels`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") # TODO(gizatt) and anything else?

        if self.with_mask:
            instances = self._forward_mask(features, instances)

        # Compute heatmap features.
        features = [features[f] for f in self.in_features]
        pred_boxes = [x.pred_boxes for x in instances]
        heatmap_features = self.heatmap_pooler(features, pred_boxes)
        instances = self._forward_heatmap(heatmap_features, instances)
        return instances

    def _forward_heatmap(self, features, instances):
        """
        Forward logic for the keypoint heatmap estimation branch.
        Args:
            features (list[Tensor]): #level input features for heatmap estimation.
            instances (list[Instances]): the per-image instances to train/predict meshes.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.
        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new field "pred_keypoint_heatmap" and return it.
        """
        
        if self.training:
            losses = {}
            heatmap_estimate = self.heatmap_head(features)
            loss_heatmap = self.heatmap_head.heatmap_loss(
                heatmap_estimate, instances,
                loss_weight=self.heatmap_loss_weight
            )
            losses.update({"loss_keypoint_heatmap": loss_heatmap})
            return losses

        else:
            heatmap_estimate = self.heatmap_head(features)
            num_instances_per_image = [len(i) for i in instances]
            pred_heatmap_by_instance_group = heatmap_estimate.split(num_instances_per_image)
            for heatmap_estimate_k, instances_k in zip(
                    pred_heatmap_by_instance_group, instances):
                instances_k.pred_keypoint_heatmap = heatmap_estimate_k
            return instances

if __name__ == "__main__":
    # Test out the XenRCNN head and XenCOCO data loaders
    pass
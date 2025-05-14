import torch
import torch.nn as nn
import torch.functional as F
from transformers.models.mask2former.modeling_mask2former import Mask2FormerForUniversalSegmentation, Mask2FormerLoss
from torch import Tensor
from typing import Dict, List, Tuple
import numpy as np
from .loss import FocalLoss

    
class Mask2FormerFocalLoss(Mask2FormerLoss):
    def __init__(self, config, weight_dict, class_weights):
        super().__init__(config, weight_dict)
        self.class_weights = class_weights

    def loss_labels(
        self, class_queries_logits: Tensor, class_labels: List[Tensor], indices: Tuple[np.array]
    ) -> Dict[str, Tensor]:

        pred_logits = class_queries_logits
        batch_size, num_queries, _ = pred_logits.shape

        # criterion = nn.CrossEntropyLoss(weight=self.empty_weight)
        criterion = FocalLoss(weight=self.class_weights)

        idx = self._get_predictions_permutation_indices(indices)  # shape of (batch_size, num_queries)
        target_classes_o = torch.cat(
            [target[j] for target, (_, j) in zip(class_labels, indices)]
        )  # shape of (batch_size, num_queries)
        target_classes = torch.full(
            (batch_size, num_queries), fill_value=self.num_labels, dtype=torch.int64, device=pred_logits.device
        )
        target_classes[idx] = target_classes_o
        # Permute target_classes (batch_size, num_queries, num_labels) -> (batch_size, num_labels, num_queries)
        pred_logits_transposed = pred_logits.transpose(1, 2)
        loss_ce = criterion(pred_logits_transposed, target_classes)
        losses = {"loss_cross_entropy": loss_ce}
        return losses
    
class Mask2FormerFocal(Mask2FormerForUniversalSegmentation):
    def __init__(self, config, class_weights=None):
        # weight_dict 정의
        weight_dict = {"loss_cross_entropy": 2.0, "loss_mask": 5.0, "loss_dice": 5.0}
        if class_weights is None:
            class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0])

        super().__init__(config)
        self.criterion = Mask2FormerFocalLoss(config, weight_dict, class_weights)

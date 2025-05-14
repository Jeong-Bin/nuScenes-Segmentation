import torch
import torch.nn as nn
import torch.nn.functional as F
import types

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight.to(inputs.device), reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

def apply_focal_loss(model, class_weights, alpha=1.0, gamma=2.0):
    """Mask2Former의 Cross Entropy Loss를 Focal Loss로 교체하는 함수"""
    if hasattr(model, 'module'):
        model = model.module

    focal_loss_fn = FocalLoss(alpha=alpha, gamma=gamma, weight=class_weights).cuda()
    
    def new_loss_labels(self, src_logits, target_labels, indices):
        # 매칭 인덱스 계산 로직을 직접 구현
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        idx = (batch_idx, src_idx)
        
        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(target_labels, indices)])
        
        # src_logits에서 매칭된 예측값만 선택
        src_logits = src_logits[idx]
        
        # print(f"After matching - src_logits shape: {src_logits.shape}")
        # print(f"After matching - target_classes shape: {target_classes_o.shape}")
        
        loss_ce = focal_loss_fn(src_logits, target_classes_o)
        losses = {'loss_cross_entropy': loss_ce}
        
        return losses
    
    # loss_labels 함수 교체
    model.criterion.loss_labels = types.MethodType(new_loss_labels, model.criterion)
    
    return model
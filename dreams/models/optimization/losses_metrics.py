import torch
import torch.nn.functional as F
from torch import nn
import torchmetrics


class SmoothIoULoss(nn.Module):
    def __init__(self):
        super(SmoothIoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        intersection = (inputs * targets).sum(axis=-1)
        total = (inputs + targets).sum(axis=-1)
        union = total - intersection
        iou = (intersection + smooth) / (union + smooth)
        return 1 - iou.mean()


class CosSimLoss(nn.Module):
    def __init__(self):
        super(CosSimLoss, self).__init__()

    def forward(self, inputs, targets):
        return 1 - F.cosine_similarity(inputs, targets).mean()


# TODO: Lovasz loss

class FocalLoss(nn.Module):
    def __init__(self, gamma, alpha=None, binary=False, return_softmax_out=False):
        """
        https://arxiv.org/pdf/1708.02002v2.pdf
        :param alpha: A vector summing up to one for multi-class classification, a positive-class scalar from (0, 1)
                      range for binary classification.
        :param return_softmax_out: If True, the return value of forward method is `(loss, softmax probabilities)` instead
               of `loss`.
        """
        super(FocalLoss, self).__init__()
        self.binary = binary
        self.gamma = gamma
        assert gamma >= 0
        self.alpha = alpha
        if self.alpha is not None:
            if not self.binary:
                raise NotImplementedError('Alpha weighting is currently implemented only for binary classification.')
            assert (isinstance(alpha, float) and 0 < alpha < 1) or (isinstance(alpha, torch.Tensor) and alpha.sum() == 1)
        self.return_softmax_out = return_softmax_out

    def forward(self, inputs, targets):
        """
        :param inputs: Class logits of shape (..., num_classes).
        :param targets: One-hot class labels (..., num_classes).
        :return: Unreduced focal loss of shape (...).
        """

        if not self.binary:

            # Compute cross-entropy
            p = F.softmax(inputs, dim=-1)
            loss = F.nll_loss(p.log(), torch.argmax(targets, dim=-1), reduction='none')
            if self.gamma == 0:
                if self.return_softmax_out:
                    return loss, p
                return loss

            # Weight with focal loss terms
            p_t = (targets * p).sum(dim=-1)
            fl_term = (1 - p_t) ** self.gamma
            loss = fl_term * loss

            if self.return_softmax_out:
                return loss, p
            return loss

        else:
            weight = torch.ones(targets.shape, dtype=targets.dtype, device=targets.device)
            targets_mask = targets > 0

            if self.alpha is not None:
                weight[targets_mask] = self.alpha
                weight[~targets_mask] = 1 - self.alpha

            if self.gamma > 0:
                weight[targets_mask] *= (1 - inputs[targets_mask]) ** self.gamma
                weight[~targets_mask] *= inputs[~targets_mask] ** self.gamma

            return F.binary_cross_entropy(inputs, targets, weight=weight.detach())


class FingerprintMetrics(torchmetrics.MetricCollection):
    """TODO: threshold argument for binary metrics"""
    def __init__(self, prefix=None, device=None):
        super(FingerprintMetrics, self).__init__([
            torchmetrics.classification.BinaryJaccardIndex(),
            torchmetrics.classification.BinaryRecall(),
            torchmetrics.classification.BinaryPrecision(),
            torchmetrics.classification.BinaryAccuracy(),
            torchmetrics.classification.BinaryAUROC(),
            # torchmetrics.classification.BinaryAveragePrecision(),
            torchmetrics.CosineSimilarity(reduction='mean')
        ], prefix=prefix)
        self.to(device=device)

#
# class BCMetrics(torchmetrics.MetricCollection):
#     def __init__(self, prefix=None, device=None):
#         super(BCMetrics, self).__init__([
#             torchmetrics.classification.BinaryRecall(),
#             torchmetrics.classification.BinaryPrecision(),
#             torchmetrics.classification.BinaryAccuracy(),
#             torchmetrics.classification.BinaryF1Score(),
#             torchmetrics.classification.BinarySpecificity()
#         ], prefix=prefix)
#         self.to(device=device)

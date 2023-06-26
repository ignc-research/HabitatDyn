import torch
import torch.nn.functional as F
import torch.nn as nn


def iou(pred_mask, gt_mask):
  """Calculates the IoU of two masks.

  Args:
    pred_mask: A torch.Tensor of shape (batch_size, height, width).
    gt_mask: A torch.Tensor of shape (batch_size, height, width).

  Returns:
    A torch.Tensor of shape (batch_size,).
  """

  pred_mask = pred_mask.bool()
  gt_mask = gt_mask.bool()
  # Compute the intersection and union of the masks.
  intersection = torch.sum(pred_mask * gt_mask, dim=[1, 2])
  union = torch.sum(pred_mask, dim=[1, 2]) + torch.sum(gt_mask, dim=[1, 2])

  # a mask to filter out images that dose not have object
  mask = torch.sum(gt_mask, dim=[1, 2])
  intersection = torch.where(mask > 0, intersection, torch.zeros_like(intersection))
  union = torch.where(mask > 0, union, torch.ones_like(union))

  # Calculate the IoU.
  iou = intersection / union

  return iou[iou.nonzero()]


def prf_metrics(pred, target):
    """ calculate precision, recall and f1

    Args:
        pred 
        target 

    Returns:
        precision, recall, f1
    """
    assert pred.shape == target.shape
    pred = pred.bool()
    target = target.bool()

    # a mask to filter out images that dose not have object
    mask = torch.sum(target, dim=[1, 2])

    # True positive
    tp = (pred & target).sum(dim=(1, 2)).float()
    # False positive
    fp = (pred & ~target).sum(dim=(1, 2)).float()
    # False negative
    fn = (~pred & target).sum(dim=(1, 2)).float()

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * ((precision * recall) / (precision + recall + 1e-12))

    # return precision[precision.nonzero()], recall[recall.nonzero()], f1[f1.nonzero()]
    return precision[mask.bool()], recall[mask.bool()], f1[mask.bool()]

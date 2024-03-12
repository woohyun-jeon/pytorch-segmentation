import numpy as np
import torch
import torch.nn.functional as F


def estimate_accuracy(pred_mask, gt_mask):
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        correct = torch.eq(pred_mask, gt_mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())

    return accuracy


def estimate_miou(pred_mask, gt_mask, num_classes, epsilon=1e-10):
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)

        pred_mask = pred_mask.contiguous().view(-1)
        gt_mask = gt_mask.view(-1)

        iou_per_class = []
        for cls in range(num_classes):
            true_pred = pred_mask == cls
            true_gt = gt_mask == cls

            if true_gt.long().sum().item() == 0:
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_pred, true_gt).sum().float().item()
                union = torch.logical_or(true_pred, true_gt).sum().float().item()

                iou = (intersect + epsilon) / (union + epsilon)
                iou_per_class.append(iou)

    return np.nanmean(iou_per_class)
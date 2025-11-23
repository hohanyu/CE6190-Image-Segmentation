import torch
import numpy as np
from sklearn.metrics import confusion_matrix


def compute_iou(pred, target, num_classes=21, ignore_index=255):
    #IoU for each class and mean IoU
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    if pred.ndim == 3:
        pred = pred.reshape(-1)
        target = target.reshape(-1)
    elif pred.ndim == 2:
        pred = pred.flatten()
        target = target.flatten()
    
    #filter out ignore index
    valid_mask = (target != ignore_index)
    pred = pred[valid_mask]
    target = target[valid_mask]
    
    # compute confusion matrix
    cm = confusion_matrix(target, pred, labels=np.arange(num_classes))
    
    #IoU per class
    iou_per_class = np.zeros(num_classes)
    for i in range(num_classes):
        intersection = cm[i, i]
        union = cm[i, :].sum() + cm[:, i].sum() - intersection
        if union > 0:
            iou_per_class[i] = intersection / union
        else:

            iou_per_class[i] = np.nan  
    
    mean_iou = np.nanmean(iou_per_class)
    
    return iou_per_class, mean_iou


def compute_pixel_accuracy(pred, target, ignore_index=255):
    #handle tensors
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    #flatten
    if pred.ndim == 3:
        pred = pred.reshape(-1)
        target = target.reshape(-1)
    elif pred.ndim == 2:
        pred = pred.flatten()
        target = target.flatten()
    
    valid_mask = (target != ignore_index)
    pred = pred[valid_mask]
    target = target[valid_mask]
    
    correct = (pred == target).sum()
    total = len(target)
    
    if total > 0:
        return correct / total
    else:
        return 0.0


def compute_dice_coefficient(pred, target, class_idx=1):
    #convert tensors
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    #flatten to 1d
    if pred.ndim == 3:
        pred = pred.reshape(-1)
        target = target.reshape(-1)
    elif pred.ndim == 2:
        pred = pred.flatten()
        target = target.flatten()
    
    # binarize
    pred_binary = (pred == class_idx).astype(np.float32)
    target_binary = (target == class_idx).astype(np.float32)
    
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum()
    
    #edge case no foreground in either
    if union == 0:
        return 1.0
    
    dice = 2.0 * intersection / union
    return dice


def evaluate_segmentation(pred, target, num_classes=21, is_binary=False, ignore_index=255):
    #compute all metrics
    metrics = {}
    #compute pixel accuracy
    metrics['pixel_acc'] = compute_pixel_accuracy(pred, target, ignore_index)
    #compute other metrics based on dataset type
    if is_binary:
        metrics['dice'] = compute_dice_coefficient(pred, target, class_idx=1)
        iou_per_class, mean_iou = compute_iou(pred, target, num_classes=2, ignore_index=ignore_index)
        metrics['iou'] = mean_iou
        metrics['iou_foreground'] = iou_per_class[1] if len(iou_per_class) > 1 else 0.0
    else:
        iou_per_class, mean_iou = compute_iou(pred, target, num_classes=num_classes, ignore_index=ignore_index)
        metrics['mIoU'] = mean_iou
        metrics['iou_per_class'] = iou_per_class
    
    return metrics


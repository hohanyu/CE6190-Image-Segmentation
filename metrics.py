"""
Evaluation metrics for segmentation
"""
import torch
import numpy as np
from sklearn.metrics import confusion_matrix


def compute_iou(pred, target, num_classes=21, ignore_index=255):
    """
    Compute Intersection over Union (IoU) for each class
    
    Args:
        pred: Predicted segmentation mask (H, W) or (B, H, W)
        target: Ground truth mask (H, W) or (B, H, W)
        num_classes: Number of classes
        ignore_index: Class index to ignore
    
    Returns:
        iou_per_class: IoU for each class (num_classes,)
        mean_iou: Mean IoU across all classes
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    # Flatten if needed
    if pred.ndim == 3:
        pred = pred.reshape(-1)
        target = target.reshape(-1)
    elif pred.ndim == 2:
        pred = pred.flatten()
        target = target.flatten()
    
    # Remove ignored pixels
    valid_mask = (target != ignore_index)
    pred = pred[valid_mask]
    target = target[valid_mask]
    
    # Compute confusion matrix
    cm = confusion_matrix(target, pred, labels=np.arange(num_classes))
    
    # Compute IoU for each class
    iou_per_class = np.zeros(num_classes)
    for i in range(num_classes):
        intersection = cm[i, i]
        union = cm[i, :].sum() + cm[:, i].sum() - intersection
        if union > 0:
            iou_per_class[i] = intersection / union
        else:
            iou_per_class[i] = np.nan
    
    # Mean IoU (excluding NaN values)
    mean_iou = np.nanmean(iou_per_class)
    
    return iou_per_class, mean_iou


def compute_pixel_accuracy(pred, target, ignore_index=255):
    """
    Compute pixel accuracy
    
    Args:
        pred: Predicted segmentation mask (H, W) or (B, H, W)
        target: Ground truth mask (H, W) or (B, H, W)
        ignore_index: Class index to ignore
    
    Returns:
        pixel_acc: Pixel accuracy
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    # Flatten if needed
    if pred.ndim == 3:
        pred = pred.reshape(-1)
        target = target.reshape(-1)
    elif pred.ndim == 2:
        pred = pred.flatten()
        target = target.flatten()
    
    # Remove ignored pixels
    valid_mask = (target != ignore_index)
    pred = pred[valid_mask]
    target = target[valid_mask]
    
    # Compute accuracy
    correct = (pred == target).sum()
    total = len(target)
    
    return correct / total if total > 0 else 0.0


def compute_dice_coefficient(pred, target, class_idx=1):
    """
    Compute Dice coefficient for binary segmentation
    
    Args:
        pred: Predicted binary mask (H, W) or (B, H, W)
        target: Ground truth binary mask (H, W) or (B, H, W)
        class_idx: Class index to compute Dice for (default: 1 for foreground)
    
    Returns:
        dice: Dice coefficient
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    # Flatten if needed
    if pred.ndim == 3:
        pred = pred.reshape(-1)
        target = target.reshape(-1)
    elif pred.ndim == 2:
        pred = pred.flatten()
        target = target.flatten()
    
    # Binary masks: 1 for class_idx, 0 otherwise
    pred_binary = (pred == class_idx).astype(np.float32)
    target_binary = (target == class_idx).astype(np.float32)
    
    # Compute Dice
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum()
    
    if union == 0:
        return 1.0  # Both empty, perfect match
    
    dice = 2.0 * intersection / union
    return dice


def evaluate_segmentation(pred, target, num_classes=21, is_binary=False, ignore_index=255):
    """
    Comprehensive evaluation for segmentation
    
    Args:
        pred: Predicted mask
        target: Ground truth mask
        num_classes: Number of classes
        is_binary: Whether this is binary segmentation
        ignore_index: Class index to ignore
    
    Returns:
        metrics: Dictionary of metrics
    """
    metrics = {}
    
    # Pixel accuracy
    metrics['pixel_acc'] = compute_pixel_accuracy(pred, target, ignore_index)
    
    if is_binary:
        # For binary segmentation, compute Dice and IoU for foreground class
        metrics['dice'] = compute_dice_coefficient(pred, target, class_idx=1)
        iou_per_class, mean_iou = compute_iou(pred, target, num_classes=2, ignore_index=ignore_index)
        metrics['iou'] = mean_iou
        metrics['iou_foreground'] = iou_per_class[1] if len(iou_per_class) > 1 else 0.0
    else:
        # For multi-class segmentation
        iou_per_class, mean_iou = compute_iou(pred, target, num_classes=num_classes, ignore_index=ignore_index)
        metrics['mIoU'] = mean_iou
        metrics['iou_per_class'] = iou_per_class
    
    return metrics


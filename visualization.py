"""
Visualization utilities for segmentation results
"""
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch


def visualize_segmentation(image, pred_mask, gt_mask=None, num_classes=21, 
                          class_names=None, save_path=None, show=True):
    """
    Visualize segmentation results
    
    Args:
        image: Original image (PIL Image, numpy array, or tensor)
        pred_mask: Predicted mask (numpy array or tensor)
        gt_mask: Ground truth mask (optional, numpy array or tensor)
        num_classes: Number of classes
        class_names: List of class names (optional)
        save_path: Path to save figure (optional)
        show: Whether to display the figure
    """
    # Convert inputs to numpy arrays
    if isinstance(image, torch.Tensor):
        if image.dim() == 4:
            image = image[0]
        if image.dim() == 3 and image.shape[0] == 3:
            # Denormalize ImageNet stats
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = image.permute(1, 2, 0).numpy()
            image = image * std + mean
            image = np.clip(image, 0, 1)
        else:
            image = image.numpy()
    elif isinstance(image, Image.Image):
        image = np.array(image)
    image = image.astype(np.float32)
    if image.max() > 1:
        image = image / 255.0
    
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    
    if gt_mask is not None:
        if isinstance(gt_mask, torch.Tensor):
            gt_mask = gt_mask.cpu().numpy()
        num_cols = 3
    else:
        num_cols = 2
    
    fig, axes = plt.subplots(1, num_cols, figsize=(5*num_cols, 5))
    if num_cols == 1:
        axes = [axes]
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Predicted mask
    if num_classes == 2:
        # Binary segmentation - use colormap
        axes[1].imshow(pred_mask, cmap='gray')
        axes[1].set_title('Predicted Mask')
    else:
        # Multi-class - use color map
        axes[1].imshow(pred_mask, cmap='tab20', vmin=0, vmax=num_classes-1)
        axes[1].set_title('Predicted Mask')
    axes[1].axis('off')
    
    # Ground truth (if provided)
    if gt_mask is not None:
        if num_classes == 2:
            axes[2].imshow(gt_mask, cmap='gray')
            axes[2].set_title('Ground Truth')
        else:
            axes[2].imshow(gt_mask, cmap='tab20', vmin=0, vmax=num_classes-1)
            axes[2].set_title('Ground Truth')
        axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_overlay(image, pred_mask, gt_mask=None, alpha=0.5, save_path=None, show=True):
    """
    Visualize segmentation with overlay on original image
    
    Args:
        image: Original image
        pred_mask: Predicted mask
        gt_mask: Ground truth mask (optional)
        alpha: Transparency for overlay
        save_path: Path to save figure
        show: Whether to display
    """
    # Convert to numpy
    if isinstance(image, torch.Tensor):
        if image.dim() == 4:
            image = image[0]
        if image.dim() == 3 and image.shape[0] == 3:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = image.permute(1, 2, 0).numpy()
            image = image * std + mean
            image = np.clip(image, 0, 1)
        else:
            image = image.numpy()
    elif isinstance(image, Image.Image):
        image = np.array(image)
    image = image.astype(np.float32)
    if image.max() > 1:
        image = image / 255.0
    
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    if gt_mask is not None and isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.cpu().numpy()
    
    num_cols = 2 if gt_mask is None else 3
    fig, axes = plt.subplots(1, num_cols, figsize=(5*num_cols, 5))
    if num_cols == 1:
        axes = [axes]
    
    # Create colored overlay for predictions
    pred_colored = np.zeros_like(image)
    pred_colored[:, :, 0] = pred_mask > 0  # Red channel
    pred_overlay = image * (1 - alpha) + pred_colored * alpha
    
    axes[0].imshow(pred_overlay)
    axes[0].set_title('Prediction Overlay')
    axes[0].axis('off')
    
    if gt_mask is not None:
        gt_colored = np.zeros_like(image)
        gt_colored[:, :, 1] = gt_mask > 0  # Green channel
        gt_overlay = image * (1 - alpha) + gt_colored * alpha
        
        axes[1].imshow(gt_overlay)
        axes[1].set_title('Ground Truth Overlay')
        axes[1].axis('off')
        
        # Combined overlay
        combined_colored = np.zeros_like(image)
        combined_colored[:, :, 0] = pred_mask > 0  # Red for pred
        combined_colored[:, :, 1] = gt_mask > 0     # Green for GT
        combined_overlay = image * (1 - alpha) + combined_colored * alpha
        
        axes[2].imshow(combined_overlay)
        axes[2].set_title('Combined (Red=Pred, Green=GT)')
        axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def save_comparison_grid(images, preds, gts, save_path, num_classes=21, max_samples=10):
    """
    Save a grid of comparison images
    
    Args:
        images: List of images
        preds: List of predicted masks
        gts: List of ground truth masks
        save_path: Path to save figure
        num_classes: Number of classes
        max_samples: Maximum number of samples to show
    """
    n_samples = min(len(images), max_samples)
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4*n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        # Original
        img = images[i]
        if isinstance(img, torch.Tensor):
            if img.dim() == 3 and img.shape[0] == 3:
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = img.permute(1, 2, 0).numpy()
                img = img * std + mean
                img = np.clip(img, 0, 1)
            else:
                img = img.numpy()
        elif isinstance(img, Image.Image):
            img = np.array(img)
        if img.max() > 1:
            img = img / 255.0
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Sample {i+1} - Original')
        axes[i, 0].axis('off')
        
        # Prediction
        pred = preds[i]
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if num_classes == 2:
            axes[i, 1].imshow(pred, cmap='gray')
        else:
            axes[i, 1].imshow(pred, cmap='tab20', vmin=0, vmax=num_classes-1)
        axes[i, 1].set_title('Prediction')
        axes[i, 1].axis('off')
        
        # Ground truth
        gt = gts[i]
        if isinstance(gt, torch.Tensor):
            gt = gt.cpu().numpy()
        if num_classes == 2:
            axes[i, 2].imshow(gt, cmap='gray')
        else:
            axes[i, 2].imshow(gt, cmap='tab20', vmin=0, vmax=num_classes-1)
        axes[i, 2].set_title('Ground Truth')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


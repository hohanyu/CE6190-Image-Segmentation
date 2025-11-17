"""
Model loading and inference for pretrained segmentation models
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import torch.nn.functional as F
import numpy as np
from PIL import Image


class SegmentationModel:
    """Wrapper for pretrained segmentation models"""
    
    def __init__(self, model_name='deeplabv3_resnet101', device='cuda'):
        """
        Args:
            model_name: 'deeplabv3_resnet101' or 'fcn_resnet50'
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        # Load pretrained model
        try:
            # Try new torchvision API (v0.13+)
            if model_name == 'deeplabv3_resnet101':
                self.model = models.segmentation.deeplabv3_resnet101(
                    weights=models.segmentation.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS
                )
            elif model_name == 'fcn_resnet50':
                self.model = models.segmentation.fcn_resnet50(
                    weights=models.segmentation.FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS
                )
            else:
                raise ValueError(f"Unknown model: {model_name}")
        except (AttributeError, TypeError):
            # Fallback to older API
            if model_name == 'deeplabv3_resnet101':
                self.model = models.segmentation.deeplabv3_resnet101(
                    pretrained=True
                )
            elif model_name == 'fcn_resnet50':
                self.model = models.segmentation.fcn_resnet50(
                    pretrained=True
                )
            else:
                raise ValueError(f"Unknown model: {model_name}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def preprocess(self, image, target_size=None):
        """
        Preprocess image for inference
        
        Args:
            image: PIL Image or numpy array
            target_size: (H, W) tuple or None to keep original size
        
        Returns:
            Preprocessed tensor
        """
        if not isinstance(image, Image.Image):
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            else:
                image = Image.open(image).convert('RGB')
        
        if target_size:
            image = transforms.Resize(target_size)(image)
        
        image_tensor = transforms.ToTensor()(image)
        image_tensor = self.normalize(image_tensor)
        
        return image_tensor.unsqueeze(0).to(self.device)
    
    def predict(self, image, target_size=None, return_logits=False):
        """
        Perform segmentation prediction
        
        Args:
            image: PIL Image, numpy array, or path to image
            target_size: (H, W) tuple for resizing
            return_logits: If True, return raw logits instead of predictions
        
        Returns:
            Prediction mask (H, W) as numpy array
        """
        with torch.no_grad():
            # Preprocess
            input_tensor = self.preprocess(image, target_size)
            
            # Get original size for resizing output
            if isinstance(image, Image.Image):
                orig_size = image.size[::-1]  # PIL uses (W, H), we need (H, W)
            else:
                if isinstance(image, str):
                    img = Image.open(image)
                else:
                    img = Image.fromarray(image)
                orig_size = img.size[::-1]
            
            # Forward pass
            output = self.model(input_tensor)['out']
            
            if return_logits:
                return output.cpu().numpy()
            
            # Get predictions
            pred = output.argmax(dim=1).squeeze().cpu().numpy()
            
            # Resize to original size if needed
            if target_size and target_size != orig_size:
                import cv2
                pred = cv2.resize(pred.astype(np.uint8), 
                                (orig_size[1], orig_size[0]), 
                                interpolation=cv2.INTER_NEAREST)
            
            return pred
    
    def predict_batch(self, images, target_size=None):
        """
        Predict on a batch of images
        
        Args:
            images: List of PIL Images or numpy arrays
            target_size: (H, W) tuple for resizing
        
        Returns:
            List of prediction masks
        """
        results = []
        for img in images:
            results.append(self.predict(img, target_size))
        return results


def load_model(model_name='deeplabv3_resnet101', device='cuda'):
    """Convenience function to load a model"""
    return SegmentationModel(model_name, device)


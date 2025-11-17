"""
Dataset loaders for Pascal VOC 2012 and ISIC 2018
"""
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import VOCSegmentation
import cv2


class VOCDataset(Dataset):
    """Pascal VOC 2012 dataset loader"""
    
    def __init__(self, root='./data/VOC2012', split='val', transform=None, target_transform=None):
        """
        Args:
            root: Root directory of VOC dataset
            split: 'train' or 'val'
            transform: Transform to apply to images
            target_transform: Transform to apply to masks
        """
        self.dataset = VOCSegmentation(
            root=root,
            year='2012',
            image_set=split,
            download=True,
            transform=None,
            target_transform=None
        )
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, mask = self.dataset[idx]
        
        # Convert mask to numpy array for processing
        mask = np.array(mask)
        # VOC masks have class indices 0-20, but some pixels are 255 (boundary/void)
        # We'll treat 255 as background (0)
        mask[mask == 255] = 0
        
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            mask = self.target_transform(mask)
        else:
            # Convert to tensor if no transform provided
            mask = torch.from_numpy(mask).long()
        
        return image, mask


class ISICDataset(Dataset):
    """ISIC 2018 dataset loader for binary segmentation"""
    
    def __init__(self, root='./data/ISIC2018', split='test', transform=None, target_transform=None, max_samples=None):
        """
        Args:
            root: Root directory containing ISIC images and masks
            split: 'train' or 'test'
            transform: Transform to apply to images
            target_transform: Transform to apply to masks
            max_samples: Maximum number of samples to load (None for all)
        """
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # ISIC 2018 structure: images in one folder, masks in another
        image_dir = os.path.join(root, 'ISIC2018_Task1-2_Training_Input')
        mask_dir = os.path.join(root, 'ISIC2018_Task1_Training_GroundTruth')
        
        if not os.path.exists(image_dir):
            raise ValueError(f"ISIC image directory not found: {image_dir}")
        if not os.path.exists(mask_dir):
            raise ValueError(f"ISIC mask directory not found: {mask_dir}")
        
        # Get all image files
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        
        if max_samples:
            self.image_files = self.image_files[:max_samples]
        
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        # Load image
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Load mask (same name but with _segmentation suffix)
        mask_name = img_name.replace('.jpg', '_segmentation.png')
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
            mask = np.array(mask)
            # Binary mask: 0 for background, 1 for lesion
            mask = (mask > 128).astype(np.uint8)
        else:
            # If mask doesn't exist, create empty mask
            mask = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
        
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            mask = self.target_transform(mask)
        else:
            mask = torch.from_numpy(mask).long()
        
        return image, mask, img_name


def get_voc_transform(size=(512, 512)):
    """Get transform for VOC dataset"""
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_isic_transform(size=(512, 512)):
    """Get transform for ISIC dataset"""
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_mask_transform(size=(512, 512)):
    """Get transform for masks (resize only, no normalization)"""
    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST)
    ])


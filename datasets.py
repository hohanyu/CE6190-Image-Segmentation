import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import OxfordIIITPet
import cv2


class ISICDataset(Dataset):
    
    def __init__(self, root='./data/ISIC2018', split='test', transform=None, target_transform=None, max_samples=None):

        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        image_dir = os.path.join(root, 'ISIC2018_Task1-2_Training_Input')
        mask_dir = os.path.join(root, 'ISIC2018_Task1_Training_GroundTruth')
        
        if not os.path.exists(image_dir):
            raise ValueError(f"ISIC image directory not found: {image_dir}")
        if not os.path.exists(mask_dir):
            raise ValueError(f"ISIC mask directory not found: {mask_dir}")
        
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        
        if max_samples:
            self.image_files = self.image_files[:max_samples]
        
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        mask_name = img_name.replace('.jpg', '_segmentation.png')
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
            mask_array = np.array(mask)
            mask_array = (mask_array > 128).astype(np.uint8)
            mask = Image.fromarray(mask_array, mode='L')
        else:
            mask = Image.new('L', image.size, 0)
        
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            mask = self.target_transform(mask)
            if isinstance(mask, Image.Image):
                mask = np.array(mask)
            mask = torch.from_numpy(mask).long() if isinstance(mask, np.ndarray) else mask
        else:
            mask = torch.from_numpy(np.array(mask)).long()
        
        return image, mask, img_name


class PetDataset(Dataset):

    def __init__(self, root='./data/OxfordPets', split='test', transform=None,
                 target_transform=None, max_samples=None):

        self.transform = transform
        self.target_transform = target_transform
        try:
            self.dataset = OxfordIIITPet(
                root=root,
                split=split,
                target_types='segmentation',
                download=True
            )
        except Exception as e:
            print(f"Warning: Failed to load Oxford-IIIT Pet dataset: {e}")
            raise

        self.indices = list(range(len(self.dataset)))
        if max_samples:
            self.indices = self.indices[:max_samples]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image, mask = self.dataset[real_idx]
        img_path = None
        if hasattr(self.dataset, '_images'):
            img_path = self.dataset._images[real_idx]
        elif hasattr(self.dataset, 'images'):
            img_path = self.dataset.images[real_idx]
        img_name = os.path.basename(img_path) if img_path else f"pet_{real_idx}"

        mask = np.array(mask) - 1
        mask = np.clip(mask, 0, 2)
        mask = Image.fromarray(mask.astype(np.uint8))

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            mask = self.target_transform(mask)
            if isinstance(mask, Image.Image):
                mask = np.array(mask)
            mask = torch.from_numpy(mask).long() if isinstance(mask, np.ndarray) else mask
        else:
            mask = torch.from_numpy(np.array(mask)).long()

        return image, mask, img_name


def get_isic_transform(size=(512, 512)):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_pet_transform(size=(512, 512)):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_mask_transform(size=(512, 512)):
    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST)
    ])


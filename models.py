
import torch
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
from PIL import Image


class SegmentationModel:
    
    def __init__(self, model_name='deeplabv3_resnet101', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        # to handle different torchvision versions
        model_loaded = False
        try:
            if model_name == 'deeplabv3_resnet101':
                self.model = models.segmentation.deeplabv3_resnet101(
                    weights=models.segmentation.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS
                )
                model_loaded = True
            elif model_name == 'deeplabv3_resnet50':
                self.model = models.segmentation.deeplabv3_resnet50(
                    weights=models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS
                )
                model_loaded = True
            elif model_name == 'fcn_resnet50':
                self.model = models.segmentation.fcn_resnet50(
                    weights=models.segmentation.FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS
                )
                model_loaded = True
            elif model_name == 'fcn_resnet101':
                self.model = models.segmentation.fcn_resnet101(
                    weights=models.segmentation.FCN_ResNet101_Weights.COCO_WITH_VOC_LABELS
                )
                model_loaded = True
        except (AttributeError, TypeError):
            model_loaded = False
        
        if not model_loaded:
            #fallback to old pretrained=True
            if model_name == 'deeplabv3_resnet101':
                self.model = models.segmentation.deeplabv3_resnet101(pretrained=True)
            elif model_name == 'deeplabv3_resnet50':
                self.model = models.segmentation.deeplabv3_resnet50(pretrained=True)
            elif model_name == 'fcn_resnet50':
                self.model = models.segmentation.fcn_resnet50(pretrained=True)
            elif model_name == 'fcn_resnet101':
                self.model = models.segmentation.fcn_resnet101(pretrained=True)
            else:
                raise ValueError("Unknown model: " + model_name)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        #imagenet normalization
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean=mean_vals, std=std_vals)
    
    def preprocess(self, image, target_size=None):
        #convert to tensor
        if isinstance(image, Image.Image):
            pass
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        else:
            image = Image.open(image).convert('RGB')
        
        if target_size:
            image = transforms.Resize(target_size)(image)
        
        image_tensor = transforms.ToTensor()(image)
        image_tensor = self.normalize(image_tensor)
        
        return image_tensor.unsqueeze(0).to(self.device)
    
    def predict(self, image, target_size=None, return_logits=False):
        with torch.no_grad():
            input_tensor = self.preprocess(image, target_size)
            
            # get original size for resizing
            if isinstance(image, Image.Image):
                #(W,H) ->(H,W)
                orig_size = image.size[::-1]  
            elif isinstance(image, str):
                img = Image.open(image)
                orig_size = img.size[::-1]
            else:
                img = Image.fromarray(image)
                orig_size = img.size[::-1]
            
            output = self.model(input_tensor)['out']
            
            if return_logits:
                return output.cpu().numpy()
            
            pred = output.argmax(dim=1).squeeze().cpu().numpy()
            
            #resize back to original if needed
            if target_size and target_size != orig_size:
                import cv2  
                pred = cv2.resize(pred.astype(np.uint8), 
                                (orig_size[1], orig_size[0]), 
                                interpolation=cv2.INTER_NEAREST)
            
            return pred
    
    def predict_batch(self, images, target_size=None):
        results = []
        for img in images:
            results.append(self.predict(img, target_size))
        return results


def load_model(model_name='deeplabv3_resnet101', device='cuda'):
    return SegmentationModel(model_name, device)


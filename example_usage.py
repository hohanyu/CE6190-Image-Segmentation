"""
Example script showing how to use individual components
"""
import torch
from models import load_model
from datasets import VOCDataset, ISICDataset, get_voc_transform, get_isic_transform, get_mask_transform
from metrics import evaluate_segmentation
from visualization import visualize_segmentation


def example_single_prediction():
    """Example: Single image prediction"""
    print("Example 1: Single Image Prediction")
    print("-" * 50)
    
    # Load model
    model = load_model('deeplabv3_resnet101', device='cuda')
    
    # Load a single image (example with PIL)
    from PIL import Image
    import numpy as np
    
    # Create a dummy image for demonstration
    dummy_image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
    
    # Predict
    prediction = model.predict(dummy_image, target_size=(512, 512))
    print(f"Prediction shape: {prediction.shape}")
    print(f"Unique classes: {np.unique(prediction)}")


def example_dataset_evaluation():
    """Example: Evaluate on a small subset of VOC dataset"""
    print("\nExample 2: Dataset Evaluation")
    print("-" * 50)
    
    # Load model
    model = load_model('fcn_resnet50', device='cuda')
    
    # Create dataset
    transform = get_voc_transform(size=(512, 512))
    mask_transform = get_mask_transform(size=(512, 512))
    
    try:
        dataset = VOCDataset(
            root='./data/VOC2012',
            split='val',
            transform=transform,
            target_transform=mask_transform
        )
        
        # Evaluate on first 5 samples
        print(f"Dataset size: {len(dataset)}")
        
        for i in range(min(5, len(dataset))):
            image, mask = dataset[i]
            
            # Predict
            with torch.no_grad():
                output = model.model(image.unsqueeze(0).to(model.device))['out']
                pred = output.argmax(dim=1).squeeze().cpu().numpy()
            
            # Resize if needed
            if pred.shape != mask.shape:
                import torch.nn.functional as F
                pred_tensor = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float()
                pred = F.interpolate(pred_tensor, size=mask.shape, mode='nearest').squeeze().numpy().astype(int)
            
            # Evaluate
            metrics = evaluate_segmentation(pred, mask.numpy(), num_classes=21, is_binary=False)
            print(f"Sample {i+1}: mIoU={metrics['mIoU']:.4f}, Pixel Acc={metrics['pixel_acc']:.4f}")
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure VOC dataset is downloaded to ./data/VOC2012")


def example_visualization():
    """Example: Visualize segmentation results"""
    print("\nExample 3: Visualization")
    print("-" * 50)
    
    from PIL import Image
    import numpy as np
    
    # Create dummy data
    dummy_image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
    dummy_pred = np.random.randint(0, 21, (512, 512), dtype=np.uint8)
    dummy_gt = np.random.randint(0, 21, (512, 512), dtype=np.uint8)
    
    # Visualize
    visualize_segmentation(
        dummy_image,
        dummy_pred,
        dummy_gt,
        num_classes=21,
        save_path='./example_visualization.png',
        show=False
    )
    print("Visualization saved to ./example_visualization.png")


if __name__ == '__main__':
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Run examples
    example_single_prediction()
    example_dataset_evaluation()
    example_visualization()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("=" * 50)


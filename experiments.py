"""
Main experiment script for segmentation evaluation
"""
import os
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader
import json

from datasets import VOCDataset, ISICDataset, get_voc_transform, get_isic_transform, get_mask_transform
from models import load_model
from metrics import evaluate_segmentation, compute_iou, compute_pixel_accuracy, compute_dice_coefficient
from visualization import visualize_segmentation, save_comparison_grid


def evaluate_model_on_dataset(model, dataset, dataset_name, device='cuda', batch_size=1, 
                             save_results_dir=None, visualize_samples=5):
    """
    Evaluate a model on a dataset
    
    Args:
        model: SegmentationModel instance
        dataset: Dataset instance
        dataset_name: Name of dataset ('voc' or 'isic')
        device: Device to use
        batch_size: Batch size for inference
        save_results_dir: Directory to save results
        visualize_samples: Number of samples to visualize
    
    Returns:
        Dictionary of metrics and sample results
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    is_binary = (dataset_name == 'isic')
    num_classes = 2 if is_binary else 21
    
    all_metrics = []
    sample_results = []
    
    model.model.eval()
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {dataset_name}")):
            if is_binary:
                images, masks, img_names = batch
            else:
                images, masks = batch
                img_names = [f"sample_{idx}_{i}" for i in range(len(images))]
            
            images = images.to(device)
            masks = masks.to(device)
            
            # Get predictions
            outputs = model.model(images)['out']
            preds = outputs.argmax(dim=1)
            
            # Resize predictions to match ground truth
            if preds.shape[1:] != masks.shape[1:]:
                preds = torch.nn.functional.interpolate(
                    preds.unsqueeze(1).float(),
                    size=masks.shape[1:],
                    mode='nearest'
                ).squeeze(1).long()
            
            # Evaluate each sample in batch
            for i in range(len(images)):
                pred = preds[i].cpu().numpy()
                mask = masks[i].cpu().numpy()
                
                # Compute metrics
                metrics = evaluate_segmentation(
                    pred, mask, 
                    num_classes=num_classes,
                    is_binary=is_binary
                )
                all_metrics.append(metrics)
                
                # Store sample for visualization
                if len(sample_results) < visualize_samples:
                    sample_results.append({
                        'image': images[i].cpu(),
                        'pred': pred,
                        'gt': mask,
                        'name': img_names[i] if isinstance(img_names, list) else img_names,
                        'metrics': metrics
                    })
    
    # Aggregate metrics
    if is_binary:
        avg_metrics = {
            'dice': np.mean([m['dice'] for m in all_metrics]),
            'iou': np.mean([m['iou'] for m in all_metrics]),
            'pixel_acc': np.mean([m['pixel_acc'] for m in all_metrics])
        }
    else:
        avg_metrics = {
            'mIoU': np.mean([m['mIoU'] for m in all_metrics]),
            'pixel_acc': np.mean([m['pixel_acc'] for m in all_metrics])
        }
        # Per-class IoU
        all_iou_per_class = [m['iou_per_class'] for m in all_metrics]
        avg_iou_per_class = np.nanmean(all_iou_per_class, axis=0)
        avg_metrics['iou_per_class'] = avg_iou_per_class.tolist()
    
    # Save results
    if save_results_dir:
        os.makedirs(save_results_dir, exist_ok=True)
        
        # Save metrics
        with open(os.path.join(save_results_dir, f'{dataset_name}_metrics.json'), 'w') as f:
            json.dump(avg_metrics, f, indent=2)
        
        # Save sample visualizations
        vis_dir = os.path.join(save_results_dir, 'visualizations', dataset_name)
        os.makedirs(vis_dir, exist_ok=True)
        
        for sample in sample_results:
            save_path = os.path.join(vis_dir, f"{sample['name']}.png")
            visualize_segmentation(
                sample['image'],
                sample['pred'],
                sample['gt'],
                num_classes=num_classes,
                save_path=save_path,
                show=False
            )
    
    return avg_metrics, sample_results


def hyperparameter_analysis(model_name, dataset_name, dataset_root, resolutions=[256, 384, 512], 
                           device='cuda', save_dir=None, max_samples=None):
    """
    Analyze the effect of different input resolutions
    
    Args:
        model_name: Name of model to evaluate
        dataset_name: 'voc' or 'isic'
        dataset_root: Root directory of the dataset
        resolutions: List of resolutions to test
        device: Device to use
        save_dir: Directory to save results
        max_samples: Maximum samples for ISIC dataset
    
    Returns:
        DataFrame with results for each resolution
    """
    results = []
    
    for res in resolutions:
        print(f"\nEvaluating {model_name} at resolution {res}x{res}")
        
        # Create dataset with specific resolution
        if dataset_name == 'voc':
            transform = get_voc_transform(size=(res, res))
            mask_transform = get_mask_transform(size=(res, res))
            test_dataset = VOCDataset(
                root=dataset_root,
                split='val',
                transform=transform,
                target_transform=mask_transform
            )
        else:
            transform = get_isic_transform(size=(res, res))
            mask_transform = get_mask_transform(size=(res, res))
            test_dataset = ISICDataset(
                root=dataset_root,
                split='test',
                transform=transform,
                target_transform=mask_transform,
                max_samples=max_samples or 500
            )
        
        # Load model
        model = load_model(model_name, device)
        
        # Evaluate
        metrics, _ = evaluate_model_on_dataset(
            model, test_dataset, dataset_name, device=device,
            save_results_dir=None  # Don't save individual results for hyperparameter analysis
        )
        
        metrics['resolution'] = res
        metrics['model'] = model_name
        results.append(metrics)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        df.to_csv(os.path.join(save_dir, f'{model_name}_{dataset_name}_hyperparameter_analysis.csv'), index=False)
    
    return df


def find_good_bad_cases(model, dataset, dataset_name, device='cuda', num_cases=5):
    """
    Find good and bad segmentation cases
    
    Args:
        model: SegmentationModel instance
        dataset: Dataset instance
        dataset_name: 'voc' or 'isic'
        device: Device to use
        num_cases: Number of good/bad cases to find
    
    Returns:
        good_cases: List of good cases
        bad_cases: List of bad cases
    """
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    is_binary = (dataset_name == 'isic')
    num_classes = 2 if is_binary else 21
    
    case_scores = []
    
    model.model.eval()
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc="Finding good/bad cases")):
            if is_binary:
                images, masks, img_names = batch
            else:
                images, masks = batch
                img_names = [f"sample_{idx}"]
            
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model.model(images)['out']
            preds = outputs.argmax(dim=1)
            
            if preds.shape[1:] != masks.shape[1:]:
                preds = torch.nn.functional.interpolate(
                    preds.unsqueeze(1).float(),
                    size=masks.shape[1:],
                    mode='nearest'
                ).squeeze(1).long()
            
            pred = preds[0].cpu().numpy()
            mask = masks[0].cpu().numpy()
            
            # Compute score (use mIoU or Dice)
            if is_binary:
                score = compute_dice_coefficient(pred, mask, class_idx=1)
            else:
                _, mIoU = compute_iou(pred, mask, num_classes=num_classes)
                score = mIoU
            
            case_scores.append({
                'idx': idx,
                'score': score,
                'image': images[0].cpu(),
                'pred': pred,
                'gt': mask,
                'name': img_names[0] if isinstance(img_names, list) else img_names
            })
    
    # Sort by score
    case_scores.sort(key=lambda x: x['score'], reverse=True)
    
    good_cases = case_scores[:num_cases]
    bad_cases = case_scores[-num_cases:]
    
    return good_cases, bad_cases


def run_all_experiments(save_dir='./results', device='cuda', resolutions=[256, 384, 512]):
    """
    Run all experiments and generate results
    
    Args:
        save_dir: Directory to save all results
        device: Device to use
        resolutions: Resolutions for hyperparameter analysis
    """
    os.makedirs(save_dir, exist_ok=True)
    
    models = ['deeplabv3_resnet101', 'fcn_resnet50']
    datasets_config = [
        {'name': 'voc', 'root': './data/VOC2012', 'split': 'val'},
        {'name': 'isic', 'root': './data/ISIC2018', 'split': 'test', 'max_samples': 500}
    ]
    
    all_results = []
    
    # Main evaluation
    print("=" * 60)
    print("Main Evaluation")
    print("=" * 60)
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name}")
        print(f"{'='*60}")
        
        model = load_model(model_name, device)
        
        for dataset_config in datasets_config:
            dataset_name = dataset_config['name']
            print(f"\n--- Dataset: {dataset_name.upper()} ---")
            
            # Create dataset
            if dataset_name == 'voc':
                transform = get_voc_transform(size=(512, 512))
                mask_transform = get_mask_transform(size=(512, 512))
                dataset = VOCDataset(
                    root=dataset_config['root'],
                    split=dataset_config['split'],
                    transform=transform,
                    target_transform=mask_transform
                )
            else:
                transform = get_isic_transform(size=(512, 512))
                mask_transform = get_mask_transform(size=(512, 512))
                dataset = ISICDataset(
                    root=dataset_config['root'],
                    split=dataset_config['split'],
                    transform=transform,
                    target_transform=mask_transform,
                    max_samples=dataset_config.get('max_samples', None)
                )
            
            # Evaluate
            results_dir = os.path.join(save_dir, model_name, dataset_name)
            metrics, samples = evaluate_model_on_dataset(
                model, dataset, dataset_name, device=device,
                save_results_dir=results_dir,
                visualize_samples=10
            )
            
            metrics['model'] = model_name
            metrics['dataset'] = dataset_name
            all_results.append(metrics)
            
            # Find good/bad cases
            print(f"\nFinding good/bad cases for {model_name} on {dataset_name}...")
            good_cases, bad_cases = find_good_bad_cases(
                model, dataset, dataset_name, device=device, num_cases=5
            )
            
            # Save good/bad cases
            cases_dir = os.path.join(results_dir, 'cases')
            os.makedirs(cases_dir, exist_ok=True)
            
            for i, case in enumerate(good_cases):
                save_path = os.path.join(cases_dir, f'good_case_{i+1}_{case["name"]}.png')
                visualize_segmentation(
                    case['image'],
                    case['pred'],
                    case['gt'],
                    num_classes=2 if dataset_name == 'isic' else 21,
                    save_path=save_path,
                    show=False
                )
            
            for i, case in enumerate(bad_cases):
                save_path = os.path.join(cases_dir, f'bad_case_{i+1}_{case["name"]}.png')
                visualize_segmentation(
                    case['image'],
                    case['pred'],
                    case['gt'],
                    num_classes=2 if dataset_name == 'isic' else 21,
                    save_path=save_path,
                    show=False
                )
    
    # Hyperparameter analysis
    print("\n" + "=" * 60)
    print("Hyperparameter Analysis")
    print("=" * 60)
    
    for model_name in models:
        for dataset_config in datasets_config:
            dataset_name = dataset_config['name']
            print(f"\nHyperparameter analysis: {model_name} on {dataset_name}")
            
            hp_dir = os.path.join(save_dir, 'hyperparameter_analysis')
            df = hyperparameter_analysis(
                model_name, 
                dataset_name,
                dataset_config['root'],
                resolutions=resolutions,
                device=device,
                save_dir=hp_dir,
                max_samples=dataset_config.get('max_samples', None)
            )
    
    # Create summary table
    print("\n" + "=" * 60)
    print("Creating Summary Table")
    print("=" * 60)
    
    summary_data = []
    for result in all_results:
        row = {
            'Model': result['model'],
            'Dataset': result['dataset'].upper()
        }
        if result['dataset'] == 'isic':
            row['Dice'] = f"{result['dice']:.4f}"
            row['IoU'] = f"{result['iou']:.4f}"
            row['Pixel Acc'] = f"{result['pixel_acc']:.4f}"
        else:
            row['mIoU'] = f"{result['mIoU']:.4f}"
            row['Pixel Acc'] = f"{result['pixel_acc']:.4f}"
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(save_dir, 'summary_results.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")
    print("\n" + summary_df.to_string())
    
    print("\n" + "=" * 60)
    print("All experiments completed!")
    print(f"Results saved to: {save_dir}")
    print("=" * 60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run segmentation experiments')
    parser.add_argument('--save_dir', type=str, default='./results',
                       help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--resolutions', type=int, nargs='+', default=[256, 384, 512],
                       help='Resolutions for hyperparameter analysis')
    
    args = parser.parse_args()
    
    run_all_experiments(
        save_dir=args.save_dir,
        device=args.device,
        resolutions=args.resolutions
    )


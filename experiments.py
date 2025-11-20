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

from datasets import (
    ISICDataset,
    PetDataset,
    get_isic_transform,
    get_pet_transform,
    get_mask_transform,
)
from models import load_model
from metrics import evaluate_segmentation, compute_iou, compute_pixel_accuracy, compute_dice_coefficient
from visualization import visualize_segmentation, save_comparison_grid


DATASET_DEFAULTS = {
    'pet': {
        'default_root': './data/OxfordPets',
        'default_split': 'test',
        'default_max_samples': None,
        'num_classes': 3,
        'is_binary': False,
    },
    'isic': {
        'default_root': './data/ISIC2018',
        'default_split': 'test',
        'default_max_samples': 500,
        'num_classes': 2,
        'is_binary': True,
    },
}


def build_dataset_instance(dataset_name, root, split, size=(512, 512), max_samples=None):
    mask_transform = get_mask_transform(size=size)
    if dataset_name == 'isic':
        transform = get_isic_transform(size=size)
        dataset = ISICDataset(
            root=root,
            split=split,
            transform=transform,
            target_transform=mask_transform,
            max_samples=max_samples
        )
    elif dataset_name == 'pet':
        transform = get_pet_transform(size=size)
        dataset = PetDataset(
            root=root,
            split=split,
            transform=transform,
            target_transform=mask_transform,
            max_samples=max_samples
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return dataset


def evaluate_model_on_dataset(model, dataset, dataset_name, num_classes, is_binary,
                             device='cuda', batch_size=1, save_results_dir=None,
                             visualize_samples=5):
    """Evaluate model on dataset and compute metrics"""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    all_metrics = []
    sample_results = []
    
    model.model.eval()
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {dataset_name}")):
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                images, masks, img_names = batch
            else:
                images, masks = batch
                img_names = [f"sample_{idx}_{i}" for i in range(len(images))]
            
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
            
            for i in range(len(images)):
                pred = preds[i].cpu().numpy()
                mask = masks[i].cpu().numpy()
                
                if isinstance(img_names, (list, tuple)):
                    img_name = img_names[i] if i < len(img_names) else f"sample_{idx}_{i}"
                else:
                    img_name = img_names
                
                if isinstance(img_name, tuple):
                    img_name = img_name[0] if len(img_name) > 0 else f"sample_{idx}_{i}"
                
                metrics = evaluate_segmentation(
                    pred, mask, 
                    num_classes=num_classes,
                    is_binary=is_binary
                )
                all_metrics.append(metrics)
                
                if len(sample_results) < visualize_samples:
                    sample_results.append({
                        'image': images[i].cpu(),
                        'pred': pred,
                        'gt': mask,
                        'name': img_name,
                        'metrics': metrics
                    })
    
    if is_binary:
        avg_metrics = {
            'dice': float(np.mean([m['dice'] for m in all_metrics])),
            'iou': float(np.mean([m['iou'] for m in all_metrics])),
            'pixel_acc': float(np.mean([m['pixel_acc'] for m in all_metrics]))
        }
    else:
        avg_metrics = {
            'mIoU': float(np.mean([m['mIoU'] for m in all_metrics])),
            'pixel_acc': float(np.mean([m['pixel_acc'] for m in all_metrics]))
        }
        all_iou_per_class = [m['iou_per_class'] for m in all_metrics]
        avg_iou_per_class = np.nanmean(all_iou_per_class, axis=0)
        avg_metrics['iou_per_class'] = [float(x) if not np.isnan(x) else None for x in avg_iou_per_class.tolist()]
    
    if save_results_dir:
        os.makedirs(save_results_dir, exist_ok=True)
        
        metrics_to_save = {}
        for key, value in avg_metrics.items():
            if isinstance(value, (np.integer, np.floating)):
                metrics_to_save[key] = float(value)
            elif isinstance(value, np.ndarray):
                metrics_to_save[key] = [float(x) if not np.isnan(x) else None for x in value.tolist()]
            elif isinstance(value, list):
                metrics_to_save[key] = [float(x) if isinstance(x, (np.integer, np.floating)) else x for x in value]
            else:
                metrics_to_save[key] = value
        
        with open(os.path.join(save_results_dir, f'{dataset_name}_metrics.json'), 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        
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


def hyperparameter_analysis(model_name, dataset_name, dataset_config, resolutions=[256, 384, 512],
                           device='cuda', save_dir=None):
    results = []
    
    for res in resolutions:
        print(f"\nEvaluating {model_name} at resolution {res}x{res}")
        
        test_dataset = build_dataset_instance(
            dataset_name,
            root=dataset_config['root'],
            split=dataset_config['split'],
            size=(res, res),
            max_samples=dataset_config.get('max_samples')
        )
        
        model = load_model(model_name, device)
        
        metrics, _ = evaluate_model_on_dataset(
            model,
            test_dataset,
            dataset_name,
            dataset_config['num_classes'],
            dataset_config['is_binary'],
            device=device,
            save_results_dir=None
        )
        
        metrics['resolution'] = res
        metrics['model'] = model_name
        results.append(metrics)
    
    df = pd.DataFrame(results)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        df.to_csv(os.path.join(save_dir, f'{model_name}_{dataset_name}_hyperparameter_analysis.csv'), index=False)
    
    return df


def find_good_bad_cases(model, dataset, dataset_name, num_classes, is_binary,
                        device='cuda', num_cases=5):
    """Find good and bad segmentation cases for qualitative analysis"""
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    
    case_scores = []
    
    model.model.eval()
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc="Finding good/bad cases")):
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
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
            
            if is_binary:
                if isinstance(img_names, (list, tuple)):
                    img_name = img_names[0] if len(img_names) > 0 else f"sample_{idx}"
                else:
                    img_name = img_names
            else:
                img_name = f"sample_{idx}"
            
            if isinstance(img_name, tuple):
                img_name = img_name[0] if len(img_name) > 0 else f"sample_{idx}"
            
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
                'name': img_name
            })
    
    case_scores.sort(key=lambda x: x['score'], reverse=True)
    
    good_cases = case_scores[:num_cases]
    bad_cases = case_scores[-num_cases:]
    
    return good_cases, bad_cases


def run_all_experiments(save_dir='./results', device='cuda', resolutions=[256, 384, 512],
                        datasets=('pet', 'isic'), dataset_roots=None,
                        dataset_splits=None, dataset_max_samples=None):
    os.makedirs(save_dir, exist_ok=True)
    
    models = ['deeplabv3_resnet101', 'fcn_resnet50']
    dataset_roots = dataset_roots or {}
    dataset_splits = dataset_splits or {}
    dataset_max_samples = dataset_max_samples or {}
    
    dataset_configs = []
    for name in datasets:
        if name not in DATASET_DEFAULTS:
            raise ValueError(f"Unsupported dataset '{name}'. Available: {list(DATASET_DEFAULTS.keys())}")
        defaults = DATASET_DEFAULTS[name]
        dataset_configs.append({
            'name': name,
            'root': dataset_roots.get(name, defaults['default_root']),
            'split': dataset_splits.get(name, defaults['default_split']),
            'num_classes': defaults['num_classes'],
            'is_binary': defaults['is_binary'],
            'max_samples': dataset_max_samples.get(name, defaults['default_max_samples']),
        })
    
    all_results = []
    
    print("=" * 60)
    print("Main Evaluation")
    print("=" * 60)
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name}")
        print(f"{'='*60}")
        
        model = load_model(model_name, device)
        
        for dataset_config in dataset_configs:
            dataset_name = dataset_config['name']
            print(f"\n--- Dataset: {dataset_name.upper()} ---")
            
            dataset = build_dataset_instance(
                dataset_name,
                root=dataset_config['root'],
                split=dataset_config['split'],
                size=(512, 512),
                max_samples=dataset_config.get('max_samples')
            )
            
            results_dir = os.path.join(save_dir, model_name, dataset_name)
            metrics, samples = evaluate_model_on_dataset(
                model,
                dataset,
                dataset_name,
                dataset_config['num_classes'],
                dataset_config['is_binary'],
                device=device,
                save_results_dir=results_dir,
                visualize_samples=10
            )
            
            metrics['model'] = model_name
            metrics['dataset'] = dataset_name
            metrics['is_binary'] = dataset_config['is_binary']
            all_results.append(metrics)
            
            print(f"\nFinding good/bad cases for {model_name} on {dataset_name}...")
            good_cases, bad_cases = find_good_bad_cases(
                model,
                dataset,
                dataset_name,
                dataset_config['num_classes'],
                dataset_config['is_binary'],
                device=device,
                num_cases=5
            )
            
            cases_dir = os.path.join(results_dir, 'cases')
            os.makedirs(cases_dir, exist_ok=True)
            
            for i, case in enumerate(good_cases):
                save_path = os.path.join(cases_dir, f'good_case_{i+1}_{case["name"]}.png')
                visualize_segmentation(
                    case['image'],
                    case['pred'],
                    case['gt'],
                    num_classes=dataset_config['num_classes'],
                    save_path=save_path,
                    show=False
                )
            
            for i, case in enumerate(bad_cases):
                save_path = os.path.join(cases_dir, f'bad_case_{i+1}_{case["name"]}.png')
                visualize_segmentation(
                    case['image'],
                    case['pred'],
                    case['gt'],
                    num_classes=dataset_config['num_classes'],
                    save_path=save_path,
                    show=False
                )
    
    print("\n" + "=" * 60)
    print("Hyperparameter Analysis")
    print("=" * 60)
    
    for model_name in models:
        for dataset_config in dataset_configs:
            dataset_name = dataset_config['name']
            print(f"\nHyperparameter analysis: {model_name} on {dataset_name}")
            
            hp_dir = os.path.join(save_dir, 'hyperparameter_analysis')
            df = hyperparameter_analysis(
                model_name,
                dataset_name,
                dataset_config,
                resolutions=resolutions,
                device=device,
                save_dir=hp_dir,
            )
    
    print("\n" + "=" * 60)
    print("Creating Summary Table")
    print("=" * 60)
    
    summary_data = []
    for result in all_results:
        row = {
            'Model': result['model'],
            'Dataset': result['dataset'].upper()
        }
        if result.get('is_binary'):
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
    print("All experiments completed")
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


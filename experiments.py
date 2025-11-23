import os
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader
import json

from datasets import ISICDataset, PetDataset
from datasets import get_isic_transform, get_pet_transform, get_mask_transform
from models import load_model
from metrics import evaluate_segmentation, compute_iou, compute_pixel_accuracy, compute_dice_coefficient
from visualization import visualize_segmentation, save_comparison_grid


# dataset configs
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
    # get transforms
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
        raise ValueError("Unknown dataset: " + dataset_name)
    return dataset


def evaluate_model_on_dataset(model, dataset, dataset_name, num_classes, is_binary,
                             device='cuda', batch_size=1, save_results_dir=None,
                             visualize_samples=5):
    #main eval
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    all_metrics = []
    sample_results = []
    
    model.model.eval()
    # print(f"evaluating on {len(dataset)} samples")
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {dataset_name}")):
            # unpack batch
            if isinstance(batch, list) or isinstance(batch, tuple):
                if len(batch) == 3:
                    images, masks, img_names = batch
                else:
                    images, masks = batch
                    img_names = []
                    for i in range(len(images)):
                        img_names.append(f"sample_{idx}_{i}")
            else:
                images, masks = batch
                img_names = []
                for i in range(len(images)):
                    img_names.append(f"sample_{idx}_{i}")
            
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
                
                # get image name
                if isinstance(img_names, list) or isinstance(img_names, tuple):
                    if i < len(img_names):
                        img_name = img_names[i]
                    else:
                        img_name = f"sample_{idx}_{i}"
                else:
                    img_name = img_names
                
                # handle tuple names from dataloader
                if isinstance(img_name, tuple):
                    if len(img_name) > 0:
                        img_name = img_name[0]
                    else:
                        img_name = f"sample_{idx}_{i}"
                
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
    
    #aggregate metrics
    if is_binary:
        dice_vals = []
        iou_vals = []
        acc_vals = []
        for m in all_metrics:
            dice_vals.append(m['dice'])
            iou_vals.append(m['iou'])
            acc_vals.append(m['pixel_acc'])
        avg_metrics = {
            'dice': float(np.mean(dice_vals)),
            'iou': float(np.mean(iou_vals)),
            'pixel_acc': float(np.mean(acc_vals))
        }
    else:
        miou_vals = [m['mIoU'] for m in all_metrics]
        acc_vals = [m['pixel_acc'] for m in all_metrics]
        avg_metrics = {
            'mIoU': float(np.mean(miou_vals)),
            'pixel_acc': float(np.mean(acc_vals))
        }
        # Per-class IoU for multi-class datasets
        all_iou_per_class = []
        for m in all_metrics:
            all_iou_per_class.append(m['iou_per_class'])
        avg_iou_per_class = np.nanmean(all_iou_per_class, axis=0)
        iou_list = []
        for x in avg_iou_per_class.tolist():
            if not np.isnan(x):
                iou_list.append(float(x))
            else:
                iou_list.append(None)
        avg_metrics['iou_per_class'] = iou_list
    
    if save_results_dir:
        os.makedirs(save_results_dir, exist_ok=True)
        
        metrics_to_save = {}
        for key, value in avg_metrics.items():
            if isinstance(value, np.integer) or isinstance(value, np.floating):
                metrics_to_save[key] = float(value)
            elif isinstance(value, np.ndarray):
                temp_list = []
                for x in value.tolist():
                    if not np.isnan(x):
                        temp_list.append(float(x))
                    else:
                        temp_list.append(None)
                metrics_to_save[key] = temp_list
            elif isinstance(value, list):
                temp_list = []
                for x in value:
                    if isinstance(x, np.integer) or isinstance(x, np.floating):
                        temp_list.append(float(x))
                    else:
                        temp_list.append(x)
                metrics_to_save[key] = temp_list
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
    #different resolutions
    results = []
    
    for res in resolutions:
        print(f"\nevaluating {model_name} at resolution {res}x{res}")
        
        test_dataset = build_dataset_instance(
            dataset_name,
            root=dataset_config['root'],
            split=dataset_config['split'],
            size=(res, res),
            max_samples=dataset_config.get('max_samples')
        )
        
        # reload modelf or each resolution
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
    # best and worst cases for visualization
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    
    case_scores = []
    
    model.model.eval()
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc="finding good/bad cases")):
            if isinstance(batch, list) or isinstance(batch, tuple):
                if len(batch) >= 3:
                    images, masks, img_names = batch[0], batch[1], batch[2]
                elif len(batch) >= 2:
                    images, masks = batch[0], batch[1]
                    img_names = [f"sample_{idx}"]
                else:
                    raise ValueError(f"Unexpected batch format: expected at least 2 elements, got {len(batch)}")
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
            
            # get image name for this sample
            if isinstance(img_names, list) or isinstance(img_names, tuple):
                if len(img_names) > 0:
                    img_name = img_names[0]
                else:
                    img_name = f"sample_{idx}"
            else:
                img_name = img_names
            
            # sometimes dataloader returns tuple
            if isinstance(img_name, tuple):
                if len(img_name) > 0:
                    img_name = img_name[0]
                else:
                    img_name = f"sample_{idx}"
            
            # score is based on dataset type
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
    
    #sort by score and get top or bottom cases
    case_scores.sort(key=lambda x: x['score'], reverse=True)
    
    good_cases = case_scores[:num_cases]
    bad_cases = case_scores[-num_cases:]
    
    return good_cases, bad_cases


def ablation_study(dataset_name, dataset_config, device='cuda', save_dir=None, resolution=512):
    # test different model components
    ablation_results = []
    
    # model variants to test
    model_variants = [
        ('fcn_resnet50', 'FCN-ResNet50 (baseline)', False, False, 'ResNet50'),
        ('fcn_resnet101', 'FCN-ResNet101 (deeper backbone)', False, False, 'ResNet101'),
        ('deeplabv3_resnet50', 'DeepLabv3+-ResNet50 (ASPP+decoder, shallow)', True, True, 'ResNet50'),
        ('deeplabv3_resnet101', 'DeepLabv3+-ResNet101 (full, deep)', True, True, 'ResNet101'),
    ]
    
    print(f"\n{'='*60}")
    print(f"Ablation Study: {dataset_name.upper()}")
    print(f"{'='*60}")
    
    dataset = build_dataset_instance(
        dataset_name,
        root=dataset_config['root'],
        split=dataset_config['split'],
        size=(resolution, resolution),
        max_samples=dataset_config.get('max_samples')
    )
    
    for model_name, description, has_aspp, has_decoder, backbone in model_variants:
        print(f"\nTesting: {description}")
        print(f"  Backbone: {backbone}")
        print(f"  ASPP: {has_aspp}")
        print(f"  Decoder: {has_decoder}")
        
        try:
            model = load_model(model_name, device)
            
            metrics, _ = evaluate_model_on_dataset(
                model,
                dataset,
                dataset_name,
                dataset_config['num_classes'],
                dataset_config['is_binary'],
                device=device,
                save_results_dir=None,
                visualize_samples=0
            )
            
            result = {
                'model': model_name,
                'description': description,
                'backbone': backbone,
                'has_aspp': has_aspp,
                'has_decoder': has_decoder,
                'resolution': resolution,
            }
            
            # store metrics
            if dataset_config['is_binary']:
                result['dice'] = metrics['dice']
                result['iou'] = metrics['iou']
                result['pixel_acc'] = metrics['pixel_acc']
            else:
                result['mIoU'] = metrics['mIoU']
                result['pixel_acc'] = metrics['pixel_acc']
            
            ablation_results.append(result)
            print(f"  Results: {result}")
            
        except Exception as e:
            print(f"  Failed to load {model_name}: {e}")
            continue
    
    # save results
    df = pd.DataFrame(ablation_results)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, f'ablation_study_{dataset_name}.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nAblation results saved to {csv_path}")
    
    print(f"\n{'='*60}")
    print("Ablation Study Summary")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    
    return df


def run_all_experiments(save_dir='./results', device='cuda', resolutions=[256, 384, 512],
                        datasets=('pet', 'isic'), dataset_roots=None,
                        dataset_splits=None, dataset_max_samples=None):
    os.makedirs(save_dir, exist_ok=True)
    
    models = ['deeplabv3_resnet101', 'fcn_resnet50']
    dataset_roots = dataset_roots or {}
    dataset_splits = dataset_splits or {}
    dataset_max_samples = dataset_max_samples or {}
    
    #build dataset configs
    dataset_configs = []
    for name in datasets:
        if name not in DATASET_DEFAULTS:
            avail = list(DATASET_DEFAULTS.keys())
            raise ValueError("Unsupported dataset '" + name + "'. Available: " + str(avail))
        defaults = DATASET_DEFAULTS[name]
        #build config dict
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
            
            #build dataset at 512x512 for main eval
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
            
            # Find qualitative examples
            print(f"\nfinding good/bad cases for {model_name} on {dataset_name}...")
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
    
    # create summary csv file
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
    print("Ablation Study")
    print("=" * 60)
    
    # run ablation study at fixed resolution (512)
    for dataset_config in dataset_configs:
        dataset_name = dataset_config['name']
        ablation_dir = os.path.join(save_dir, 'ablation_study')
        ablation_study(
            dataset_name,
            dataset_config,
            device=device,
            save_dir=ablation_dir,
            resolution=512
        )
    
    print("\n" + "=" * 60)
    print("All experiments completed")
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


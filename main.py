import argparse
from experiments import run_all_experiments, DATASET_DEFAULTS


def main():
    parser = argparse.ArgumentParser(description='Run segmentation experiments')
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        default=['pet', 'isic'],
        choices=list(DATASET_DEFAULTS.keys()),
        help='Datasets to evaluate (choose any subset)'
    )
    parser.add_argument(
        '--save_dir', 
        type=str, 
        default='./results',
        help='Directory to save all results'
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for inference'
    )
    parser.add_argument(
        '--resolutions', 
        type=int, 
        nargs='+', 
        default=[256, 384, 512],
        help='Resolutions for hyperparameter analysis'
    )
    parser.add_argument(
        '--pet_root',
        type=str,
        default='./data/OxfordPets',
        help='Root directory for Oxford-IIIT Pet dataset'
    )
    parser.add_argument(
        '--pet_split',
        type=str,
        default='test',
        help='Split for Oxford-IIIT Pet (trainval/test)'
    )
    parser.add_argument(
        '--pet_max_samples',
        type=int,
        default=0,
        help='Optional max samples for Pet dataset (0 = all)'
    )
    parser.add_argument(
        '--isic_root',
        type=str,
        default='./data/ISIC2018',
        help='Root directory for ISIC 2018 dataset'
    )
    parser.add_argument(
        '--isic_split',
        type=str,
        default='test',
        help='Split for ISIC dataset (default: test)'
    )
    parser.add_argument(
        '--isic_max_samples',
        type=int,
        default=500,
        help='Maximum number of ISIC samples to evaluate'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Segmentation Experiments")
    print("=" * 70)
    print(f"Results will be saved to: {args.save_dir}")
    print(f"Device: {args.device}")
    print(f"Resolutions for hyperparameter analysis: {args.resolutions}")
    print(f"Datasets: {args.datasets}")
    print("=" * 70)

    dataset_roots = {
        'pet': args.pet_root,
        'isic': args.isic_root,
    }
    dataset_splits = {
        'pet': args.pet_split,
        'isic': args.isic_split,
    }
    dataset_max_samples = {
        'pet': None if args.pet_max_samples <= 0 else args.pet_max_samples,
        'isic': None if args.isic_max_samples <= 0 else args.isic_max_samples,
    }
    
    run_all_experiments(
        save_dir=args.save_dir,
        device=args.device,
        resolutions=args.resolutions,
        datasets=args.datasets,
        dataset_roots=dataset_roots,
        dataset_splits=dataset_splits,
        dataset_max_samples=dataset_max_samples,
    )


if __name__ == '__main__':
    main()


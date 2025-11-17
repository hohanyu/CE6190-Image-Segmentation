"""
Main entry point for running segmentation experiments
"""
import argparse
from experiments import run_all_experiments


def main():
    parser = argparse.ArgumentParser(
        description='Run segmentation experiments for Assignment 1',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
        '--voc_root',
        type=str,
        default='./data/VOC2012',
        help='Root directory for Pascal VOC 2012 dataset'
    )
    parser.add_argument(
        '--isic_root',
        type=str,
        default='./data/ISIC2018',
        help='Root directory for ISIC 2018 dataset'
    )
    parser.add_argument(
        '--isic_max_samples',
        type=int,
        default=500,
        help='Maximum number of ISIC samples to evaluate'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Segmentation Review Assignment - Experiment Runner")
    print("=" * 70)
    print(f"Results will be saved to: {args.save_dir}")
    print(f"Device: {args.device}")
    print(f"Resolutions for hyperparameter analysis: {args.resolutions}")
    print("=" * 70)
    
    run_all_experiments(
        save_dir=args.save_dir,
        device=args.device,
        resolutions=args.resolutions
    )


if __name__ == '__main__':
    main()


# Segmentation Review Assignment

This repository contains code for evaluating pretrained segmentation models (DeepLabv3+ and FCN-ResNet50) on multiple datasets (Oxford-IIIT Pet, ISIC 2018, Pascal VOC 2012).

## Project Structure

```
.
├── main.py                 # Main entry point
├── experiments.py          # Main experiment runner
├── models.py              # Model loading and inference
├── datasets.py            # Dataset loaders
├── metrics.py             # Evaluation metrics
├── visualization.py       # Visualization utilities
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Installation

1. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note**: If you don't have `pip` available, use `pip3` instead. On macOS with Homebrew, you may need to use a virtual environment to avoid system package conflicts.

3. Download datasets:
   - **Oxford-IIIT Pet**: Automatically downloaded via torchvision when first used (default in this repo)
   - **ISIC 2018**: Download from [ISIC Archive](https://challenge2018.isic-archive.com/) and extract to `./data/ISIC2018/`
     - Training images: `ISIC2018_Task1-2_Training_Input/`
     - Training masks: `ISIC2018_Task1_Training_GroundTruth/`
   - **Pascal VOC 2012 (optional)**: Auto-downloads via torchvision. If the official mirror is slow, download `VOCtrainval_11-May-2012.tar` manually and extract into `./data/VOC2012/`.

## Usage

### Run All Experiments

**Important**: Make sure your virtual environment is activated first:
```bash
source venv/bin/activate  # On macOS/Linux
```

Then run all experiments (evaluation, hyperparameter analysis, good/bad case finding). By default, both Oxford-IIIT Pet (in-domain) and ISIC 2018 (out-of-domain) are evaluated:

```bash
python main.py --save_dir ./results --device cuda
```

If you don't have CUDA available, use `--device cpu` instead.

### Command Line Arguments

- `--datasets`: List of datasets to evaluate (any subset of `pet`, `isic`, `voc`)
- `--save_dir`: Directory to save results (default: `./results`)
- `--device`: Device to use - `cuda` or `cpu` (default: `cuda`)
- `--resolutions`: Resolutions for hyperparameter analysis (default: `256 384 512`)
- `--pet_root`, `--pet_split`, `--pet_max_samples`: Settings for Oxford-IIIT Pet
- `--isic_root`, `--isic_split`, `--isic_max_samples`: Settings for ISIC 2018
- `--voc_root`, `--voc_split`, `--voc_max_samples`: Settings for Pascal VOC

### Example

```bash
# Evaluate Pet + ISIC on CPU with custom sample counts
python main.py \
    --datasets pet isic \
    --device cpu \
    --save_dir ./my_results \
    --pet_max_samples 2000 \
    --isic_max_samples 800
```

## Output Structure

After running experiments, results will be saved in the following structure:

```
results/
├── deeplabv3_resnet101/
│   ├── pet/
│   │   ├── pet_metrics.json
│   │   ├── visualizations/
│   │   └── cases/
│   ├── isic/
│   │   ├── isic_metrics.json
│   │   └── ...
│   └── voc/ (if enabled)
├── fcn_resnet50/ (same structure)
├── hyperparameter_analysis/
│   └── <model>_<dataset>_hyperparameter_analysis.csv
└── summary_results.csv
```

## Evaluation Metrics

The code computes the following metrics:

- **mIoU (Mean Intersection over Union)**: For multi-class datasets (Pet, VOC)
- **Pixel Accuracy**: Overall pixel classification accuracy
- **Dice Coefficient**: For binary segmentation (ISIC)
- **Per-class IoU**: Individual class performance (multi-class datasets)

## Models

- **DeepLabv3+ (ResNet-101)**: State-of-the-art segmentation model with ASPP and encoder-decoder architecture
- **FCN-ResNet50**: Fully Convolutional Network with ResNet-50 backbone

Both models are pretrained on COCO with VOC-style labels and used without fine-tuning.

## Datasets

- **Oxford-IIIT Pet** (default in-domain benchmark): 37 breeds, segmentation masks with background/pet/border classes. Auto-download.
- **ISIC 2018** (default out-of-domain benchmark): Binary lesion segmentation. Requires manual download.
- **Pascal VOC 2012** (optional in-domain benchmark): 21 semantic classes. Auto-download (large ~2GB tarball).

## Notes

- All models are used in zero-shot inference mode (no training or fine-tuning)
- Images are resized to specified resolutions and normalized using ImageNet statistics
- Results include quantitative metrics, visualizations, and analysis of good/bad cases

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision 0.15+
- See `requirements.txt` for full list


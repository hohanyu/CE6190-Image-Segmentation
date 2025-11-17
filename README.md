# Segmentation Review Assignment

This repository contains code for evaluating pretrained segmentation models (DeepLabv3+ and FCN-ResNet50) on Pascal VOC 2012 and ISIC 2018 datasets.

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
   - **Pascal VOC 2012**: The dataset will be automatically downloaded when you run the code (using torchvision's built-in downloader)
   - **ISIC 2018**: Download from [ISIC Archive](https://challenge2018.isic-archive.com/) and extract to `./data/ISIC2018/`
     - Training images: `ISIC2018_Task1-2_Training_Input/`
     - Training masks: `ISIC2018_Task1_Training_GroundTruth/`

## Usage

### Run All Experiments

**Important**: Make sure your virtual environment is activated first:
```bash
source venv/bin/activate  # On macOS/Linux
```

Then run all experiments (evaluation, hyperparameter analysis, good/bad case finding):

```bash
python main.py --save_dir ./results --device cuda
```

If you don't have CUDA available, use `--device cpu` instead.

### Command Line Arguments

- `--save_dir`: Directory to save results (default: `./results`)
- `--device`: Device to use - `cuda` or `cpu` (default: `cuda`)
- `--resolutions`: Resolutions for hyperparameter analysis (default: `256 384 512`)
- `--voc_root`: Root directory for VOC dataset (default: `./data/VOC2012`)
- `--isic_root`: Root directory for ISIC dataset (default: `./data/ISIC2018`)
- `--isic_max_samples`: Maximum number of ISIC samples (default: `500`)

### Example

```bash
# Run with custom settings
python main.py \
    --save_dir ./my_results \
    --device cuda \
    --resolutions 256 384 512 640 \
    --isic_max_samples 1000
```

## Output Structure

After running experiments, results will be saved in the following structure:

```
results/
├── deeplabv3_resnet101/
│   ├── voc/
│   │   ├── voc_metrics.json
│   │   ├── visualizations/
│   │   └── cases/
│   │       ├── good_case_*.png
│   │       └── bad_case_*.png
│   └── isic/
│       ├── isic_metrics.json
│       ├── visualizations/
│       └── cases/
├── fcn_resnet50/
│   └── (same structure)
├── hyperparameter_analysis/
│   ├── deeplabv3_resnet101_voc_hyperparameter_analysis.csv
│   ├── deeplabv3_resnet101_isic_hyperparameter_analysis.csv
│   ├── fcn_resnet50_voc_hyperparameter_analysis.csv
│   └── fcn_resnet50_isic_hyperparameter_analysis.csv
└── summary_results.csv
```

## Evaluation Metrics

The code computes the following metrics:

- **mIoU (Mean Intersection over Union)**: For multi-class segmentation (VOC)
- **Pixel Accuracy**: Overall pixel classification accuracy
- **Dice Coefficient**: For binary segmentation (ISIC)
- **Per-class IoU**: Individual class performance (VOC)

## Models

- **DeepLabv3+ (ResNet-101)**: State-of-the-art segmentation model with ASPP and encoder-decoder architecture
- **FCN-ResNet50**: Fully Convolutional Network with ResNet-50 backbone

Both models are pretrained on COCO with VOC-style labels and used without fine-tuning.

## Datasets

- **Pascal VOC 2012**: 21 classes (20 object classes + background), validation set (1,449 images)
- **ISIC 2018**: Binary segmentation (lesion vs background), subset of 500 test images

## Notes

- All models are used in zero-shot inference mode (no training or fine-tuning)
- Images are resized to specified resolutions and normalized using ImageNet statistics
- Results include quantitative metrics, visualizations, and analysis of good/bad cases

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision 0.15+
- See `requirements.txt` for full list


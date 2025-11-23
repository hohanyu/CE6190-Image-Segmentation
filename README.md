# Image Segmentation CE6190 Assignment 1

Evaluation code for pretrained segmentation models (DeepLabv3+ and FCN-ResNet50) on Pet and ISIC datasets. Just runs inference, no training.

## Installation

Create a virtual environment:
```bash
python3 -m venv venv
# macOS/Linux
source venv/bin/activate  
# Windows: venv\Scripts\activate
```

Install packages:
```bash
pip install -r requirements.txt
```

### Datasets

- **Oxford-IIIT Pet**: auto-downloads via torchvision when you first run it
- **ISIC 2018**: download from [ISIC Archive](https://challenge2018.isic-archive.com/) and extract to `./data/ISIC2018/`
  - Images: `ISIC2018_Task1-2_Training_Input/`
  - Masks: `ISIC2018_Task1_Training_GroundTruth/`

## Usage

Activate venv:
```bash
source venv/bin/activate
```

Run experiments (defaults to both Pet and ISIC):
```bash
python main.py --save_dir ./results --device cuda
```

Use `--device cpu` if no GPU.

### Arguments

Main:
- `--datasets`: `pet`, `isic`, or both (default: both)
- `--save_dir`: Where to save (default: `./results`)
- `--device`: `cuda` or `cpu` (default: `cuda`)
- `--resolutions`: For hyperparameter analysis, like `256 384 512`

Per dataset:
- `--pet_root`, `--pet_split`, `--pet_max_samples`: Pet settings
- `--isic_root`, `--isic_split`, `--isic_max_samples`: ISIC settings

Example:
```bash
python main.py \
    --datasets pet isic \
    --device cpu \
    --save_dir ./my_results \
    --pet_max_samples 2000 \
    --isic_max_samples 800
```

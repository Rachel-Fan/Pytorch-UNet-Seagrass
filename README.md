# 🌿 U-Net Seagrass — High-Resolution Drone Imagery Segmentation

This repository provides a **U-Net–based segmentation pipeline** for eelgrass mapping using high-resolution drone imagery (Alaska / California / Washington / Oregon / British Columbia / ).  
It supports **RGB**, **index channels**, and **custom multi-channel inputs** (e.g., RGB + Index + GLCM), and includes tiling, training, evaluation, and metric computation.

---

## 📂 Repository Structure

```text
unet-seagrass/
│
├── train/                          # Training scripts
│   ├── dataset/                    # Custom eelgrass dataset loader
│   ├── transforms/                 # Data augmentation
│   ├── work_dirs/                  # Logs, checkpoints, configs
│   └── train_unet.py               # Main U-Net training script
│
├── eval/                           # Evaluation + metrics
│   ├── eval_unet.py
│   └── metrics/                    # IoU, Dice, F-score, boundary IoU
│
├── utils/
│   ├── tiling/                     # Ortho/index 512×512 tile generator
│   ├── image_utils.py
│   └── augment.py
│
├── models/                         # U-Net architectures (SMP, custom UNet)
│   ├── unet_smp.py
│   └── unet_custom.py
│
├── splits/                         # train/valid/test .txt file lists
├── pretrained/                     # (Optional) pre-trained backbone weights
├── requirements.txt
└── README.md
```

---

## 📦 Installation

Tested on:

- Python 3.8–3.10  
- PyTorch 1.12–2.2  
- CUDA 11.8 / 12.x  

Install environment:

```bash
conda create -n unet python=3.10 -y
conda activate unet
pip install -r requirements.txt
```

(Recommended) Install segmentation-models-pytorch:

```bash
pip install segmentation-models-pytorch
```

---

## 📁 Dataset Format

Eelgrass tiles follow the 512×512 PNG structure:

```text
data/
├── BC/
│   ├── train/
│   │   ├── image/
│   │   └── mask/
│   ├── valid/
│   └── test/
├── OR/
├── WA/
└── AK/
```

Tile naming convention:

```
<site>_<region>_<year>_rowXX_colYY.png
```

Example:

```
BH_WA_19_row10_col50.png
```

Dataset list files (`train.txt`, `valid.txt`, `test.txt`) follow format:

```
BH_WA_19_row10_col50
BH_WA_19_row12_col33
...
```

(no extension)

---

## 🚀 Training

Basic U-Net (ResNet34 encoder):

```bash
python train/train_unet.py \
    --data-root /path/to/data \
    --split-root ./splits \
    --output ./train/work_dirs/unet_run1 \
    --encoder resnet34 \
    --epochs 40 \
    --batch-size 8
```

Multi-channel training (RGB + index):

```bash
python train/train_unet.py \
    --input-channels 4 \
    --modalities rgb index \
    --data-root /path/to/data \
    --output ./train/work_dirs/unet_4ch
```

Training outputs:

```
train/work_dirs/
    ├── checkpoints/
    ├── logs/
    └── config.json
```

---

## 🧪 Evaluation

Run evaluation on a trained checkpoint:

```bash
python eval/eval_unet.py \
    --data-root /path/to/data \
    --split ./splits/test.txt \
    --checkpoint ./train/work_dirs/unet_run1/best_model.pth \
    --output ./eval/results
```

Metrics include:

- IoU  
- Dice  
- Precision / Recall  
- Accuracy  
- Boundary IoU  
- Hausdorff Distance  

All results saved as CSV.

---

## 📊 Tiling Pipeline (Ortho → 512×512 Tiles)

Use the tiling tool to convert orthomosaics into dataset tiles:

```
utils/tiling/tile_pair.py
```

Outputs:

```
tiles/
├── image/
├── mask/ (if available)
└── manifest.csv
```

Supports:

- custom overlap  
- edge alignment  
- paired image + index extraction  

---

## 📝 TODO

- [ ] Add DDP multi-GPU training  
- [ ] Add mixed-precision training (AMP)  
- [ ] Upload pre-trained regional U-Net models  
- [ ] Add visualization notebook  
- [ ] Publish evaluation benchmark results  

---

## 📄 License

MIT License.  
Please cite this repository if used in your research.


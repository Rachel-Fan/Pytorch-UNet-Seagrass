# -*- coding: utf-8 -*-
"""
U-Net 训练脚本（零参数硬编码版，适配你的数据路径）
- 数据目录（已硬编码）：
    E:\Eelgrass_processed_images_2025\Alaska
      ├─ image\     (RGB 主图)
      ├─ index\     (mask，单通道，0..K-1 或 0/1)
      ├─ glcm\      (可选，单通道，额外特征)
      └─ splits\
          ├─ train.txt
          ├─ val.txt
          └─ test.txt
- 指标：Dice / mIoU / Acc / Prec / Rec / F1 / Boundary IoU / Boundary F1
- 可视化：混淆矩阵 PNG、PR 曲线 PNG（每个 epoch）
- 依赖：torch, torchvision, numpy, pillow, tqdm, matplotlib, scikit-learn, scipy
"""

import os
import csv
import logging
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 你的 UNet 在 unet/unet_mode.py 下
from unet.unet_model import UNet

# 独立评估模块（与本文件同目录）
from evaluate import eval_on_loader, save_confusion_matrix_figure, save_pr_curves_figure


# ==========================
#         配置区
# ==========================
BASE_DIR = Path(r"E:\Eelgrass_processed_images_2025\Alaska")
IMAGE_DIR = BASE_DIR / "image"
INDEX_DIR = BASE_DIR / "index"   # mask
GLCM_DIR  = BASE_DIR / "glcm"    # 可选
SPLITS_DIR = BASE_DIR / "splits"
OUT_DIR    = BASE_DIR / "checkpoints"

# 数据与模型设置
CLASSES: int = 1                # 1=二分类；K>1=多分类
BILINEAR: bool = False          # UNet 上采样：False=转置卷积；True=双线性
SCALE: float = 1.0              # 下采样比例，例如 0.5
EXTRA_MODE: Optional[str] = "append4"  # None | "append4" | "replace_red"
INCLUDE_BACKGROUND: bool = False       # 多分类时，边界/PR 宏平均是否包含背景(0类)

# 训练超参
EPOCHS: int = 20
BATCH_SIZE: int = 2
LR: float = 1e-4
WEIGHT_DECAY: float = 1e-8
MOMENTUM: float = 0.99
GRAD_CLIP: float = 1.0
WORKERS: int = 4
AMP: bool = True                # 混合精度

# 类不平衡（2选1）
POS_WEIGHT: float = 0.0         # 二分类专用；>1 前景稀少时可增大
CLASS_WEIGHTS: Optional[List[float]] = None  # 多分类专用，例如 [1.0, 2.0, 0.5]

# PR 曲线
PR_POINTS: int = 40             # 仅用于旧版 evaluate；现在用 sklearn 会自适应
# ==========================


# ==========================
#       数据集定义
# ==========================
class SegDataset(Dataset):
    """
    - images_dir: RGB 主图目录（image/）
    - masks_dir : 单通道 mask 目录（index/）
    - split_list_file: 列表文件（每行一个基名，不带扩展名）
    - dir_extra: 可选单通道目录（glcm/）
    - extra_mode: None | 'append4' | 'replace_red'
    """
    def __init__(self,
                 images_dir: Path,
                 masks_dir: Path,
                 split_list_file: Path,
                 scale: float = 1.0,
                 dir_extra: Optional[Path] = None,
                 extra_mode: Optional[str] = None,
                 valid_exts: Optional[List[str]] = None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.scale = scale
        self.dir_extra = Path(dir_extra) if dir_extra else None
        self.extra_mode = extra_mode
        self.valid_exts = set(e.lower() for e in (valid_exts or [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]))

        with open(split_list_file, "r", encoding="utf-8") as f:
            self.ids = [line.strip() for line in f if line.strip()]
        if not self.ids:
            raise RuntimeError(f"Split list is empty: {split_list_file}")

        # 扫描 mask 唯一值，映射到 0..K-1（保持与原始 BasicDataset 逻辑一致）
        vals = []
        for stem in self.ids:
            mpath = self._find_first(self.masks_dir, stem)
            m = np.asarray(Image.open(mpath).convert("L"))
            vals.append(np.unique(m))
        self.mask_values = sorted(np.unique(np.concatenate(vals)).tolist())

    def __len__(self):
        return len(self.ids)

    def _find_first(self, folder: Path, stem: str) -> Path:
        for ext in self.valid_exts:
            p = folder / f"{stem}{ext}"
            if p.exists():
                return p
        matches = list(folder.glob(f"{stem}.*"))
        if not matches:
            raise FileNotFoundError(f"File not found for {stem} in {folder}")
        return matches[0]

    @staticmethod
    def _resize(img: Image.Image, scale: float, is_mask: bool):
        if scale == 1.0:
            return img
        w, h = img.size
        newW, newH = int(w * scale), int(h * scale)
        assert newW > 0 and newH > 0
        return img.resize((newW, newH), Image.NEAREST if is_mask else Image.BILINEAR)

    def _pil_to_chw_float(self, pil_img: Image.Image) -> np.ndarray:
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        arr = np.asarray(pil_img).transpose(2, 0, 1).astype(np.float32)
        if (arr > 1).any():
            arr /= 255.0
        return arr

    def _mask_to_long(self, pil_mask: Image.Image) -> np.ndarray:
        m = np.asarray(pil_mask.convert("L"))
        h, w = m.shape
        out = np.zeros((h, w), dtype=np.int64)
        for i, v in enumerate(self.mask_values):
            out[m == v] = i
        return out

    def __getitem__(self, idx):
        stem = self.ids[idx]
        ip = self._find_first(self.images_dir, stem)
        mp = self._find_first(self.masks_dir, stem)

        img = Image.open(ip)
        mask = Image.open(mp)

        assert img.size == mask.size, f"Size mismatch for {stem}: {img.size} vs {mask.size}"

        img = self._resize(img, self.scale, is_mask=False)
        mask = self._resize(mask, self.scale, is_mask=True)

        img_arr = self._pil_to_chw_float(img)

        # glcm 作为单通道 extra
        if self.dir_extra is not None and self.extra_mode is not None:
            ep = self._find_first(self.dir_extra, stem)
            extra = Image.open(ep)
            if extra.mode != "L":
                extra = extra.convert("L")
            extra = self._resize(extra, self.scale, is_mask=False)
            extra_arr = np.asarray(extra).astype(np.float32)
            if (extra_arr > 1).any():
                extra_arr /= 255.0
            extra_arr = extra_arr[None, ...]
            if self.extra_mode == "replace_red":
                img_arr[0:1, ...] = extra_arr
            elif self.extra_mode == "append4":
                img_arr = np.concatenate([img_arr, extra_arr], axis=0)

        mask_arr = self._mask_to_long(mask)
        return {
            "image": torch.from_numpy(img_arr.copy()).float().contiguous(),
            "mask": torch.from_numpy(mask_arr.copy()).long().contiguous(),
        }


# ==========================
#        训练主程序
# ==========================
def build_loaders():
    dir_img = IMAGE_DIR
    dir_mask = INDEX_DIR
    dir_extra = GLCM_DIR if GLCM_DIR.exists() and EXTRA_MODE is not None else None
    dir_splits = SPLITS_DIR

    train_set = SegDataset(dir_img, dir_mask, dir_splits / "train.txt", scale=SCALE,
                           dir_extra=dir_extra, extra_mode=EXTRA_MODE)
    val_set   = SegDataset(dir_img, dir_mask, dir_splits / "val.txt",   scale=SCALE,
                           dir_extra=dir_extra, extra_mode=EXTRA_MODE)
    test_set  = SegDataset(dir_img, dir_mask, dir_splits / "test.txt",  scale=SCALE,
                           dir_extra=dir_extra, extra_mode=EXTRA_MODE)

    num_workers = min(WORKERS, os.cpu_count() or 0)
    loader_args = dict(batch_size=BATCH_SIZE, num_workers=num_workers,
                       pin_memory=True, persistent_workers=(num_workers > 0))
    return (
        DataLoader(train_set, shuffle=True,  **loader_args),
        DataLoader(val_set,   shuffle=False, drop_last=True,  **loader_args),
        DataLoader(test_set,  shuffle=False, drop_last=False, **loader_args),
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 构建数据
    train_loader, val_loader, test_loader = build_loaders()

    # 探测输入通道
    in_ch = next(iter(train_loader))["image"].shape[1]
    model = UNet(n_channels=in_ch, n_classes=CLASSES, bilinear=BILINEAR)
    model = model.to(memory_format=torch.channels_last).to(device)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
                                    momentum=MOMENTUM, foreach=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP)

    # Loss
    if CLASSES == 1:
        pos_w = torch.tensor([POS_WEIGHT], dtype=torch.float32, device=device) if POS_WEIGHT > 0 else None
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    else:
        class_w = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=device) if CLASS_WEIGHTS else None
        criterion = nn.CrossEntropyLoss(weight=class_w)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "metrics_log.csv"
    wrote_header = csv_path.exists()
    best_val = -1.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", unit="batch")
        running_loss = 0.0

        for batch in pbar:
            images = batch["image"].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            masks  = batch["mask"].to(device=device, dtype=torch.long)

            with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=AMP):
                logits = model(images)
                if CLASSES == 1:
                    loss = criterion(logits.squeeze(1), masks.float())
                else:
                    loss = criterion(logits, masks)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss_epoch = running_loss / max(1, len(train_loader))

        # —— 验证（完整指标 + 图件）——
        val_metrics, val_confmat, val_pr = eval_on_loader(
            model, val_loader, device,
            num_classes=CLASSES, amp=AMP,
            pr_points=PR_POINTS, include_background=INCLUDE_BACKGROUND
        )

        scheduler.step(val_metrics["dice"])
        logging.info(
            ("[Val] Dice={dice:.4f} | mIoU={miou:.4f} | Acc={acc:.4f} | P={prec:.4f} | R={rec:.4f} | F1={f1:.4f} | "
             "bIoU={b_iou:.4f} | bF1={b_f1:.4f} | LR={lr:.2e}")
            .format(lr=optimizer.param_groups[0]['lr'], **val_metrics)
        )

        # 图件：混淆矩阵 & PR 曲线
        save_confusion_matrix_figure(val_confmat, OUT_DIR / f"val_confmat_epoch{epoch}.png",
                                     class_names=[str(i) for i in range(CLASSES)])
        save_pr_curves_figure(val_pr, OUT_DIR / f"val_pr_epoch{epoch}.png")

        # 每轮写 CSV
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not wrote_header:
                w.writerow(["epoch","train_loss","val_dice","val_miou","val_acc","val_prec","val_rec","val_f1","val_bIoU","val_bF1"])
                wrote_header = True
            w.writerow([
                epoch, f"{train_loss_epoch:.6f}",
                f"{val_metrics['dice']:.6f}", f"{val_metrics['miou']:.6f}",
                f"{val_metrics['acc']:.6f}",  f"{val_metrics['prec']:.6f}",
                f"{val_metrics['rec']:.6f}",  f"{val_metrics['f1']:.6f}",
                f"{val_metrics['b_iou']:.6f}", f"{val_metrics['b_f1']:.6f}",
            ])

        # 保存
        torch.save(model.state_dict(), OUT_DIR / f"checkpoint_epoch{epoch}.pth")
        if val_metrics["dice"] > best_val:
            best_val = val_metrics["dice"]
            torch.save(model.state_dict(), OUT_DIR / "best.pth")
            logging.info(f"✅ New best (Dice={best_val:.4f}) saved to {OUT_DIR / 'best.pth'}")

    # —— Test 汇报一次（不必画图）——
    test_metrics, _, _ = eval_on_loader(
        model, test_loader, device,
        num_classes=CLASSES, amp=AMP,
        pr_points=PR_POINTS, include_background=INCLUDE_BACKGROUND
    )
    logging.info(
        "[Test] Dice={dice:.4f} | mIoU={miou:.4f} | Acc={acc:.4f} | P={prec:.4f} | R={rec:.4f} | F1={f1:.4f} | bIoU={b_iou:.4f} | bF1={b_f1:.4f}"
        .format(**test_metrics)
    )


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError:
        logging.error("OOM detected. Retrying after empty cache.")
        torch.cuda.empty_cache()
        main()

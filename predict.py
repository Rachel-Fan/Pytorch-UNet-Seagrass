# -*- coding: utf-8 -*-
"""
批量预测并保存：
- class mask（按类别 id，uint8 PNG）
- overlay（预测边界/区域叠加在原图上，便于质检）
- （可选）probability map（仅二分类）

模型：加载 checkpoints/best.pth
依赖：torch, numpy, pillow, tqdm
"""

import os
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from unet.unet_model import UNet

# 与 train.py 一致的路径与设置
BASE_DIR = Path(r"E:\Eelgrass_processed_images_2025\Alaska")
IMAGE_DIR = BASE_DIR / "image"
INDEX_DIR = BASE_DIR / "index"   # 仅用于 size/名称匹配，不用读取掩膜
GLCM_DIR  = BASE_DIR / "glcm"
SPLITS_DIR = BASE_DIR / "splits"
CKPT_PATH  = BASE_DIR / "checkpoints" / "best.pth"

# 输出
OUT_DIR = BASE_DIR / "predictions"   # 将在其下创建 split 子目录

# 选择预测哪个 split：'train' | 'val' | 'test'
SPLIT = "test"

# 模型/数据集设置（需与训练一致）
CLASSES: int = 1
BILINEAR: bool = False
SCALE: float = 1.0
EXTRA_MODE: Optional[str] = "append4"  # None | 'append4' | 'replace_red'
WORKERS: int = 4
BATCH_SIZE: int = 1   # 保存逐张；设 1 便于文件名对应
AMP: bool = True

# 保存哪些产物
SAVE_MASK: bool = True
SAVE_OVERLAY: bool = True
SAVE_PROB: bool = True   # 仅二分类有效
OVERLAY_ALPHA: float = 0.4


# ---------------- Dataset（与 train.py 同逻辑） ----------------
class SegDataset(Dataset):
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

    def __getitem__(self, idx):
        stem = self.ids[idx]
        ip = self._find_first(self.images_dir, stem)
        img = Image.open(ip)
        img_orig = img.copy()  # for overlay

        img = self._resize(img, SCALE, is_mask=False)
        img_arr = self._pil_to_chw_float(img)

        if GLCM_DIR.exists() and EXTRA_MODE is not None:
            ep = self._find_first(GLCM_DIR, stem)
            extra = Image.open(ep)
            if extra.mode != "L":
                extra = extra.convert("L")
            extra = self._resize(extra, SCALE, is_mask=False)
            extra_arr = np.asarray(extra).astype(np.float32)
            if (extra_arr > 1).any():
                extra_arr /= 255.0
            extra_arr = extra_arr[None, ...]
            if EXTRA_MODE == "replace_red":
                img_arr[0:1, ...] = extra_arr
            elif EXTRA_MODE == "append4":
                img_arr = np.concatenate([img_arr, extra_arr], axis=0)

        return {
            "image": torch.from_numpy(img_arr.copy()).float().contiguous(),
            "stem": stem,
            "img_orig": img_orig  # PIL
        }


# ----------------- 可视化与保存 -----------------
def _palette_k(k: int) -> np.ndarray:
    rng = np.random.default_rng(2025)
    pal = rng.integers(0, 256, size=(k, 3), dtype=np.uint8)
    pal[0] = np.array([0,0,0], dtype=np.uint8)  # 背景=黑色
    return pal

def save_mask_png(mask: np.ndarray, out_path: Path):
    """mask: HxW uint8（类别 id）"""
    Image.fromarray(mask.astype(np.uint8)).save(out_path)

def save_overlay(img_pil: Image.Image, mask: np.ndarray, out_path: Path, alpha: float = 0.4):
    img = np.asarray(img_pil.convert("RGB")).astype(np.uint8)
    if CLASSES == 1:
        # 二分类：前景绿色
        color = np.array([0,255,0], dtype=np.uint8)
        over = img.copy()
        over[mask==1] = (alpha*color + (1-alpha)*over[mask==1]).astype(np.uint8)
    else:
        pal = _palette_k(CLASSES)
        color_mask = pal[mask]  # HxWx3
        over = (alpha*color_mask + (1-alpha)*img).astype(np.uint8)
    Image.fromarray(over).save(out_path)

def save_prob_png(prob: np.ndarray, out_path: Path):
    """prob: HxW float32 [0,1] -> 0..255"""
    arr = np.clip(prob, 0, 1) * 255.0
    Image.fromarray(arr.astype(np.uint8)).save(out_path)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_file = SPLITS_DIR / f"{SPLIT}.txt"
    ds = SegDataset(IMAGE_DIR, INDEX_DIR, split_file, scale=SCALE, dir_extra=GLCM_DIR, extra_mode=EXTRA_MODE)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=min(WORKERS, os.cpu_count() or 0),
                        pin_memory=True, persistent_workers=(WORKERS>0))

    # 探测通道并构建模型
    in_ch = next(iter(loader))["image"].shape[1]
    model = UNet(n_channels=in_ch, n_classes=CLASSES, bilinear=BILINEAR).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    out_split_dir = OUT_DIR / SPLIT
    (out_split_dir / "mask").mkdir(parents=True, exist_ok=True)
    if SAVE_OVERLAY: (out_split_dir / "overlay").mkdir(parents=True, exist_ok=True)
    if SAVE_PROB and CLASSES == 1: (out_split_dir / "prob").mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Predict {SPLIT}", unit="img")
        for batch in pbar:
            x = batch["image"].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            stems = batch["stem"]
            imgs_orig = batch["img_orig"]

            with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=AMP):
                logits = model(x)

            for i in range(x.size(0)):
                stem = stems[i]
                img_pil = imgs_orig[i]

                if CLASSES == 1:
                    prob = torch.sigmoid(logits[i:i+1]).squeeze().cpu().numpy()
                    mask = (prob > 0.5).astype(np.uint8)
                    if SAVE_PROB:
                        save_prob_png(prob, out_split_dir / "prob" / f"{stem}.png")
                else:
                    pred = torch.argmax(logits[i], dim=0).cpu().numpy().astype(np.uint8)
                    mask = pred

                if SAVE_MASK:
                    save_mask_png(mask, out_split_dir / "mask" / f"{stem}.png")
                if SAVE_OVERLAY:
                    save_overlay(img_pil, mask, out_split_dir / "overlay" / f"{stem}.png", alpha=OVERLAY_ALPHA)

    print(f"✅ Done. Saved to: {out_split_dir.resolve()}")


if __name__ == "__main__":
    main()

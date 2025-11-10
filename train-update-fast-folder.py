# -*- coding: utf-8 -*-
"""
train_update_fast_accum_folders.py
— 稳定+提速训练（直接读取 train/valid/test 目录；RGB 或 4 通道；梯度累积；BCE+Dice）
目录结构：
  BASE/train/image, BASE/train/index, (可选) BASE/train/glcm
  BASE/valid/image, BASE/valid/index, (可选) BASE/valid/glcm
  BASE/test /image, BASE/test /index, (可选) BASE/test /glcm
"""

# —— 在 import torch 之前彻底禁用动态编译（避免 triton/inductor 报错）——
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLED"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

from pathlib import Path
from typing import Optional, Dict, List
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from unet.unet_model import UNet

# ===================== 硬编码路径 & 配置（按需修改） =====================
BASE_DIR = Path(r"D:\Eelgrass_Process_2025_Bo\DroneVision_Model_data\OR")

TR_IMG   = BASE_DIR / "train" / "image"
TR_MASK  = BASE_DIR / "train" / "index"
TR_GLCM  = BASE_DIR / "train" / "glcm"     # 可不存在

VA_IMG   = BASE_DIR / "valid" / "image"
VA_MASK  = BASE_DIR / "valid" / "index"
VA_GLCM  = BASE_DIR / "valid" / "glcm"     # 可不存在

TE_IMG   = BASE_DIR / "test"  / "image"
TE_MASK  = BASE_DIR / "test"  / "index"
TE_GLCM  = BASE_DIR / "test"  / "glcm"     # 可不存在；仅说明结构，用不到也无所谓

DIR_CKPT = BASE_DIR / "train1" / "checkpoints"  # 保存模型

# ===== 数据/训练设置 =====
IMG_SIZE    = 512                 # 可调 384/448/640 平衡速度与精度
EXTRA_MODE  = None                # None | 'append4' | 'replace_red'
CLASSES     = 1                   # 二分类
BILINEAR    = False

# 训练超参（稳定优先）
EPOCHS        = 10
BATCH_SIZE    = 4                 # 单次前向 batch
ACCUM_STEPS   = 4                 # 梯度累积 ⇒ 等效 batch = 4*4=16
LR            = 3e-5
WEIGHT_DECAY  = 1e-4
GRAD_CLIP     = 1.0

# DataLoader
NUM_WORKERS     = min(os.cpu_count() or 0, 8)
PREFETCH_FACTOR = 2
PIN_MEMORY      = True
PERSISTENT      = True

# AMP/内核/随机性
AMP_ENABLED  = False              # 先关，稳定优先；跑稳后可开 True
CUDNN_BENCH  = True
SEED         = 0

SAVE_EVERY_EPOCH = False

# ===================== 通用工具 =====================
VALID_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]

def strip_known_ext(name: str) -> str:
    lower = name.lower()
    for ext in VALID_EXTS:
        if lower.endswith(ext):
            return name[: -len(ext)]
    return name

def pil_to_chw_float(pil_img: Image.Image) -> np.ndarray:
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    arr = np.asarray(pil_img).transpose(2, 0, 1).astype(np.float32)
    if (arr > 1).any():
        arr /= 255.0
    return arr

def resize_longest_side(pil_img: Image.Image, target: int, is_mask: bool) -> Image.Image:
    w, h = pil_img.size
    if max(w, h) == target:
        return pil_img
    scale = float(target) / float(max(w, h))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return pil_img.resize((new_w, new_h), Image.NEAREST if is_mask else Image.BILINEAR)

def pad_to_square(pil_img: Image.Image, target: int, fill=0) -> Image.Image:
    w, h = pil_img.size
    if (w, h) == (target, target):
        return pil_img
    canvas = Image.new(pil_img.mode, (target, target), color=fill)
    canvas.paste(pil_img, (0, 0))
    return canvas

def find_by_stem(folder: Path, stem: str) -> Path:
    """在 folder 里按 stem 匹配文件（大小写/扩展名兼容）。"""
    folder = Path(folder)
    target = strip_known_ext(stem).lower()
    # 快路径
    for ext in VALID_EXTS:
        p = folder / f"{target}{ext}"
        if p.exists():
            return p
    # 遍历兜底
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in VALID_EXTS and p.stem.lower() == target:
            return p
    raise FileNotFoundError(f"not found: {folder}\\{stem}.*")

# ===================== 数据集（直接扫描文件夹） =====================
class SegFolderDataset(Dataset):
    """
    - 直接读取 <split>/image 下的所有图像文件
    - 掩膜到 <split>/index 中按同名配对
    - 可选 GLCM 到 <split>/glcm（append4/replace_red）
    - 训练/验证输出：image (CxSxS)，mask (SxS, long)，stem
    """
    def __init__(self, image_dir: Path, mask_dir: Path,
                 glcm_dir: Optional[Path] = None,
                 extra_mode: Optional[str] = None,
                 img_size: int = 512):
        self.image_dir = Path(image_dir)
        self.mask_dir  = Path(mask_dir)
        self.glcm_dir  = Path(glcm_dir) if (glcm_dir and Path(glcm_dir).exists()) else None
        self.extra_mode = extra_mode
        self.img_size  = img_size

        # 收集所有 image 文件
        files = []
        for p in self.image_dir.iterdir():
            if p.is_file() and p.suffix.lower() in VALID_EXTS:
                files.append(p)
        if not files:
            raise RuntimeError(f"No images found in {self.image_dir}")
        # stems
        self.items: List[str] = [p.stem for p in sorted(files)]

        # 快速验证 mask 是否存在（抽查）
        miss = 0
        for s in self.items[:min(200, len(self.items))]:
            try:
                _ = find_by_stem(self.mask_dir, s)
            except FileNotFoundError:
                miss += 1
        if miss > 0:
            print(f"[warn] first {min(200,len(self.items))} samples: {miss} masks missing in {self.mask_dir}. "
                  f"Missing samples将报错退出。")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        stem = self.items[idx]
        ip = find_by_stem(self.image_dir, stem)
        mp = find_by_stem(self.mask_dir,  stem)

        img0 = Image.open(ip).convert("RGB")
        msk0 = Image.open(mp)    # 不转 L，让映射在后续统一二值化

        # 预处理：等比缩放到 IMG_SIZE → 右/下 pad 成正方形
        img_p = pad_to_square(resize_longest_side(img0, self.img_size, is_mask=False), self.img_size, fill=0)
        msk_p = pad_to_square(resize_longest_side(msk0, self.img_size, is_mask=True),  self.img_size, fill=0)

        img_arr = pil_to_chw_float(img_p)        # CxSxS
        mask_arr = np.array(msk_p.convert("L"))  # SxS; 后续训练里会强制二值化

        # 可选：GLCM 注入
        if self.glcm_dir is not None and self.extra_mode is not None:
            gp = find_by_stem(self.glcm_dir, stem)
            g = Image.open(gp)
            if g.mode != "L":
                g = g.convert("L")
            g_p = pad_to_square(resize_longest_side(g, self.img_size, is_mask=False), self.img_size, fill=0)
            ga  = np.asarray(g_p).astype(np.float32)
            if (ga > 1).any():
                ga /= 255.0
            ga = ga[None, ...]  # 1xSxS
            if self.extra_mode == "replace_red":
                img_arr[0:1, ...] = ga
            elif self.extra_mode == "append4":
                img_arr = np.concatenate([img_arr, ga], axis=0)

        return {
            "image": torch.from_numpy(img_arr.copy()).float().contiguous(),
            "mask":  torch.from_numpy(mask_arr.copy()).long().contiguous(),
            "stem":  stem
        }

# ===================== 验证（Dice/mIoU，二分类） =====================
@torch.no_grad()
def quick_validate(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool = False):
    model.eval()
    dice_sum = 0.0
    miou_sum = 0.0
    n_batches = 0
    for batch in loader:
        x = batch["image"].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        y = batch["mask"].to(device=device, dtype=torch.long)
        y_bin = (y > 0).float()

        with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
            logits = model(x)
        prob = torch.sigmoid(logits).squeeze(1)

        inter = (prob * y_bin).sum(dim=(1,2))
        denom = prob.sum(dim=(1,2)) + y_bin.sum(dim=(1,2)) + 1e-6
        dice = (2*inter/denom).mean().item()

        pred = (prob > 0.5).long()
        inter_b = ((pred==1) & (y==1)).sum().float()
        union_b = ((pred==1) | (y==1)).sum().float() + 1e-6
        miou = (inter_b/union_b).item()

        dice_sum += dice
        miou_sum += miou
        n_batches += 1
    model.train()
    if n_batches == 0:
        return 0.0, 0.0
    return dice_sum/n_batches, miou_sum/n_batches

# ===================== 主程序 =====================
def main():
    # —— 内核 & 随机种子 ——
    if CUDNN_BENCH:
        torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    torch.manual_seed(SEED); np.random.seed(SEED)

    DIR_CKPT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"INFO Using device: {device}")

    # —— 构建数据集/加载器（直接指向 train/valid 文件夹） ——
    train_set = SegFolderDataset(
        TR_IMG, TR_MASK,
        glcm_dir=(TR_GLCM if EXTRA_MODE else None),
        extra_mode=EXTRA_MODE, img_size=IMG_SIZE
    )
    val_set = SegFolderDataset(
        VA_IMG, VA_MASK,
        glcm_dir=(VA_GLCM if EXTRA_MODE else None),
        extra_mode=EXTRA_MODE, img_size=IMG_SIZE
    )

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR, persistent_workers=PERSISTENT
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR, persistent_workers=PERSISTENT
    )

    # —— Model：动态探测输入通道（append4/replace_red 后可能是 4 通道） ——
    sample = next(iter(DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0)))
    in_ch = sample["image"].shape[1]
    model = UNet(n_channels=in_ch, n_classes=CLASSES, bilinear=BILINEAR).to(
        device=device, memory_format=torch.channels_last
    )

    # —— Loss & Optimizer（BCE + Dice，强制二值掩膜） ——
    if CLASSES != 1:
        raise ValueError("本脚本为二分类（CLASSES=1）。如需多分类请切换到 CrossEntropy 分支。")
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, verbose=True)
    scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu', enabled=AMP_ENABLED)

    best_dice = -1.0
    step_in_epoch = 0

    # —— Train Loop（梯度累积） ——
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", unit="batch")
        for batch in pbar:
            images = batch["image"].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            masks  = batch["mask"].to(device=device, dtype=torch.long)
            masks_bin = (masks > 0).float()

            with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=AMP_ENABLED):
                logits = model(images)  # (B,1,H,W)
                # BCE
                loss_bce = criterion(logits.squeeze(1), masks_bin)
                # Dice(soft)
                prob = torch.sigmoid(logits).squeeze(1)
                inter = (prob * masks_bin).sum(dim=(1,2))
                denom = prob.sum(dim=(1,2)) + masks_bin.sum(dim=(1,2)) + 1e-6
                dice = (2*inter/denom).mean()
                loss = loss_bce + (1.0 - dice)

            # 非有限保护
            if not torch.isfinite(loss):
                print("[warn] non-finite loss, skip batch")
                optimizer.zero_grad(set_to_none=True)
                continue

            # 梯度累积
            loss = loss / ACCUM_STEPS
            if AMP_ENABLED:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            step_in_epoch += 1
            if step_in_epoch % ACCUM_STEPS == 0:
                if AMP_ENABLED:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += float(loss.item()) * ACCUM_STEPS
            pbar.set_postfix(loss=f"{(loss.item()*ACCUM_STEPS):.4f}")

        # —— 收尾：最后不足 ACCUM_STEPS 的残梯度也 step 一次 ——
        if step_in_epoch % ACCUM_STEPS != 0:
            if AMP_ENABLED:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        step_in_epoch = 0

        # —— Validation ——
        val_dice, val_miou = quick_validate(model, val_loader, device, amp=AMP_ENABLED)
        scheduler.step(val_dice)
        print(f"Epoch {epoch}: train_loss={epoch_loss/len(train_loader):.4f} | val_dice={val_dice:.4f} | val_mIoU={val_miou:.4f} | lr={optimizer.param_groups[0]['lr']:.2e}")

        # —— Save ——
        if SAVE_EVERY_EPOCH:
            DIR_CKPT.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), DIR_CKPT / f"checkpoint_epoch{epoch}.pth")

        if val_dice > best_dice:
            best_dice = val_dice
            DIR_CKPT.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), DIR_CKPT / "best.pth")
            print(f"  ✅ New best! Saved to {DIR_CKPT / 'best.pth'} (val_dice={best_dice:.4f})")

if __name__ == "__main__":
    main()

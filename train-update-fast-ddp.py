# -*- coding: utf-8 -*-
"""
train_update_fast_ddp.py — 读取 train/valid/test 文件夹结构，支持单机多卡 DDP，梯度累积/缓存/AMP/二值分割
目录要求:
BASE_DIR/
  train/
    image/*.png|jpg|tif
    index/*.png|jpg|tif
  valid/
    image/
    index/
  test/           # 可选（此脚本训练时未使用）
    image/
    index/
可选：BASE_DIR/glcm/ 以及 BASE_DIR/cache/{image,index,glcm}/*.pt 用于加速
"""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLED"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
from PIL import Image
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

# 你的 UNet 实现
from unet.unet_model import UNet

# ===================== 硬编码路径 & 配置 =====================
# 可被环境变量 BASE_DIR 覆盖
BASE_DIR   = Path(os.environ.get("BASE_DIR", "/content/drive/MyDrive/Drone AI/DroneVision_Model_data/OR"))

DIR_TRAIN  = BASE_DIR / "train"
DIR_VALID  = BASE_DIR / "valid"     # 注意：你确认了文件夹名是 "valid"
DIR_TEST   = BASE_DIR / "test"      # 可选

DIR_IMG_TRAIN  = DIR_TRAIN / "image"
DIR_IDX_TRAIN  = DIR_TRAIN / "index"
DIR_IMG_VALID  = DIR_VALID / "image"
DIR_IDX_VALID  = DIR_VALID / "index"

DIR_GLCM       = BASE_DIR / "glcm"  # 可选：额外通道（单通道）
DIR_CACHE      = BASE_DIR / "cache" # 可选：离线缓存
DIR_CKPT       = BASE_DIR / "train_ddp" / "checkpoints"

# ===== 训练/数据设置 =====
IMG_SIZE        = 512
EXTRA_MODE      = None              # None | 'append4' | 'replace_red'  （使用 GLCM）
CLASSES         = 1                 # 二分类
BILINEAR        = False

EPOCHS        = 10
BATCH_SIZE    = 4                   # 单进程单步 batch
ACCUM_STEPS   = 4                   # 梯度累积 => 等效总 batch = BATCH_SIZE * ACCUM_STEPS * world_size
LR            = 3e-5
WEIGHT_DECAY  = 1e-4
GRAD_CLIP     = 1.0

NUM_WORKERS     = min(os.cpu_count() or 0, 8)
PREFETCH_FACTOR = 2
PIN_MEMORY      = True
PERSISTENT      = True

AMP_ENABLED  = False                # 先关 AMP，稳定后再开
CUDNN_BENCH  = True
SEED         = 0

SAVE_EVERY_EPOCH = False

# ===================== 工具函数 =====================
VALID_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]

def find_first(folder: Path, stem: str) -> Path:
    """在 folder 下按 stem（不带扩展或带扩展）寻找首个匹配文件（大小写不敏感）"""
    folder = Path(folder).resolve()
    target_stem = Path(stem).stem.lower()
    for ext in VALID_EXTS:
        p = (folder / f"{target_stem}{ext}").resolve()
        if p.exists():
            return p
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in VALID_EXTS and p.stem.lower() == target_stem:
            return p.resolve()
    raise FileNotFoundError(f"not found: {folder}/{stem}.*")

def pil_to_chw_float(pil_img: Image.Image) -> np.ndarray:
    """PIL -> CxHxW float[0,1]（RGB）"""
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
    new_w = int(round(w * scale)); new_h = int(round(h * scale))
    return pil_img.resize((new_w, new_h), Image.NEAREST if is_mask else Image.BILINEAR)

def pad_to_square(pil_img: Image.Image, target: int, fill=0) -> Image.Image:
    w, h = pil_img.size
    if (w, h) == (target, target):
        return pil_img
    canvas = Image.new(pil_img.mode, (target, target), color=fill)
    canvas.paste(pil_img, (0, 0))
    return canvas

def load_tensor(path: Path) -> torch.Tensor:
    """读取 cache .pt；兼容 dict({'tensor': ...}) 或直接 tensor"""
    obj = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(obj, dict) and "tensor" in obj:
        return obj["tensor"]
    if torch.is_tensor(obj):
        return obj
    raise TypeError(f"Unexpected object in {path}: {type(obj)}")

# ===================== DDP 辅助 =====================
def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0

def get_world_size():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1

def reduce_mean(t: torch.Tensor):
    """跨进程平均，返回各进程相同结果"""
    if not is_dist_avail_and_initialized():
        return t
    rt = t.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= get_world_size()
    return rt

# ===================== Dataset (folder-based) =====================
class SegFolderDataset(Dataset):
    """
    split_dir/
      image/*.png|jpg|tif
      index/*.png|jpg|tif
    掩膜匹配规则：
      1) index/下同名文件
      2) 或 <stem>_index.<ext>
    可选：EXTRA_MODE + GLCM（append4 或 replace_red）
    可选：cache/{image,index,glcm}/*.pt
    """
    def __init__(self,
                 split_dir: Path,
                 extra_dir: Optional[Path] = None,
                 extra_mode: Optional[str] = None,
                 img_size: int = 512,
                 cache_dir: Optional[Path] = None):
        self.split_dir = Path(split_dir)
        self.image_dir = self.split_dir / "image"
        self.index_dir = self.split_dir / "index"
        self.extra_dir = Path(extra_dir) if extra_dir else None
        self.extra_mode = extra_mode
        self.img_size = img_size

        assert self.image_dir.exists(), f"Missing: {self.image_dir}"
        assert self.index_dir.exists(), f"Missing: {self.index_dir}"

        all_imgs = [f for f in os.listdir(self.image_dir) if Path(f).suffix.lower() in VALID_EXTS]
        pairs: List[Tuple[Path, Path, str]] = []
        for fname in sorted(all_imgs):
            img_p = self.image_dir / fname
            stem = Path(fname).stem
            m1 = self.index_dir / fname
            m2 = self.index_dir / f"{stem}_index{Path(fname).suffix}"
            if m1.exists():
                pairs.append((img_p, m1, stem))
            elif m2.exists():
                pairs.append((img_p, m2, stem))
        if not pairs:
            raise RuntimeError(f"No (image, mask) pairs under {self.split_dir}")
        self.pairs = pairs

        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_cache = False
        if self.cache_dir and (self.cache_dir / "image").exists() and (self.cache_dir / "index").exists():
            if (self.extra_dir is not None and self.extra_mode is not None):
                self.use_cache = (self.cache_dir / "glcm").exists()
            else:
                self.use_cache = True

        self.mask_values = None
        if not self.use_cache:
            vals = []
            sample_n = min(500, len(self.pairs))
            for i in range(sample_n):
                _, mp, _ = self.pairs[i]
                vals.append(np.unique(np.asarray(Image.open(mp).convert("L"))))
            self.mask_values = sorted(np.unique(np.concatenate(vals)).tolist())

    def __len__(self):
        return len(self.pairs)

    def _mask_to_long(self, pil_mask: Image.Image) -> np.ndarray:
        m = np.asarray(pil_mask.convert("L"))
        out = np.zeros_like(m, dtype=np.int64)
        for i, v in enumerate(self.mask_values):
            out[m == v] = i
        return out

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_p, msk_p, stem = self.pairs[idx]

        # 缓存直读
        if self.use_cache:
            img_t = load_tensor(self.cache_dir / "image" / f"{stem}.pt")   # CxSxS
            msk_t = load_tensor(self.cache_dir / "index" / f"{stem}.pt")   # SxS
            if self.extra_dir is not None and self.extra_mode is not None:
                glcm_t = load_tensor(self.cache_dir / "glcm" / f"{stem}.pt")  # 1xSxS
                if self.extra_mode == "replace_red":
                    img_t[0:1, ...] = glcm_t
                elif self.extra_mode == "append4":
                    img_t = torch.cat([img_t, glcm_t], dim=0)
            return {"image": img_t.float().contiguous(),
                    "mask":  msk_t.long().contiguous(),
                    "stem":  stem}

        # 现场预处理
        img = Image.open(img_p).convert("RGB")
        msk = Image.open(msk_p)

        extra_arr = None
        if self.extra_dir is not None and self.extra_mode is not None:
            gp = find_first(self.extra_dir, stem)
            g = Image.open(gp).convert("L")
            g = pad_to_square(resize_longest_side(g, self.img_size, is_mask=False), self.img_size, fill=0)
            ga = np.asarray(g).astype(np.float32)
            if (ga > 1).any():
                ga /= 255.0
            extra_arr = ga[None, ...]  # 1xSxS

        img_prep = pad_to_square(resize_longest_side(img, self.img_size, is_mask=False), self.img_size, fill=0)
        msk_prep = pad_to_square(resize_longest_side(msk, self.img_size, is_mask=True),  self.img_size, fill=0)

        img_arr  = pil_to_chw_float(img_prep)
        mask_arr = self._mask_to_long(msk_prep)

        if extra_arr is not None and self.extra_mode is not None:
            if self.extra_mode == "replace_red":
                img_arr[0:1, ...] = extra_arr
            elif self.extra_mode == "append4":
                img_arr = np.concatenate([img_arr, extra_arr], axis=0)

        return {"image": torch.from_numpy(img_arr.copy()).float().contiguous(),
                "mask":  torch.from_numpy(mask_arr.copy()).long().contiguous(),
                "stem":  stem}

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

# ===================== 主程序（含 DDP 初始化） =====================
def main():
    # cuDNN
    if CUDNN_BENCH:
        torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    torch.manual_seed(SEED); np.random.seed(SEED)

    # ==== DDP init ====
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    use_ddp = (local_rank >= 0)
    if use_ddp:
        backend = os.environ.get("DIST_BACKEND", "nccl")  # Windows 可设 gloo
        dist.init_process_group(backend=backend, timeout=datetime.timedelta(seconds=1800))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if get_rank() == 0:
        print(f"[rank {get_rank()}] device={device} world_size={get_world_size()}")
        print(f"BASE_DIR = {BASE_DIR}")

    # 目录创建
    if get_rank() == 0:
        DIR_CKPT.mkdir(parents=True, exist_ok=True)

    # === 数据集 & 采样器 ===
    extra_needed = (DIR_GLCM.exists() and EXTRA_MODE is not None)
    use_cache = DIR_CACHE.exists() and (DIR_CACHE / "image").exists() and (DIR_CACHE / "index").exists()
    if extra_needed:
        use_cache = use_cache and (DIR_CACHE / "glcm").exists()

    train_set = SegFolderDataset(
        split_dir=DIR_TRAIN,
        extra_dir=DIR_GLCM if EXTRA_MODE else None,
        extra_mode=EXTRA_MODE,
        img_size=IMG_SIZE,
        cache_dir=DIR_CACHE if use_cache else None
    )
    val_set = SegFolderDataset(
        split_dir=DIR_VALID,
        extra_dir=DIR_GLCM if EXTRA_MODE else None,
        extra_mode=EXTRA_MODE,
        img_size=IMG_SIZE,
        cache_dir=DIR_CACHE if use_cache else None
    )

    if use_ddp:
        train_sampler = DistributedSampler(train_set, shuffle=True, drop_last=True)
        val_sampler   = DistributedSampler(val_set,   shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler   = None

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=PERSISTENT
    )
    val_loader = DataLoader(
        val_set,
        batch_size=max(1, BATCH_SIZE//2),
        shuffle=False,
        sampler=val_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=PERSISTENT
    )

    # === 模型 ===
    sample = next(iter(DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0)))
    in_ch = sample["image"].shape[1]
    model = UNet(n_channels=in_ch, n_classes=CLASSES, bilinear=BILINEAR).to(
        device=device, memory_format=torch.channels_last
    )

    if CLASSES != 1:
        raise ValueError("此脚本为二分类（CLASSES=1）。多分类请改 CrossEntropy 分支。")

    if use_ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False
        )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, verbose=(get_rank()==0))
    scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu', enabled=AMP_ENABLED)

    best_dice = -1.0
    step_in_epoch = 0

    # === 训练循环 ===
    for epoch in range(1, EPOCHS + 1):
        if use_ddp:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        iter_obj = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", unit="batch") if get_rank()==0 else train_loader
        for batch in iter_obj:
            images = batch["image"].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            masks  = batch["mask"].to(device=device, dtype=torch.long)
            masks_bin = (masks > 0).float()

            with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=AMP_ENABLED):
                logits = model(images)  # (B,1,H,W)
                loss_bce = criterion(logits.squeeze(1), masks_bin)
                prob = torch.sigmoid(logits).squeeze(1)
                inter = (prob * masks_bin).sum(dim=(1,2))
                denom = prob.sum(dim=(1,2)) + masks_bin.sum(dim=(1,2)) + 1e-6
                dice = (2*inter/denom).mean()
                loss = loss_bce + (1.0 - dice)

            if not torch.isfinite(loss):
                if get_rank()==0:
                    print("[warn] non-finite loss, skip batch")
                optimizer.zero_grad(set_to_none=True)
                continue

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
            if get_rank()==0 and isinstance(iter_obj, tqdm):
                iter_obj.set_postfix(loss=f"{(loss.item()*ACCUM_STEPS):.4f}")

        # 收尾：不足 ACCUM_STEPS 的残梯度再 step 一次
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

        # 验证 & 跨卡聚合
        val_dice, val_miou = quick_validate(model, val_loader, device, amp=AMP_ENABLED)
        val_dice_t = reduce_mean(torch.tensor(val_dice, device=device))
        val_miou_t = reduce_mean(torch.tensor(val_miou, device=device))
        val_dice = val_dice_t.item(); val_miou = val_miou_t.item()

        scheduler.step(val_dice)

        if get_rank()==0:
            print(f"Epoch {epoch}: "
                  f"train_loss={epoch_loss/len(train_loader):.4f} | "
                  f"val_dice={val_dice:.4f} | val_mIoU={val_miou:.4f} | "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")

        # 保存
        if SAVE_EVERY_EPOCH and get_rank()==0:
            DIR_CKPT.mkdir(parents=True, exist_ok=True)
            torch.save(
                (model.module.state_dict() if isinstance(model, DDP) else model.state_dict()),
                DIR_CKPT / f"checkpoint_epoch{epoch}.pth"
            )

        if get_rank()==0 and val_dice > best_dice:
            best_dice = val_dice
            DIR_CKPT.mkdir(parents=True, exist_ok=True)
            torch.save(
                (model.module.state_dict() if isinstance(model, DDP) else model.state_dict()),
                DIR_CKPT / "best.pth"
            )
            print(f"  ✅ New best! Saved to {DIR_CKPT / 'best.pth'} (val_dice={best_dice:.4f})")

    # 退出进程组
    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()

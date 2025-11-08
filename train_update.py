# -*- coding: utf-8 -*-
"""
train.py — 快速版（PA-SAM风格输入 + 离线缓存 + DataLoader优化）
路径硬编码到 E:\Eelgrass_processed_images_2025\Alaska
- 读取 splits/train.txt, splits/val.txt
- 若 BASE/cache/{image,index,glcm} 存在 -> 直接加载 .pt 张量（大幅提速）
- 默认 IMG_SIZE=768；可调 640 进一步提速
"""

# >>> put these lines at the very top of train_update.py, BEFORE importing torch <<<
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"      # 完全禁用 torch._dynamo
os.environ["TORCHINDUCTOR_DISABLED"] = "1"   # 禁用 Inductor / Triton 后端
# （可选）屏蔽动态编译的其它路径
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import os
from pathlib import Path
from typing import Optional, List, Dict
import numpy as np
from PIL import Image


import torch
try:
    import torch._dynamo as dynamo
    dynamo.config.suppress_errors = True  # 即使触发也静默回退
    dynamo.reset()
    dynamo.disable()                      # 彻底禁用 dynamo
except Exception:
    pass

import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from unet.unet_model import UNet

try:
    from utils.dice_score import dice_loss as ext_dice_loss
    HAS_EXT_DICE = True
except Exception:
    HAS_EXT_DICE = False


# ===================== 硬编码路径 & 配置 =====================
BASE_DIR   = Path(r"E:\Eelgrass_processed_images_2025\Alaska")
DIR_IMG    = BASE_DIR / "image"
DIR_MASK   = BASE_DIR / "index"
DIR_GLCM   = BASE_DIR / "glcm"               # 可选
DIR_SPLITS = BASE_DIR / "splits"
DIR_CKPT   = BASE_DIR / "rbg/train-run1/checkpoints"
DIR_CACHE  = BASE_DIR / "cache"              # 若存在将启用加速

# 数据/模型设置
IMG_SIZE   = 768                              # 之前 1024 -> 768（或 640 继续提速）
EXTRA_MODE: Optional[str] = "None"         # None | 'append4' | 'replace_red'
CLASSES    = 1                                # 1=二分类；>1=多分类
BILINEAR   = False
# ==== config ====
USE_COMPILE = False

# 训练超参
EPOCHS       = 20
BATCH_SIZE   = 8                              # 降低 IMG_SIZE 后可适当加大 Batch
LR           = 1e-4
WEIGHT_DECAY = 1e-8
MOMENTUM     = 0.99
GRAD_CLIP    = 1.0

# DataLoader
NUM_WORKERS     = min(os.cpu_count() or 0, 8)
PREFETCH_FACTOR = 4
PIN_MEMORY      = True
PERSISTENT      = True

# AMP & 内核加速
AMP_ENABLED = True
CUDNN_BENCH = True

# 其他
SAVE_EVERY_EPOCH = True
SEED = 0
# ===========================================================


# ----------------- 工具函数 -----------------
VALID_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]

def find_first(folder: Path, stem: str) -> Path:
    for ext in VALID_EXTS:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    m = list(folder.glob(stem + ".*"))
    if not m:
        raise FileNotFoundError(f"not found: {folder}/{stem}.*")
    return m[0]

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


# ----------------- Dataset（训练/验证用：定长输出，支持缓存） -----------------
class SegDataset(Dataset):
    """
    - 若 BASE/cache/{image,index,glcm} 存在 -> 直接加载 .pt
    - 否则：PA-SAM风格前处理（最长边缩放到 IMG_SIZE，右/下 pad 到方形）
    - extra_mode：None | 'append4' | 'replace_red'
    - 训练/验证仅返回定长 image/mask，避免 collate 报错
    """
    def __init__(self,
                 images_dir: Path,
                 masks_dir: Path,
                 split_list_file: Path,
                 dir_extra: Optional[Path] = None,
                 extra_mode: Optional[str] = None,
                 img_size: int = 768,
                 cache_dir: Optional[Path] = None):
        self.images_dir = Path(images_dir)
        self.masks_dir  = Path(masks_dir)
        self.dir_extra  = Path(dir_extra) if dir_extra else None
        self.extra_mode = extra_mode
        self.img_size   = img_size

        self.stems = [s.strip() for s in Path(split_list_file).read_text(encoding="utf-8").splitlines() if s.strip()]
        if not self.stems:
            raise RuntimeError(f"Split list is empty: {split_list_file}")

        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_cache = False
        if self.cache_dir and (self.cache_dir / "image").exists() and (self.cache_dir / "index").exists():
            # 若使用了 glcm 且需要 extra 通道，也要求 cache/glcm 存在
            if (self.dir_extra is not None and self.extra_mode is not None):
                self.use_cache = (self.cache_dir / "glcm").exists()
            else:
                self.use_cache = True

        # 非缓存时，需要构建 mask 值映射（应对标签不从 0/1 开始的情况）
        if not self.use_cache:
            vals = []
            for stem in self.stems[:min(500, len(self.stems))]:  # 加速初始化：采样扫描
                mp = find_first(self.masks_dir, stem)
                vals.append(np.unique(np.asarray(Image.open(mp).convert("L"))))
            self.mask_values = sorted(np.unique(np.concatenate(vals)).tolist())
        else:
            self.mask_values = None  # 缓存里已经是整齐的类别ID张量
        print(f"[SegDataset] use_cache={self.use_cache}  cache_dir={self.cache_dir}")


    def __len__(self):
        return len(self.stems)

    def _mask_to_long(self, pil_mask: Image.Image) -> np.ndarray:
        # 非缓存路径下使用
        m = np.asarray(pil_mask.convert("L"))
        h, w = m.shape
        out = np.zeros((h, w), dtype=np.int64)
        for i, v in enumerate(self.mask_values):
            out[m == v] = i
        return out

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        stem = self.stems[idx]

        # ---------- 1) 优先走缓存 ----------
        if self.use_cache:
            # img_t = torch.load(self.cache_dir / "image" / f"{stem}.pt")["tensor"]  # CxSxS (float)
            # msk_t = torch.load(self.cache_dir / "index" / f"{stem}.pt")["tensor"]  # SxS   (long/int64)
            img_t  = load_tensor(self.cache_dir / "image" / f"{stem}.pt")  # CxSxS (float)
            msk_t  = load_tensor(self.cache_dir / "index" / f"{stem}.pt")  # SxS   (long)

            if self.dir_extra is not None and self.extra_mode is not None:
                glcm_t = load_tensor(self.cache_dir / "glcm"  / f"{stem}.pt")  # 1xSxS (float)
                if self.extra_mode == "replace_red":
                    img_t[0:1, ...] = glcm_t
                elif self.extra_mode == "append4":
                    img_t = torch.cat([img_t, glcm_t], dim=0)
            return {
                "image": img_t.float().contiguous(),
                "mask":  msk_t.long().contiguous(),
                "stem":  stem
            }

        # ---------- 2) 非缓存：现场做 PA-SAM 风格前处理 ----------
        ip = find_first(self.images_dir, stem)
        mp = find_first(self.masks_dir,  stem)

        img = Image.open(ip).convert("RGB")
        msk = Image.open(mp)

        # GLCM 可选
        extra_arr = None
        if self.dir_extra is not None and self.extra_mode is not None:
            gp = find_first(self.dir_extra, stem)
            extra = Image.open(gp)
            if extra.mode != "L":
                extra = extra.convert("L")
            extra_rs  = resize_longest_side(extra, self.img_size, is_mask=False)
            extra_pad = pad_to_square(extra_rs, self.img_size, fill=0)
            ea = np.asarray(extra_pad).astype(np.float32)
            if (ea > 1).any():
                ea /= 255.0
            extra_arr = ea[None, ...]  # 1xSxS

        # SAM风格：img/mask resize → pad
        img_rs  = resize_longest_side(img,  self.img_size, is_mask=False)
        msk_rs  = resize_longest_side(msk,  self.img_size, is_mask=True)
        img_pad = pad_to_square(img_rs,  self.img_size, fill=0)
        msk_pad = pad_to_square(msk_rs,  self.img_size, fill=0)

        img_arr = pil_to_chw_float(img_pad)
        mask_arr = self._mask_to_long(msk_pad)

        # 4通道拼接/替换
        if extra_arr is not None and self.extra_mode is not None:
            if self.extra_mode == "replace_red":
                img_arr[0:1, ...] = extra_arr
            elif self.extra_mode == "append4":
                img_arr = np.concatenate([img_arr, extra_arr], axis=0)

        return {
            "image": torch.from_numpy(img_arr.copy()).float().contiguous(),
            "mask":  torch.from_numpy(mask_arr.copy()).long().contiguous(),
            "stem":  stem
        }


# ----------------- 简易验证（Dice/mIoU） -----------------
@torch.no_grad()
def quick_validate(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool = True, num_classes: int = 1):
    model.eval()
    dice_sum = 0.0
    miou_sum = 0.0
    n_batches = 0
    for batch in loader:
        x = batch["image"].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        y = batch["mask"].to(device=device, dtype=torch.long)
        with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
            logits = model(x)
        if num_classes > 1:
            prob = F.softmax(logits, dim=1).float()
            oh = F.one_hot(y, num_classes=num_classes).permute(0,3,1,2).float()
            inter = (prob*oh).sum(dim=(2,3))
            denom = prob.sum(dim=(2,3)) + oh.sum(dim=(2,3)) + 1e-6
            dice = (2*inter/denom).mean().item()
            pred = logits.argmax(dim=1)
            ious = []
            for c in range(num_classes):
                p = (pred==c); t=(y==c)
                inter_c = (p & t).sum().float()
                union_c = (p | t).sum().float() + 1e-6
                ious.append(inter_c/union_c)
            miou = torch.stack(ious).mean().item()
        else:
            prob = torch.sigmoid(logits).squeeze(1)
            inter = (prob*y.float()).sum(dim=(1,2))
            denom = prob.sum(dim=(1,2)) + y.float().sum(dim=(1,2)) + 1e-6
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

def load_tensor(path: Path):
    # PyTorch 2.4+ 支持 weights_only 这个安全开关
    obj = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(obj, dict) and "tensor" in obj:
        return obj["tensor"]
    elif torch.is_tensor(obj):
        return obj
    else:
        raise TypeError(f"Unexpected object in {path}: {type(obj)}")


def main():
    # ---- 内核 & 随机种子 ----
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

    # ---- DataLoaders（统一 val.txt 命名）----
    train_set = SegDataset(DIR_IMG, DIR_MASK, DIR_SPLITS / "train.txt",
                           dir_extra=DIR_GLCM if EXTRA_MODE else None,
                           extra_mode=EXTRA_MODE, img_size=IMG_SIZE,
                           cache_dir=DIR_CACHE if DIR_CACHE.exists() else None)
    val_set   = SegDataset(DIR_IMG, DIR_MASK, DIR_SPLITS / "val.txt",
                           dir_extra=DIR_GLCM if EXTRA_MODE else None,
                           extra_mode=EXTRA_MODE, img_size=IMG_SIZE,
                           cache_dir=DIR_CACHE if DIR_CACHE.exists() else None)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              prefetch_factor=PREFETCH_FACTOR, persistent_workers=PERSISTENT)
    val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              prefetch_factor=PREFETCH_FACTOR, persistent_workers=PERSISTENT)

    def estimate_pos_weight(stems, cache_dir: Path, sample_n: int = 1000):
        stems = stems[:min(sample_n, len(stems))]
        pos = 0
        neg = 0
        for s in stems:
            p = cache_dir / "index" / f"{s}.pt"
            obj = torch.load(p, map_location="cpu", weights_only=True)
            m = obj["tensor"] if isinstance(obj, dict) else obj
            pos += int((m == 1).sum())
            neg += int((m == 0).sum())
        pos = max(pos, 1); neg = max(neg, 1)
        # BCEWithLogitsLoss 的 pos_weight 定义为 负类/正类
        return torch.tensor([neg/pos], dtype=torch.float32)

    # 放在构建 train_loader / val_loader 之后
    pos_weight = None
    if CLASSES == 1 and DIR_CACHE.exists():
        pos_weight = estimate_pos_weight(train_set.stems, DIR_CACHE, sample_n=1000).to(device)
        print("pos_weight =", float(pos_weight))

    # ---- Model ----
    # 动态探测输入通道（看 append4/replace_red 后是否为 4 通道）
    sample = next(iter(DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0)))
    in_ch = sample["image"].shape[1]
    model = UNet(n_channels=in_ch, n_classes=CLASSES, bilinear=BILINEAR).to(device=device,
                                                                        memory_format=torch.channels_last)
    if USE_COMPILE:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("torch.compile enabled.")
        except Exception as e:
            print("torch.compile skipped:", e)
        
    # ---- Loss & Optimizer ----
    if CLASSES > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else nn.BCEWithLogitsLoss()


    optimizer = optim.RMSprop(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, verbose=True)
    scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu', enabled=AMP_ENABLED)

    best_dice = -1.0

    # ---- Train Loop ----
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", unit="batch")
        for batch in pbar:
            images = batch["image"].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            masks  = batch["mask"].to(device=device, dtype=torch.long)

            with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=AMP_ENABLED):
                logits = model(images)
                if CLASSES == 1:
                    loss = criterion(logits.squeeze(1), masks.float())
                    if HAS_EXT_DICE:
                        loss = loss + ext_dice_loss(torch.sigmoid(logits.squeeze(1)), masks.float(), multiclass=False)
                    else:
                        prob = torch.sigmoid(logits).squeeze(1)
                        inter = (prob*masks.float()).sum(dim=(1,2))
                        denom = prob.sum(dim=(1,2)) + masks.float().sum(dim=(1,2)) + 1e-6
                        dice = (2*inter/denom).mean()
                        loss = loss + (1.0 - dice)
                else:
                    loss = criterion(logits, masks)
                    if HAS_EXT_DICE:
                        onehot = F.one_hot(masks, num_classes=CLASSES).permute(0,3,1,2).float()
                        loss = loss + ext_dice_loss(F.softmax(logits, dim=1).float(), onehot, multiclass=True)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # ---- Validation ----
        val_dice, val_miou = quick_validate(model, val_loader, device, amp=AMP_ENABLED, num_classes=CLASSES)
        scheduler.step(val_dice)
        print(f"Epoch {epoch}: train_loss={epoch_loss/len(train_loader):.4f} | val_dice={val_dice:.4f} | val_mIoU={val_miou:.4f} | lr={optimizer.param_groups[0]['lr']:.2e}")

        # 保存
        if SAVE_EVERY_EPOCH:
            (DIR_CKPT / f"checkpoint_epoch{epoch}.pth").parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), DIR_CKPT / f"checkpoint_epoch{epoch}.pth")

        if val_dice > best_dice:
            best_dice = val_dice
            DIR_CKPT.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), DIR_CKPT / "best.pth")
            print(f"  ✅ New best! Saved to {DIR_CKPT / 'best.pth'} (val_dice={best_dice:.4f})")


if __name__ == "__main__":
    main()

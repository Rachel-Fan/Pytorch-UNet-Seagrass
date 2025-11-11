# -*- coding: utf-8 -*-
"""
train-update-fast-ddp.py — Full-dataset quick benchmark trainer (UNet, binary seg)
- Folder layout:
  BASE_DIR/
    train/{image,index}
    valid/{image,index}
    (test/{image,index} optional)
  Optional subsets: --train-split/--valid-split text files (one stem or stem.ext per line)

Key features:
- Fast I/O & compute: AMP + TF32 + channels_last + larger batch
- Class imbalance: BCEWithLogits(pos_weight) + Dice loss to boost recall
- OneCycleLR for quick convergence in ~10–12 epochs
- DDP-ready (single/multi-GPU); works fine on single GPU (LOCAL_RANK unset)
- Logging: training_log.csv; checkpoints every 2 epochs; best.pth
"""

import os, math, json, datetime, csv
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLED"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

# ================== CONFIG (edit defaults here or via CLI env) ==================
# Point to local SSD copy for speed (e.g., /content/BB after rsync)
BASE_DIR   = Path(os.environ.get("BASE_DIR", "/content/BB"))

# I/O
IMG_SIZE        = 512
EXTRA_MODE      = None        # None | 'append4' | 'replace_red' (GLCM not used here)
CLASSES         = 1           # binary
BILINEAR        = False

# Speed/quality knobs
EPOCHS        = int(os.environ.get("EPOCHS", 12))
BATCH_SIZE    = int(os.environ.get("BATCH_SIZE", 16))   # A100: try 16–32; lower if OOM
ACCUM_STEPS   = int(os.environ.get("ACCUM_STEPS", 1))   # if BATCH smaller, set e.g. 2
LR            = float(os.environ.get("LR", 3e-4))
WEIGHT_DECAY  = float(os.environ.get("WEIGHT_DECAY", 1e-4))
GRAD_CLIP     = float(os.environ.get("GRAD_CLIP", 1.0))

NUM_WORKERS     = int(os.environ.get("NUM_WORKERS", 8))
PREFETCH_FACTOR = int(os.environ.get("PREFETCH_FACTOR", 4))
PIN_MEMORY      = True
PERSISTENT      = True

AMP_ENABLED  = True
CUDNN_BENCH  = True
SEED         = 0

SAVE_EVERY_EPOCH = True  # save checkpoint every 2 epochs

# Subset txt (optional). If provided, restricts to listed stems.
TRAIN_SPLIT_TXT = os.environ.get("TRAIN_SPLIT", "") or None
VALID_SPLIT_TXT = os.environ.get("VALID_SPLIT", "") or None

# pos_weight from your earlier sample (≈7.8). Tune 8–12 if recall too low.
POS_WEIGHT_VAL = float(os.environ.get("POS_WEIGHT", 8.0))

# Checkpoint & logs
DIR_CKPT = BASE_DIR / "train_ddp" / "checkpoints"
LOG_CSV  = BASE_DIR / "train_ddp" / "training_log.csv"

# ================== Utils ==================
VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

def find_first(folder: Path, stem: str) -> Path:
    folder = Path(folder).resolve()
    target = Path(stem).stem.lower()
    for ext in VALID_EXTS:
        p = (folder / f"{target}{ext}").resolve()
        if p.exists(): return p
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in VALID_EXTS and p.stem.lower() == target:
            return p.resolve()
    raise FileNotFoundError(f"not found: {folder}/{stem}.*")

def pil_to_chw_float(pil_img: Image.Image) -> np.ndarray:
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    arr = np.asarray(pil_img).transpose(2, 0, 1).astype(np.float32)
    if (arr > 1).any(): arr /= 255.0
    return arr

def resize_longest_side(pil_img: Image.Image, target: int, is_mask: bool) -> Image.Image:
    w, h = pil_img.size
    if max(w, h) == target: return pil_img
    s = float(target) / float(max(w, h))
    nw, nh = int(round(w * s)), int(round(h * s))
    return pil_img.resize((nw, nh), Image.NEAREST if is_mask else Image.BILINEAR)

def pad_to_square(pil_img: Image.Image, target: int, fill=0) -> Image.Image:
    w, h = pil_img.size
    if (w, h) == (target, target): return pil_img
    canvas = Image.new(pil_img.mode, (target, target), color=fill)
    canvas.paste(pil_img, (0, 0))
    return canvas

# ================== DDP helpers ==================
def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0

def get_world_size():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1

# ================== Dataset (folder-based, optional subset) ==================
class SegFolderDataset(Dataset):
    def __init__(self, split_dir: Path, img_size: int, split_txt: Optional[str] = None):
        self.split_dir = Path(split_dir)
        self.image_dir = self.split_dir / "image"
        self.index_dir = self.split_dir / "index"
        self.img_size = img_size
        assert self.image_dir.exists(), f"Missing: {self.image_dir}"
        assert self.index_dir.exists(), f"Missing: {self.index_dir}"

        wanted = None
        if split_txt:
            stems = []
            for line in Path(split_txt).read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if not s: continue
                stems.append(Path(s).stem)
            wanted = set(stems)

        idx_map = {}
        for p in self.index_dir.iterdir():
            if p.is_file() and p.suffix.lower() in VALID_EXTS:
                idx_map[p.stem] = p

        pairs: List[Tuple[Path, Path, str]] = []
        for p in self.image_dir.iterdir():
            if not (p.is_file() and p.suffix.lower() in VALID_EXTS):
                continue
            stem = p.stem
            if wanted and (stem not in wanted):  # restrict to subset if given
                continue
            m1 = idx_map.get(stem, None)
            m2 = idx_map.get(stem + "_index", None) if m1 is None else None
            msk_p = m1 if m1 is not None else m2
            if msk_p is not None:
                pairs.append((p, msk_p, stem))

        if not pairs:
            raise RuntimeError(f"No (image,mask) pairs under {self.split_dir}")
        self.pairs = sorted(pairs, key=lambda x: x[2])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ip, mp, stem = self.pairs[idx]
        img = Image.open(ip).convert("RGB")
        msk = Image.open(mp)

        img_p = pad_to_square(resize_longest_side(img, IMG_SIZE, is_mask=False), IMG_SIZE, fill=0)
        msk_p = pad_to_square(resize_longest_side(msk, IMG_SIZE, is_mask=True),  IMG_SIZE, fill=0)

        x = torch.from_numpy(pil_to_chw_float(img_p).copy()).float()
        y = torch.from_numpy(np.asarray(msk_p.convert("L")).copy()).long()
        y = (y > 0).long()  # force binary

        return {"image": x, "mask": y, "stem": stem}

# ================== Model ==================
# Your UNet implementation
from unet.unet_model import UNet

# ================== Loss (BCE pos_weight + Dice) ==================
def build_loss(device, pos_weight_val: float = 8.0):
    pos_w = torch.tensor([pos_weight_val], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    def dice_loss_from_logits(logits, targets, eps=1e-6):
        probs = torch.sigmoid(logits).squeeze(1)
        targets = targets.float()
        inter = (probs * targets).sum(dim=(1,2))
        denom = probs.sum(dim=(1,2)) + targets.sum(dim=(1,2)) + eps
        dice = (2*inter / denom).mean()
        return 1.0 - dice

    def loss_fn(logits, targets):
        return bce(logits.squeeze(1), targets.float()) + 0.5 * dice_loss_from_logits(logits, targets)

    return loss_fn

# ================== Quick validate ==================
@torch.no_grad()
def quick_validate(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool = True):
    model.eval()
    dice_sum = 0.0
    miou_sum = 0.0
    n_batches = 0
    for batch in loader:
        x = batch["image"].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        y = batch["mask"].to(device=device, dtype=torch.long)
        yb = (y > 0).float()
        with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
            logits = model(x)
        prob = torch.sigmoid(logits).squeeze(1)
        inter = (prob * yb).sum(dim=(1,2))
        denom = prob.sum(dim=(1,2)) + yb.sum(dim=(1,2)) + 1e-6
        dice = (2*inter/denom).mean().item()

        pred = (prob > 0.5).long()
        tp = ((pred==1) & (y==1)).sum().float()
        fp = ((pred==1) & (y==0)).sum().float()
        fn = ((pred==0) & (y==1)).sum().float()
        union = tp + fp + fn
        miou = (tp/union).item() if union > 0 else 0.0

        dice_sum += dice
        miou_sum += miou
        n_batches += 1
    model.train()
    if n_batches == 0: return 0.0, 0.0
    return dice_sum/n_batches, miou_sum/n_batches

# ================== Main ==================
def main():
    # Seeds & backend
    torch.manual_seed(SEED); np.random.seed(SEED)
    if CUDNN_BENCH: torch.backends.cudnn.benchmark = True
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # DDP init
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    use_ddp = (local_rank >= 0)
    if use_ddp:
        backend = os.environ.get("DIST_BACKEND", "nccl")
        dist.init_process_group(backend=backend, timeout=datetime.timedelta(seconds=1800))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if get_rank()==0:
        print(f"[rank {get_rank()}] device={device} world_size={get_world_size()}")
        print(f"BASE_DIR = {BASE_DIR}")

    # Dirs
    DIR_CKPT.mkdir(parents=True, exist_ok=True)
    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    if get_rank()==0 and not LOG_CSV.exists():
        with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["epoch","train_loss","val_dice","val_mIoU","lr"])

    # Datasets
    dir_train = BASE_DIR / "train"
    dir_valid = BASE_DIR / "valid"
    train_set = SegFolderDataset(dir_train, IMG_SIZE, split_txt=TRAIN_SPLIT_TXT)
    valid_set = SegFolderDataset(dir_valid, IMG_SIZE, split_txt=VALID_SPLIT_TXT)

    if use_ddp:
        train_sampler = DistributedSampler(train_set, shuffle=True, drop_last=True)
        valid_sampler = DistributedSampler(valid_set, shuffle=False, drop_last=False)
    else:
        train_sampler = None
        valid_sampler = None

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR, persistent_workers=PERSISTENT
    )
    valid_loader = DataLoader(
        valid_set, batch_size=max(1, BATCH_SIZE//2),
        shuffle=False, sampler=valid_sampler,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR, persistent_workers=PERSISTENT
    )

    # Model
    sample = next(iter(DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0)))
    in_ch = sample["image"].shape[1]
    model = UNet(n_channels=in_ch, n_classes=CLASSES, bilinear=BILINEAR).to(
        device=device, memory_format=torch.channels_last
    )
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # Loss / Optim / Scheduler
    loss_fn = build_loss(device, pos_weight_val=POS_WEIGHT_VAL)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, ACCUM_STEPS)))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, epochs=EPOCHS, steps_per_epoch=steps_per_epoch,
        pct_start=0.15, div_factor=10.0, final_div_factor=100.0, anneal_strategy="cos"
    )
    scaler = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)

    best_dice = -1.0
    step_in_epoch = 0

    # Train loop
    for epoch in range(1, EPOCHS + 1):
        if use_ddp: train_sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        iterator = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", unit="batch") if get_rank()==0 else train_loader

        for batch in iterator:
            images = batch["image"].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            masks  = batch["mask"].to(device=device, dtype=torch.long)
            masks_bin = (masks > 0).float()

            with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=AMP_ENABLED):
                logits = model(images)
                loss = loss_fn(logits, masks_bin)

            # gradient accumulation
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

                scheduler.step()  # <<< update LR after each optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += float(loss.item()) * ACCUM_STEPS
            if get_rank()==0 and isinstance(iterator, tqdm):
                iterator.set_postfix(loss=f"{(loss.item()*ACCUM_STEPS):.4f}",
                                     lr=f"{scheduler.get_last_lr()[0]:.2e}")

        # Tail: if leftover grads < ACCUM_STEPS, still step once
        if step_in_epoch % ACCUM_STEPS != 0:
            if AMP_ENABLED:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        step_in_epoch = 0

        # Validation
        val_dice, val_miou = quick_validate(model, valid_loader, device, amp=AMP_ENABLED)
        if get_rank()==0:
            print(f"Epoch {epoch}: train_loss={epoch_loss/len(train_loader):.4f} | "
                  f"val_dice={val_dice:.4f} | val_mIoU={val_miou:.4f} | "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

            # CSV log
            with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([epoch, f"{epoch_loss/len(train_loader):.6f}",
                            f"{val_dice:.6f}", f"{val_miou:.6f}", f"{scheduler.get_last_lr()[0]:.6e}"])

        # Save checkpoints
        if get_rank()==0 and SAVE_EVERY_EPOCH and (epoch % 2 == 0):
            torch.save((model.module.state_dict() if isinstance(model, DDP) else model.state_dict()),
                       DIR_CKPT / f"checkpoint_epoch{epoch}.pth")

        if get_rank()==0 and val_dice > best_dice:
            best_dice = val_dice
            torch.save((model.module.state_dict() if isinstance(model, DDP) else model.state_dict()),
                       DIR_CKPT / "best.pth")
            print(f"  ✅ New best! Saved to {DIR_CKPT / 'best.pth'} (val_dice={best_dice:.4f})")

    if is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()

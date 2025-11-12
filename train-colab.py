# -*- coding: utf-8 -*-
"""
train_update_fast_ddp.py
- Reads train/valid/test folder structure with image/ + index/
- Optional split files (--train-split/--valid-split/--test-split) to restrict to subsets
- Single-GPU or multi-GPU via DDP (torchrun)
- Gradient accumulation, channels_last, optional GLCM extra channel
- Logs to CSV, checkpoints every 2 epochs, saves best.pth / best_model.pth / last.pth
"""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLED"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import argparse
import csv
import time
import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Set

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

# your UNet
from unet.unet_model import UNet

# ===================== defaults (can be overridden by CLI) =====================
DEFAULT_BASE_DIR = Path(os.environ.get("BASE_DIR", "/content/drive/MyDrive/Drone AI/DroneVision_Model_data/OR"))

IMG_SIZE        = 512
EXTRA_MODE      = None          # None | 'append4' | 'replace_red'
CLASSES         = 1             # binary
BILINEAR        = False

EPOCHS        = 20
BATCH_SIZE    = 4
ACCUM_STEPS   = 4               # effective batch = BATCH_SIZE * ACCUM_STEPS * world_size
LR            = 3e-4
WEIGHT_DECAY  = 1e-4
GRAD_CLIP     = 1.0

NUM_WORKERS     = min(os.cpu_count() or 0, 8)
PREFETCH_FACTOR = 2
PIN_MEMORY      = True
PERSISTENT      = True

AMP_ENABLED  = False            # start stable; turn on later if needed
CUDNN_BENCH  = True
SEED         = 0

SAVE_EVERY_EPOCH = False        # per-epoch saving (in addition to every 2 epochs below)

# ===================== helpers =====================
VALID_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]

def find_first(folder: Path, stem: str) -> Path:
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
    obj = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(obj, dict) and "tensor" in obj:
        return obj["tensor"]
    if torch.is_tensor(obj):
        return obj
    raise TypeError(f"Unexpected object in {path}: {type(obj)}")

def load_split_file(txt_path: Path) -> Set[str]:
    """Return a lowercased set of stems from a split file; ignores blanks and lines starting with #."""
    stems = set()
    lines = Path(txt_path).read_text(encoding="utf-8").splitlines()
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        stems.add(Path(ln).stem.lower())
    if not stems:
        raise RuntimeError(f"Empty split file: {txt_path}")
    return stems

# ===== DDP helpers =====
def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0

def get_world_size():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1

def reduce_mean(t: torch.Tensor):
    if not is_dist_avail_and_initialized():
        return t
    rt = t.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= get_world_size()
    return rt

# ===================== dataset =====================
class SegFolderDataset(Dataset):
    """
    split_dir/
      image/*.png|jpg|tif
      index/*.png|jpg|tif
    Mask pairing:
      1) index/same_name
      2) <stem>_index.<ext>
    Optional: EXTRA_MODE + GLCM; cache/{image,index,glcm}/*.pt
    Optional: stems_allow (subset filter)
    """
    def __init__(self,
                 split_dir: Path,
                 extra_dir: Optional[Path] = None,
                 extra_mode: Optional[str] = None,
                 img_size: int = 512,
                 cache_dir: Optional[Path] = None,
                 stems_allow: Optional[Set[str]] = None):
        self.split_dir = Path(split_dir)
        self.image_dir = self.split_dir / "image"
        self.index_dir = self.split_dir / "index"
        self.extra_dir = Path(extra_dir) if extra_dir else None
        self.extra_mode = extra_mode
        self.img_size = img_size
        self.stems_allow = {s.lower() for s in stems_allow} if stems_allow else None

        assert self.image_dir.exists(), f"Missing: {self.image_dir}"
        assert self.index_dir.exists(), f"Missing: {self.index_dir}"

        all_imgs = [f for f in os.listdir(self.image_dir) if Path(f).suffix.lower() in VALID_EXTS]
        pairs: List[Tuple[Path, Path, str]] = []
        for fname in sorted(all_imgs):
            img_p = self.image_dir / fname
            stem = Path(fname).stem
            if self.stems_allow and (stem.lower() not in self.stems_allow):
                continue
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
            extra_arr = ga[None, ...]

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

# ===================== validation (binary) =====================
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

# ===================== main =====================
def main():
    # cuDNN, seeds
    if CUDNN_BENCH:
        torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    torch.manual_seed(SEED); np.random.seed(SEED)

    # CLI
    parser = argparse.ArgumentParser(description="UNet training with optional split subset & DDP")
    parser.add_argument("--base-dir", type=str, default=str(DEFAULT_BASE_DIR), help="BASE_DIR root")
    parser.add_argument("--train-dir", type=str, default=None, help="override train dir (optional)")
    parser.add_argument("--valid-dir", type=str, default=None, help="override valid dir (optional)")
    parser.add_argument("--test-dir",  type=str, default=None, help="override test dir (optional)")
    parser.add_argument("--train-split", type=str, default=None, help="txt with image names for train subset")
    parser.add_argument("--valid-split", type=str, default=None, help="txt with image names for valid subset")
    parser.add_argument("--test-split",  type=str, default=None, help="txt with image names for test subset")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).expanduser()
    dir_train = Path(args.train_dir).expanduser() if args.train_dir else base_dir / "train"
    dir_valid = Path(args.valid_dir).expanduser() if args.valid_dir else base_dir / "valid"
    dir_test  = Path(args.test_dir).expanduser()  if args.test_dir  else base_dir / "test"

    stems_train = load_split_file(Path(args.train_split)) if args.train_split else None
    stems_valid = load_split_file(Path(args.valid_split)) if args.valid_split else None
    stems_test  = load_split_file(Path(args.test_split))  if args.test_split  else None

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

    if get_rank() == 0:
        print(f"[rank {get_rank()}] device={device} world_size={get_world_size()}")
        print(f"BASE_DIR = {base_dir}")

    # paths for ckpts & logs (under base_dir/train_ddp)
    dir_ckpt = base_dir / "train_ddp" / "checkpoints"
    dir_logs = base_dir / "train_ddp" / "logs"
    log_csv  = dir_logs / "training_log.csv"

    if get_rank() == 0:
        dir_ckpt.mkdir(parents=True, exist_ok=True)
        dir_logs.mkdir(parents=True, exist_ok=True)
        if not log_csv.exists():
            with open(log_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["epoch", "train_loss", "val_dice", "val_mIoU", "lr", "time_sec"])

    # datasets & loaders
    extra_dir = base_dir / "glcm" if EXTRA_MODE else None
    cache_dir = base_dir / "cache"
    extra_needed = (extra_dir is not None and extra_dir.exists())
    use_cache = cache_dir.exists() and (cache_dir / "image").exists() and (cache_dir / "index").exists()
    if extra_needed:
        use_cache = use_cache and (cache_dir / "glcm").exists()

    train_set = SegFolderDataset(
        split_dir=dir_train,
        extra_dir=extra_dir,
        extra_mode=EXTRA_MODE,
        img_size=IMG_SIZE,
        cache_dir=cache_dir if use_cache else None,
        stems_allow=stems_train
    )
    val_set = SegFolderDataset(
        split_dir=dir_valid,
        extra_dir=extra_dir,
        extra_mode=EXTRA_MODE,
        img_size=IMG_SIZE,
        cache_dir=cache_dir if use_cache else None,
        stems_allow=stems_valid
    )
    # (optional) test_set if you need later:
    # test_set = SegFolderDataset(split_dir=dir_test, ... stems_allow=stems_test)

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

    # model
    sample = next(iter(DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0)))
    in_ch = sample["image"].shape[1]
    model = UNet(n_channels=in_ch, n_classes=CLASSES, bilinear=BILINEAR).to(
        device=device, memory_format=torch.channels_last
    )

    if CLASSES != 1:
        raise ValueError("This script is for binary segmentation (CLASSES=1).")

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, verbose=(get_rank()==0))
    scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu', enabled=AMP_ENABLED)

    best_dice = -1.0
    step_in_epoch = 0

    # training loop
    for epoch in range(1, EPOCHS + 1):
        if use_ddp:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        it = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", unit="batch") if get_rank()==0 else train_loader
        for batch in it:
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
            if get_rank()==0 and isinstance(it, tqdm):
                it.set_postfix(loss=f"{(loss.item()*ACCUM_STEPS):.4f}")

        # tail step if remaining grads exist
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

        # validate
        val_dice, val_miou = quick_validate(model, val_loader, device, amp=AMP_ENABLED)
        val_dice_t = reduce_mean(torch.tensor(val_dice, device=device))
        val_miou_t = reduce_mean(torch.tensor(val_miou, device=device))
        val_dice = val_dice_t.item()
        val_miou = val_miou_t.item()

        scheduler.step(val_dice)

        if get_rank()==0:
            print(f"Epoch {epoch}: "
                  f"train_loss={epoch_loss/len(train_loader):.4f} | "
                  f"val_dice={val_dice:.4f} | val_mIoU={val_miou:.4f} | "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")

            # append CSV log
            with open(log_csv, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    epoch,
                    round(epoch_loss/len(train_loader), 6),
                    round(val_dice, 6),
                    round(val_miou, 6),
                    float(optimizer.param_groups[0]['lr']),
                    round(time.time(), 3)
                ])

        # save every 2 epochs
        if get_rank()==0 and (epoch % 2 == 0):
            ckpt_path = dir_ckpt / f"checkpoint_epoch{epoch}.pth"
            torch.save(
                (model.module.state_dict() if isinstance(model, DDP) else model.state_dict()),
                ckpt_path
            )

        # best model
        if get_rank()==0 and val_dice > best_dice:
            best_dice = val_dice
            state = (model.module.state_dict() if isinstance(model, DDP) else model.state_dict())
            torch.save(state, dir_ckpt / "best.pth")
            torch.save(state, dir_ckpt / "best_model.pth")
            print(f"  ✅ New best! Saved to {dir_ckpt/'best.pth'} (val_dice={best_dice:.4f})")

    # save last epoch
    if get_rank()==0:
        state = (model.module.state_dict() if isinstance(model, DDP) else model.state_dict())
        torch.save(state, dir_ckpt / "last.pth")

    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
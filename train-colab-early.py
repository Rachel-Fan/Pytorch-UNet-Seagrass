# -*- coding: utf-8 -*-
"""
train_colab_early.py — UNet binary segmentation trainer (Colab/CLI)
- EarlyStopping + best/last checkpoints + CSV logs
- CosineAnnealingLR with linear warmup
- AMP + channels_last + grad accumulation
- Read from BASE_DIR/{train,valid}/{image,index} or split txts (stems)
"""

import os, csv, math, time, random, argparse
from pathlib import Path
from typing import Optional, Set, Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

# speed knobs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# import your UNet (assumes repo root contains /unet)
import sys
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.append(str(Path(__file__).resolve().parent))
from unet.unet_model import UNet

VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# ---------------- utils ----------------
def seed_everything(seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def resize_longest_side(img: Image.Image, target: int, is_mask: bool) -> Image.Image:
    w, h = img.size
    if max(w, h) == target:
        return img
    s = float(target) / float(max(w, h))
    new_w, new_h = int(round(w * s)), int(round(h * s))
    return img.resize((new_w, new_h), Image.NEAREST if is_mask else Image.BILINEAR)

def pad_to_square(img: Image.Image, target: int, fill=0) -> Image.Image:
    w, h = img.size
    if (w, h) == (target, target):
        return img
    canvas = Image.new(img.mode, (target, target), color=fill)
    canvas.paste(img, (0, 0))
    return canvas

def pil_to_chw_float(img: Image.Image) -> np.ndarray:
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.asarray(img).transpose(2, 0, 1).astype(np.float32)
    if (arr > 1).any():
        arr /= 255.0
    return arr

def load_split_file(txt_path: Optional[str]) -> Optional[Set[str]]:
    if not txt_path:
        return None
    p = Path(txt_path)
    if not p.exists():
        raise FileNotFoundError(f"split file not found: {p}")
    stems: Set[str] = set()
    for ln in p.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if s and not s.startswith("#"):
            stems.add(Path(s).stem.lower())
    if not stems:
        raise RuntimeError(f"split file is empty: {p}")
    return stems

def estimate_pos_weight_quick(mask_dir: Path, sample_n=2000) -> float:
    paths = [q for q in mask_dir.iterdir() if q.is_file() and q.suffix.lower() in VALID_EXTS]
    if not paths:
        return 1.0
    if len(paths) > sample_n:
        random.seed(0); paths = random.sample(paths, sample_n)
    non_empty = 0
    for q in tqdm(paths, total=len(paths), desc="scan FG ratio", leave=False):
        try:
            im = Image.open(q).convert("L")
            if im.getbbox() is not None:
                non_empty += 1
        except Exception:
            pass
    r = non_empty / max(1, len(paths))  # approx prob(mask has FG)
    r = max(r, 1e-6)
    posw = max(2.0, min(30.0, 1.0 / r))
    print(f"[loss] pos_weight ≈ {posw:.2f} (r={r:.5f})")
    return float(posw)

# ---------------- dataset ----------------
class FolderDataset(Dataset):
    """BASE/split/{image,index}; mask pair by <stem> or <stem>_index; optional stems filter."""
    def __init__(self, base_dir: Path, split: str, img_size: int = 512,
                 stems_allow: Optional[Set[str]] = None, use_native_size: bool = False):
        self.base = Path(base_dir)
        self.split = split
        self.img_size = img_size
        self.use_native_size = use_native_size

        self.img_dir = self.base / split / "image"
        self.msk_dir = self.base / split / "index"
        assert self.img_dir.exists() and self.msk_dir.exists(), f"Missing {self.img_dir} or {self.msk_dir}"

        idx_map: Dict[str, Path] = {}
        for p in self.msk_dir.iterdir():
            if p.is_file() and p.suffix.lower() in VALID_EXTS:
                idx_map[p.stem.lower()] = p

        pairs: List[Tuple[Path, Path, str]] = []
        for p in self.img_dir.iterdir():
            if not (p.is_file() and p.suffix.lower() in VALID_EXTS):
                continue
            stem = p.stem
            if stems_allow and (stem.lower() not in stems_allow):
                continue
            m1 = idx_map.get(stem.lower(), None)
            m2 = idx_map.get((stem + "_index").lower(), None)
            mp = m1 if m1 is not None else m2
            if mp is not None:
                pairs.append((p, mp, stem))
        if not pairs:
            raise RuntimeError(f"No (image,mask) pairs under {self.base}/{self.split}")
        self.pairs = sorted(pairs, key=lambda x: x[2])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ip, mp, _ = self.pairs[i]
        img = Image.open(ip).convert("RGB")
        msk = Image.open(mp).convert("L")
        if self.use_native_size:
            img_p, msk_p = img, msk
        else:
            img_p = pad_to_square(resize_longest_side(img, self.img_size, is_mask=False), self.img_size, fill=0)
            msk_p = pad_to_square(resize_longest_side(msk, self.img_size, is_mask=True),  self.img_size, fill=0)
        x = torch.from_numpy(pil_to_chw_float(img_p).copy()).float()
        y = torch.from_numpy((np.asarray(msk_p) > 0).astype(np.int64)).long()
        return x, y

# ---------------- metrics ----------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device, thr=0.5):
    model.eval()
    dice_sum, miou_sum, n = 0.0, 0.0, 0
    for x, y in tqdm(loader, desc="Valid", leave=False):
        x = x.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        y = y.to(device=device, dtype=torch.long)
        logits = model(x).squeeze(1)      # (B,H,W)
        prob = torch.sigmoid(logits)
        pred = (prob > thr).long()

        inter = (prob * y.float()).sum(dim=(1,2))
        denom = prob.sum(dim=(1,2)) + y.float().sum(dim=(1,2)) + 1e-6
        dice = (2*inter/denom).mean().item()

        inter_b = ((pred==1) & (y==1)).sum(dim=(1,2)).float()
        union_b = ((pred==1) | (y==1)).sum(dim=(1,2)).float() + 1e-6
        miou = (inter_b/union_b).mean().item()

        dice_sum += dice; miou_sum += miou; n += 1
    model.train()
    return (dice_sum/n if n else 0.0), (miou_sum/n if n else 0.0)

# -------------- early stopping --------------
class EarlyStopping:
    def __init__(self, patience=6, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.bad = 0
        self.stop = False
    def step(self, score: float) -> bool:
        if self.best is None:
            self.best = score; return False
        if score < self.best + self.min_delta:
            self.bad += 1
            if self.bad >= self.patience:
                self.stop = True
        else:
            self.best = score; self.bad = 0
        return self.stop

# -------------- train one epoch --------------
from torch.amp import autocast, GradScaler
def train_one_epoch(model, loader, optimizer, criterion, scaler, device, accum_steps: int, scheduler=None):
    model.train()
    epoch_loss = 0.0
    accum = 0  # how many micro-batches accumulated since last step

    optimizer.zero_grad(set_to_none=True)

    use_amp = isinstance(scaler, torch.amp.grad_scaler.GradScaler) and scaler.is_enabled()

    it = tqdm(loader, desc="Train", leave=False)
    for step, (x, y) in enumerate(it):
        x = x.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        y = y.to(device=device, dtype=torch.long)
        yf = (y > 0).float()

        with torch.autocast(device_type=('cuda' if device.type == 'cuda' else 'cpu'), enabled=use_amp):
            logits = model(x).squeeze(1)            # (B,H,W)
            bce = criterion(logits, yf)
            prob = torch.sigmoid(logits)
            inter = (prob * yf).sum(dim=(1,2))
            denom = prob.sum(dim=(1,2)) + yf.sum(dim=(1,2)) + 1e-6
            dice = 1.0 - (2.0 * inter / denom).mean()  # dice loss
            loss = (bce + dice)

        # grad accumulation
        loss_to_scale = loss / accum_steps
        if use_amp:
            scaler.scale(loss_to_scale).backward()
        else:
            loss_to_scale.backward()

        accum += 1
        epoch_loss += float(loss.item())

        # do an optimizer step only when we hit accum_steps
        if accum == accum_steps:
            if use_amp:
                # Unscale once per optimizer step
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            accum = 0

        it.set_postfix(loss=f"{loss.item():.4f}")

    # tail flush (do this at most once, and only if we have leftover grads)
    if accum > 0:
        if use_amp:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # (optional) step a per-iteration scheduler here; for ReduceLROnPlateau keep it in eval()
    return epoch_loss / max(1, len(loader))

# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser(description="UNet training with EarlyStopping (Colab/CLI)")
    parser.add_argument("--base-dir", type=str, required=True, help="Dataset root, with train/valid/{image,index}")
    parser.add_argument("--out-root", type=str, required=True, help="Where to save checkpoints/ and logs/")
    parser.add_argument("--train-split", type=str, default=None, help="Optional split txt for train (stems)")
    parser.add_argument("--valid-split", type=str, default=None, help="Optional split txt for valid (stems)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accum-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--use-native-size", action="store_true", help="Use image native size instead of 512-square")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = Path(args.base_dir)
    out_root = Path(args.out_root)
    dir_ckpt = out_root / "checkpoints"
    dir_logs = out_root / "logs"
    dir_ckpt.mkdir(parents=True, exist_ok=True)
    dir_logs.mkdir(parents=True, exist_ok=True)
    log_csv = dir_logs / "training_log.csv"
    if not log_csv.exists():
        with open(log_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch","train_loss","val_dice","val_mIoU","lr","time_epoch_sec"])

    stems_train = load_split_file(args.train_split)
    stems_valid = load_split_file(args.valid_split)

    train_ds = FolderDataset(base, "train", img_size=args.img_size, stems_allow=stems_train, use_native_size=args.use_native_size)
    valid_ds = FolderDataset(base, "valid", img_size=args.img_size, stems_allow=stems_valid, use_native_size=args.use_native_size)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers>0))
    valid_dl = DataLoader(valid_ds, batch_size=max(1, args.batch_size//2), shuffle=False,
                          num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers>0))

    # model / loss / opt
    sample = next(iter(DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=0)))
    in_ch = sample[0].shape[1]
    model = UNet(n_channels=in_ch, n_classes=1, bilinear=False).to(device=device, memory_format=torch.channels_last)

    posw = estimate_pos_weight_quick(base/"train"/"index", sample_n=2000)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(posw, device=device))

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # cosine + warmup (step per optimizer.step)
    steps_per_epoch = math.ceil(len(train_dl) / max(1, args.accum_steps))
    warmup_steps = steps_per_epoch * max(0, args.warmup_epochs)
    total_cosine_steps = steps_per_epoch * max(0, args.epochs - args.warmup_epochs)
    cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_cosine_steps))

    if warmup_steps > 0:
        def lr_lambda(step):
            if step < warmup_steps:
                return max(1e-3, (step + 1)/float(warmup_steps))
            return 1.0
        warmup = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        class WarmupThenCosine:
            def __init__(self, warm, cos): self.warm, self.cos, self.t = warm, cos, 0
            def step(self):
                self.t += 1
                if self.t <= warmup_steps: self.warm.step()
                else: self.cos.step()
        scheduler = WarmupThenCosine(warmup, cosine)
    else:
        scheduler = cosine

    scaler = torch.amp.GradScaler('cuda' if device.type=='cuda' else 'cpu', enabled=True)
    stopper = EarlyStopping(patience=args.patience, min_delta=args.min_delta)

    print(f"[rank 0] device={device} | BASE_DIR={base}")
    print(f"[data] train={len(train_ds)}  valid={len(valid_ds)}  | img_size={'native' if args.use_native_size else args.img_size}")
    print(f"[train] bs={args.batch_size}, accum={args.accum_steps}, lr={args.lr}, wd={args.weight_decay}")

    best_dice = -1.0
    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_dl, optimizer, criterion, scaler, device, args.accum_steps, scheduler)
        val_dice, val_miou = evaluate(model, valid_dl, device, thr=0.5)
        t1 = time.time()

        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f} | val_dice={val_dice:.4f} | val_mIoU={val_miou:.4f} | lr={lr_now:.2e}")

        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch, round(train_loss,6), round(val_dice,6), round(val_miou,6), float(lr_now), round(t1-t0,3)])

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), dir_ckpt/"best.pth")
            torch.save(model.state_dict(), dir_ckpt/"best_model.pth")
            print(f"  ✅ New best! Saved to {dir_ckpt/'best.pth'} (val_dice={best_dice:.4f})")

        if stopper.step(val_dice):
            print(f"⏹️ Early stopping triggered (patience={args.patience}).")
            break

    torch.save(model.state_dict(), dir_ckpt/"last.pth")
    print(f"Saved last to {dir_ckpt/'last.pth'}")

if __name__ == "__main__":
    main()

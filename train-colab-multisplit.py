# train-colab-multisplit.py
# ==========================================================
# - 跨 {train,valid,test} 自动配对（同名或 *_index 掩膜）
# - 可传全局/局部 split 文本（stem 或 stem.ext 混合均可）
# - 可指定 out_dir（包含 checkpoints/ 与 logs/）
# - AMP(autocast + GradScaler)，梯度累积，channels_last，TF32
# - BCEWithLogits + Dice 复合损失；val 上 ReduceLROnPlateau
# - 保存 best.pth / last.pth 与 CSV 训练日志
# ==========================================================

import os, csv, time, json, math, random, argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, Iterable, List

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ---- 你的 UNet ----
from unet.unet_model import UNet

# ---------------- 基本设置（可被 CLI 覆盖） ----------------
DEFAULT_BASE_DIR = "/content/drive/MyDrive/Drone AI/DroneVision_Model_data/OR"
DEFAULT_OUT_DIR  = None  # 若为 None，则使用 BASE_DIR/train_ddp
IMG_SIZE   = None        # None 表示不缩放（你的 tiles 本来是 512x512 就很快）
EPOCHS     = 20
BATCH_SIZE = 8
ACCUM_STEPS = 2          # 有效 batch = BATCH_SIZE * ACCUM_STEPS
LR         = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP  = 1.0
NUM_WORKERS = 2          # Drive 读写友好值
PIN_MEMORY  = True
PERSISTENT  = False      # True 容易在 Colab 卡住，这里默认 False
SEED        = 0
AMP_ENABLED = True       # A100/L4 上建议开
BILINEAR    = False      # UNet 上采样方式

VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# ---------------- 工具函数 ----------------
def set_seed(seed: int = 0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def to_ch_last(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous(memory_format=torch.channels_last)

def safe_mean(a: List[float]) -> float:
    if not a: return 0.0
    arr = np.asarray(a, dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if arr.size else 0.0

# ---------------- 跨三子目录搜索 ----------------
def _find_with_ext(folder: Path, stem: str) -> Optional[Path]:
    for ext in VALID_EXTS:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None

def _find_mask(idx_dir: Path, stem: str) -> Optional[Path]:
    p = _find_with_ext(idx_dir, stem)
    if p is not None:
        return p
    return _find_with_ext(idx_dir, f"{stem}_index")

def _iter_image_stems(img_dir: Path) -> Iterable[Tuple[str, Path]]:
    for p in img_dir.iterdir():
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            yield p.stem, p

class MultiSplitDataset(Dataset):
    """
    给一个“全局名单”或局部名单（split_txt），在 base_dir/{train,valid,test}/{image,index} 里搜索并配对。
    返回：
        x: [3,H,W] (float, 0..1)
        y: [1,H,W] (float, 0/1)
    """
    def __init__(self,
                 base_dir: str | Path,
                 split_txt: Optional[str | Path] = None,
                 search_splits: Tuple[str, ...] = ("train","valid","test"),
                 img_size: Optional[int] = IMG_SIZE):
        self.base = Path(base_dir)
        self.search_splits = tuple(search_splits)
        self.img_size = img_size

        stem2pair: Dict[str, Tuple[Path, Path]] = {}
        missing_mask = 0

        for split in self.search_splits:
            img_dir = self.base / split / "image"
            idx_dir = self.base / split / "index"
            if not (img_dir.exists() and idx_dir.exists()):
                continue
            for stem, ip in _iter_image_stems(img_dir):
                if stem in stem2pair:
                    continue
                mp = _find_mask(idx_dir, stem)
                if mp is None:
                    missing_mask += 1
                    continue
                stem2pair[stem] = (ip, mp)

        if split_txt:
            raw = [ln.strip() for ln in Path(split_txt).read_text(encoding="utf-8").splitlines() if ln.strip()]
            wanted = [Path(r).stem for r in raw]
            pairs: List[Tuple[Path, Path, str]] = []
            miss = 0
            for s in wanted:
                if s in stem2pair:
                    ip, mp = stem2pair[s]
                    pairs.append((ip, mp, s))
                else:
                    miss += 1
            if miss > 0:
                print(f"[MultiSplit] ⚠️ {miss} names from list not found or unpaired in searched splits.")
        else:
            pairs = [(ip, mp, s) for s, (ip, mp) in stem2pair.items()]

        if not pairs:
            raise RuntimeError("No (image,mask) pairs found across splits.")
        if missing_mask > 0:
            print(f"[MultiSplit] ⚠️ skipped {missing_mask} images without a matching mask.")

        self.pairs = sorted(pairs, key=lambda t: t[2])
        self.t_img = transforms.ToTensor()  # [0,1]
        self.t_msk = transforms.ToTensor()

        # 预处理：可选 resize 到正方形
        if self.img_size is not None:
            self.resize_img = transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
            self.resize_msk = transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.NEAREST)
        else:
            self.resize_img = None
            self.resize_msk = None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        ip, mp, _ = self.pairs[idx]
        img = Image.open(ip).convert("RGB")
        msk = Image.open(mp).convert("L")
        if self.resize_img is not None:
            img = self.resize_img(img)
            msk = self.resize_msk(msk)
        x = self.t_img(img)               # [3,H,W]
        y = (self.t_msk(msk) > 0).float() # [1,H,W]
        return x, y

# ---------------- 评估与损失 ----------------
@torch.no_grad()
def evaluate(model, loader, device, thr: float = 0.5, amp: bool = True):
    model.eval()
    dice_sum, iou_sum, n = 0.0, 0.0, 0

    for x, y in tqdm(loader, desc="Valid", leave=False):
        # x: (B,C,H,W)
        # y: (B,H,W) 或 (B,1,H,W)
        x = x.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        y = y.to(device=device, dtype=torch.long)

        # 统一 y 维度到 (B,H,W)
        if y.dim() == 4 and y.size(1) == 1:
            y = y.squeeze(1)
        y = y.float()  # 后面要算概率/集合

        with torch.autocast(device_type='cuda', enabled=amp):
            logits = model(x)              # (B,1,H,W)
            logits = logits.squeeze(1)     # -> (B,H,W)
            prob   = torch.sigmoid(logits) # (B,H,W)

        # 二值化
        pred = (prob > thr).long()         # (B,H,W)

        # IoU
        inter = ((pred == 1) & (y == 1)).sum(dim=(1, 2)).float()
        union = ((pred == 1) | (y == 1)).sum(dim=(1, 2)).float().clamp(min=1e-6)
        iou   = (inter / union).mean().item()

        # Dice
        fp = ((pred == 1) & (y == 0)).sum(dim=(1, 2)).float()
        fn = ((pred == 0) & (y == 1)).sum(dim=(1, 2)).float()
        dice = (2 * inter / (2 * inter + fp + fn).clamp(min=1e-6)).mean().item()

        dice_sum += dice
        iou_sum  += iou
        n += 1

    return (dice_sum / n) if n else 0.0, (iou_sum / n) if n else 0.0

def make_loss(pos_weight: float | None, device: torch.device):
    if pos_weight is None:
        return nn.BCEWithLogitsLoss()
    pw = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pw)

def estimate_pos_weight(dataset: Dataset, max_batches: int = 64, batch_size: int = 8) -> float:
    """粗估正样本稀疏度：pos_weight ≈ (1-r)/r，r 为正像素比例；限制在 [3, 50]。"""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    pos_px = 0
    tot_px = 0
    with torch.no_grad():
        for i, (_, y) in enumerate(loader):
            pos_px += int(y.sum().item())
            tot_px += int(y.numel())
            if i + 1 >= max_batches:
                break
    r = (pos_px / max(tot_px, 1)) if tot_px > 0 else 0.01
    if r <= 0: r = 0.01
    pw = (1.0 - r) / r
    pw = float(max(3.0, min(50.0, pw)))
    print(f"[loss] pos_weight ≈ {pw:.2f} (r={r:.5f})")
    return pw

# ---------------- 训练一个 epoch ----------------
def train_one_epoch(model, loader, optimizer, criterion, scaler, device, accum_steps: int, scheduler=None):
    model.train()
    epoch_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    it = loader
    if isinstance(loader, DataLoader):
        from tqdm import tqdm
        it = tqdm(loader, desc="Train", leave=False)

    for step, (x, y) in enumerate(it):
        x = to_ch_last(x.to(device, non_blocking=True))
        y = y.to(device, non_blocking=True).squeeze(1)  # [B,H,W]
        with torch.autocast(device_type=device.type, enabled=AMP_ENABLED):
            logits = model(x).squeeze(1)     # [B,H,W]
            bce = criterion(logits, y)
            prob = torch.sigmoid(logits)
            inter = (prob * y).sum(dim=(1,2))
            denom = prob.sum(dim=(1,2)) + y.sum(dim=(1,2)) + 1e-6
            dice = 1.0 - (2*inter/denom).mean()
            loss = bce + dice

        loss = loss / accum_steps
        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        epoch_loss += float(loss.item()) * accum_steps

    return epoch_loss / max(1, len(loader))

# ---------------- 主函数 ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR)
    parser.add_argument("--train-split", type=str, default=None, help="全局/局部名单（可选）")
    parser.add_argument("--valid-split", type=str, default=None, help="全局/局部名单（可选）")
    parser.add_argument("--search-splits", type=str, nargs="+", default=["train","valid","test"])
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR, help="保存目录（含 checkpoints/、logs/）")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--accum-steps", type=int, default=ACCUM_STEPS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--grad-clip", type=float, default=GRAD_CLIP)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE if IMG_SIZE is not None else -1, help="-1 表示不缩放")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--pos-weight", type=float, default=-1.0, help="-1 表示自动估计")
    args = parser.parse_args()

    # 设备&加速
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    base_dir = Path(args.base_dir)
    out_root = Path(args.out_dir) if args.out_dir else (base_dir / "train_ddp")
    ckpt_dir = out_root / "checkpoints"
    logs_dir = out_root / "logs"
    ensure_dir(ckpt_dir); ensure_dir(logs_dir)

    log_csv = logs_dir / "training_log.csv"
    if not log_csv.exists():
        with open(log_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch","train_loss","val_dice","val_mIoU","lr","time_sec"])

    img_size = None if args.img_size in (-1, 0) else int(args.img_size)

    # 数据
    train_ds = MultiSplitDataset(base_dir, split_txt=args.train_split,
                                 search_splits=tuple(args.search_splits),
                                 img_size=img_size)
    valid_ds = MultiSplitDataset(base_dir, split_txt=args.valid_split,
                                 search_splits=tuple(args.search_splits),
                                 img_size=img_size)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=PIN_MEMORY,
                          persistent_workers=PERSISTENT)
    valid_dl = DataLoader(valid_ds, batch_size=max(1, args.batch_size//2), shuffle=False,
                          num_workers=min(2, args.num_workers), pin_memory=PIN_MEMORY,
                          persistent_workers=False)

    # 模型
    model = UNet(n_channels=3, n_classes=1, bilinear=BILINEAR).to(device)
    model = model.to(memory_format=torch.channels_last)

    # pos_weight
    if args.pos_weight is not None and args.pos_weight > 0:
        pos_w = args.pos_weight
        print(f"[loss] pos_weight = {pos_w:.2f} (from args)")
    else:
        pos_w = estimate_pos_weight(train_ds, max_batches=64, batch_size=8)

    criterion = make_loss(pos_w, device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3)  # 无 verbose 参数
    scaler = torch.amp.GradScaler(device.type, enabled=AMP_ENABLED)

    best_dice = -1.0
    print(f"[rank 0] device={device} | BASE_DIR={base_dir}")
    print(f"[data] train={len(train_ds)}  valid={len(valid_ds)}  | img_size={img_size or 'native'}")
    print(f"[train] bs={args.batch_size}, accum={args.accum_steps}, lr={args.lr}, wd={args.weight_decay}")

    # 训练
    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_dl, optimizer, criterion, scaler, device, args.accum_steps, scheduler=None)
        val_dice, val_miou = evaluate(model, valid_dl, device, thr=0.5)
        scheduler.step(val_dice)

        # 日志
        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{args.epochs}: "
              f"train_loss={train_loss:.4f} | val_dice={val_dice:.4f} | val_mIoU={val_miou:.4f} | lr={lr_now:.2e}")

        with open(log_csv, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([epoch, round(train_loss,6), round(val_dice,6), round(val_miou,6), float(lr_now), round(time.time(),3)])

        # 保存
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), ckpt_dir / "best.pth")
            torch.save(model.state_dict(), ckpt_dir / "best_model.pth")
            print(f"  ✅ New best! Saved → {ckpt_dir/'best.pth'} (val_dice={best_dice:.4f})")

        if epoch % 2 == 0:
            torch.save(model.state_dict(), ckpt_dir / f"checkpoint_epoch{epoch}.pth")

    # last
    torch.save(model.state_dict(), ckpt_dir / "last.pth")
    print("Done.")

if __name__ == "__main__":
    main()

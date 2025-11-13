#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval-metrics-pretrained.py — Evaluate SMP Unet(resnet34, imagenet) baseline
Outputs:
  1) CSV per-image metrics
  2) JSON summary with:
     average_iou, average_boundary_iou, average_accuracy,
     average_dice, average_precision, average_recall
Usage example:
  python eval-metrics-pretrained.py \
    --base-dir "/content/dataset/BB" \
    --split test \
    --out-dir "/content/drive/MyDrive/Drone AI/ModelRuns/UNet/BB/pretrained_eval" \
    --batch-size 32 --img-size 512
"""

import os, csv, json, time, argparse
from pathlib import Path
from typing import List, Optional
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# Helpers
# ----------------------------
VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
EPS = 1e-6

def pil_resize_pad(im: Image.Image, size: int, is_mask: bool):
    w, h = im.size
    if max(w, h) != size:
        s = float(size) / float(max(w, h))
        nw, nh = int(round(w*s)), int(round(h*s))
        im = im.resize((nw, nh), Image.NEAREST if is_mask else Image.BILINEAR)
    if im.size != (size, size):
        canvas = Image.new(im.mode, (size, size), 0)
        canvas.paste(im, (0, 0))
        im = canvas
    return im

def to_tensor_rgb(im: Image.Image):
    if im.mode != "RGB": im = im.convert("RGB")
    arr = np.asarray(im, dtype=np.float32)
    if arr.max() > 1.0: arr /= 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1))  # CxHxW

def load_split_stems(txt_path: Path) -> List[str]:
    return [Path(line.strip()).stem for line in txt_path.read_text(encoding="utf-8").splitlines() if line.strip()]

def match_mask(idx_dir: Path, img_path: Path):
    m1 = idx_dir / img_path.name
    if m1.exists(): return m1
    m2 = idx_dir / f"{img_path.stem}_index{img_path.suffix}"
    if m2.exists(): return m2
    return None

# ----------------------------
# Dataset
# ----------------------------
class EvalDataset(Dataset):
    def __init__(self, base_dir: Path, split: str = "test", img_size: int = 512, split_txt: Optional[Path] = None):
        self.img_size = img_size
        img_dir = base_dir / split / "image"
        idx_dir = base_dir / split / "index"
        assert img_dir.exists(), f"Missing: {img_dir}"
        assert idx_dir.exists(), f"Missing: {idx_dir}"

        imgs = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]
        if split_txt:
            stems = set(load_split_stems(split_txt))
            imgs = [p for p in imgs if p.stem in stems]

        pairs = []
        for ip in imgs:
            mp = match_mask(idx_dir, ip)
            if mp is not None:
                pairs.append((ip, mp))
        if not pairs:
            raise RuntimeError("No valid (image,mask) pairs found.")
        self.pairs = sorted(pairs, key=lambda t: t[0].stem)

    def __len__(self): return len(self.pairs)

    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        img = pil_resize_pad(Image.open(ip).convert("RGB"), self.img_size, is_mask=False)
        msk = pil_resize_pad(Image.open(mp).convert("L"),   self.img_size, is_mask=True)
        x = to_tensor_rgb(img)
        y = torch.from_numpy((np.asarray(msk) > 0).astype(np.int64))
        return {"image": x, "mask": y, "stem": ip.stem}

# ----------------------------
# Model: SMP Unet(resnet34, imagenet)
# ----------------------------
def build_smp_unet(in_ch: int):
    try:
        import segmentation_models_pytorch as smp
    except Exception as e:
        raise RuntimeError(
            "segmentation_models_pytorch not installed. "
            "Install with: pip install -q segmentation-models-pytorch"
        ) from e
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=in_ch,
        classes=1,
    )
    return model

# ----------------------------
# Evaluation (vectorized + boundary IoU)
# ----------------------------
@torch.no_grad()
def evaluate(model, dl, device, csv_path, thr=0.5, boundary_tol=1):
    model.eval()

    all_iou, all_dice, all_prec, all_rec, all_acc, all_biou = [], [], [], [], [], []

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "filename","H","W","pixels_pred","pixels_gt","inter","union",
            "iou","boundary_iou","accuracy","dice","precision","recall",
            "tp","fp","fn","tn"
        ])

        for batch in tqdm(dl, desc="Eval"):
            x = batch["image"].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            y = batch["mask"].to(device=device, dtype=torch.long)              # (B,H,W)
            stems = batch["stem"]
            B, H, W = y.shape

            # AMP 推理
            with torch.autocast(device_type=('cuda' if device.type=='cuda' else 'cpu'), enabled=True):
                logits = model(x)                                              # (B,1,H,W)
            prob = torch.sigmoid(logits.squeeze(1))                            # (B,H,W)
            pred = (prob > thr).to(torch.long)

            # 批量统计
            y1 = (y == 1)
            p1 = (pred == 1)
            tp = (p1 & y1).sum(dim=(1,2)).float()
            fp = (p1 & ~y1).sum(dim=(1,2)).float()
            fn = (~p1 & y1).sum(dim=(1,2)).float()
            tn = ((pred == 0) & (y == 0)).sum(dim=(1,2)).float()

            inter = tp
            union = tp + fp + fn
            pixels_pred = tp + fp
            pixels_gt   = tp + fn
            total_pix   = float(H*W)

            iou  = (inter / (union + EPS)).cpu().numpy()
            dice = ((2*tp) / (2*tp + fp + fn + EPS)).cpu().numpy()
            prec = (tp / (tp + fp + EPS)).cpu().numpy()
            rec  = (tp / (tp + fn + EPS)).cpu().numpy()
            acc  = ((tp + tn) / (total_pix + EPS)).cpu().numpy()

            # Boundary IoU（1px 边界 + 可选膨胀容差）
            b_pred = pred.unsqueeze(1).float()   # (B,1,H,W)
            b_gt   = y.unsqueeze(1).float()

            def erode(b):
                return 1.0 - torch.max_pool2d(1.0 - b, kernel_size=3, stride=1, padding=1)
            bp = torch.clamp(b_pred - erode(b_pred), 0, 1)
            bg = torch.clamp(b_gt   - erode(b_gt),   0, 1)

            def dilate(b, k):
                for _ in range(k):
                    b = torch.max_pool2d(b, kernel_size=3, stride=1, padding=1)
                return b
            if boundary_tol > 0:
                bp = dilate(bp, boundary_tol)
                bg = dilate(bg, boundary_tol)

            inter_b = (bp * bg).sum(dim=(1,2,3))
            union_b = torch.clamp(bp + bg, 0, 1).sum(dim=(1,2,3))
            biou = torch.where(union_b > 0, inter_b / (union_b + EPS), torch.ones_like(union_b)).cpu().numpy()

            # 累积
            all_iou.extend(iou.tolist())
            all_dice.extend(dice.tolist())
            all_prec.extend(prec.tolist())
            all_rec .extend(rec.tolist())
            all_acc.extend(acc.tolist())
            all_biou.extend(biou.tolist())

            # 写 CSV
            tp  = tp.cpu().numpy(); fp = fp.cpu().numpy(); fn = fn.cpu().numpy(); tn = tn.cpu().numpy()
            inter = inter.cpu().numpy(); union = union.cpu().numpy()
            pixels_pred = pixels_pred.cpu().numpy(); pixels_gt = pixels_gt.cpu().numpy()

            for i in range(B):
                w.writerow([
                    stems[i], H, W, int(pixels_pred[i]), int(pixels_gt[i]),
                    int(inter[i]), int(union[i]),
                    float(iou[i]), float(biou[i]), float(acc[i]),
                    float(dice[i]), float(prec[i]), float(rec[i]),
                    int(tp[i]), int(fp[i]), int(fn[i]), int(tn[i])
                ])

    return {
        "average_iou":          float(np.mean(all_iou)) if all_iou else 0.0,
        "average_boundary_iou": float(np.mean(all_biou)) if all_biou else 0.0,
        "average_accuracy":     float(np.mean(all_acc)) if all_acc else 0.0,
        "average_dice":         float(np.mean(all_dice)) if all_dice else 0.0,
        "average_precision":    float(np.mean(all_prec)) if all_prec else 0.0,
        "average_recall":       float(np.mean(all_rec)) if all_rec else 0.0,
    }

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser("Evaluate SMP pretrained Unet (resnet34, imagenet)")
    ap.add_argument("--base-dir", required=True, help="Dataset root with split folders")
    ap.add_argument("--split", default="test", choices=["train","valid","test"])
    ap.add_argument("--split-txt", type=str, default=None, help="Optional stems list (txt)")
    ap.add_argument("--out-dir", type=str, required=True, help="Output folder for CSV/JSON")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--fast-boundary", action="store_true", help="Set boundary tol=0 (faster)")
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    base = Path(args.base_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    split_txt = Path(args.split_txt) if args.split_txt else None
    if split_txt and not split_txt.exists():
        raise FileNotFoundError(f"split txt not found: {split_txt}")

    print(f"[split] Using split = {args.split}")
    if split_txt:
        print(f"[subset] Using subset from {split_txt}")

    ds = EvalDataset(base, split=args.split, img_size=args.img_size, split_txt=split_txt)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # infer channels from a sample
    sample = next(iter(DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)))
    in_ch = sample["image"].shape[1]

    print("⚙️ Using SMP Unet (resnet34, imagenet) as pretrained baseline")
    model = build_smp_unet(in_ch).to(device)
    model.eval()

    csv_path = out_dir / f"{args.split}_detailed_metrics_smp.csv"
    metrics = evaluate(
        model, dl, device, csv_path,
        thr=args.thr,
        boundary_tol=(0 if args.fast_boundary else 1)  # 1px 容差；fast 模式为 0

    )

    # save summary
    summary = {
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
        "command_line": " ".join(os.sys.argv),
        "split": args.split,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "metrics": metrics
    }
    js_path = out_dir / f"{args.split}_summary_smp.json"
    with open(js_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n✅ Saved per-image metrics →", csv_path)
    print("✅ Saved summary JSON     →", js_path)
    print("\n—— Summary ——")
    for k, v in metrics.items():
        print(f"{k:>22s}: {v:.6f}")

if __name__ == "__main__":
    main()

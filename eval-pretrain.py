#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval-pretrain.py — Evaluate SMP Unet(resnet34, imagenet) baseline
Outputs:
  1) per_image_metrics.csv
  2) summary.json (average_iou, average_boundary_iou, average_accuracy,
                   average_dice, average_precision, average_recall,
                   average_hausdorff)
"""

import os, sys, json, time, argparse
from pathlib import Path
from typing import Optional, List
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

# =========================
# Helpers
# =========================
VALID_EXTS = {".png",".jpg",".jpeg",".tif",".tiff",".bmp"}
EPS = 1e-6

def resize_longest_side(pil_img: Image.Image, target: int, is_mask: bool) -> Image.Image:
    w, h = pil_img.size
    if max(w, h) == target:
        return pil_img
    s = float(target) / float(max(w, h))
    new_w, new_h = int(round(w * s)), int(round(h * s))
    return pil_img.resize((new_w, new_h), Image.NEAREST if is_mask else Image.BILINEAR)

def pad_to_square(pil_img: Image.Image, target: int, fill=0) -> Image.Image:
    if pil_img.size == (target, target):
        return pil_img
    canvas = Image.new(pil_img.mode, (target, target), color=fill)
    canvas.paste(pil_img, (0, 0))
    return canvas

def pil_to_chw_float(pil_img: Image.Image) -> np.ndarray:
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    arr = np.asarray(pil_img).transpose(2,0,1).astype(np.float32)
    if (arr > 1).any():
        arr /= 255.0
    return arr

def safe_mean(xs: List[float]) -> float:
    if not xs: return 0.0
    arr = np.array(xs, dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if arr.size else 0.0

# =========================
# Boundary IoU & Hausdorff
# =========================
from scipy import ndimage as ndi

def _binary_boundary(mask_bool: np.ndarray, tol: int = 1) -> np.ndarray:
    """粗糙一层：边界=mask ^ erosion(mask, tol)。tol=1 ~ 1px 边界厚度。"""
    tol = max(1, int(tol))
    eroded = ndi.binary_erosion(mask_bool, structure=np.ones((3,3), bool),
                                iterations=tol, border_value=0)
    return mask_bool ^ eroded

def boundary_iou(pred_bool: np.ndarray, gt_bool: np.ndarray, tol: int = 1) -> float:
    if pred_bool.size == 0 or gt_bool.size == 0:
        return 0.0
    if not pred_bool.any() and not gt_bool.any():
        return 1.0  # 两者都空，边界完美一致
    pb = _binary_boundary(pred_bool, tol=1)
    gb = _binary_boundary(gt_bool,  tol=1)
    if tol > 1:
        se = np.ones((2*tol+1, 2*tol+1), dtype=bool)
        pb = ndi.binary_dilation(pb, structure=se)
        gb = ndi.binary_dilation(gb, structure=se)
    inter = np.logical_and(pb, gb).sum()
    union = np.logical_or(pb, gb).sum()
    return float(inter) / float(union) if union > 0 else 0.0

def hausdorff_symmetric(pred_bool: np.ndarray, gt_bool: np.ndarray) -> float:
    """对称 Hausdorff（像素单位）。单侧空返回 max(H,W)，双空返回 0。"""
    H, W = pred_bool.shape
    if not pred_bool.any() and not gt_bool.any():
        return 0.0
    if not pred_bool.any() or not gt_bool.any():
        return float(max(H, W))
    dt_pred = ndi.distance_transform_edt(~pred_bool)
    dt_gt   = ndi.distance_transform_edt(~gt_bool)
    h_ab = float(dt_gt[pred_bool].max()) if pred_bool.any() else 0.0
    h_ba = float(dt_pred[gt_bool].max()) if gt_bool.any() else 0.0
    return max(h_ab, h_ba)

# =========================
# Dataset
# =========================
class SplitFolderDataset(Dataset):
    def __init__(self, base_dir: Path, split: str, img_size: int, split_txt: Optional[Path] = None):
        self.base = Path(base_dir)
        self.img_size = img_size
        img_dir = self.base / split / "image"
        idx_dir = self.base / split / "index"
        assert img_dir.exists() and idx_dir.exists(), f"Missing {img_dir} or {idx_dir}"

        wanted = None
        if split_txt:
            wanted = set(Path(split_txt).read_text(encoding="utf-8").split())
            wanted = {Path(s).stem for s in wanted}
            print(f"[subset] Using subset from {split_txt}")

        idx_map = {p.stem: p for p in idx_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS}

        pairs = []
        for p in img_dir.iterdir():
            if not (p.is_file() and p.suffix.lower() in VALID_EXTS):
                continue
            stem = p.stem
            if wanted and (stem not in wanted):
                continue
            mp = idx_map.get(stem) or idx_map.get(stem + "_index")
            if mp is not None:
                pairs.append((p, mp, stem))
        if not pairs:
            raise RuntimeError(f"No (image,mask) pairs under {img_dir}")
        self.pairs = sorted(pairs, key=lambda t: t[2])

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        ip, mp, stem = self.pairs[idx]
        img = Image.open(ip).convert("RGB")
        msk = Image.open(mp).convert("L")

        img = pad_to_square(resize_longest_side(img, self.img_size, False), self.img_size, 0)
        msk = pad_to_square(resize_longest_side(msk, self.img_size, True),  self.img_size, 0)

        x = torch.from_numpy(pil_to_chw_float(img).copy()).float()
        y = torch.from_numpy((np.asarray(msk) > 0).astype(np.int64)).long()
        return {"image": x, "mask": y, "stem": stem}

# =========================
# Model: SMP Unet baseline
# =========================
def build_smp_unet(in_ch: int):
    import segmentation_models_pytorch as smp
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=in_ch,
        classes=1,
    )
    return model

# =========================
# Evaluation
# =========================
@torch.no_grad()
def run_eval(model: torch.nn.Module, loader: DataLoader, device, thr: float,
             csv_path: Path, boundary_tol: int):
    model.eval()

    per_iou, per_biou, per_acc, per_dice, per_prec, per_rec, per_hd = [], [], [], [], [], [], []

    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "filename","H","W",
            "iou","boundary_iou","accuracy","dice","precision","recall","hausdorff",
            "tp","fp","fn","tn"
        ])

        pbar = tqdm(loader, desc="Eval", leave=True)
        for batch in pbar:
            x = batch["image"].to(device=device, dtype=torch.float32)
            y = batch["mask"].to(device=device, dtype=torch.long)
            stems = batch["stem"]
            logits = model(x).squeeze(1)
            prob = torch.sigmoid(logits)
            pred = (prob > thr).long()

            # per-sample
            B, H, W = pred.shape
            for i in range(B):
                p = pred[i].cpu().numpy().astype(np.uint8)
                g = y[i].cpu().numpy().astype(np.uint8)

                tp = int(((p==1) & (g==1)).sum())
                fp = int(((p==1) & (g==0)).sum())
                fn = int(((p==0) & (g==1)).sum())
                tn = int(((p==0) & (g==0)).sum())
                tot = tp + fp + fn + tn

                iou   = (tp / (tp + fp + fn + EPS))
                dice  = (2*tp / (2*tp + fp + fn + EPS))
                prec  = (tp / (tp + fp + EPS))
                rec   = (tp / (tp + fn + EPS))
                acc   = ((tp + tn) / (tot + EPS))

                biou  = boundary_iou(p.astype(bool), g.astype(bool), tol=boundary_tol)
                hd    = hausdorff_symmetric(p.astype(bool), g.astype(bool))

                per_iou.append(iou)
                per_biou.append(biou)
                per_acc.append(acc)
                per_dice.append(dice)
                per_prec.append(prec)
                per_rec.append(rec)
                per_hd.append(hd)

                w.writerow([
                    stems[i], H, W,
                    float(iou), float(biou), float(acc), float(dice), float(prec), float(rec), float(hd),
                    tp, fp, fn, tn
                ])

    summary = {
        "average_iou":           safe_mean(per_iou),
        "average_boundary_iou":  safe_mean(per_biou),
        "average_accuracy":      safe_mean(per_acc),
        "average_dice":          safe_mean(per_dice),
        "average_precision":     safe_mean(per_prec),
        "average_recall":        safe_mean(per_rec),
        "average_hausdorff":     safe_mean(per_hd),
    }
    return summary

# =========================
# Main
# =========================
def parse_args():
    ap = argparse.ArgumentParser("Evaluate pretrained SMP Unet baseline")
    ap.add_argument("--base-dir", required=True, help="Dataset root (has split/image, split/index)")
    ap.add_argument("--split", default="test", choices=["train","valid","test"], help="Which split folder to use")
    ap.add_argument("--split-txt", type=str, default=None, help="Optional subset txt (stems)")
    ap.add_argument("--out-dir", type=str, default="./eval_pretrain_out", help="Output folder")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=0, help="Dataloader workers (Drive 建议 0)")
    ap.add_argument("--boundary-tol", type=int, default=1, help="Boundary IoU tolerance (px)")
    ap.add_argument("--fast-boundary", action="store_true",
                    help="Fast boundary mode (equivalent to --boundary-tol 0)")
    return ap.parse_args()

def main():
    args = parse_args()
    base = Path(args.base_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    split_txt_path = Path(args.split_txt) if args.split_txt else None
    print(f"[split] Using split = {args.split}")
    if split_txt_path:
        pass  # 提示在 Dataset 里打印

    ds = SplitFolderDataset(base, args.split, img_size=args.img_size, split_txt=split_txt_path)
    print(f"[split] Using {len(ds)} images in {base/args.split/'image'}")

    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=False  # 在 GDrive 上更稳
    )

    # build pretrained SMP Unet
    print("⚙️ Using SMP Unet (resnet34, imagenet) as pretrained baseline")
    in_ch = next(iter(DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)))["image"].shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_smp_unet(in_ch).to(device)
    model.eval()

    # boundary tolerance
    boundary_tol = 0 if args.fast_boundary else max(0, int(args.boundary_tol))

    # run
    csv_path = out_dir / "per_image_metrics.csv"
    summary = run_eval(model, dl, device, args.thr, csv_path, boundary_tol=boundary_tol)

    # save summary
    js = {
        "args": vars(args),
        "summary": {k: float(v) for k, v in summary.items()}
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(js, f, indent=2, ensure_ascii=False)

    # print summary
    print("\n===== Summary =====")
    for k, v in summary.items():
        print(f"{k}: {v:.6f}")
    print(f"\n✅ Per-image CSV → {csv_path}")
    print(f"✅ Summary JSON  → {out_dir/'summary.json'}")

if __name__ == "__main__":
    main()

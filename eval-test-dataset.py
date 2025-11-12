# -*- coding: utf-8 -*-
"""
Evaluate UNet on test set (folder-based OR split-based)
--------------------------------------------------------
Outputs:
  1. CSV per-image metrics
  2. JSON summary (command args + 6 key metrics)

Metrics:
  mean_iou_overall, mean_dice_overall,
  mean_precision_overall, mean_recall_overall,
  mean_iou_non_empty_gt, mean_dice_non_empty_gt
"""

import os, csv, json, time, argparse, sys
from pathlib import Path
from typing import List, Optional
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from unet.unet_model import UNet

VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# ---------- util ----------

# --- 放到文件顶部 ---

def safe_mean(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(x.mean()) if x.size else 0.0

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
    return [Path(line.strip()).stem for line in txt_path.read_text().splitlines() if line.strip()]

def match_mask(idx_dir: Path, img_path: Path):
    m1 = idx_dir / img_path.name
    if m1.exists(): return m1
    m2 = idx_dir / f"{img_path.stem}_index{img_path.suffix}"
    if m2.exists(): return m2
    return None

# ---------- dataset ----------
class TestDataset(Dataset):
    def __init__(self, base_dir: Path, img_size=512, test_split: Optional[Path]=None):
        self.img_size = img_size
        img_dir = base_dir / "test" / "image"
        idx_dir = base_dir / "test" / "index"
        assert img_dir.exists(), f"Missing: {img_dir}"
        assert idx_dir.exists(), f"Missing: {idx_dir}"

        imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in VALID_EXTS]
        if test_split:
            stems = set(load_split_stems(test_split))
            imgs = [p for p in imgs if p.stem in stems]

        self.pairs = [(ip, match_mask(idx_dir, ip)) for ip in imgs if match_mask(idx_dir, ip)]
        if not self.pairs:
            raise RuntimeError("No valid (image,mask) pairs found.")

    def __len__(self): return len(self.pairs)

    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        img = pil_resize_pad(Image.open(ip).convert("RGB"), self.img_size, is_mask=False)
        msk = pil_resize_pad(Image.open(mp).convert("L"),   self.img_size, is_mask=True)
        x = to_tensor_rgb(img)
        y = torch.from_numpy((np.asarray(msk) > 0).astype(np.int64))
        return {"image": x, "mask": y, "stem": ip.stem}


# ---------- metric ----------
def compute_metrics(pred, gt):
    tp = np.logical_and(pred==1, gt==1).sum()
    fp = np.logical_and(pred==1, gt==0).sum()
    fn = np.logical_and(pred==0, gt==1).sum()
    tn = np.logical_and(pred==0, gt==0).sum()
    inter = tp
    union = tp+fp+fn
    iou = inter / (union + 1e-6)
    dice = (2*tp) / (2*tp+fp+fn+1e-6)
    prec = tp / (tp+fp+1e-6)
    rec  = tp / (tp+fn+1e-6)
    return iou, dice, prec, rec, tp, fp, fn, tn, inter, union

# ---------- eval ----------
import json, time, math
import numpy as np

def _safe_div(a, b):
    return float(a) / float(b) if b and b != 0 else 0.0

def _nanmean(xs):
    if not xs:
        return 0.0
    arr = np.array(xs, dtype=float)
    # if all are NaN, nanmean would raise warning; handle that
    if np.all(np.isnan(arr)):
        return 0.0
    return float(np.nanmean(arr))

@torch.no_grad()
def evaluate(model, dl, device, csv_path):
    model.eval()
    import csv

    all_iou, all_dice, all_prec, all_rec = [], [], [], []
    nz_iou, nz_dice = [], []   # non-empty GT only

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename","H","W","pixels_pred","pixels_gt","inter","union",
                    "iou","dice","precision","recall","tp","fp","fn","tn"])

        for batch in tqdm(dl, desc="Evaluating"):
            imgs = batch["image"].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            ys   = batch["mask"].to(device=device, dtype=torch.long)
            stems= batch["stem"]

            logits = model(imgs)
            prob   = torch.sigmoid(logits.squeeze(1))  # (B,H,W)
            pred   = (prob > 0.5).to(torch.long)

            for i in range(pred.shape[0]):
                p = pred[i]
                y = ys[i]

                H, W = p.shape[-2], p.shape[-1]
                tp = int(((p==1) & (y==1)).sum().item())
                fp = int(((p==1) & (y==0)).sum().item())
                fn = int(((p==0) & (y==1)).sum().item())
                tn = int(((p==0) & (y==0)).sum().item())

                inter = tp
                union = tp + fp + fn
                pixels_pred = tp + fp
                pixels_gt   = tp + fn

                iou = _safe_div(inter, union)                       # Jaccard
                dice = _safe_div(2*tp, 2*tp + fp + fn)              # F1/Dice
                precision = _safe_div(tp, tp + fp)
                recall    = _safe_div(tp, tp + fn)

                all_iou.append(iou)
                all_dice.append(dice)
                all_prec.append(precision)
                all_rec.append(recall)
                if pixels_gt > 0:
                    nz_iou.append(iou)
                    nz_dice.append(dice)

                w.writerow([
                    stems[i], H, W, pixels_pred, pixels_gt, inter, union,
                    iou, dice, precision, recall, tp, fp, fn, tn
                ])

    # robust means (no NaN)
    metrics = {
        "mean_iou_overall":      _nanmean(all_iou),
        "mean_dice_overall":     _nanmean(all_dice),
        "mean_precision_overall":_nanmean(all_prec),
        "mean_recall_overall":   _nanmean(all_rec),
        "mean_iou_non_empty_gt": _nanmean(nz_iou),
        "mean_dice_non_empty_gt":_nanmean(nz_dice),
    }
    return metrics

def save_summary_json(out_dir, cmdline_str, metrics):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    js = {
        "timestamp": ts,
        "command_line": cmdline_str,
        "metrics": {k: float(v) for k, v in metrics.items()}  # ensure plain floats
    }
    with open(out_dir / "test-summary.json", "w", encoding="utf-8") as f:
        json.dump(js, f, indent=2, ensure_ascii=False)
    print("✅ summary saved to", out_dir / "test-summary.json")


# ---------- main ----------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base-dir",required=True)
    ap.add_argument("--model-path",required=True)
    ap.add_argument("--test-split",default=None)
    ap.add_argument("--img-size",type=int,default=512)
    ap.add_argument("--batch-size",type=int,default=8)
    ap.add_argument("--out-dir",default="eval_results")
    args=ap.parse_args()

    base=Path(args.base_dir)
    out_dir=base/args.out_dir
    out_dir.mkdir(parents=True,exist_ok=True)

    ds=TestDataset(base,img_size=args.img_size,
                   test_split=Path(args.test_split) if args.test_split else None)
    dl=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=2,pin_memory=True)
    sample=next(iter(dl))
    in_ch=sample["image"].shape[1]

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True

    model=UNet(n_channels=in_ch,n_classes=1,bilinear=False).to(device)
    ckpt=torch.load(args.model_path,map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    csv_path=out_dir/"test_detailed_metrics.csv"
    t0=time.strftime("%Y-%m-%d_%H-%M-%S")
    metrics=evaluate(model,dl,device,csv_path)

    # summarize JSON
    summary={
        "timestamp":t0,
        "command_line":" ".join(sys.argv),
        "metrics":metrics
    }
    json_path=out_dir/"test_summary.json"
    with open(json_path,"w",encoding="utf-8") as f:
        json.dump(summary,f,indent=2)
    print(f"\n✅ Saved per-image metrics → {csv_path}")
    print(f"✅ Saved summary JSON → {json_path}")

if __name__=="__main__":
    main()

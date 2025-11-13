# -*- coding: utf-8 -*-
"""
Evaluate UNet on test set (folder-based OR split-based)
--------------------------------------------------------
Outputs:
  1. CSV per-image metrics
  2. JSON summary (command args + 6 key metrics)

Metrics (summary keys):
  average_iou, average_boundary_iou, average_accuracy,
  average_dice, average_precision, average_recall
"""

import os, csv, json, time, argparse, sys
from pathlib import Path
from typing import List, Optional
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from unet.unet_model import UNet

VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
EPS = 1e-6

# ---------- util ----------
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

# ---------- dataset ----------
class TestDataset(Dataset):
    def __init__(self, base_dir: Path, img_size=512, test_split: Optional[Path]=None):
        self.img_size = img_size
        img_dir = base_dir / "test" / "image"
        idx_dir = base_dir / "test" / "index"
        assert img_dir.exists(), f"Missing: {img_dir}"
        assert idx_dir.exists(), f"Missing: {idx_dir}"

        imgs = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]
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

# ---------- boundary IoU helpers (PyTorch morphology) ----------
def binary_erode_pt(b: torch.Tensor, k: int = 1) -> torch.Tensor:
    """
    b: (1,1,H,W) binary {0,1}
    Erosion with 3x3 ones kernel, k iterations using min-pool trick.
    """
    x = b
    for _ in range(k):
        # erosion = 1 - max_pool(1-b)
        x = 1.0 - F.max_pool2d(1.0 - x, kernel_size=3, stride=1, padding=1)
    return x

def binary_dilate_pt(b: torch.Tensor, k: int = 1) -> torch.Tensor:
    """
    b: (1,1,H,W) binary {0,1}
    Dilation with 3x3 ones kernel, k iterations via max-pool.
    """
    x = b
    for _ in range(k):
        x = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    return x

def boundary_map_pt(b: torch.Tensor) -> torch.Tensor:
    """
    b: (1,1,H,W) binary {0,1}
    boundary = b - erode(b)
    """
    er = binary_erode_pt(b, k=1)
    return torch.clamp(b - er, 0, 1)

def boundary_iou_pt(pred: torch.Tensor, gt: torch.Tensor, tol: int = 1) -> float:
    """
    pred, gt: (H,W) int/byte {0,1}
    1-pixel wide boundaries with dilation tolerance 'tol' for both sides.
    """
    p = pred.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    g = gt.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    bp = boundary_map_pt(p)
    bg = boundary_map_pt(g)

    if tol > 0:
        bp = binary_dilate_pt(bp, k=tol)
        bg = binary_dilate_pt(bg, k=tol)

    inter = (bp * bg).sum().item()
    union = torch.clamp(bp + bg, 0, 1).sum().item()
    if union < EPS:
        # 两者都无边界（全空/全满），按完全匹配处理 → IoU=1
        return 1.0
    return float(inter / (union + EPS))

# ---------- eval ----------
@torch.no_grad()
def evaluate(model, dl, device, csv_path):
    model.eval()

    all_iou, all_dice, all_prec, all_rec, all_acc = [], [], [], [], []
    all_biou = []

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "filename","H","W","pixels_pred","pixels_gt","inter","union",
            "iou","boundary_iou","accuracy","dice","precision","recall",
            "tp","fp","fn","tn"
        ])

        for batch in tqdm(dl, desc="Evaluating"):
            imgs = batch["image"].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            ys   = batch["mask"].to(device=device, dtype=torch.long)
            stems= batch["stem"]

            logits = model(imgs)                # (B,1,H,W)
            prob   = torch.sigmoid(logits.squeeze(1))  # (B,H,W)
            pred   = (prob > 0.5).to(torch.long)

            B, H, W = pred.shape
            for i in range(B):
                p = pred[i]
                y = ys[i]

                tp = int(((p==1) & (y==1)).sum().item())
                fp = int(((p==1) & (y==0)).sum().item())
                fn = int(((p==0) & (y==1)).sum().item())
                tn = int(((p==0) & (y==0)).sum().item())

                inter = tp
                union = tp + fp + fn
                pixels_pred = tp + fp
                pixels_gt   = tp + fn
                total_pix   = H * W

                iou   = inter / (union + EPS)
                dice  = (2*tp) / (2*tp + fp + fn + EPS)
                prec  = tp / (tp + fp + EPS)
                rec   = tp / (tp + fn + EPS)
                acc   = (tp + tn) / (total_pix + EPS)
                biou  = boundary_iou_pt(p, y, tol=1)  # 容差=1像素

                all_iou.append(iou)
                all_dice.append(dice)
                all_prec.append(prec)
                all_rec.append(rec)
                all_acc.append(acc)
                all_biou.append(biou)

                w.writerow([
                    stems[i], H, W, pixels_pred, pixels_gt, inter, union,
                    iou, biou, acc, dice, prec, rec,
                    tp, fp, fn, tn
                ])

    metrics = {
        "average_iou":            float(np.mean(all_iou)) if all_iou else 0.0,
        "average_boundary_iou":   float(np.mean(all_biou)) if all_biou else 0.0,
        "average_accuracy":       float(np.mean(all_acc)) if all_acc else 0.0,
        "average_dice":           float(np.mean(all_dice)) if all_dice else 0.0,
        "average_precision":      float(np.mean(all_prec)) if all_prec else 0.0,
        "average_recall":         float(np.mean(all_rec)) if all_rec else 0.0,
    }
    return metrics

def save_summary_json(out_dir, cmdline_str, metrics):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    js = {
        "timestamp": ts,
        "command_line": cmdline_str,
        "metrics": {k: float(v) for k, v in metrics.items()}
    }
    with open(out_dir / "test-summary.json", "w", encoding="utf-8") as f:
        json.dump(js, f, indent=2, ensure_ascii=False)
    print("\n===== SUMMARY =====")
    for k, v in js["metrics"].items():
        print(f"{k}: {v:.6f}")
    print("===================")
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
    sample=next(iter(DataLoader(ds,batch_size=1,shuffle=False,num_workers=0)))
    in_ch=sample["image"].shape[1]

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True

    model=UNet(n_channels=in_ch,n_classes=1,bilinear=False).to(device)
    ckpt=torch.load(args.model_path,map_location=device)
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    csv_path=out_dir/"test_detailed_metrics.csv"
    t0=time.strftime("%Y-%m-%d_%H-%M-%S")
    metrics=evaluate(model,dl,device,csv_path)

    save_summary_json(out_dir, " ".join(sys.argv), metrics)
    print(f"✅ Saved per-image metrics → {csv_path}")

if __name__=="__main__":
    main()

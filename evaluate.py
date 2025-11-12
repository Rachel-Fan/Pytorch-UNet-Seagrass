#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate.py — Evaluate trained model or SMP pretrained baseline on test subset
Author: RF | 2025-11 | Compatible with train/valid/test splits + split txt
"""

import os, sys, json, time, argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

# ==========================================================
#  ARGUMENTS
# ==========================================================
parser = argparse.ArgumentParser(description="Evaluate segmentation model on test subset")
parser.add_argument("--base-dir", required=True, help="Base dataset directory containing train/valid/test folders")
parser.add_argument("--split", default="test", choices=["train","valid","test"], help="Split folder to use")
parser.add_argument("--split-txt", type=str, default=None, help="Optional txt file listing subset (e.g., test.txt)")
parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint path (.pth)")
parser.add_argument("--out-dir", type=str, default="./eval_output", help="Output folder for metrics JSON")
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--img-size", type=int, default=512)
parser.add_argument("--thr", type=float, default=0.5, help="Probability threshold")
parser.add_argument("--use-smp-pretrained", action="store_true",
                    help="Use segmentation_models_pytorch Unet(resnet34, imagenet) instead of checkpoint")
args = parser.parse_args()

BASE_DIR = Path(args.base_dir)
SPLIT = args.split
OUT_DIR = Path(args.out_dir); OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
#  HELPERS
# ==========================================================
VALID_EXTS = {".png",".jpg",".jpeg",".tif",".tiff",".bmp"}

def resize_longest_side(pil_img, target, is_mask):
    w, h = pil_img.size
    if max(w, h) == target: return pil_img
    s = float(target) / float(max(w, h))
    return pil_img.resize((int(round(w*s)), int(round(h*s))), Image.NEAREST if is_mask else Image.BILINEAR)

def pad_to_square(pil_img, target, fill=0):
    w, h = pil_img.size
    if (w, h) == (target, target): return pil_img
    canvas = Image.new(pil_img.mode, (target, target), color=fill)
    canvas.paste(pil_img, (0, 0))
    return canvas

def pil_to_chw_float(pil_img):
    if pil_img.mode != "RGB": pil_img = pil_img.convert("RGB")
    arr = np.asarray(pil_img).transpose(2,0,1).astype(np.float32)
    if (arr > 1).any(): arr /= 255.0
    return arr

# ==========================================================
#  DATASET
# ==========================================================
class SplitFolderDataset(Dataset):
    def __init__(self, base_dir, split, img_size, split_txt=None):
        self.base = Path(base_dir)
        img_dir = self.base / split / "image"
        idx_dir = self.base / split / "index"
        assert img_dir.exists() and idx_dir.exists(), f"Missing {img_dir} or {idx_dir}"

        wanted = None
        if split_txt:
            wanted = set(Path(split_txt).read_text(encoding="utf-8").split())
            wanted = {Path(s).stem for s in wanted}
            print(f"[subset] Using {len(wanted)} entries from {split_txt}")

        idx_map = {p.stem: p for p in idx_dir.iterdir() if p.suffix.lower() in VALID_EXTS}
        pairs = []
        for p in img_dir.iterdir():
            if not (p.is_file() and p.suffix.lower() in VALID_EXTS): continue
            stem = p.stem
            if wanted and (stem not in wanted): continue
            m = idx_map.get(stem) or idx_map.get(stem + "_index")
            if m is not None:
                pairs.append((p, m, stem))
        if not pairs:
            raise RuntimeError("No (image,mask) pairs found in " + str(img_dir))
        self.pairs = sorted(pairs, key=lambda t: t[2])
        self.img_size = img_size

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        ip, mp, stem = self.pairs[idx]
        img = Image.open(ip).convert("RGB")
        msk = Image.open(mp).convert("L")

        img = pad_to_square(resize_longest_side(img, self.img_size, False), self.img_size, 0)
        msk = pad_to_square(resize_longest_side(msk, self.img_size, True),  self.img_size, 0)

        x = torch.from_numpy(pil_to_chw_float(img).copy()).float()
        y = torch.from_numpy(np.asarray(msk).copy()).long()
        y = (y > 0).long()
        return {"image": x, "mask": y, "stem": stem}

# ==========================================================
#  MODEL BUILDER
# ==========================================================
sys.path.append("/content/Pytorch-UNet-Seagrass")
from unet.unet_model import UNet

def build_model(in_ch: int):
    return UNet(n_channels=in_ch, n_classes=1, bilinear=False)

def build_model_smp(in_ch: int):
    import segmentation_models_pytorch as smp
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=in_ch,
        classes=1,
    )

# ==========================================================
#  EVALUATION
# ==========================================================
@torch.no_grad()
def evaluate(model, loader, device, thr):
    dices, ious, precs, recs = [], [], [], []
    for batch in loader:
        x = batch["image"].to(device, dtype=torch.float32)
        y = batch["mask"].to(device, dtype=torch.long)
        logits = model(x).squeeze(1)
        prob = torch.sigmoid(logits)
        pred = (prob > thr).long()

        y1 = (y == 1); p1 = (pred == 1)
        tp = (p1 & y1).sum(dim=(1,2)).float()
        fp = (p1 & ~y1).sum(dim=(1,2)).float()
        fn = (~p1 & y1).sum(dim=(1,2)).float()

        precision = (tp / (tp + fp + 1e-6)).mean().item()
        recall    = (tp / (tp + fn + 1e-6)).mean().item()
        dice      = (2*tp / (2*tp + fp + fn + 1e-6)).mean().item()
        iou       = (tp / (tp + fp + fn + 1e-6)).mean().item()

        precs.append(precision); recs.append(recall)
        dices.append(dice); ious.append(iou)

    return {
        "mean_iou": float(np.mean(ious)),
        "mean_dice": float(np.mean(dices)),
        "mean_precision": float(np.mean(precs)),
        "mean_recall": float(np.mean(recs)),
        
    }

# ==========================================================
#  MAIN
# ==========================================================
def main():
    ds = SplitFolderDataset(BASE_DIR, SPLIT, args.img_size, split_txt=args.split_txt)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    in_ch = next(iter(dl))["image"].shape[1]

    if args.use_smp_pretrained:
        print("⚙️ Using segmentation_models_pytorch pretrained Unet (resnet34, imagenet)")
        model = build_model_smp(in_ch)
    else:
        print(f"⚙️ Loading checkpoint: {args.ckpt}")
        model = build_model(in_ch)
        state = torch.load(args.ckpt, map_location=DEVICE)
        model.load_state_dict(state, strict=True)

    model.to(DEVICE)
    model.eval()

    print(f"[eval] split={SPLIT}, samples={len(ds)}, thr={args.thr}")
    res = evaluate(model, dl, DEVICE, args.thr)

    out_name = f"summary_{'smp' if args.use_smp_pretrained else 'ckpt'}.json"
    out_path = OUT_DIR / out_name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print("\n✅ Done. Saved:", out_path)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()

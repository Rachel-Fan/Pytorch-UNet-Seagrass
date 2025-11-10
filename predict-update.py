# -*- coding: utf-8 -*-
"""
predict_eval_full.py — 统一预测 + 全面指标评估（PA-SAM 风格）
- 输入:  splits/test.txt
- 输出:
    pred/*.png           (二值掩膜)
    pred_viz/*.jpg       (叠加可视化)
    metrics_summary.csv  (逐图+汇总指标)
"""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLED"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm
import csv

from unet.unet_model import UNet

# ---------- 配置 ----------
BASE = Path(r"D:\Eelgrass_Process_2025_Bo\DroneVision_Model_data\BC")
DIR_IMG = BASE / "image"
DIR_MASK = BASE / "index"
DIR_GLCM = BASE / "glcm"
DIR_SPLIT = BASE / "splits"
DIR_CKPT = BASE / "train2/checkpoints"
CKPT = DIR_CKPT / "best.pth"
OUT_MASK = DIR_CKPT / "pred"
OUT_VIZ = DIR_CKPT / "pred_viz"
OUT_CSV = DIR_CKPT / "metrics_summary.csv"

IMG_SIZE = 768
EXTRA_MODE = None
CLASSES = 1
BILINEAR = False
THRESH = 0.5
SAVE_VIZ = True
VALID_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]

OUT_MASK.mkdir(parents=True, exist_ok=True)
if SAVE_VIZ:
    OUT_VIZ.mkdir(parents=True, exist_ok=True)

# ---------- 工具 ----------
def strip_known_ext(name: str) -> str:
    lower = name.lower()
    for ext in VALID_EXTS:
        if lower.endswith(ext):
            return name[: -len(ext)]
    return name

def find_first(folder: Path, stem: str) -> Path:
    clean = strip_known_ext(stem)
    for ext in VALID_EXTS:
        p = folder / f"{clean}{ext}"
        if p.exists():
            return p
    cand = list(folder.glob(clean + ".*"))
    if cand:
        return cand[0]
    raise FileNotFoundError(f"not found: {folder}/{clean}.*")

def resize_longest_side(pil_img, target, is_mask):
    w, h = pil_img.size
    if max(w, h) == target:
        return pil_img
    scale = target / max(w, h)
    return pil_img.resize(
        (int(round(w * scale)), int(round(h * scale))),
        Image.NEAREST if is_mask else Image.BILINEAR,
    )

def pad_to_square(pil_img, target, fill=0):
    w, h = pil_img.size
    if (w, h) == (target, target):
        return pil_img
    canvas = Image.new(pil_img.mode, (target, target), fill)
    canvas.paste(pil_img, (0, 0))
    return canvas

def pil_to_chw_float(pil_img):
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    arr = np.asarray(pil_img).transpose(2, 0, 1).astype(np.float32)
    if arr.max() > 1:
        arr /= 255.0
    return arr

def calc_metrics(pred_bin, gt_bin):
    """计算 Dice, IoU, Precision, Recall"""
    p = pred_bin > 0
    g = gt_bin > 0
    tp = np.logical_and(p, g).sum()
    fp = np.logical_and(p, np.logical_not(g)).sum()
    fn = np.logical_and(np.logical_not(p), g).sum()
    union = tp + fp + fn
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else (1.0 if g.sum() == 0 else 0.0)
    iou = tp / union if union > 0 else (1.0 if g.sum() == 0 else 0.0)
    prec = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if g.sum() == 0 else 0.0)
    rec = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if g.sum() == 0 else 0.0)
    return dice, iou, prec, rec, g.sum() > 0

# ---------- 主流程 ----------
@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_channels=3, n_classes=CLASSES, bilinear=BILINEAR).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()

    split_file = DIR_SPLIT / "test.txt"
    stems = [strip_known_ext(s.strip()) for s in split_file.read_text().splitlines() if s.strip()]

    records = []
    overall = {"iou": [], "dice": [], "prec": [], "rec": [], "iou_non_empty": [], "dice_non_empty": []}

    for stem in tqdm(stems, desc="Predicting"):
        img_path = find_first(DIR_IMG, stem)
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        img_rs = resize_longest_side(img, IMG_SIZE, False)
        img_pad = pad_to_square(img_rs, IMG_SIZE, 0)
        x_arr = pil_to_chw_float(img_pad)

        x = torch.from_numpy(x_arr).unsqueeze(0).to(device)
        prob = torch.sigmoid(model(x))[0, 0]
        rs_w, rs_h = img_rs.size
        prob_np = prob[:rs_h, :rs_w].cpu().numpy()
        prob_img = Image.fromarray((prob_np * 255).astype(np.uint8), "L")
        prob_up = prob_img.resize((orig_w, orig_h), Image.BILINEAR)
        pred_bin = ((np.array(prob_up) / 255.0) > THRESH).astype(np.uint8) * 255

        Image.fromarray(pred_bin, "L").save(OUT_MASK / f"{stem}.png")

        if SAVE_VIZ:
            rgba = img.convert("RGBA")
            ov = np.zeros((orig_h, orig_w, 4), np.uint8)
            ov[..., 0] = (pred_bin > 0).astype(np.uint8) * 255
            ov[..., 3] = (pred_bin > 0).astype(np.uint8) * 120
            blended = Image.alpha_composite(rgba, Image.fromarray(ov, "RGBA")).convert("RGB")
            blended.save(OUT_VIZ / f"{stem}.jpg")

        try:
            gt_path = find_first(DIR_MASK, stem)
            gt = Image.open(gt_path).convert("L").resize((orig_w, orig_h), Image.NEAREST)
            gt_bin = (np.array(gt) > 0).astype(np.uint8) * 255
            dice, iou, prec, rec, has_gt = calc_metrics(pred_bin, gt_bin)
            records.append([stem, iou, dice, prec, rec, has_gt])
            overall["iou"].append(iou)
            overall["dice"].append(dice)
            overall["prec"].append(prec)
            overall["rec"].append(rec)
            if has_gt:
                overall["iou_non_empty"].append(iou)
                overall["dice_non_empty"].append(dice)
        except Exception:
            records.append([stem, np.nan, np.nan, np.nan, np.nan, False])

    mean_iou = np.nanmean(overall["iou"])
    mean_dice = np.nanmean(overall["dice"])
    mean_prec = np.nanmean(overall["prec"])
    mean_rec = np.nanmean(overall["rec"])
    mean_iou_ne = np.nanmean(overall["iou_non_empty"]) if overall["iou_non_empty"] else 0.0
    mean_dice_ne = np.nanmean(overall["dice_non_empty"]) if overall["dice_non_empty"] else 0.0

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "stem", "iou", "dice", "precision", "recall", "has_gt",
        ])
        w.writerows(records)
        w.writerow([])
        w.writerow(["mean_iou_overall", mean_iou])
        w.writerow(["mean_dice_overall", mean_dice])
        w.writerow(["mean_precision_overall", mean_prec])
        w.writerow(["mean_recall_overall", mean_rec])
        w.writerow(["mean_iou_non_empty_gt", mean_iou_ne])
        w.writerow(["mean_dice_non_empty_gt", mean_dice_ne])

    print("✅ Done.")
    print(f"   mean_iou_overall={mean_iou:.4f}")
    print(f"   mean_dice_overall={mean_dice:.4f}")
    print(f"   mean_precision_overall={mean_prec:.4f}")
    print(f"   mean_recall_overall={mean_rec:.4f}")
    print(f"   mean_iou_non_empty_gt={mean_iou_ne:.4f}")
    print(f"   mean_dice_non_empty_gt={mean_dice_ne:.4f}")
    print(f"Results saved to {OUT_CSV}")


if __name__ == "__main__":
    main()

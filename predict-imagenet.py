# -*- coding: utf-8 -*-
"""
predict_smp_imagenet.py — 使用 segmentation_models_pytorch 的 UNet(ResNet34, encoder_weights='imagenet')
在 test.txt 上做预测（未在你数据上训练，仅用 ImageNet 预训练的 encoder）

输出：
- 概率热图（pred_smp_prob）
- 固定阈值(0.5)二值掩膜（pred_smp_bin_fixed）
- 自适应阈值(top-K%)二值掩膜（pred_smp_bin_topk）
- 可视化叠加（pred_smp_viz_fixed / pred_smp_viz_topk）
- 汇总统计 CSV（pred_smp_stats.csv）

路径硬编码到：E:\Eelgrass_processed_images_2025\Alaska
"""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLED"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

from pathlib import Path
import csv
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F

# pip install segmentation-models-pytorch timm
import segmentation_models_pytorch as smp

# ====== 硬编码路径 & 选项（与训练保持一致）======
BASE      = Path(r"E:\Eelgrass_processed_images_2025\Alaska")
DIR_IMG   = BASE / "image"
DIR_GLCM  = BASE / "glcm"
DIR_SPLIT = BASE / "splits"

OUT_PROB       = BASE / "pred_smp_prob"
OUT_BIN_FIXED  = BASE / "pred_smp_bin_fixed"
OUT_BIN_TOPK   = BASE / "pred_smp_bin_topk"
OUT_VIZ_FIXED  = BASE / "pred_smp_viz_fixed"
OUT_VIZ_TOPK   = BASE / "pred_smp_viz_topk"
CSV_STATS      = BASE / "pred_smp_stats.csv"

IMG_SIZE    = 768                 # 与 train_update.py 保持一致（可改 640/512 加速）
EXTRA_MODE  = "append4"           # None | "append4" | "replace_red"
THRESH_FIX  = 0.5                 # 固定阈值
TOPK_PERCENT = 1.0                # 自适应阈值：每张图 top-K% 为正（如 1.0 表示 top 1%）

SAVE_VIZ    = True

VALID_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]


def find_first(folder: Path, stem: str) -> Path:
    for ext in VALID_EXTS:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    m = list(folder.glob(stem + ".*"))
    if not m:
        raise FileNotFoundError(f"not found: {folder}/{stem}.*")
    return m[0]


def resize_longest_side(pil_img: Image.Image, target: int, is_mask: bool) -> Image.Image:
    w, h = pil_img.size
    if max(w, h) == target:
        return pil_img
    scale = float(target) / float(max(w, h))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return pil_img.resize((new_w, new_h), Image.NEAREST if is_mask else Image.BILINEAR)


def pad_to_square(pil_img: Image.Image, target: int, fill=0) -> Image.Image:
    w, h = pil_img.size
    if (w, h) == (target, target):
        return pil_img
    canvas = Image.new(pil_img.mode, (target, target), color=fill)
    canvas.paste(pil_img, (0, 0))
    return canvas


def pil_to_chw_float(pil_img: Image.Image) -> np.ndarray:
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    arr = np.asarray(pil_img).transpose(2, 0, 1).astype(np.float32)
    if (arr > 1).any():
        arr /= 255.0
    return arr


@torch.no_grad()
def main():
    # 输出目录
    OUT_PROB.mkdir(parents=True, exist_ok=True)
    OUT_BIN_FIXED.mkdir(parents=True, exist_ok=True)
    OUT_BIN_TOPK.mkdir(parents=True, exist_ok=True)
    if SAVE_VIZ:
        OUT_VIZ_FIXED.mkdir(parents=True, exist_ok=True)
        OUT_VIZ_TOPK.mkdir(parents=True, exist_ok=True)

    stems = [s.strip() for s in (DIR_SPLIT / "test.txt").read_text(encoding="utf-8").splitlines() if s.strip()]
    if not stems:
        raise RuntimeError("splits/test.txt is empty or missing.")

    # —— 构建 SMP UNet（ResNet34 + ImageNet 预训练 encoder），输入通道与 EXTRA_MODE 对齐 —— #
    in_ch = 3
    if EXTRA_MODE in ("append4", "replace_red"):
        in_ch = 4 if EXTRA_MODE == "append4" else 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",   # 只加载 ImageNet 的 encoder 预训练权重
        in_channels=in_ch,            # 如果是4通道，首层权重会自动适配（第4通道通常随机/复制初始化）
        classes=1,                    # 二分类输出 1通道
        activation=None               # 输出 logits，自行做 sigmoid
    ).to(device).eval()

    # 统计 CSV
    with open(CSV_STATS, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "stem",
            "pos_ratio_fixed_%", "pos_ratio_topk_%",
            "prob_min", "prob_max", "prob_mean",
            "topk_percent"
        ])

        pbar = tqdm(stems, desc="Predicting with SMP(ResNet34, ImageNet encoder)")
        for stem in pbar:
            # 读原图
            ipath = find_first(DIR_IMG, stem)
            img0  = Image.open(ipath).convert("RGB")
            orig_w, orig_h = img0.size

            # 预处理：resize→pad→CHW
            img_rs  = resize_longest_side(img0, IMG_SIZE, is_mask=False)
            img_pad = pad_to_square(img_rs, IMG_SIZE, fill=0)
            x_arr   = pil_to_chw_float(img_pad)

            # 可选 GLCM
            if EXTRA_MODE in ("append4", "replace_red"):
                gp = find_first(DIR_GLCM, stem)
                g   = Image.open(gp)
                if g.mode != "L":
                    g = g.convert("L")
                g_rs  = resize_longest_side(g, IMG_SIZE, is_mask=False)
                g_pad = pad_to_square(g_rs, IMG_SIZE, fill=0)
                g_arr = np.asarray(g_pad).astype(np.float32)
                if (g_arr > 1).any():
                    g_arr /= 255.0
                g_arr = g_arr[None, ...]  # 1xSxS
                if EXTRA_MODE == "replace_red":
                    x_arr[0:1, ...] = g_arr
                else:
                    x_arr = np.concatenate([x_arr, g_arr], axis=0)

            x = torch.from_numpy(x_arr).unsqueeze(0).to(device=device, dtype=torch.float32)

            # 前向
            logit = model(x)                  # (1,1,S,S)
            prob  = torch.sigmoid(logit)[0,0] # (S,S)
            prob_np_full = prob.detach().cpu().numpy()

            # 去 pad（pad 在右下角）
            rs_w, rs_h = img_rs.size
            prob_np = prob_np_full[:rs_h, :rs_w]

            # 还原到原图尺寸（概率图）
            prob_img = Image.fromarray((prob_np * 255).astype(np.uint8), mode="L")
            prob_up  = prob_img.resize((orig_w, orig_h), Image.BILINEAR)
            prob_up_np = np.asarray(prob_up).astype(np.float32) / 255.0

            # —— 固定阈值 0.5 —— #
            pred_fix = (prob_up_np >= THRESH_FIX).astype(np.uint8) * 255
            Image.fromarray(pred_fix, mode="L").save(OUT_BIN_FIXED / f"{stem}.png")

            # —— 自适应阈值 top-K% —— #
            k = float(np.clip(TOPK_PERCENT, 0.05, 10.0))  # 0.05%~10% 合理区间
            q = 100.0 - k                                 # 例如 k=1.0 → 99分位为阈值
            thr = float(np.percentile(prob_up_np, q))
            pred_topk = (prob_up_np >= thr).astype(np.uint8) * 255
            Image.fromarray(pred_topk, mode="L").save(OUT_BIN_TOPK / f"{stem}.png")

            # 概率热图
            prob_up.save(OUT_PROB / f"{stem}.png")

            # 叠加可视化
            if SAVE_VIZ:
                def overlay(mask255, out_dir):
                    rgb = img0.copy().convert("RGBA")
                    rr = (mask255 > 0).astype(np.uint8) * 255
                    ov = np.stack([rr, np.zeros_like(rr), np.zeros_like(rr), (rr*0.4).astype(np.uint8)], axis=-1)
                    blended = Image.alpha_composite(rgb, Image.fromarray(ov, mode="RGBA")).convert("RGB")
                    blended.save(out_dir / f"{stem}.jpg")
                overlay(pred_fix, OUT_VIZ_FIXED)
                overlay(pred_topk, OUT_VIZ_TOPK)

            # 统计
            pos_fix  = float(pred_fix.mean() / 255.0) * 100.0
            pos_topk = float(pred_topk.mean() / 255.0) * 100.0
            writer.writerow([
                stem,
                f"{pos_fix:.3f}", f"{pos_topk:.3f}",
                f"{prob_up_np.min():.5f}", f"{prob_up_np.max():.5f}", f"{prob_up_np.mean():.5f}",
                f"{TOPK_PERCENT:.2f}"
            ])

    print(f"✅ prob heatmaps : {OUT_PROB}")
    print(f"✅ bin(fixed@0.5): {OUT_BIN_FIXED}")
    print(f"✅ bin(topK)     : {OUT_BIN_TOPK}")
    if SAVE_VIZ:
        print(f"✅ viz fixed     : {OUT_VIZ_FIXED}")
        print(f"✅ viz topK      : {OUT_VIZ_TOPK}")
    print(f"✅ stats csv     : {CSV_STATS}")


if __name__ == "__main__":
    main()

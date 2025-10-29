# -*- coding: utf-8 -*-
"""
predict.py — 读取 test.txt，生成同尺寸预测掩膜（0/1 PNG）及可选叠加图
硬编码数据根目录：E:\Eelgrass_processed_images_2025\Alaska
要求：best.pth 已在 checkpoints 目录
依赖：torch, numpy, pillow, tqdm
"""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLED"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F

from unet.unet_model import UNet  # 你仓库里的 UNet 入口

# ===== 路径 & 选项（与训练保持一致）=====
BASE      = Path(r"E:\Eelgrass_processed_images_2025\Alaska")
DIR_IMG   = BASE / "image"
DIR_MASK  = BASE / "index"        # 若 test 有 GT，可用于简单对比统计；没有也可运行
DIR_GLCM  = BASE / "glcm"
DIR_SPLIT = BASE / "splits"
DIR_CKPT  = BASE / "checkpoints"
OUT_DIR   = BASE / "pred"         # 二值掩膜输出目录
VIZ_DIR   = BASE / "pred_viz"     # 可选可视化叠加

IMG_SIZE    = 768                 # 与 train_update.py 一致
EXTRA_MODE  = "append4"           # None | "append4" | "replace_red"
THRESH      = 0.5                 # 默认阈值
SAVE_VIZ    = True                # 保存叠加可视化

VALID_EXTS = [".png",".jpg",".jpeg",".tif",".tiff",".bmp"]

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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_VIZ:
        VIZ_DIR.mkdir(parents=True, exist_ok=True)

    stems = [s.strip() for s in (DIR_SPLIT / "test.txt").read_text(encoding="utf-8").splitlines() if s.strip()]
    if not stems:
        raise RuntimeError("splits/test.txt is empty or missing.")

    # 探测输入通道数
    sample_img = Image.open(find_first(DIR_IMG, stems[0])).convert("RGB")
    c_in = 3
    if EXTRA_MODE in ("append4","replace_red"):
        c_in = 4 if EXTRA_MODE == "append4" else 3  # replace_red 仍是3通道
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_channels=c_in, n_classes=1, bilinear=False).to(device)
    ckpt = DIR_CKPT / "best.pth"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    pbar = tqdm(stems, desc="Predicting test set")
    for stem in pbar:
        ipath = find_first(DIR_IMG, stem)
        img0  = Image.open(ipath).convert("RGB")
        orig_w, orig_h = img0.size

        # 预处理（与训练一致）
        img_rs  = resize_longest_side(img0, IMG_SIZE, is_mask=False)
        img_pad = pad_to_square(img_rs, IMG_SIZE, fill=0)
        x_arr   = pil_to_chw_float(img_pad)

        if EXTRA_MODE in ("append4","replace_red"):
            gp = find_first(DIR_GLCM, stem)
            g   = Image.open(gp)
            if g.mode != "L":
                g = g.convert("L")
            g_rs  = resize_longest_side(g, IMG_SIZE, is_mask=False)
            g_pad = pad_to_square(g_rs, IMG_SIZE, fill=0)
            g_arr = np.asarray(g_pad).astype(np.float32)
            if (g_arr > 1).any(): g_arr /= 255.0
            g_arr = g_arr[None, ...]  # 1xSxS
            if EXTRA_MODE == "replace_red":
                x_arr[0:1, ...] = g_arr
            else:
                x_arr = np.concatenate([x_arr, g_arr], axis=0)  # 4xSxS

        x = torch.from_numpy(x_arr).unsqueeze(0).to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        logit = model(x)                    # (1,1,S,S)
        prob  = torch.sigmoid(logit)[0,0]   # (S,S)
        # 去 pad：这里 pad 是右下角方向；原缩放后尺寸：
        rs_w, rs_h = img_rs.size
        prob_np = prob[:rs_h, :rs_w].detach().cpu().numpy()

        # 还原到原图尺寸
        prob_img = Image.fromarray((prob_np*255).astype(np.uint8), mode="L")
        prob_up  = prob_img.resize((orig_w, orig_h), Image.BILINEAR)
        pred_bin = (np.array(prob_up) / 255.0) > THRESH
        pred_bin = (pred_bin.astype(np.uint8) * 255)

        # 保存掩膜
        out_mask = OUT_DIR / f"{stem}.png"
        Image.fromarray(pred_bin, mode="L").save(out_mask)

        if SAVE_VIZ:
            # 简易叠加：将掩膜以红色覆盖
            rgb = img0.copy().convert("RGBA")
            overlay = Image.fromarray(np.zeros((orig_h, orig_w, 4), dtype=np.uint8), mode="RGBA")
            rr = (pred_bin > 0).astype(np.uint8) * 255
            ov = np.stack([rr, np.zeros_like(rr), np.zeros_like(rr), (rr*0.4).astype(np.uint8)], axis=-1)  # 半透明 red
            overlay = Image.fromarray(ov, mode="RGBA")
            blended = Image.alpha_composite(rgb, overlay).convert("RGB")
            blended.save(VIZ_DIR / f"{stem}.jpg")

    print(f"✅ Done. Masks @ {OUT_DIR}")
    if SAVE_VIZ:
        print(f"✅ Viz  @ {VIZ_DIR}")

if __name__ == "__main__":
    main()

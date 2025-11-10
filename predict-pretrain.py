# -*- coding: utf-8 -*-
"""
compare_unet_pretrained_vs_scratch.py
— 对比 Segmentation Models PyTorch (smp) 的 UNet：
   1) 从零权重 (encoder_weights=None)
   2) 预训练 encoder (encoder_weights='imagenet')

输入:  BASE/splits/test.txt （每行是不带扩展名的stem）
输出:
  BASE/pred_scratch/*.png
  BASE/pred_pretrained/*.png
  BASE/metrics_compare_scratch_vs_pretrained.csv
可选: BASE/pred_viz_scratch/*.jpg, BASE/pred_viz_pretrained/*.jpg 叠加可视化
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

try:
    import segmentation_models_pytorch as smp
except Exception as e:
    raise RuntimeError(
        "需要安装 segmentation_models_pytorch：\n"
        "  pip install -U segmentation-models-pytorch\n"
        "（若在 Windows + CUDA，确保 torch/torchvision 已正确安装与GPU匹配）"
    ) from e

# ================== 路径与配置（按需修改） ==================
BASE       = Path(r"D:\Eelgrass_Process_2025_Bo\DroneVision_Model_data\BC")
DIR_IMG    = BASE / "image"
DIR_MASK   = BASE / "index"          # 若无GT也可运行，只是无法算指标
DIR_SPLIT  = BASE / "splits"
OUT_SCR    = BASE / "pred_scratch"
OUT_PRE    = BASE / "pred_pretrained"
OUT_VIZ_S  = BASE / "pred_viz_scratch"
OUT_VIZ_P  = BASE / "pred_viz_pretrained"
OUT_CSV    = BASE / "metrics_compare_scratch_vs_pretrained.csv"

IMG_SIZE     = 768                   # 与现有流程对齐（可改小如640提速）
BACKBONE     = "resnet34"            # 可改：'resnet18','resnet50','timm-efficientnet-b0',等
THRESH       = 0.5
SAVE_VIZ     = True                  # 保存红色叠加可视化
USE_AMP      = True                  # 推理混精以提速
VALID_EXTS   = [".png",".jpg",".jpeg",".tif",".tiff",".bmp"]

for p in [OUT_SCR, OUT_PRE, (OUT_VIZ_S if SAVE_VIZ else None), (OUT_VIZ_P if SAVE_VIZ else None)]:
    if p is not None:
        p.mkdir(parents=True, exist_ok=True)


# ================== 工具函数 ==================
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
    cands = list(folder.glob(clean + ".*"))
    if cands:
        return cands[0]
    raise FileNotFoundError(f"not found: {folder}\\{clean}.*")

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

def calc_metrics(pred_bin: np.ndarray, gt_bin: np.ndarray):
    """返回 (dice, iou, precision, recall, has_gt)"""
    p = pred_bin > 0
    g = gt_bin > 0
    tp = np.logical_and(p, g).sum()
    fp = np.logical_and(p, ~g).sum()
    fn = np.logical_and(~p, g).sum()

    denom_dice = (2 * tp + fp + fn)
    dice = (2 * tp) / denom_dice if denom_dice > 0 else (1.0 if g.sum() == 0 else 0.0)

    union = tp + fp + fn
    iou = (tp / union) if union > 0 else (1.0 if g.sum() == 0 else 0.0)

    prec = (tp / (tp + fp)) if (tp + fp) > 0 else (1.0 if g.sum() == 0 else 0.0)
    rec  = (tp / (tp + fn)) if (tp + fn) > 0 else (1.0 if g.sum() == 0 else 0.0)

    return dice, iou, prec, rec, (g.sum() > 0)


# ================== 主逻辑 ==================
@torch.no_grad()
def run_one_model(model: torch.nn.Module, stems, tag_out_dir: Path, tag_viz_dir: Path, device):
    """对给定 model 逐图推理并返回逐图指标列表（若有GT）。
       返回 records: List[[stem, iou, dice, prec, rec, has_gt]]"""
    model.eval()
    records = []

    for stem in tqdm(stems, desc=f"Predicting ({tag_out_dir.name})"):
        ip = find_first(DIR_IMG, stem)
        img0 = Image.open(ip).convert("RGB")
        orig_w, orig_h = img0.size

        img_rs  = resize_longest_side(img0, IMG_SIZE, is_mask=False)
        img_pad = pad_to_square(img_rs, IMG_SIZE, fill=0)
        x_arr   = pil_to_chw_float(img_pad)

        x = torch.from_numpy(x_arr).unsqueeze(0).to(device=device, dtype=torch.float32)
        with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=USE_AMP):
            logit = model(x)                 # (1,1,S,S)
            prob  = torch.sigmoid(logit)[0,0].float()

        # 去 pad → 回到缩放后尺寸
        rs_w, rs_h = img_rs.size
        prob_np = prob[:rs_h, :rs_w].detach().cpu().numpy()

        # 复原到原图
        prob_img = Image.fromarray((prob_np*255).astype(np.uint8), mode="L")
        prob_up  = prob_img.resize((orig_w, orig_h), Image.BILINEAR)
        pred_bin = ((np.array(prob_up)/255.0) > THRESH).astype(np.uint8)*255

        # 保存掩膜
        Image.fromarray(pred_bin, mode="L").save(tag_out_dir / f"{stem}.png")

        # 可视化叠加
        if SAVE_VIZ:
            rgba = img0.convert("RGBA")
            ov = np.zeros((orig_h, orig_w, 4), np.uint8)
            ov[..., 0] = (pred_bin > 0).astype(np.uint8) * 255
            ov[..., 3] = (pred_bin > 0).astype(np.uint8) * 120
            blended = Image.alpha_composite(rgba, Image.fromarray(ov, "RGBA")).convert("RGB")
            blended.save(tag_viz_dir / f"{stem}.jpg")

        # 指标（若有GT）
        try:
            mp = find_first(DIR_MASK, stem)
            gt = Image.open(mp).convert("L").resize((orig_w, orig_h), Image.NEAREST)
            gt_bin = (np.array(gt) > 0).astype(np.uint8) * 255
            dice, iou, prec, rec, has_gt = calc_metrics(pred_bin, gt_bin)
            records.append([stem, iou, dice, prec, rec, has_gt])
        except Exception:
            records.append([stem, np.nan, np.nan, np.nan, np.nan, False])

    return records


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_file = DIR_SPLIT / "test.txt"
    stems = [strip_known_ext(s.strip()) for s in split_file.read_text(encoding="utf-8").splitlines() if s.strip()]
    if not stems:
        raise RuntimeError("splits/test.txt 为空或缺失")

    # --------- 构建两套模型：Scratch vs Pretrained ---------
    # SMP UNet: logits shape [B, classes, H, W]，这里 classes=1（二分类）
    model_scratch = smp.Unet(
        encoder_name=BACKBONE,
        encoder_weights=None,      # 从零权重
        in_channels=3,
        classes=1
    ).to(device)

    model_pretrained = smp.Unet(
        encoder_name=BACKBONE,
        encoder_weights="imagenet",  # 预训练encoder权重
        in_channels=3,
        classes=1
    ).to(device)

    # --------- 逐模型推理 ---------
    rec_s = run_one_model(model_scratch,  stems, OUT_SCR, (OUT_VIZ_S if SAVE_VIZ else OUT_SCR), device)
    rec_p = run_one_model(model_pretrained, stems, OUT_PRE, (OUT_VIZ_P if SAVE_VIZ else OUT_PRE), device)

    # --------- 合并结果并写出 CSV ---------
    # 将两个 records 对齐同一个 stem 顺序（都按 stems 遍历生成，顺序一致）
    assert len(rec_s) == len(rec_p) == len(stems)

    # 汇总统计容器
    S = {"iou": [], "dice": [], "prec": [], "rec": [], "iou_ne": [], "dice_ne": []}
    P = {"iou": [], "dice": [], "prec": [], "rec": [], "iou_ne": [], "dice_ne": []}

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "stem",
            "iou_scratch","dice_scratch","precision_scratch","recall_scratch",
            "iou_pretrained","dice_pretrained","precision_pretrained","recall_pretrained",
            "has_gt"
        ])
        for (stem_s, iou_s, dice_s, prec_s, rec_s_, has_gt_s), \
            (stem_p, iou_p, dice_p, prec_p, rec_p_, has_gt_p) in zip(rec_s, rec_p):
            # 保护性：stem 应一致
            assert stem_s == stem_p
            w.writerow([stem_s, iou_s, dice_s, prec_s, rec_s_, iou_p, dice_p, prec_p, rec_p_, bool(has_gt_s or has_gt_p)])

            # 聚合
            if not np.isnan(iou_s):  S["iou"].append(iou_s)
            if not np.isnan(dice_s): S["dice"].append(dice_s)
            if not np.isnan(prec_s): S["prec"].append(prec_s)
            if not np.isnan(rec_s_): S["rec"].append(rec_s_)
            if (has_gt_s is True) and (not np.isnan(iou_s)):  S["iou_ne"].append(iou_s)
            if (has_gt_s is True) and (not np.isnan(dice_s)): S["dice_ne"].append(dice_s)

            if not np.isnan(iou_p):  P["iou"].append(iou_p)
            if not np.isnan(dice_p): P["dice"].append(dice_p)
            if not np.isnan(prec_p): P["prec"].append(prec_p)
            if not np.isnan(rec_p_): P["rec"].append(rec_p_)
            if (has_gt_p is True) and (not np.isnan(iou_p)):  P["iou_ne"].append(iou_p)
            if (has_gt_p is True) and (not np.isnan(dice_p)): P["dice_ne"].append(dice_p)

        # 空行 + 汇总
        w.writerow([])
        def safe_mean(arr): return float(np.nanmean(arr)) if len(arr) else 0.0

        w.writerow(["[SCRATCH] mean_iou_overall",        safe_mean(S["iou"])])
        w.writerow(["[SCRATCH] mean_dice_overall",       safe_mean(S["dice"])])
        w.writerow(["[SCRATCH] mean_precision_overall",  safe_mean(S["prec"])])
        w.writerow(["[SCRATCH] mean_recall_overall",     safe_mean(S["rec"])])
        w.writerow(["[SCRATCH] mean_iou_non_empty_gt",   safe_mean(S["iou_ne"])])
        w.writerow(["[SCRATCH] mean_dice_non_empty_gt",  safe_mean(S["dice_ne"])])

        w.writerow([])
        w.writerow(["[PRETRAINED] mean_iou_overall",        safe_mean(P["iou"])])
        w.writerow(["[PRETRAINED] mean_dice_overall",       safe_mean(P["dice"])])
        w.writerow(["[PRETRAINED] mean_precision_overall",  safe_mean(P["prec"])])
        w.writerow(["[PRETRAINED] mean_recall_overall",     safe_mean(P["rec"])])
        w.writerow(["[PRETRAINED] mean_iou_non_empty_gt",   safe_mean(P["iou_ne"])])
        w.writerow(["[PRETRAINED] mean_dice_non_empty_gt",  safe_mean(P["dice_ne"])])

    print("✅ Done.")
    print(f"CSV => {OUT_CSV}")
    print(f"Scratch masks:    {OUT_SCR}")
    print(f"Pretrained masks: {OUT_PRE}")
    if SAVE_VIZ:
        print(f"Scratch viz:     {OUT_VIZ_S}")
        print(f"Pretrained viz:  {OUT_VIZ_P}")


if __name__ == "__main__":
    main()

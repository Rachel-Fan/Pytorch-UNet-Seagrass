# -*- coding: utf-8 -*-
"""
eval-metrics.py — HQ-SAM风格 + 扩展7项指标的统一评测脚本
与 train_colab_early.py 的数据读取/预处理保持一致：
- 从 BASE_DIR/{split}/{image,index} 读取 (或通过 --split-txt 指定若干 stem)
- 图像等比缩放到最长边=img_size，然后左上对齐填充到正方形(img_size,img_size)
- mask >0 视为前景

模型：
- 默认：本地 UNet (unet.unet_model.UNet)，--ckpt 载入
- 可选：--use-smp 时用 segmentation_models_pytorch.Unet(resnet34, imagenet) 作为“预训练基线”

输出：
- per_image.csv：逐样本 IoU / BoundaryIoU / Acc / Dice / Prec / Recall / Hausdorff
- summary.json / summary.csv：上述7项的宏平均
"""

import os, json, csv, argparse, math, random
from pathlib import Path
from typing import Optional, Set, Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 尽量与训练脚本保持一致的加速设置
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ---------------- repo 本地 UNet ----------------
import sys
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.append(str(Path(__file__).resolve().parent))
from unet.unet_model import UNet

VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# ---------------------- 实用函数 ----------------------
def seed_everything(seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def resize_longest_side(img: Image.Image, target: int, is_mask: bool) -> Image.Image:
    w, h = img.size
    if max(w, h) == target:
        return img
    s = float(target) / float(max(w, h))
    new_w, new_h = int(round(w * s)), int(round(h * s))
    return img.resize((new_w, new_h), Image.NEAREST if is_mask else Image.BILINEAR)

def pad_to_square(img: Image.Image, target: int, fill=0) -> Image.Image:
    w, h = img.size
    if (w, h) == (target, target):
        return img
    canvas = Image.new(img.mode, (target, target), color=fill)
    canvas.paste(img, (0, 0))
    return canvas

def pil_to_chw_float(img: Image.Image) -> np.ndarray:
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.asarray(img).transpose(2, 0, 1).astype(np.float32)
    if (arr > 1).any():
        arr /= 255.0
    return arr

def load_split_file(txt_path: Optional[str]) -> Optional[Set[str]]:
    if not txt_path:
        return None
    p = Path(txt_path)
    if not p.exists():
        raise FileNotFoundError(f"split file not found: {p}")
    stems: Set[str] = set()
    for ln in p.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if s and not s.startswith("#"):
            stems.add(Path(s).stem.lower())
    if not stems:
        raise RuntimeError(f"split file is empty: {p}")
    return stems

# ---------------------- 数据集 ----------------------
class EvalFolderDataset(Dataset):
    """与训练同风格：BASE_DIR/split/{image,index}；可用 --split-txt 过滤 stems。"""
    def __init__(self, base_dir: str, split_txt: Optional[str], img_size: int = 512,
                 split: str = "test", use_native_size: bool = False):
        self.base = Path(base_dir)
        self.split = split
        self.img_size = img_size
        self.use_native = use_native_size

        self.img_dir = self.base / split / "image"
        self.msk_dir = self.base / split / "index"
        assert self.img_dir.exists() and self.msk_dir.exists(), f"Missing {self.img_dir} or {self.msk_dir}"

        allow = load_split_file(split_txt) if split_txt else None

        # 建立 mask 索引
        idx_map: Dict[str, Path] = {}
        for mp in self.msk_dir.iterdir():
            if mp.is_file() and mp.suffix.lower() in VALID_EXTS:
                idx_map[mp.stem.lower()] = mp

        pairs: List[Tuple[Path, Path, str]] = []
        for ip in self.img_dir.iterdir():
            if not (ip.is_file() and ip.suffix.lower() in VALID_EXTS):
                continue
            stem = ip.stem
            if allow and (stem.lower() not in allow):
                continue
            m1 = idx_map.get(stem.lower(), None)
            m2 = idx_map.get((stem + "_index").lower(), None)
            mp = m1 if m1 is not None else m2
            if mp is not None:
                pairs.append((ip, mp, stem))
        if not pairs:
            raise RuntimeError(f"No (image,mask) pairs under {self.base}/{self.split} with filter={bool(allow)}")
        self.pairs = sorted(pairs, key=lambda x: x[2])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ip, mp, stem = self.pairs[i]
        img = Image.open(ip).convert("RGB")
        msk = Image.open(mp).convert("L")
        H, W = img.size[1], img.size[0]  # 原始高宽（注意 PIL 是 (W,H)）

        if self.use_native:
            img_p, msk_p = img, msk
        else:
            img_p = pad_to_square(resize_longest_side(img, self.img_size, is_mask=False), self.img_size, 0)
            msk_p = pad_to_square(resize_longest_side(msk, self.img_size, is_mask=True),  self.img_size, 0)

        x = torch.from_numpy(pil_to_chw_float(img_p).copy()).float()
        y = torch.from_numpy((np.asarray(msk_p) > 0).astype(np.int64)).long()
        return {
            "image": x,           # (C,H',W')
            "mask": y,            # (H',W') {0,1}
            "stem": stem,
            "orig_size": (H, W),  # 原始(H,W) 仅记录
        }

# ---------------------- 模型构建 ----------------------
def build_model_local(in_ch: int, ckpt: Optional[str], device):
    model = UNet(n_channels=in_ch, n_classes=1, bilinear=False).to(device=device, memory_format=torch.channels_last)
    if ckpt:
        state = torch.load(ckpt, map_location=device)
        # 兼容 DataParallel / 额外键
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        cleaned = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[7:]
            cleaned[nk] = v
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing or unexpected:
            print("[warn] state_dict mismatched:",
                  f"\n  missing({len(missing)}): {missing}\n  unexpected({len(unexpected)}): {unexpected}")
    print(f"⚙️ Using local UNet (ckpt={ckpt})")
    model.eval()
    return model

def build_model_smp(in_ch: int, device, encoder="resnet34", weights="imagenet"):
    import segmentation_models_pytorch as smp
    model = smp.Unet(encoder_name=encoder, encoder_weights=weights, in_channels=in_ch, classes=1)
    model = model.to(device)
    print("⚙️ Using segmentation_models_pytorch pretrained Unet "
          f"(encoder={encoder}, weights={weights})")
    model.eval()
    return model

# ---------------------- 度量计算 ----------------------
def _to_numpy_mask(t: torch.Tensor) -> np.ndarray:
    """(H,W) long/bool/float -> uint8 {0,1}"""
    if isinstance(t, torch.Tensor):
        a = t.detach().cpu().numpy()
    else:
        a = t
    if a.dtype != np.uint8:
        a = (a > 0.5).astype(np.uint8) if a.dtype != np.uint8 else a
        a = (a > 0).astype(np.uint8)
    return a

def confusion_from_binary(pred: torch.Tensor, gt: torch.Tensor):
    """pred, gt: (H,W) {0,1} tensor"""
    pred = pred.to(torch.bool)
    gt = gt.to(torch.bool)
    TP = (pred & gt).sum().item()
    TN = (~pred & ~gt).sum().item()
    FP = (pred & ~gt).sum().item()
    FN = (~pred & gt).sum().item()
    return TP, TN, FP, FN

def dice_from_counts(TP, FP, FN, eps=1e-6):
    return (2*TP) / (2*TP + FP + FN + eps)

def iou_from_counts(TP, FP, FN, eps=1e-6):
    return TP / (TP + FP + FN + eps)

def precision_from_counts(TP, FP, eps=1e-6):
    return TP / (TP + FP + eps)

def recall_from_counts(TP, FN, eps=1e-6):
    return TP / (TP + FN + eps)

def accuracy_from_counts(TP, TN, FP, FN, eps=1e-6):
    return (TP + TN) / (TP + TN + FP + FN + eps)

def boundary_map(mask: np.ndarray, k: int = 3) -> np.ndarray:
    """
    近似边界：B = M XOR erode(M)（3x3 结构元素）
    """
    from scipy.ndimage import binary_erosion
    m = (mask > 0)
    er = binary_erosion(m, structure=np.ones((k, k), dtype=bool), border_value=0)
    b = (m ^ er).astype(np.uint8)
    return b

def boundary_iou(pred: np.ndarray, gt: np.ndarray, eps=1e-6) -> float:
    """pred, gt: uint8 {0,1}; 基于二值边界图的 IoU"""
    bp = boundary_map(pred)
    bg = boundary_map(gt)
    inter = np.logical_and(bp, bg).sum()
    union = np.logical_or(bp, bg).sum()
    return float(inter) / float(union + eps)

def hausdorff95(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Hausdorff 95% (对称)：若双方都空则返回 0；一空一非空则返回图像对角线近似。
    用边界像素的点集计算。
    """
    from scipy.spatial.distance import cdist

    def coords_of_boundary(m: np.ndarray) -> np.ndarray:
        b = boundary_map(m)
        ys, xs = np.nonzero(b)
        if len(xs) == 0:
            # 如果没有边界像素，退化为所有前景像素
            ys, xs = np.nonzero(m)
        return np.stack([ys, xs], axis=1) if len(xs) > 0 else np.zeros((0, 2), dtype=np.int32)

    A = coords_of_boundary(pred)
    B = coords_of_boundary(gt)

    H, W = pred.shape
    diag = math.hypot(H, W)

    if len(A) == 0 and len(B) == 0:
        return 0.0
    if len(A) == 0 or len(B) == 0:
        return float(diag)

    D = cdist(A.astype(np.float32), B.astype(np.float32), metric="euclidean")
    if D.size == 0:
        return float(diag)

    # directed HD95
    d_ab = np.percentile(D.min(axis=1), 95)
    d_ba = np.percentile(D.min(axis=0), 95)
    return float(max(d_ab, d_ba))

# ---------------------- 主流程 ----------------------
def parse_args():
    ap = argparse.ArgumentParser("Evaluate (HQ-SAM style + extended metrics)")
    ap.add_argument("--base-dir", type=str, required=True, help="Dataset root with split/{image,index}")
    ap.add_argument("--split", type=str, default="test", choices=["train","valid","test"])
    ap.add_argument("--split-txt", type=str, default=None, help="Optional txt (stems) to subset within the split")
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--use-native-size", action="store_true")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--thr", type=float, default=0.5, help="prob threshold to binarize")
    ap.add_argument("--ckpt", type=str, default=None, help="Local UNet checkpoint path")
    ap.add_argument("--use-smp", action="store_true", help="Use SMP Unet(resnet34, imagenet) as pretrained baseline")
    ap.add_argument("--smp-encoder", type=str, default="resnet34")
    ap.add_argument("--smp-weights", type=str, default="imagenet")
    ap.add_argument("--out-dir", type=str, required=True, help="Where to write metrics")
    ap.add_argument("--limit", type=int, default=0, help="Eval at most N images (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()

@torch.no_grad()
def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据集
    ds = EvalFolderDataset(args.base_dir, args.split_txt,
                           img_size=args.img_size, split=args.split,
                           use_native_size=args.use_native_size)
    if args.limit and args.limit > 0:
        # 子集化
        orig_pairs = ds.pairs
        ds.pairs = orig_pairs[:min(args.limit, len(orig_pairs))]
    print(f"[split] Using full split = {args.split}")
    print(f"[split] Using {len(ds)} images in {ds.img_dir}")

    # 构建一个样本，决定 in_channels
    sample = ds[0]
    in_ch = sample["image"].shape[0]

    # 模型
    if args.use_smp:
        model = build_model_smp(in_ch, device, encoder=args.smp_encoder, weights=args.smp_weights)
    else:
        model = build_model_local(in_ch, args.ckpt, device)

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=2, pin_memory=True, persistent_workers=True)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    per_image_csv = out_root / "per_image.csv"
    summary_json = out_root / "summary.json"
    summary_csv  = out_root / "summary.csv"

    # 写 per-image 表头
    with open(per_image_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "stem",
            "iou",
            "boundary_iou",
            "accuracy",
            "dice",
            "precision",
            "recall",
            "hausdorff"
        ])

    # 累积器（宏平均）
    acc_iou = []
    acc_biou = []
    acc_acc = []
    acc_dice = []
    acc_prec = []
    acc_rec = []
    acc_haus = []

    model.eval()
    for batch in tqdm(dl, desc="Eval"):
        x = batch["image"].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)  # (B,C,H,W)
        y = batch["mask"].to(device=device, dtype=torch.long)                                        # (B,H,W)
        stems = batch["stem"]

        logits = model(x).squeeze(1)               # (B,H,W)
        prob = torch.sigmoid(logits)
        pred = (prob > args.thr).long()            # (B,H,W)

        # 逐样本统计
        rows = []
        for i in range(pred.size(0)):
            p_i = pred[i]           # (H,W)
            g_i = y[i]              # (H,W)

            TP, TN, FP, FN = confusion_from_binary(p_i, g_i)
            iou  = iou_from_counts(TP, FP, FN)
            dice = dice_from_counts(TP, FP, FN)
            pre  = precision_from_counts(TP, FP)
            rec  = recall_from_counts(TP, FN)
            acc  = accuracy_from_counts(TP, TN, FP, FN)

            # boundary / hausdorff 基于 numpy 计算
            p_np = _to_numpy_mask(p_i)
            g_np = _to_numpy_mask(g_i)

            biou = boundary_iou(p_np, g_np)
            h95  = hausdorff95(p_np, g_np)

            rows.append((stems[i], iou, biou, acc, dice, pre, rec, h95))
            # accumulate
            acc_iou.append(iou)
            acc_biou.append(biou)
            acc_acc.append(acc)
            acc_dice.append(dice)
            acc_prec.append(pre)
            acc_rec.append(rec)
            acc_haus.append(h95)

        # 写批量到 per_image.csv
        with open(per_image_csv, "a", newline="") as f:
            w = csv.writer(f)
            for r in rows:
                w.writerow(list(r))

    # 汇总
    def avg(v): return float(np.mean(v)) if len(v) else 0.0
    summary = {
        "average_iou":           round(avg(acc_iou), 6),
        "average_boundary_iou":  round(avg(acc_biou), 6),
        "average_accuracy":      round(avg(acc_acc), 6),
        "average_dice":          round(avg(acc_dice), 6),
        "average_precision":     round(avg(acc_prec), 6),
        "average_recall":        round(avg(acc_rec), 6),
        "average_hausdorff":     round(avg(acc_haus), 6),
        "count": int(len(acc_iou)),
        "split": args.split,
        "base_dir": str(args.base_dir),
        "img_size": ("native" if args.use_native_size else args.img_size),
        "model": ("smp_unet" if args.use_smp else "local_unet"),
        "ckpt": (None if args.use_smp else args.ckpt),
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(summary.keys()))
        w.writerow(list(summary.values()))

    # 打印summary到终端
    print("\n📊 Average Metrics Summary")
    print("-" * 40)
    for k, v in summary.items():
        if k.startswith("average_"):
            print(f"{k:24s}: {v:.6f}")
    print("-" * 40)
    print(f"Count: {summary['count']} | Split: {summary['split']} | Model: {summary['model']}")

    print("\n✅ Done.")
    print(f"  • Per-image: {per_image_csv}")
    print(f"  • Summary  : {summary_json} / {summary_csv}")

if __name__ == "__main__":
    main()

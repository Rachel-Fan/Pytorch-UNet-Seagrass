# -*- coding: utf-8 -*-
"""
评估模块（sklearn/scipy 版）
提供：
- eval_on_loader(model, loader, device, num_classes, amp=False, include_background=False)
    -> (metrics: dict, cm: numpy.ndarray[K,K], pr_curves: dict)
- save_confusion_matrix_figure(cm, out_path, class_names=None)
- save_pr_curves_figure(pr_curves, out_path)

说明：
- metrics 包含：dice, miou, acc, prec, rec, f1, b_iou, b_f1
- cm：混淆矩阵（真实为行，预测为列）
- pr_curves：
    二分类: {'binary': {'thresholds','precision','recall','auc'}}
    多分类: {'macro': {...}, 'per_class': {c: {...}}}
依赖：torch, numpy, matplotlib, scikit-learn, scipy
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, auc, confusion_matrix
from scipy.ndimage import binary_erosion, distance_transform_edt


# ---------- 基础指标 ----------
def _dice_multiclass(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    prob = F.softmax(logits, dim=1).float()
    C = prob.size(1)
    oh = F.one_hot(target, num_classes=C).permute(0,3,1,2).float()
    inter = (prob*oh).sum(dim=(2,3))
    denom = prob.sum(dim=(2,3)) + oh.sum(dim=(2,3)) + eps
    return (2*inter/denom).mean().item()

def _dice_binary(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    prob = torch.sigmoid(logits).squeeze(1)
    tgt = target.float()
    inter = (prob*tgt).sum(dim=(1,2))
    denom = prob.sum(dim=(1,2)) + tgt.sum(dim=(1,2)) + eps
    return (2*inter/denom).mean().item()

def _iou_binary(pred: torch.Tensor, tgt: torch.Tensor, eps: float = 1e-6) -> float:
    inter = ((pred==1) & (tgt==1)).sum().float()
    union = ((pred==1) | (tgt==1)).sum().float() + eps
    return (inter/union).item()

def _iou_multiclass(pred: torch.Tensor, tgt: torch.Tensor, C: int, eps: float = 1e-6) -> float:
    ious = []
    for c in range(C):
        p = (pred==c); t=(tgt==c)
        inter = (p & t).sum().float()
        union = (p | t).sum().float() + eps
        ious.append(inter/union)
    return torch.stack(ious).mean().item()

def _acc_prec_rec_f1_from_cm(cm: np.ndarray, eps: float = 1e-6):
    C = cm.shape[0]
    acc = cm.trace() / (cm.sum() + eps)
    precs, recs, f1s = [], [], []
    for c in range(C):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp + eps)
        rec  = tp / (tp + fn + eps)
        f1   = 2*prec*rec / (prec+rec + eps)
        precs.append(prec); recs.append(rec); f1s.append(f1)
    return float(acc), float(np.mean(precs)), float(np.mean(recs)), float(np.mean(f1s))


# ---------- Boundary（binary_erosion 版） ----------
def _boundary_from_binary(mask01: np.ndarray) -> np.ndarray:
    """mask01: HxW 0/1 ndarray -> boundary(bool) via erosion-xor"""
    er = binary_erosion(mask01.astype(bool),
                        structure=np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=bool),
                        border_value=0)
    return mask01.astype(bool) ^ er

def _boundary_iou_f1_binary_np(pred01: np.ndarray, tgt01: np.ndarray, eps: float = 1e-6) -> Tuple[float,float]:
    pb = _boundary_from_binary(pred01)
    tb = _boundary_from_binary(tgt01)
    inter = np.logical_and(pb, tb).sum()
    union = np.logical_or(pb, tb).sum() + eps
    iou = inter / union

    tp = inter
    fp = np.logical_and(pb, ~tb).sum()
    fn = np.logical_and(~pb, tb).sum()
    prec = tp / (tp + fp + eps)
    rec  = tp / (tp + fn + eps)
    f1   = 2*prec*rec / (prec+rec + eps)
    return float(iou), float(f1)

def _boundary_iou_f1_multiclass(pred: torch.Tensor, tgt: torch.Tensor, C: int,
                                include_background: bool = False, eps: float = 1e-6) -> Tuple[float,float]:
    preds_np = pred.cpu().numpy()
    tgt_np   = tgt.cpu().numpy()
    classes = list(range(C))
    if not include_background and C > 1:
        classes = classes[1:]
    ious, f1s = [], []
    for c in classes:
        i, f = _boundary_iou_f1_binary_np((preds_np==c).astype(np.uint8), (tgt_np==c).astype(np.uint8), eps=eps)
        ious.append(i); f1s.append(f)
    if len(ious)==0:
        return 0.0, 0.0
    return float(np.mean(ious)), float(np.mean(f1s))


# ---------- Hausdorff95（可选，当前未返回；如需可加入 metrics） ----------
def _hausdorff95_binary_np(pred01: np.ndarray, tgt01: np.ndarray) -> float:
    if pred01.max()==0 or tgt01.max()==0:
        return np.nan
    pb = _boundary_from_binary(pred01)
    tb = _boundary_from_binary(tgt01)
    if pb.sum()==0 or tb.sum()==0:
        return np.nan
    dt_pred = distance_transform_edt(~pb)
    dt_tgt  = distance_transform_edt(~tb)
    d1 = dt_tgt[pb]
    d2 = dt_pred[tb]
    all_d = np.concatenate([d1, d2]).astype(np.float32)
    if all_d.size == 0:
        return np.nan
    return float(np.percentile(all_d, 95))


# ---------- PR 曲线 ----------
def _pr_binary(prob: np.ndarray, tgt01: np.ndarray) -> Dict[str, np.ndarray]:
    y_true = tgt01.reshape(-1).astype(np.uint8)
    y_score = prob.reshape(-1).astype(np.float32)
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    return {"thresholds": thresholds, "precision": precision, "recall": recall, "auc": float(pr_auc)}

def _pr_multiclass(logits: torch.Tensor, tgt: torch.Tensor, C: int, include_background: bool = False):
    prob = F.softmax(logits, dim=1).float().cpu().numpy()   # N x C x H x W
    y = tgt.cpu().numpy()                                   # N x H x W
    classes = list(range(C))
    if not include_background and C > 1:
        classes = classes[1:]
    per_class: Dict[int, Dict[str, np.ndarray]] = {}
    prs = []; rcs = []
    for c in classes:
        pc = prob[:, c, ...]
        tc = (y == c).astype(np.uint8)
        cur = _pr_binary(pc, tc)
        per_class[c] = cur
        # 统一到固定 recall 栅格求 macro 平均曲线
        Rg = np.linspace(0,1,200)
        # PRC 输出的 recall 是升序，precision 与 recall 同长
        prs.append(np.interp(Rg, cur["recall"], cur["precision"]))
        rcs.append(Rg)
    if len(prs)==0:
        macro = {"thresholds": np.array([]), "precision": np.zeros(200), "recall": np.linspace(0,1,200), "auc": 0.0}
    else:
        Pm = np.mean(np.stack(prs, axis=0), axis=0)
        Rm = np.linspace(0,1,200)
        macro = {"thresholds": np.array([]), "precision": Pm, "recall": Rm, "auc": float(auc(Rm, Pm))}
    return macro, per_class


# ---------- 可视化 ----------
def save_confusion_matrix_figure(cm: np.ndarray, out_path, class_names: Optional[list] = None):
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm.astype(np.float64), interpolation='nearest')
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax)
    if class_names:
        ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right'); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]:.0f}", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

def save_pr_curves_figure(pr_curves: Dict, out_path):
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111)
    if "binary" in pr_curves:
        pr = pr_curves["binary"]
        ax.plot(pr["recall"], pr["precision"], label=f"binary (AUC={pr['auc']:.3f})")
    else:
        mac = pr_curves["macro"]
        ax.plot(mac["recall"], mac["precision"], label=f"macro (AUC={mac['auc']:.3f})", linewidth=2)
        for c, d in pr_curves.get("per_class", {}).items():
            ax.plot(d["recall"], d["precision"], alpha=0.35, linewidth=1, label=f"class {c} (AUC={d['auc']:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("PR Curve")
    ax.grid(True, linewidth=0.3)
    ax.legend(fontsize=8, ncol=1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ---------- 主评估入口 ----------
@torch.no_grad()
def eval_on_loader(model, loader, device, num_classes: int,
                   amp: bool = False, pr_points: int = 40,
                   include_background: bool = False):
    """
    返回：
      metrics: dict(dice, miou, acc, prec, rec, f1, b_iou, b_f1)
      cm: KxK numpy array
      pr_curves: dict
    """
    model.eval()
    dice_sum = 0.0
    miou_sum = 0.0
    cm_total = np.zeros((num_classes, num_classes), dtype=np.int64)
    b_iou_list, b_f1_list = [], []

    all_logits = []
    all_targets = []

    for batch in loader:
        imgs = batch["image"].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        gts  = batch["mask"].to(device=device, dtype=torch.long)

        with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
            logits = model(imgs)

        if num_classes > 1:
            dice_sum += _dice_multiclass(logits, gts)
            preds = logits.argmax(dim=1)
            miou_sum += _iou_multiclass(preds, gts, num_classes)
            bi, bf = _boundary_iou_f1_multiclass(preds, gts, num_classes, include_background=include_background)
        else:
            dice_sum += _dice_binary(logits, gts)
            prob = torch.sigmoid(logits).squeeze(1)
            preds = (prob > 0.5).long()
            miou_sum += _iou_binary(preds, gts)
            # 二分类 boundary（逐样本）
            pn = preds.cpu().numpy().astype(np.uint8)
            tn = gts.cpu().numpy().astype(np.uint8)
            bi_list = []; bf_list = []
            for i in range(pn.shape[0]):
                iou_b, f1_b = _boundary_iou_f1_binary_np(pn[i], tn[i])
                bi_list.append(iou_b); bf_list.append(f1_b)
            bi = float(np.nanmean(bi_list)); bf = float(np.nanmean(bf_list))

        b_iou_list.append(bi); b_f1_list.append(bf)

        # 混淆矩阵（真实为行，预测为列）
        cm_batch = confusion_matrix(
            gts.view(-1).cpu().numpy(), preds.view(-1).cpu().numpy(),
            labels=list(range(num_classes))
        )
        cm_total += cm_batch

        all_logits.append(logits.cpu())
        all_targets.append(gts.cpu())

    n_batches = max(1, len(loader))
    dice = dice_sum / n_batches
    miou = miou_sum / n_batches
    b_iou = float(np.nanmean(b_iou_list))
    b_f1  = float(np.nanmean(b_f1_list))
    acc, prec, rec, f1 = _acc_prec_rec_f1_from_cm(cm_total)

    # PR 曲线
    logits_all = torch.cat(all_logits, dim=0)
    targets_all = torch.cat(all_targets, dim=0)
    pr_curves: Dict = {}
    if num_classes == 1:
        prob = torch.sigmoid(logits_all).squeeze(1).cpu().numpy()
        tgt01 = targets_all.cpu().numpy().astype(np.uint8)
        pr_curves["binary"] = _pr_binary(prob, tgt01)
    else:
        macro, per_class = _pr_multiclass(logits_all, targets_all, num_classes, include_background=include_background)
        pr_curves["macro"] = macro
        pr_curves["per_class"] = per_class

    metrics = dict(dice=dice, miou=miou, acc=acc, prec=prec, rec=rec, f1=f1,
                   b_iou=b_iou, b_f1=b_f1)
    return metrics, cm_total, pr_curves

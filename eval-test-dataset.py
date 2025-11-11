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
        return {"image": x, "mask": y, "stem": ip.stem, "size": y.shape}

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
@torch.no_grad()
def evaluate(model, loader, device, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fcsv = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(fcsv)
    writer.writerow(["filename","H","W","pixels_pred","pixels_gt","inter","union",
                     "iou","dice","precision","recall","tp","fp","fn","tn"])

    ious,dices,precs,recs=[],[],[],[]
    ious_ne,dices_ne=[],[]
    for batch in tqdm(loader, desc="Evaluating", unit="batch"):
        imgs=batch["image"].to(device=device,dtype=torch.float32,memory_format=torch.channels_last)
        gts=batch["mask"].cpu().numpy()
        logits=model(imgs)
        probs=torch.sigmoid(logits).squeeze(1).cpu().numpy()
        preds=(probs>0.5).astype(np.uint8)
        for stem,pred,gt,sz in zip(batch["stem"],preds,gts,batch["size"]):
            H,W=sz
            iou,dice,prec,rec,tp,fp,fn,tn,inter,union=compute_metrics(pred,gt)
            writer.writerow([stem,H,W,int(pred.sum()),int(gt.sum()),
                             inter,union,iou,dice,prec,rec,tp,fp,fn,tn])
            ious.append(iou);dices.append(dice);precs.append(prec);recs.append(rec)
            if gt.sum()>0:
                ious_ne.append(iou);dices_ne.append(dice)
    fcsv.close()

    return {
        "mean_iou_overall":np.mean(ious).item(),
        "mean_dice_overall":np.mean(dices).item(),
        "mean_precision_overall":np.mean(precs).item(),
        "mean_recall_overall":np.mean(recs).item(),
        "mean_iou_non_empty_gt":np.mean(ious_ne).item() if ious_ne else float("nan"),
        "mean_dice_non_empty_gt":np.mean(dices_ne).item() if dices_ne else float("nan"),
    }

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

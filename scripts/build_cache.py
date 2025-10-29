from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

BASE = Path(r"E:\Eelgrass_processed_images_2025\Alaska")
IMG_DIR, IDX_DIR, GLCM_DIR = BASE/"image", BASE/"index", BASE/"glcm"
SPLITS = BASE/"splits"
CACHE  = BASE/"cache"
IMG_SIZE = 768  # 与 train.py 保持一致

VALID_EXTS = [".png",".jpg",".jpeg",".tif",".tiff",".bmp"]
def find(folder, stem):
    for e in VALID_EXTS:
        p = folder/f"{stem}{e}"
        if p.exists(): return p
    m = list(folder.glob(stem+".*"))
    if not m: raise FileNotFoundError(stem)
    return m[0]

def resize_longest_side(pil, target, is_mask):
    w,h = pil.size
    if max(w,h)==target: return pil
    s = float(target)/max(w,h)
    nw,nh = int(round(w*s)), int(round(h*s))
    return pil.resize((nw,nh), Image.NEAREST if is_mask else Image.BILINEAR)

def pad_square(pil, target, fill=0):
    w,h = pil.size
    if (w,h)==(target,target): return pil
    c = Image.new(pil.mode, (target,target), color=fill)
    c.paste(pil,(0,0))
    return c

def pil_to_chw_float(p):
    if p.mode!="RGB": p = p.convert("RGB")
    a = np.asarray(p).transpose(2,0,1).astype(np.float32)
    if (a>1).any(): a/=255.0
    return torch.from_numpy(a).contiguous()

def mask_to_long(pil_mask, mask_values):
    m = np.asarray(pil_mask.convert("L"))
    out = np.zeros_like(m, dtype=np.int64)
    for i,v in enumerate(mask_values): out[m==v]=i
    return torch.from_numpy(out).contiguous()

def gather_mask_values(stems):
    vals=[]
    for s in stems[:min(500,len(stems))]:
        mp = find(IDX_DIR,s)
        vals.append(np.unique(np.asarray(Image.open(mp).convert("L"))))
    return sorted(np.unique(np.concatenate(vals)).tolist())

def build(split):
    stems = [x.strip() for x in (SPLITS/f"{split}.txt").read_text(encoding="utf-8").splitlines() if x.strip()]
    mv = gather_mask_values(stems)
    (CACHE/"image").mkdir(parents=True, exist_ok=True)
    (CACHE/"index").mkdir(parents=True, exist_ok=True)
    if GLCM_DIR.exists(): (CACHE/"glcm").mkdir(parents=True, exist_ok=True)

    for s in tqdm(stems, desc=f"cache {split}"):
        ip = find(IMG_DIR,s); mp = find(IDX_DIR,s)
        img = Image.open(ip).convert("RGB")
        msk = Image.open(mp)

        img = pad_square(resize_longest_side(img, IMG_SIZE, False), IMG_SIZE)
        msk = pad_square(resize_longest_side(msk, IMG_SIZE, True ), IMG_SIZE)

        img_t = pil_to_chw_float(img)                       # CxSxS
        msk_t = mask_to_long(msk, mv)                       # SxS
        torch.save({"tensor": img_t}, CACHE/"image"/f"{s}.pt")
        torch.save({"tensor": msk_t}, CACHE/"index"/f"{s}.pt")

        if GLCM_DIR.exists():
            gp = find(GLCM_DIR,s)
            g = Image.open(gp)
            if g.mode!="L": g=g.convert("L")
            g  = pad_square(resize_longest_side(g, IMG_SIZE, False), IMG_SIZE)
            ga = np.asarray(g).astype(np.float32)
            if (ga>1).any(): ga/=255.0
            torch.save({"tensor": torch.from_numpy(ga)[None,...].contiguous()}, CACHE/"glcm"/f"{s}.pt")

if __name__=="__main__":
    for sp in ["train","val","test"]:
        build(sp)
    print("✅ cache done:", CACHE)

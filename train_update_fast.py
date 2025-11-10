
'''
# -*- coding: utf-8 -*-
"""
train_update_fast_accum.py — 小显存稳态训练版（RGB 或 4通道，离线cache + 梯度累积）
硬编码数据根目录：E:\Eelgrass_processed_images_2025\Alaska

与之前版本的区别：
- IMG_SIZE=448（仍 OOM 可降到 384）
- BATCH_SIZE=2 + ACCUM_STEPS=8（等效总batch=16）
- 更保守的 DataLoader：num_workers=2, pin_memory=False, persistent_workers=False
- 训练循环里做梯度累积，降低单步显存峰值
"""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLED"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
# 注意：你的平台不支持 expandable_segments，这里不再设置它

from pathlib import Path
from typing import Optional, Dict
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from unet.unet_model import UNet

# ===================== 硬编码路径 & 配置 =====================
BASE_DIR   = Path(r"D:\Eelgrass_Process_2025_Bo\DroneVision_Model_data\BC")
DIR_IMG    = BASE_DIR / "image"
DIR_MASK   = BASE_DIR / "index"
DIR_GLCM   = BASE_DIR / "glcm"         # 可选
DIR_SPLITS = BASE_DIR / "splits"
DIR_CKPT   = BASE_DIR / "train1/checkpoints"
DIR_CACHE  = BASE_DIR / "train1/cache"

# —— 数据/模型设置 ——
IMG_SIZE    = 448                       # 若仍 OOM：改 384
EXTRA_MODE: Optional[str] = None        # None | 'append4' | 'replace_red'
CLASSES     = 1                         # 二分类
BILINEAR    = False

# —— 训练超参 ——（小显存安全组合）
EPOCHS        = 20
BATCH_SIZE    = 2
ACCUM_STEPS   = 8                       # 等效总batch = BATCH_SIZE * ACCUM_STEPS
LR            = 1e-4
WEIGHT_DECAY  = 1e-8
MOMENTUM      = 0.99
GRAD_CLIP     = 1.0

# —— DataLoader ——（更稳的设置）
NUM_WORKERS      = 2
PREFETCH_FACTOR  = 2
PIN_MEMORY       = False
PERSISTENT       = False

# —— AMP & CUDNN ——
AMP_ENABLED   = True
CUDNN_BENCH   = True

SAVE_EVERY_EPOCH = False
SEED = 0

# ===================== 工具函数 =====================
VALID_EXTS = [".png",".jpg",".jpeg",".tif",".tiff",".bmp"]

def find_first(folder: Path, name: str) -> Path:
    """
    支持 name 为 “stem” 或 “带扩展名的文件名”
    1) 若直接存在同名文件，则返回
    2) 否则按常见扩展名补全
    3) 否则 glob 一次
    """
    p = Path(name)
    # 直接命中：带扩展名的完整文件名
    if p.suffix:
        direct = folder / p.name
        if direct.exists():
            return direct
        # 若没命中，继续用 stem 走扩展名匹配
        stem = p.stem
    else:
        stem = p.name

    # 常见扩展名匹配
    for ext in VALID_EXTS:
        cand = folder / f"{stem}{ext}"
        if cand.exists():
            return cand

    # 最后兜底：任意扩展名
    m = list(folder.glob(stem + ".*"))
    if m:
        return m[0]

    raise FileNotFoundError(f"not found: {folder}/{stem}.*")


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

def load_tensor(path: Path):
    obj = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(obj, dict) and "tensor" in obj:
        return obj["tensor"]
    elif torch.is_tensor(obj):
        return obj
    else:
        raise TypeError(f"Unexpected object in {path}: {type(obj)}")

# ===================== Dataset =====================
class SegDataset(Dataset):
    """
    - 若 cache/{image,index,glcm} 存在 -> 直接加载 .pt（CxSxS / SxS）
    - 否则：SAM风格 resize→pad 到 IMG_SIZE
    - extra_mode: None | 'append4' | 'replace_red'
    """
    def __init__(self,
                 images_dir: Path,
                 masks_dir: Path,
                 split_list_file: Path,
                 dir_extra: Optional[Path] = None,
                 extra_mode: Optional[str] = None,
                 img_size: int = 768,
                 cache_dir: Optional[Path] = None):
        self.images_dir = Path(images_dir)
        self.masks_dir  = Path(masks_dir)
        self.dir_extra  = Path(dir_extra) if dir_extra else None
        self.extra_mode = extra_mode
        self.img_size   = img_size

        # self.stems = [s.strip() for s in Path(split_list_file).read_text(encoding="utf-8").splitlines() if s.strip()]
        raw_lines = Path(split_list_file).read_text(encoding="utf-8").splitlines()
        self.stems = [Path(s.strip()).stem for s in raw_lines if s.strip()]

        if not self.stems:
            raise RuntimeError(f"Split list is empty: {split_list_file}")

        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_cache = False
        if self.cache_dir and (self.cache_dir / "image").exists() and (self.cache_dir / "index").exists():
            if (self.dir_extra is not None and self.extra_mode is not None):
                self.use_cache = (self.cache_dir / "glcm").exists()
            else:
                self.use_cache = True

        if not self.use_cache:
            # 扫描少量样本确定 mask 值映射
            vals = []
            for stem in self.stems[:min(500, len(self.stems))]:
                mp = find_first(self.masks_dir, stem)
                vals.append(np.unique(np.asarray(Image.open(mp).convert("L"))))
            self.mask_values = sorted(np.unique(np.concatenate(vals)).tolist())
        else:
            self.mask_values = None

    def __len__(self):
        return len(self.stems)

    def _mask_to_long(self, pil_mask: Image.Image) -> np.ndarray:
        m = np.asarray(pil_mask.convert("L"))
        h, w = m.shape
        out = np.zeros((h, w), dtype=np.int64)
        for i, v in enumerate(self.mask_values):
            out[m == v] = i
        return out

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        stem = self.stems[idx]

        if self.use_cache:
            img_t  = load_tensor(self.cache_dir / "image" / f"{stem}.pt")  # CxSxS (float)
            msk_t  = load_tensor(self.cache_dir / "index" / f"{stem}.pt")  # SxS   (long)

            if self.dir_extra is not None and self.extra_mode is not None:
                glcm_t = load_tensor(self.cache_dir / "glcm"  / f"{stem}.pt")  # 1xSxS (float)
                if self.extra_mode == "replace_red":
                    img_t[0:1, ...] = glcm_t
                elif self.extra_mode == "append4":
                    img_t = torch.cat([img_t, glcm_t], dim=0)
            return {"image": img_t.float().contiguous(),
                    "mask":  msk_t.long().contiguous(),
                    "stem":  stem}

        # 非缓存：现场预处理
        ip = find_first(self.images_dir, stem)
        mp = find_first(self.masks_dir,  stem)
        img = Image.open(ip).convert("RGB")
        msk = Image.open(mp)

        extra_arr = None
        if self.dir_extra is not None and self.extra_mode is not None:
            gp = find_first(self.dir_extra, stem)
            extra = Image.open(gp).convert("L")
            extra_rs  = resize_longest_side(extra, self.img_size, is_mask=False)
            extra_pad = pad_to_square(extra_rs, self.img_size, fill=0)
            ea = np.asarray(extra_pad).astype(np.float32)
            if (ea > 1).any():
                ea /= 255.0
            extra_arr = ea[None, ...]

        img_rs  = resize_longest_side(img, self.img_size, is_mask=False)
        msk_rs  = resize_longest_side(msk, self.img_size, is_mask=True)
        img_pad = pad_to_square(img_rs, self.img_size, fill=0)
        msk_pad = pad_to_square(msk_rs, self.img_size, fill=0)

        img_arr  = pil_to_chw_float(img_pad)
        mask_arr = self._mask_to_long(msk_pad)

        if extra_arr is not None and self.extra_mode is not None:
            if self.extra_mode == "replace_red":
                img_arr[0:1, ...] = extra_arr
            elif self.extra_mode == "append4":
                img_arr = np.concatenate([img_arr, extra_arr], axis=0)

        return {"image": torch.from_numpy(img_arr.copy()).float().contiguous(),
                "mask":  torch.from_numpy(mask_arr.copy()).long().contiguous(),
                "stem":  stem}

# ===================== 简易验证 =====================
@torch.no_grad()
def quick_validate(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool = True, num_classes: int = 1):
    model.eval()
    dice_sum = 0.0
    miou_sum = 0.0
    n_batches = 0
    for batch in loader:
        x = batch["image"].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        y = batch["mask"].to(device=device, dtype=torch.long)
        with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
            logits = model(x)
        if num_classes > 1:
            prob = F.softmax(logits, dim=1).float()
            oh = F.one_hot(y, num_classes=num_classes).permute(0,3,1,2).float()
            inter = (prob*oh).sum(dim=(2,3))
            denom = prob.sum(dim=(2,3)) + oh.sum(dim=(2,3)) + 1e-6
            dice = (2*inter/denom).mean().item()
            pred = logits.argmax(dim=1)
            ious = []
            for c in range(num_classes):
                p = (pred==c); t=(y==c)
                inter_c = (p & t).sum().float()
                union_c = (p | t).sum().float() + 1e-6
                ious.append(inter_c/union_c)
            miou = torch.stack(ious).mean().item()
        else:
            prob = torch.sigmoid(logits).squeeze(1)
            inter = (prob*y.float()).sum(dim=(1,2))
            denom = prob.sum(dim=(1,2)) + y.float().sum(dim=(1,2)) + 1e-6
            dice = (2*inter/denom).mean().item()
            pred = (prob > 0.5).long()
            inter_b = ((pred==1) & (y==1)).sum().float()
            union_b = ((pred==1) | (y==1)).sum().float() + 1e-6
            miou = (inter_b/union_b).item()
        dice_sum += dice
        miou_sum += miou
        n_batches += 1
    model.train()
    if n_batches == 0:
        return 0.0, 0.0
    return dice_sum/n_batches, miou_sum/n_batches

# ===================== 主程序 =====================
def main():
    if CUDNN_BENCH:
        torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    torch.manual_seed(SEED); np.random.seed(SEED)

    DIR_CKPT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"INFO Using device: {device}")

    # —— Datasets / Loaders（统一使用 val.txt）——
    train_set = SegDataset(DIR_IMG, DIR_MASK, DIR_SPLITS / "train.txt",
                           dir_extra=DIR_GLCM if EXTRA_MODE else None,
                           extra_mode=EXTRA_MODE, img_size=IMG_SIZE,
                           cache_dir=DIR_CACHE if DIR_CACHE.exists() else None)
    val_set   = SegDataset(DIR_IMG, DIR_MASK, DIR_SPLITS / "val.txt",
                           dir_extra=DIR_GLCM if EXTRA_MODE else None,
                           extra_mode=EXTRA_MODE, img_size=IMG_SIZE,
                           cache_dir=DIR_CACHE if DIR_CACHE.exists() else None)

    print(f"[SegDataset] use_cache={'True' if train_set.use_cache else 'False'}  cache_dir={DIR_CACHE}")

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR, persistent_workers=PERSISTENT
    )
    val_loader = DataLoader(
        val_set, batch_size=max(1, BATCH_SIZE//2), shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR, persistent_workers=PERSISTENT
    )

    # —— Model ——（动态探测通道）
    sample = next(iter(DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0)))
    in_ch = sample["image"].shape[1]
    model = UNet(n_channels=in_ch, n_classes=CLASSES, bilinear=BILINEAR).to(
        device=device, memory_format=torch.channels_last
    )

    # —— Loss / Optim —— 
    if CLASSES > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        # 二分类：用 BCEWithLogits，pos_weight 可按需调整（这里不自动统计，稳定优先）
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM, foreach=True
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, verbose=True)
    scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu', enabled=AMP_ENABLED)

    best_dice = -1.0

    # —— Train Loop（梯度累积）——
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", unit="batch")

        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(pbar, start=1):
            images = batch["image"].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            masks  = batch["mask"].to(device=device, dtype=torch.long)

            with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=AMP_ENABLED):
                logits = model(images)
                if CLASSES == 1:
                    loss = criterion(logits.squeeze(1), masks.float())
                    # 简化：不额外叠加 dice，显存更稳（如需可加，与旧版一致）
                else:
                    loss = criterion(logits, masks)

                # 关键：按累积步数缩放
                loss = loss / ACCUM_STEPS

            scaler.scale(loss).backward()

            # 每 ACCUM_STEPS 次才真正 step 一次
            if step % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                # 清理缓存，降低峰值
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            epoch_loss += float(loss.item()) * ACCUM_STEPS
            pbar.set_postfix(loss=f"{(loss.item()*ACCUM_STEPS):.4f}")

        # —— Validation —— 
        val_dice, val_miou = quick_validate(model, val_loader, device, amp=AMP_ENABLED, num_classes=CLASSES)
        scheduler.step(val_dice)
        print(f"Epoch {epoch}: train_loss={epoch_loss/len(train_loader):.4f} | val_dice={val_dice:.4f} | val_mIoU={val_miou:.4f} | lr={optimizer.param_groups[0]['lr']:.2e}")

        # —— Save —— 
        if SAVE_EVERY_EPOCH:
            DIR_CKPT.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), DIR_CKPT / f"checkpoint_epoch{epoch}.pth")

        if val_dice > best_dice:
            best_dice = val_dice
            DIR_CKPT.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), DIR_CKPT / "best.pth")
            print(f"  ✅ New best! Saved to {DIR_CKPT / 'best.pth'} (val_dice={best_dice:.4f})")

if __name__ == "__main__":
    main()
'''


# -*- coding: utf-8 -*-
"""
train_update_fast_accum.py — 稳定+提速版（PA-SAM风格输入 + 离线缓存 + 梯度累积 + NaN防护）
- 硬编码根目录：E:\Eelgrass_processed_images_2025\Alaska   ←按需修改
- 使用 splits/train.txt, splits/val.txt
- 若 BASE/cache/{image,index,glcm} 存在 -> 直接加载 .pt 张量（极大提速）
- 训练尺寸 IMG_SIZE 默认 640（较 768 更快）；二分类强制二值化掩膜
- AMP 关闭；AdamW 优化器；梯度累积减显存；DataLoader 优化；channels_last
"""

# —— 在 import torch 之前彻底禁用动态编译（避免 triton / inductor 报错）——
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLED"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

from pathlib import Path
from typing import Optional, Dict, List
import numpy as np
from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from unet.unet_model import UNet

'''
# ===================== 硬编码路径 & 配置（按需修改） =====================
BASE_DIR   = Path(r"D:\Eelgrass_Process_2025_Bo\DroneVision_Model_data\OR")
DIR_IMG    = BASE_DIR / "image"
DIR_MASK   = BASE_DIR / "index"
DIR_GLCM   = BASE_DIR / "glcm"         # 可选
DIR_SPLITS = BASE_DIR / "splits"
DIR_CKPT   = BASE_DIR / "train2/checkpoints"
DIR_CACHE  = BASE_DIR / "train2/cache"
'''

BASE_DIR = Path(r"D:\Eelgrass_Process_2025_Bo\DroneVision_Model_data\OR")

TR_IMG   = BASE_DIR / "train" / "image"
TR_MASK  = BASE_DIR / "train" / "index"
TR_GLCM  = BASE_DIR / "train" / "glcm"     # 可不存在

VA_IMG   = BASE_DIR / "valid" / "image"
VA_MASK  = BASE_DIR / "valid" / "index"
VA_GLCM  = BASE_DIR / "valid" / "glcm"     # 可不存在

TE_IMG   = BASE_DIR / "test"  / "image"
TE_MASK  = BASE_DIR / "test"  / "index"
TE_GLCM  = BASE_DIR / "test"  / "glcm"     # 可不存在；仅说明结构，用不到也无所谓

DIR_CKPT = BASE_DIR / "train1" / "checkpoints"  # 保存模型



# ===== 数据/训练设置 =====
IMG_SIZE        = 512                         # 768 → 640，速度/显存更友好
EXTRA_MODE      = None                        # None | 'append4' | 'replace_red'
CLASSES         = 1                           # 1=二分类
BILINEAR        = False

# 训练超参（稳定优先）
EPOCHS        = 10
BATCH_SIZE    = 4                            # 单次前向的 batch
ACCUM_STEPS   = 4                            # 梯度累积步数 ⇒ 等效 batch = 4*4=16
LR            = 3e-5                         # 稳定起步；跑通后可调大
WEIGHT_DECAY  = 1e-4
GRAD_CLIP     = 1.0

# DataLoader
NUM_WORKERS     = min(os.cpu_count() or 0, 8)
PREFETCH_FACTOR = 2
PIN_MEMORY      = True
PERSISTENT      = True

# AMP/内核/随机性
AMP_ENABLED  = False                         # 先关掉，跑稳后再开
CUDNN_BENCH  = True
SEED         = 0

# 其他
SAVE_EVERY_EPOCH = False

# ===================== 工具函数 =====================
VALID_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]



def find_first(folder: Path, stem: str) -> Path:
    """
    在 folder 下查找给定 stem 对应的图像文件：
    - 传入的 stem 可以是 'du_bc_19_row10_col26' 或 'du_bc_19_row10_col26.png'
    - 自动去扩展名、大小写无关匹配、标准化路径分隔符
    """
    folder = Path(folder).resolve()

    # 统一：去掉传入 stem 的扩展名，并做小写比较
    target_stem = Path(stem).stem.lower()

    # 先尝试直接拼接常见扩展名（快路径）
    for ext in VALID_EXTS:
        p = (folder / f"{target_stem}{ext}").resolve()
        if p.exists():
            return p

    # 若快路径没找到，遍历目录做小写无关匹配（兼容奇怪大小写/后缀）
    for p in folder.iterdir():
        if not p.is_file():
            continue
        if p.stem.lower() == target_stem and p.suffix.lower() in VALID_EXTS:
            return p.resolve()

    raise FileNotFoundError(f"not found (checked normalized path): {folder}\\{stem}.*")


def pil_to_chw_float(pil_img: Image.Image) -> np.ndarray:
    """PIL -> CxHxW float[0,1]（RGB）"""
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    arr = np.asarray(pil_img).transpose(2, 0, 1).astype(np.float32)
    if (arr > 1).any():
        arr /= 255.0
    return arr

def resize_longest_side(pil_img: Image.Image, target: int, is_mask: bool) -> Image.Image:
    """按最长边缩放到 target；保持比例。"""
    w, h = pil_img.size
    if max(w, h) == target:
        return pil_img
    scale = float(target) / float(max(w, h))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return pil_img.resize((new_w, new_h), Image.NEAREST if is_mask else Image.BILINEAR)

def pad_to_square(pil_img: Image.Image, target: int, fill=0) -> Image.Image:
    """右/下方向 pad 到正方形 target×target。"""
    w, h = pil_img.size
    if (w, h) == (target, target):
        return pil_img
    canvas = Image.new(pil_img.mode, (target, target), color=fill)
    canvas.paste(pil_img, (0, 0))
    return canvas

def load_tensor(path: Path) -> torch.Tensor:
    """从缓存 .pt 读取张量（兼容 dict({'tensor': ...}) 或裸 tensor）。"""
    obj = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(obj, dict) and "tensor" in obj:
        return obj["tensor"]
    elif torch.is_tensor(obj):
        return obj
    else:
        raise TypeError(f"Unexpected object in {path}: {type(obj)}")

# ===================== 数据集 =====================
class SegDataset(Dataset):
    """
    - 若 BASE/cache/{image,index,glcm} 存在 -> 直接加载 .pt（C×S×S / S×S）
    - 否则：PA-SAM 风格前处理（最长边=IMG_SIZE，右/下 pad 到正方形）
    - EXTRA_MODE：None | 'append4' | 'replace_red'
    - 训练/验证仅返回定长 image/mask/stem，便于 collate
    """
    def __init__(self,
                 images_dir: Path,
                 masks_dir: Path,
                 split_list_file: Path,
                 dir_extra: Optional[Path] = None,
                 extra_mode: Optional[str] = None,
                 img_size: int = 640,
                 cache_dir: Optional[Path] = None):
        self.images_dir = Path(images_dir)
        self.masks_dir  = Path(masks_dir)
        self.dir_extra  = Path(dir_extra) if dir_extra else None
        self.extra_mode = extra_mode
        self.img_size   = img_size

        self.stems: List[str] = [s.strip() for s in Path(split_list_file).read_text(encoding="utf-8").splitlines() if s.strip()]
        if not self.stems:
            raise RuntimeError(f"Split list is empty: {split_list_file}")

        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_cache = False
        if self.cache_dir and (self.cache_dir / "image").exists() and (self.cache_dir / "index").exists():
            # 若使用 glcm 作为 extra，则也要求 cache/glcm 存在
            if (self.dir_extra is not None and self.extra_mode is not None):
                self.use_cache = (self.cache_dir / "glcm").exists()
            else:
                self.use_cache = True

        print(f"[SegDataset] use_cache={self.use_cache}  cache_dir={self.cache_dir if self.use_cache else None}")

        # 非缓存时需要建立掩膜值映射（防非 {0,1} 标签）
        if not self.use_cache:
            vals = []
            # 初始化采样 500 个以提速
            for stem in self.stems[:min(500, len(self.stems))]:
                mp = find_first(self.masks_dir, stem)
                vals.append(np.unique(np.asarray(Image.open(mp).convert("L"))))
            self.mask_values = sorted(np.unique(np.concatenate(vals)).tolist())
        else:
            self.mask_values = None

    def __len__(self):
        return len(self.stems)

    def _mask_to_long(self, pil_mask: Image.Image) -> np.ndarray:
        """把任意灰度值集合映射到 0..K-1（非缓存路径用）"""
        m = np.asarray(pil_mask.convert("L"))
        h, w = m.shape
        out = np.zeros((h, w), dtype=np.int64)
        for i, v in enumerate(self.mask_values):
            out[m == v] = i
        return out

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        stem = self.stems[idx]

        # ---- 1) 缓存直读 ----
        if self.use_cache:
            img_t = load_tensor(self.cache_dir / "image" / f"{stem}.pt")   # CxSxS float[0,1]
            msk_t = load_tensor(self.cache_dir / "index" / f"{stem}.pt")   # SxS   long/int64

            if self.dir_extra is not None and self.extra_mode is not None:
                glcm_t = load_tensor(self.cache_dir / "glcm"  / f"{stem}.pt")  # 1xSxS float
                if self.extra_mode == "replace_red":
                    img_t[0:1, ...] = glcm_t
                elif self.extra_mode == "append4":
                    img_t = torch.cat([img_t, glcm_t], dim=0)

            return {
                "image": img_t.float().contiguous(),
                "mask":  msk_t.long().contiguous(),
                "stem":  stem
            }

        # ---- 2) 非缓存：现场做 resize/pad ----
        ip = find_first(self.images_dir, stem)
        mp = find_first(self.masks_dir,  stem)

        img = Image.open(ip).convert("RGB")
        msk = Image.open(mp)  # 不强转 L，交给映射函数

        # 可选 glcm
        extra_arr = None
        if self.dir_extra is not None and self.extra_mode is not None:
            gp = find_first(self.dir_extra, stem)
            g = Image.open(gp)
            if g.mode != "L":
                g = g.convert("L")
            g = pad_to_square(resize_longest_side(g, self.img_size, is_mask=False), self.img_size, fill=0)
            ga = np.asarray(g).astype(np.float32)
            if (ga > 1).any():
                ga /= 255.0
            extra_arr = ga[None, ...]  # 1xSxS

        # SAM 风格：先等比缩放，再右/下 pad
        img_p = pad_to_square(resize_longest_side(img, self.img_size, is_mask=False), self.img_size, fill=0)
        msk_p = pad_to_square(resize_longest_side(msk, self.img_size, is_mask=True),  self.img_size, fill=0)

        img_arr  = pil_to_chw_float(img_p)                 # CxSxS
        mask_arr = self._mask_to_long(msk_p)               # SxS（0..K-1）

        if extra_arr is not None and self.extra_mode is not None:
            if self.extra_mode == "replace_red":
                img_arr[0:1, ...] = extra_arr
            elif self.extra_mode == "append4":
                img_arr = np.concatenate([img_arr, extra_arr], axis=0)

        return {
            "image": torch.from_numpy(img_arr.copy()).float().contiguous(),
            "mask":  torch.from_numpy(mask_arr.copy()).long().contiguous(),
            "stem":  stem
        }

# ===================== 验证（Dice/mIoU，二分类） =====================
@torch.no_grad()
def quick_validate(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool = False):
    model.eval()
    dice_sum = 0.0
    miou_sum = 0.0
    n_batches = 0
    for batch in loader:
        x = batch["image"].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        y = batch["mask"].to(device=device, dtype=torch.long)
        y_bin = (y > 0).float()

        with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
            logits = model(x)
        prob = torch.sigmoid(logits).squeeze(1)

        inter = (prob * y_bin).sum(dim=(1,2))
        denom = prob.sum(dim=(1,2)) + y_bin.sum(dim=(1,2)) + 1e-6
        dice = (2*inter/denom).mean().item()

        pred = (prob > 0.5).long()
        inter_b = ((pred==1) & (y==1)).sum().float()
        union_b = ((pred==1) | (y==1)).sum().float() + 1e-6
        miou = (inter_b/union_b).item()

        dice_sum += dice
        miou_sum += miou
        n_batches += 1
    model.train()
    if n_batches == 0:
        return 0.0, 0.0
    return dice_sum/n_batches, miou_sum/n_batches

# ===================== 主程序 =====================
def main():
    # —— 内核 & 随机种子 ——
    if CUDNN_BENCH:
        torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    torch.manual_seed(SEED); np.random.seed(SEED)

    DIR_CKPT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"INFO Using device: {device}")

    # —— DataLoaders（统一 val.txt 命名）——
    use_cache = DIR_CACHE.exists() and (DIR_CACHE / "image").exists() and (DIR_CACHE / "index").exists()
    extra_needed = (DIR_GLCM.exists() and EXTRA_MODE is not None)
    if extra_needed:
        use_cache = use_cache and (DIR_CACHE / "glcm").exists()

    train_set = SegDataset(DIR_IMG, DIR_MASK, DIR_SPLITS / "train.txt",
                           dir_extra=DIR_GLCM if EXTRA_MODE else None,
                           extra_mode=EXTRA_MODE, img_size=IMG_SIZE,
                           cache_dir=DIR_CACHE if use_cache else None)
    val_set   = SegDataset(DIR_IMG, DIR_MASK, DIR_SPLITS / "valid.txt",
                           dir_extra=DIR_GLCM if EXTRA_MODE else None,
                           extra_mode=EXTRA_MODE, img_size=IMG_SIZE,
                           cache_dir=DIR_CACHE if use_cache else None)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              prefetch_factor=PREFETCH_FACTOR, persistent_workers=PERSISTENT)
    val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              prefetch_factor=PREFETCH_FACTOR, persistent_workers=PERSISTENT)

    # —— Model：动态探测输入通道（append4/replace_red 后可能是 4 通道）——
    sample = next(iter(DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0)))
    in_ch = sample["image"].shape[1]
    model = UNet(n_channels=in_ch, n_classes=CLASSES, bilinear=BILINEAR).to(
        device=device, memory_format=torch.channels_last
    )

    # —— Loss & Optimizer（BCE+Dice，强制二值掩膜）——
    if CLASSES != 1:
        raise ValueError("本脚本按二分类实现（N_CLASSES=1）。如需多分类请改 CrossEntropy 分支。")
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, verbose=True)
    scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu', enabled=AMP_ENABLED)

    best_dice = -1.0
    step_in_epoch = 0

    # —— Train Loop（梯度累积）——
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", unit="batch")
        for batch in pbar:
            images = batch["image"].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            masks  = batch["mask"].to(device=device, dtype=torch.long)

            # 强制二值化（避免非 {0,1} 标签引起数值不稳）
            masks_bin = (masks > 0).float()

            with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=AMP_ENABLED):
                logits = model(images)  # (B,1,H,W)
                # BCE
                loss_bce = criterion(logits.squeeze(1), masks_bin)
                # Dice(soft)
                prob = torch.sigmoid(logits).squeeze(1)
                inter = (prob * masks_bin).sum(dim=(1,2))
                denom = prob.sum(dim=(1,2)) + masks_bin.sum(dim=(1,2)) + 1e-6
                dice = (2*inter/denom).mean()
                loss = loss_bce + (1.0 - dice)

            # NaN/Inf 防护：直接跳过 batch
            if not torch.isfinite(loss):
                print("[warn] non-finite loss, skip batch")
                optimizer.zero_grad(set_to_none=True)
                continue

            # 梯度累积
            loss = loss / ACCUM_STEPS
            if AMP_ENABLED:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            step_in_epoch += 1
            if step_in_epoch % ACCUM_STEPS == 0:
                if AMP_ENABLED:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += float(loss.item()) * ACCUM_STEPS  # 还原记录
            pbar.set_postfix(loss=f"{(loss.item()*ACCUM_STEPS):.4f}")

        # -- 收尾：若最后不足 ACCUM_STEPS 的残梯度，仍需 step 一次 --
        if step_in_epoch % ACCUM_STEPS != 0:
            if AMP_ENABLED:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        step_in_epoch = 0

        # —— Validation —— 
        val_dice, val_miou = quick_validate(model, val_loader, device, amp=AMP_ENABLED)
        scheduler.step(val_dice)
        print(f"Epoch {epoch}: train_loss={epoch_loss/len(train_loader):.4f} | val_dice={val_dice:.4f} | val_mIoU={val_miou:.4f} | lr={optimizer.param_groups[0]['lr']:.2e}")

        # —— 保存 —— 
        if SAVE_EVERY_EPOCH:
            (DIR_CKPT / f"checkpoint_epoch{epoch}.pth").parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), DIR_CKPT / f"checkpoint_epoch{epoch}.pth")

        if val_dice > best_dice:
            best_dice = val_dice
            DIR_CKPT.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), DIR_CKPT / "best.pth")
            print(f"  ✅ New best! Saved to {DIR_CKPT / 'best.pth'} (val_dice={best_dice:.4f})")

if __name__ == "__main__":
    main()

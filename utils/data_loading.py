import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')


class SegDataset(Dataset):
    """
    通用分割数据集：
      - 固定划分：通过 split_list_file（如 data/splits/train.txt）
      - 主图：RGB（自动去 alpha）或保持 4 通道（见 extra_mode）
      - 额外单通道：dir_extra，可 append 为第 4 通道或替换红通道
      - mask：单通道 L，值域为 0..K-1（多类）或 0/1（二分类）
      - 复用你现有的 mask_values 与编码策略
    """
    def __init__(
        self,
        images_dir: str | Path,
        mask_dir: str | Path,
        split_list_file: str | Path,
        scale: float = 1.0,
        mask_suffix: str = '',
        dir_extra: str | Path | None = None,
        extra_mode: str | None = None,   # None | 'append4' | 'replace_red'
        dtype_priority: tuple[str, ...] = ('.png', '.tif', '.tiff', '.jpg', '.jpeg')
    ):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.dir_extra = Path(dir_extra) if dir_extra is not None else None
        self.extra_mode = extra_mode
        self.dtype_priority = tuple(ext.lower() for ext in dtype_priority)

        # 读取 split 列表：每行一个基名（不含扩展名）
        split_list_file = Path(split_list_file)
        if not split_list_file.exists():
            raise FileNotFoundError(f"Split list file not found: {split_list_file}")
        with open(split_list_file, 'r', encoding='utf-8') as f:
            self.ids = [line.strip() for line in f if line.strip()]

        if not self.ids:
            raise RuntimeError(f'No IDs found in split list: {split_list_file}')

        logging.info(f'[SegDataset] Building with {len(self.ids)} items from {split_list_file.name}')

        # 扫描 mask 获取 unique 值（复用你的并行逻辑）
        logging.info('[SegDataset] Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=Path(self.mask_dir), mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))
        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'[SegDataset] Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    # ---------- 工具函数：按优先级寻找文件 ----------
    def _find_with_priority(self, base_dir: Path, stem: str, suffix: str = '') -> Path:
        for ext in self.dtype_priority:
            cand = base_dir / f"{stem}{suffix}{ext}"
            if cand.exists():
                return cand
        # 兜底：通配第一个匹配
        matches = list(base_dir.glob(f"{stem}{suffix}.*"))
        if not matches:
            raise FileNotFoundError(f'File not found for stem="{stem}", suffix="{suffix}" in "{base_dir}"')
        return matches[0]

    # ---------- 统一 resize ----------
    @staticmethod
    def _resize(pil_img: Image.Image, scale: float, is_mask: bool) -> Image.Image:
        if scale == 1.0:
            return pil_img
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        return pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BILINEAR)

    def _pil_to_chw_float(self, pil_img: Image.Image) -> np.ndarray:
        """
        PIL -> CHW float32 [0,1], 自动处理 1/3/4 通道
        - 若 4 通道（RGBA），默认丢 alpha 变 RGB（除非 extra_mode='append4' 并且你希望保留外部 extra）
        """
        if pil_img.mode == 'L':
            arr = np.asarray(pil_img)  # HxW
            arr = arr[np.newaxis, ...]  # 1xHxW
        elif pil_img.mode in ('RGB', 'YCbCr', 'LAB'):
            arr = np.asarray(pil_img)  # HxWx3
            arr = arr.transpose((2, 0, 1))
        elif pil_img.mode in ('RGBA', 'CMYK'):
            # 默认去掉 alpha/多余通道，只取前 3 个通道（RGBA->RGB）
            arr = np.asarray(pil_img)[..., :3].transpose((2, 0, 1))
        else:
            # 其他模式一律转 RGB
            arr = np.asarray(pil_img.convert('RGB')).transpose((2, 0, 1))

        if (arr > 1).any():
            arr = arr.astype(np.float32) / 255.0
        else:
            arr = arr.astype(np.float32)
        return arr

    def _pil_to_mask_long(self, pil_mask: Image.Image) -> np.ndarray:
        """
        将 mask 映射到 0..K-1（与 BasicDataset 的 preprocess 一致）
        """
        pil_mask = pil_mask.convert('L')
        arr = np.asarray(pil_mask)
        h, w = arr.shape[:2]
        mask = np.zeros((h, w), dtype=np.int64)
        for i, v in enumerate(self.mask_values):
            # 注意：mask_values 可能是 list[int] 或 list[list[int]]；这里按你原逻辑处理
            if isinstance(v, (list, tuple, np.ndarray)):
                # 多通道的情况（很少见），兜底；通常不会走到
                matches = (arr == v).all(-1) if arr.ndim == 3 else (arr == v)
                mask[matches] = i
            else:
                mask[arr == v] = i
        return mask

    def __getitem__(self, idx):
        stem = self.ids[idx]

        img_path = self._find_with_priority(self.images_dir, stem, suffix='')
        mask_path = self._find_with_priority(self.mask_dir, stem, suffix=self.mask_suffix)

        img_pil = load_image(img_path)
        mask_pil = load_image(mask_path)

        # 尺寸校验（mask 与 image 一致）
        assert img_pil.size == mask_pil.size, f'Image and mask sizes differ for "{stem}": {img_pil.size} vs {mask_pil.size}'

        # 额外通道（如 GLCM）
        extra_arr = None
        if self.dir_extra is not None and self.extra_mode is not None:
            extra_path = self._find_with_priority(self.dir_extra, stem, suffix='')
            extra_pil = load_image(extra_path)
            # 单通道化
            if extra_pil.mode != 'L':
                extra_pil = extra_pil.convert('L')
            # resize
            extra_pil = self._resize(extra_pil, self.scale, is_mask=False)
            extra_arr = np.asarray(extra_pil).astype(np.float32)
            if (extra_arr > 1).any():
                extra_arr = extra_arr / 255.0
            extra_arr = extra_arr[np.newaxis, ...]  # 1xHxW

        # resize
        img_pil = self._resize(img_pil, self.scale, is_mask=False)
        mask_pil = self._resize(mask_pil, self.scale, is_mask=True)

        # to numpy (CHW)
        img_arr = self._pil_to_chw_float(img_pil)  # 3xHxW（默认 RGB），或 1xHxW（若原本 L）
        # 保证有 3 通道作为“主图”
        if img_arr.shape[0] == 1:
            img_arr = np.repeat(img_arr, 3, axis=0)

        if extra_arr is not None:
            if self.extra_mode == 'replace_red':
                # 用 extra 替换 R 通道
                img_arr[0:1, ...] = extra_arr
            elif self.extra_mode == 'append4':
                # 追加为第 4 通道
                img_arr = np.concatenate([img_arr, extra_arr], axis=0)
            else:
                pass  # None

        mask_arr = self._pil_to_mask_long(mask_pil)

        return {
            'image': torch.as_tensor(img_arr.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask_arr.copy()).long().contiguous()
        }
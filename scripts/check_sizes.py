# -*- coding: utf-8 -*-
"""
数据完整性检查（硬编码路径，零参数运行）
检查项：
1) splits 中的样本在 image/index/glcm 是否都存在
2) 尺寸是否为 512x512，且 image 与 index/glcm 尺寸一致
3) image 是否严格 3 通道；index 与 glcm 是否严格 1 通道
4) 同名多扩展名冲突提示
5) 汇总统计与问题列表；若存在问题以非零码退出

依赖：Pillow（PIL）, numpy（可选）
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image

# ====== 硬编码你的数据根路径 ======
BASE = Path(r"E:\Eelgrass_processed_images_2025\Alaska")
IMAGE = BASE / "image"
INDEX = BASE / "index"
GLCM  = BASE / "glcm"
SPLITS = BASE / "splits"

# 期望尺寸与通道
EXPECTED_SIZE = (512, 512)
IMAGE_CHANNELS = 3
INDEX_CHANNELS = 1
GLCM_CHANNELS  = 1

# 扩展名优先级（同时存在多种扩展名时，按此顺序选用，并报告冲突）
VALID_EXTS = [".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"]


def list_matches(folder: Path, stem: str) -> List[Path]:
    """返回同名不同扩展名的所有匹配"""
    matches = []
    for ext in VALID_EXTS:
        p = folder / f"{stem}{ext}"
        if p.exists():
            matches.append(p)
    if not matches:
        # 宽松回退：任意扩展
        matches = list(folder.glob(stem + ".*"))
    return matches


def pick_one_with_warning(folder: Path, stem: str, who: str, warnings: List[str]) -> Optional[Path]:
    """按优先级选一个文件，若多于一个则记录警告；若无则返回 None。"""
    matches = list_matches(folder, stem)
    if not matches:
        return None
    # 优先级选择
    for ext in VALID_EXTS:
        cand = folder / f"{stem}{ext}"
        if cand in matches:
            chosen = cand
            break
    else:
        chosen = matches[0]
    # 冲突提示（多于1个）
    others = [m for m in matches if m != chosen]
    if others:
        warnings.append(f"[{who}] multiple files for '{stem}': chosen {chosen.name}, ignored {[x.name for x in others]}")
    return chosen


def get_size_and_channels(p: Path) -> Tuple[Tuple[int, int], int, str]:
    """
    返回 (宽,高), 通道数, PIL 模式字符串
    通道数通过 PIL 模式 + numpy shape 推断，尽量不修改原像素
    """
    with Image.open(p) as im:
        mode = im.mode  # e.g., 'RGB', 'RGBA', 'L', 'P', 'I;16'
        size = im.size  # (w,h)
        # 对于 palette/索引模式 'P'，等价单通道
        if mode == "P":
            channels = 1
        elif mode in ("L", "I;16", "I", "F"):
            channels = 1
        elif mode in ("RGB",):
            channels = 3
        elif mode in ("RGBA", "LA"):
            channels = 4 if mode == "RGBA" else 2
        else:
            # 兜底：用 numpy 维度判断（不做转换）
            arr = im.__array__()  # Pillow>=9 支持
            if arr.ndim == 2:
                channels = 1
            elif arr.ndim == 3:
                channels = arr.shape[2]
            else:
                channels = -1
    return size, channels, mode


def check_split(split_name: str) -> Dict[str, List[str]]:
    """
    针对一个 split（train/valid/test）做全量检查
    返回 problems: 键为问题类型，值为样本名列表
    同时打印摘要
    """
    txt = SPLITS / f"{split_name}.txt"
    if not txt.exists():
        print(f"[{split_name}] ❌ split file not found: {txt}")
        return {"missing_split_file": [str(txt)]}

    stems = [s.strip() for s in txt.read_text(encoding="utf-8").splitlines() if s.strip()]
    print(f"\n=== Checking split: {split_name} ===")
    print(f"Total stems in {txt.name}: {len(stems)}")

    problems: Dict[str, List[str]] = {
        "missing_image": [],
        "missing_index": [],
        "missing_glcm": [],
        "image_size_mismatch": [],
        "index_size_mismatch": [],
        "glcm_size_mismatch": [],
        "image_channels_bad": [],
        "index_channels_bad": [],
        "glcm_channels_bad": [],
        "image_index_size_inconsistent": [],
        "image_glcm_size_inconsistent": [],
    }
    warnings: List[str] = []
    mode_hist_image: Dict[str, int] = {}
    mode_hist_index: Dict[str, int] = {}
    mode_hist_glcm:  Dict[str, int] = {}

    for stem in stems:
        # 定位文件
        ip = pick_one_with_warning(IMAGE, stem, who="image", warnings=warnings)
        mp = pick_one_with_warning(INDEX, stem, who="index", warnings=warnings)
        gp = pick_one_with_warning(GLCM,  stem, who="glcm",  warnings=warnings)

        if ip is None: problems["missing_image"].append(stem)
        if mp is None: problems["missing_index"].append(stem)
        if gp is None: problems["missing_glcm"].append(stem)

        # 不齐就跳过后续检查（避免重复报错）
        if ip is None or mp is None or gp is None:
            continue

        # 读取属性
        try:
            sz_i, ch_i, md_i = get_size_and_channels(ip)
        except Exception as e:
            problems["missing_image"].append(stem)
            continue
        try:
            sz_m, ch_m, md_m = get_size_and_channels(mp)
        except Exception as e:
            problems["missing_index"].append(stem)
            continue
        try:
            sz_g, ch_g, md_g = get_size_and_channels(gp)
        except Exception as e:
            problems["missing_glcm"].append(stem)
            continue

        # 模式统计
        mode_hist_image[md_i] = mode_hist_image.get(md_i, 0) + 1
        mode_hist_index[md_m] = mode_hist_index.get(md_m, 0) + 1
        mode_hist_glcm[md_g]  = mode_hist_glcm.get(md_g, 0) + 1

        # 尺寸检查：是否 512x512
        if sz_i != EXPECTED_SIZE: problems["image_size_mismatch"].append(f"{stem} {sz_i}")
        if sz_m != EXPECTED_SIZE: problems["index_size_mismatch"].append(f"{stem} {sz_m}")
        if sz_g != EXPECTED_SIZE: problems["glcm_size_mismatch"].append(f"{stem} {sz_g}")

        # 尺寸一致性：image vs index/glcm
        if sz_i != sz_m:
            problems["image_index_size_inconsistent"].append(f"{stem} img{sz_i} vs idx{sz_m}")
        if sz_i != sz_g:
            problems["image_glcm_size_inconsistent"].append(f"{stem} img{sz_i} vs glcm{sz_g}")

        # 通道检查：image 必须 3；index 必须 1；glcm 必须 1
        if ch_i != IMAGE_CHANNELS:
            problems["image_channels_bad"].append(f"{stem} ch={ch_i} mode={md_i}")
        if ch_m != INDEX_CHANNELS:
            problems["index_channels_bad"].append(f"{stem} ch={ch_m} mode={md_m}")
        if ch_g != GLCM_CHANNELS:
            problems["glcm_channels_bad"].append(f"{stem} ch={ch_g} mode={md_g}")

    # 摘要
    def count(lst): return sum(len(v) for k, v in lst.items())
    total_missing = len(problems["missing_image"]) + len(problems["missing_index"]) + len(problems["missing_glcm"])
    total_size_bad = len(problems["image_size_mismatch"]) + len(problems["index_size_mismatch"]) + len(problems["glcm_size_mismatch"])
    total_ch_bad   = len(problems["image_channels_bad"]) + len(problems["index_channels_bad"]) + len(problems["glcm_channels_bad"])
    total_incons   = len(problems["image_index_size_inconsistent"]) + len(problems["image_glcm_size_inconsistent"])

    print(f"[{split_name}] Missing files: image={len(problems['missing_image'])}, index={len(problems['missing_index'])}, glcm={len(problems['missing_glcm'])}")
    print(f"[{split_name}] Size ≠ {EXPECTED_SIZE}: image={len(problems['image_size_mismatch'])}, index={len(problems['index_size_mismatch'])}, glcm={len(problems['glcm_size_mismatch'])}")
    print(f"[{split_name}] Size inconsistent (img vs idx/glcm): idx={len(problems['image_index_size_inconsistent'])}, glcm={len(problems['image_glcm_size_inconsistent'])}")
    print(f"[{split_name}] Bad channels (image!=3 / index!=1 / glcm!=1): image={len(problems['image_channels_bad'])}, index={len(problems['index_channels_bad'])}, glcm={len(problems['glcm_channels_bad'])}")

    # 模式直方
    print(f"[{split_name}] Image modes: {mode_hist_image}")
    print(f"[{split_name}] Index modes: {mode_hist_index}")
    print(f"[{split_name}] Glcm  modes: {mode_hist_glcm}")

    # 打印部分警告与问题样本（最多 50）
    if warnings:
        print(f"[{split_name}] ⚠ Multiple-extension conflicts: {len(warnings)} (showing up to 10)")
        for w in warnings[:10]:
            print("   ", w)

    def show(title: str, items: List[str], n: int = 50):
        if items:
            print(f"--- {title} (count={len(items)}, show up to {n}) ---")
            for x in items[:n]:
                print("   ", x)

    show("missing_image", problems["missing_image"])
    show("missing_index", problems["missing_index"])
    show("missing_glcm",  problems["missing_glcm"])
    show("image_size_mismatch", problems["image_size_mismatch"])
    show("index_size_mismatch", problems["index_size_mismatch"])
    show("glcm_size_mismatch",  problems["glcm_size_mismatch"])
    show("image_index_size_inconsistent", problems["image_index_size_inconsistent"])
    show("image_glcm_size_inconsistent",  problems["image_glcm_size_inconsistent"])
    show("image_channels_bad", problems["image_channels_bad"])
    show("index_channels_bad", problems["index_channels_bad"])
    show("glcm_channels_bad",  problems["glcm_channels_bad"])

    return problems


def main():
    # 逐 split 检查
    all_problems: Dict[str, List[str]] = {}
    any_issue = False
    for split in ["train", "val", "test"]:
        probs = check_split(split)
        for k, v in probs.items():
            all_problems.setdefault(k, []).extend(v)
            if len(v) > 0:
                any_issue = True

    print("\n======= SUMMARY =======")
    for k, v in all_problems.items():
        print(f"{k:32s}: {len(v)}")

    if any_issue:
        print("\n❌ Found issues. Please fix before training.")
        sys.exit(1)
    else:
        print("\n✅ All good. Dataset looks consistent.")
        sys.exit(0)


if __name__ == "__main__":
    main()

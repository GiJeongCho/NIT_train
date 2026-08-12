"""
utils_image
===========

tracker_py `preprocess/utils_image.py` 를 그대로 복사(vendored). float/uint8 변환 헬퍼.
"""

from __future__ import annotations

import numpy as np


def to_float(img: np.ndarray) -> np.ndarray:
    """이미지를 float32 [0,1] 로 변환(best-effort).

    - uint8: [0,255] 가정
    - float: max 가 ~1.5 이하면 이미 [0,1] 로 보고, 그보다 크면 [0,255] 로 보고 나눈다.
    """
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    out = img.astype(np.float32, copy=False)
    m = float(out.max()) if out.size else 1.0
    if m > 1.5:  # 0..255 범위의 float 로 추정
        out = out / 255.0
    return out


def to_uint8(img01: np.ndarray) -> np.ndarray:
    """float [0,1] 이미지를 uint8 로(클리핑)."""
    return (np.clip(img01, 0.0, 1.0) * 255.0).astype(np.uint8)

"""
clahe_lab
=========

tracker_py `preprocess/modules/quality/clahe_lab.py` 를 복사(vendored).
Stage2(화질향상) CLAHE. 기본은 LAB 의 L(밝기) 채널에만 적용해 색을 보존한다.

원본과의 유일한 차이: import 경로(`preprocess.utils_image` → `.utils_image`).
"""

import cv2
import numpy as np

from .utils_image import to_float, to_uint8


def apply_clahe_grayscale(
    image: np.ndarray, clip_limit: float = 2.0, tile_grid_size=(8, 8)
) -> np.ndarray:
    """단일 채널(HxW) 이미지에 CLAHE. 반환은 float32 [0,1]."""
    if image.ndim != 2:
        raise ValueError("apply_clahe_grayscale expects a single-channel (HxW) image")
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid_size))
    if image.dtype != np.uint8:
        img8 = to_uint8(to_float(image))
    else:
        img8 = image
    eq = clahe.apply(img8)
    return eq.astype(np.float32) / 255.0


def preprocess_with_params(
    frame: np.ndarray,
    *,
    clip_limit: float = 2.0,
    tile_grid_size=(8, 8),
    mode: str = "lab_l",
) -> np.ndarray:
    """CLAHE 적용. 반환은 uint8 BGR."""
    if frame is None:
        return frame

    if isinstance(tile_grid_size, list):
        tile_grid_size = (int(tile_grid_size[0]), int(tile_grid_size[1]))

    m = (mode or "lab_l").strip().lower()

    if m == "grayscale":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eq01 = apply_clahe_grayscale(gray, clip_limit=float(clip_limit), tile_grid_size=tile_grid_size)
        eq8 = to_uint8(eq01)
        return cv2.cvtColor(eq8, cv2.COLOR_GRAY2BGR)

    # 기본: LAB 의 L 채널에만 CLAHE(색 안정적으로 유지)
    img8 = frame if frame.dtype == np.uint8 else to_uint8(to_float(frame))
    lab = cv2.cvtColor(img8, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    eq01 = apply_clahe_grayscale(l, clip_limit=float(clip_limit), tile_grid_size=tile_grid_size)
    l_eq = to_uint8(eq01)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


preprocess = preprocess_with_params

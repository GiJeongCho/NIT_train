"""
zero_dce
========

tracker_py `preprocess/modules/dark/zero_dce_plus.py` 를 이식(vendored).
Zero-DCE++ 저조도(야간) 보정. **가중치 파일이 있어야** 동작한다.

가중치(`Epoch99.pth`)는 리포에 포함돼 있지 않다(용량/배포 정책). 다음 위치에 파일을
넣으면 자동으로 사용된다(추론과 완전히 동일한 저조도 보정):

    NIT_train/src/v1/preprocess_vendor/zero_dce_weights/Epoch99.pth
    또는 환경변수 NIT_TRAIN_ZERODCE_WEIGHTS 로 절대경로 지정.

가중치가 없으면 `available()` 이 False 를 돌려주고, 상위(services.preprocess)가 가중치
불필요한 대체(감마 + CLAHE)로 폴백한다.

torch 는 이 모듈을 실제로 쓸 때만 import 한다(가중치 없는 환경에서 import 비용/의존 회피).
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np

_PKG_DIR = Path(__file__).resolve().parent
_DEFAULT_WEIGHTS = _PKG_DIR / "zero_dce_weights" / "Epoch99.pth"

_LOCK = threading.Lock()
_MODEL = None
_MODEL_DEVICE = None
_MODEL_KEY = None


def weights_path() -> Path:
    env = os.getenv("NIT_TRAIN_ZERODCE_WEIGHTS", "").strip()
    return Path(env) if env else _DEFAULT_WEIGHTS


def available() -> bool:
    """가중치 파일이 실제로 있는지. torch 유무는 여기서 보지 않는다(호출 시 확인)."""
    try:
        return weights_path().exists()
    except OSError:
        return False


def _select_device(device: str):
    import torch

    d = (device or "auto").strip().lower()
    if d in ("auto", "cuda", "gpu"):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


def _load_once(*, scale_factor: int, device: str):
    global _MODEL, _MODEL_DEVICE, _MODEL_KEY
    import torch

    from .zero_dce_model.model import enhance_net_nopool

    wp = weights_path()
    dev = _select_device(device)
    key = (str(wp), int(scale_factor), str(dev))
    with _LOCK:
        if _MODEL is not None and _MODEL_KEY == key:
            return
        model = enhance_net_nopool(int(scale_factor))
        if wp.exists():
            state = torch.load(str(wp), map_location=dev)
            model.load_state_dict(state)
        else:
            raise FileNotFoundError(f"Zero-DCE++ 가중치가 없습니다: {wp}")
        model.to(dev)
        model.eval()
        _MODEL = model
        _MODEL_DEVICE = dev
        _MODEL_KEY = key


def enhance(
    frame: np.ndarray,
    *,
    scale_factor: int = 12,
    brightness_gain: float = 1.0,
    device: str = "auto",
) -> np.ndarray:
    """저조도 보정. BGR uint8 입력 → BGR uint8 출력. 가중치가 없으면 예외."""
    import torch
    import torch.nn.functional as F

    _load_once(scale_factor=scale_factor, device=device)
    assert _MODEL is not None and _MODEL_DEVICE is not None

    if frame.ndim != 3 or frame.shape[2] != 3:
        return frame

    from .utils_image import to_uint8

    x = torch.from_numpy(frame.astype(np.float32) / 255.0).to(_MODEL_DEVICE)
    x = x[:, :, [2, 1, 0]].permute(2, 0, 1).unsqueeze(0)

    h, w = frame.shape[:2]
    sf = int(scale_factor) if int(scale_factor) > 0 else 12
    pad_h = (sf - (h % sf)) % sf
    pad_w = (sf - (w % sf)) % sf
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    with torch.no_grad():
        y, _ = _MODEL(x)
        if brightness_gain and abs(float(brightness_gain) - 1.0) > 1e-6:
            y = torch.clamp(y * float(brightness_gain), 0.0, 1.0)

    y = y.squeeze(0).permute(1, 2, 0)[:, :, [2, 1, 0]]
    if pad_h or pad_w:
        y = y[:h, :w, :]
    return to_uint8(y.detach().cpu().numpy())

"""
services/preprocess.py
======================

학습 프레임을 저장하기 **전에** 거치는 영상 전처리.

두 가지를 한다.

1. **주/야간 보정** — 드론 영상은 같은 표적이라도 주간(밝고 대비 큼)과 야간(어둡고
   저대비, IR 계열)에서 픽셀 분포가 크게 다르다. 야간 프레임을 그대로 학습에 넣으면
   저조도 구간에서 표적 특징이 뭉개져 성능이 떨어진다. 그래서 야간으로 판정된 프레임에만
   저조도 보정(CLAHE + 감마)을 걸어 대비를 살린다.

   판정은 **자동(밝기 임계치)** 이 기본이다. 프레임의 평균 밝기(Y, 0~255)가 임계치보다
   낮으면 야간으로 본다. 다만 자동 판정이 틀리는 영상(예: 낮인데 짙은 그늘, 흑백 렌즈)이
   있을 수 있어, **영상별로 주간/야간 고정 또는 끔** 을 지정할 수 있게 했다
   (그 지정은 학습 전 '구간' 단계에서 한다 → `video.set_preprocess`).

2. **해상도 다운스케일** — 운용 추론(tracker_py)이 640x480 으로 정규화하므로 학습도 같은
   분포로 맞추는 것이 기본이다. 다만 원본 해상도로 두고 싶은 경우가 있어 **선택** 으로
   끌 수 있게 했다(실제 리사이즈는 `video.fit_frame` 이 담당, 여기서는 판정/보정만).

야간 보정은 밝기 채널(LAB 의 L)에만 CLAHE 를 적용한다. 색을 건드리지 않아 색 왜곡이
없고, 국소 대비만 끌어올려 작은 표적의 윤곽이 살아난다. 밝은 주간 프레임에 같은 보정을
걸면 하늘/노면이 과증폭돼 오히려 노이즈가 늘기 때문에 기본적으로 주간은 손대지 않는다.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

# 주/야간 판정 모드.
#   auto  : 프레임 평균 밝기 < 임계치 → 야간 (기본)
#   day   : 항상 주간 취급(보정 안 함)
#   night : 항상 야간 취급(보정 함)
#   off   : 주/야간 보정 자체를 끔
DAYNIGHT_MODES = ("auto", "day", "night", "off")


def mean_luma(frame) -> float:
    """프레임의 평균 밝기(0~255). BT.601 Y = 0.299R+0.587G+0.114B.

    BGR 평균이 아니라 지각 밝기(Y)를 쓴다. 초록이 강한 야시(野視) 영상에서 단순 평균은
    실제보다 밝게 나와 야간을 주간으로 오판한다.
    """
    if frame is None or frame.size == 0:
        return 0.0
    b, g, r = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
    y = 0.114 * b + 0.587 * g + 0.299 * r
    return float(y.mean())


def classify(frame, threshold: float) -> str:
    """평균 밝기 임계치로 day/night 판정."""
    return "night" if mean_luma(frame) < float(threshold) else "day"


def _gamma(frame, gamma: float):
    """감마 보정. gamma>1 이면 어두운 영역을 밝게 편다(저조도에 유리)."""
    if not gamma or abs(gamma - 1.0) < 1e-3:
        return frame
    inv = 1.0 / float(gamma)
    lut = np.array([((i / 255.0) ** inv) * 255 for i in range(256)],
                   dtype=np.uint8)
    return cv2.LUT(frame, lut)


def enhance_night(frame, *, clahe_clip: float = 2.0, clahe_grid: int = 8,
                  gamma: float = 1.15):
    """야간 저조도 보정: L 채널 CLAHE(국소 대비) + 감마(전역 밝기)."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    grid = max(1, int(clahe_grid))
    clahe = cv2.createCLAHE(clipLimit=max(0.1, float(clahe_clip)),
                            tileGridSize=(grid, grid))
    l = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    return _gamma(out, gamma)


def resolve(cfg: Optional[dict] = None) -> dict:
    """전처리 설정을 서버 기본값 위에 병합해 완성한다.

    우선순위(호출자가 미리 합쳐 넘기는 것을 전제): 요청 > 영상별 저장값 > 서버 기본값.
    None/누락 필드는 서버 기본값으로 채운다.
    """
    from core.config import get_settings

    s = get_settings()
    cfg = dict(cfg or {})

    mode = str(cfg.get("daynight") or s.preprocess_daynight or "auto").lower()
    if mode not in DAYNIGHT_MODES:
        raise ValueError(f"daynight 은 {list(DAYNIGHT_MODES)} 중 하나여야 합니다: {mode!r}")

    resize = cfg.get("resize")
    resize = s.frame_resize if resize is None else bool(resize)
    return {
        "daynight": mode,
        "threshold": float(cfg.get("daynight_threshold")
                           if cfg.get("daynight_threshold") is not None
                           else s.daynight_threshold),
        "clahe_clip": float(cfg.get("night_clahe_clip")
                            if cfg.get("night_clahe_clip") is not None
                            else s.night_clahe_clip),
        "clahe_grid": int(cfg.get("night_clahe_grid")
                          if cfg.get("night_clahe_grid") is not None
                          else s.night_clahe_grid),
        "gamma": float(cfg.get("night_gamma")
                       if cfg.get("night_gamma") is not None
                       else s.night_gamma),
        "resize": resize,
        "resize_width": int(cfg.get("resize_width") or s.frame_width),
        "resize_height": int(cfg.get("resize_height") or s.frame_height),
    }


def apply_daynight(frame, resolved: dict) -> Tuple[object, str]:
    """주/야간 보정만 적용한다(리사이즈는 별도). (보정된 프레임, 판정결과) 반환.

    판정결과는 "day"|"night"|"off" 로, 프레임마다 무엇으로 처리했는지 라벨 문서에
    남겨 이후 집계/디버깅에 쓴다.
    """
    mode = resolved.get("daynight", "auto")
    if mode == "off":
        return frame, "off"
    decided = classify(frame, resolved["threshold"]) if mode == "auto" else mode
    if decided == "night":
        out = enhance_night(frame, clahe_clip=resolved["clahe_clip"],
                            clahe_grid=resolved["clahe_grid"],
                            gamma=resolved["gamma"])
        return out, "night"
    return frame, "day"

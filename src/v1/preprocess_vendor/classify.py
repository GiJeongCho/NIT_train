"""
classify
========

tracker_py `preprocess/pipeline.py` 의 auto 모드 판정 로직을 이식(vendored).

핵심: 밝기 하나가 아니라 **밝기·대비·채도·선명도** 를 종합해 dark(저조도)/fog(안개)를
**독립적으로** 판정한다. 안개는 '대비 낮음 + 채도 낮음 + 흐릿함' 이 동시에 성립할 때만
인정한다(밝기만으로는 못 잡는다). 극야간(너무 어두움)은 안개 판정에서 제외한다.

추론 서비스는 프레임 스트림에서 최근 15프레임 다수결로 플리커를 억제한다. 학습 프레임은
띄엄띄엄 샘플링되므로(초당 2장 등) 시간적 연속성이 약해, 여기서는 다수결 창을 옵션으로
두되 기본은 프레임 단위 판정을 쓴다(각 저장 프레임이 자기 특성대로 전처리되도록).

임계값 기본치는 원본과 동일:
  extreme_dark_th=15, dark_th=60, contrast_th=40, sat_th=50, sharp_th=200
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import cv2

DEFAULT_THRESHOLDS = {
    "extreme_dark_th": 15.0,
    "dark_th": 60.0,
    "contrast_th": 40.0,
    "sat_th": 50.0,
    "sharp_th": 200.0,
}


def metrics(frame) -> dict:
    """프레임 특성 지표. brightness/contrast/saturation/sharpness."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "saturation": float(hsv[:, :, 1].mean()),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }


def classify_conditions(frame, cfg: Optional[dict] = None) -> dict:
    """dark/fog 를 독립 판정. 반환: {"dark": bool, "fog": bool, "brightness": float}.

    원본 `classify_conditions()` 와 동일한 규칙:
      - brightness < extreme_dark_th               → dark(극야간), fog 제외
      - (contrast<c) and (sat<s) and (sharp<p)     → fog
      - brightness < dark_th                       → dark
    """
    cfg = {**DEFAULT_THRESHOLDS, **(cfg or {})}
    if frame is None or getattr(frame, "ndim", 0) != 3:
        return {"dark": False, "fog": False, "brightness": 0.0}

    m = metrics(frame)
    b, c, sat, sharp = m["brightness"], m["contrast"], m["saturation"], m["sharpness"]

    is_extreme_dark = b < float(cfg["extreme_dark_th"])
    is_dark = b < float(cfg["dark_th"])
    is_foggy = (c < float(cfg["contrast_th"])) and (sat < float(cfg["sat_th"])) and (sharp < float(cfg["sharp_th"]))
    return {
        "dark": bool(is_dark or is_extreme_dark),
        "fog": bool(is_foggy and not is_extreme_dark),
        "brightness": round(b, 2),
    }


class Classifier:
    """영상 1개(추출 잡 1개) 동안 유지되는 판정기.

    `window` 를 주면 최근 N프레임 다수결로 안정화한다(추론 서비스처럼). 기본은 창 없음
    (프레임 단위 판정). 임계값은 `cfg` 로 덮을 수 있다.
    """

    def __init__(self, cfg: Optional[dict] = None, window: int = 0):
        self.cfg = {**DEFAULT_THRESHOLDS, **(cfg or {})}
        self.window = int(window or 0)
        self._hist: deque = deque(maxlen=self.window) if self.window > 0 else deque()
        self.last: dict = {}

    def reset(self) -> None:
        self._hist.clear()
        self.last = {}

    def classify(self, frame) -> dict:
        raw = classify_conditions(frame, self.cfg)
        self.last = raw
        if self.window <= 0:
            return raw
        self._hist.append((raw["dark"], raw["fog"]))
        n = len(self._hist)
        if n < 3:
            return raw
        dark_votes = sum(1 for it in self._hist if it[0])
        fog_votes = sum(1 for it in self._hist if it[1])
        return {"dark": dark_votes * 2 >= n, "fog": fog_votes * 2 >= n, "brightness": raw["brightness"]}

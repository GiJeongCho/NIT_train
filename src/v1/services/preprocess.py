"""
services/preprocess.py
======================

학습 프레임을 저장하기 **전에** 거치는 영상 전처리. 추론 서비스(**tracker_py**)의 전처리를
그대로 떼어 온 `preprocess_vendor` 엔진을 오케스트레이션한다.

왜 tracker_py 것을 재사용하나
-----------------------------
학습 이미지는 **추론 입력과 똑같이** 전처리돼야 분포가 맞다. 예전에는 여기서 감마/간이
DCP 를 자체 구현했는데, 그건 추론 파이프라인과 **다른 알고리즘**이라 "학습≠추론" 이 됐다.
그래서 tracker_py 가 추론 직전에 돌리는 전처리를 그대로 복사해(`preprocess_vendor`) 쓴다.

파이프라인(추론 `run_all` 과 동일한 순서)
------------------------------------------
    (원본 프레임)
      → [야간 보정(dark)]  auto 면 저조도로 판정된 프레임에만
      → [안개 제거(fog)]   auto 면 안개로 판정된 프레임에만
      → [화질 향상(CLAHE)] 켜져 있으면 전 프레임 (추론 Stage2 기본 ON)
      → [표적 강조(emphasis)] 켜져 있으면 전 프레임 (추론 Stage3, 언샤프 마스크·기본 OFF)
      → (선택) 해상도 다운스케일 은 video.fit_frame 이 담당

세 전처리는 **서로 독립**으로 켠다. `auto` 를 켜면 프레임마다 tracker_py 와 동일한
다중지표(밝기·대비·채도·선명도)로 dark/fog 를 판정해 **해당하는 프레임에만** 적용한다.
`auto` 를 끄면 켜 둔 전처리를 **모든 프레임**에 강제 적용한다(tracker_py 의 custom 모드).

야간 보정(dark) 구현
--------------------
- `preprocess_vendor/zero_dce_weights/Epoch99.pth` 가 있으면 추론과 동일한 **Zero-DCE++**(CNN)
  로 보정한다(학습==추론 완전 일치).
- 가중치가 없으면(현재 기본) 가중치 불필요한 대체로 폴백한다: **감마(전역 밝기 상향) +
  L 채널 CLAHE(국소 대비)**. 감마가 실제 밝기 축이라 부족하면 값을 키운다.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from preprocess_vendor import classify as _classify
from preprocess_vendor import clahe_lab as _clahe
from preprocess_vendor import dcp_dehaze as _dcp
from preprocess_vendor import zero_dce as _zdce


# ── 가중치 불필요 야간 보정 폴백(감마 + CLAHE) ─────────────────────────────
def _gamma(frame, gamma: float):
    if not gamma or abs(gamma - 1.0) < 1e-3:
        return frame
    inv = 1.0 / float(gamma)
    lut = np.array([((i / 255.0) ** inv) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(frame, lut)


def _enhance_lowlight_fallback(frame, *, gamma: float, clahe_clip: float, clahe_grid: int):
    """Zero-DCE++ 가중치가 없을 때의 저조도 보정: 감마 → L 채널 CLAHE."""
    out = _gamma(frame, gamma)
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    grid = max(1, int(clahe_grid))
    clahe = cv2.createCLAHE(clipLimit=max(0.1, float(clahe_clip)), tileGridSize=(grid, grid))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def _emphasis_unsharp(frame, *, sigma: float, alpha: float):
    """Stage3 표적 강조 — 언샤프 마스크(tracker_py emphasis.unsharp_mask 이식).

    out = frame·(1+α) − blur·α. 고주파(윤곽) 성분을 α 만큼 증폭해 원거리 소형 표적의
    흐릿한 경계를 또렷하게 만든다. 좌표를 바꾸지 않는 광도 변환이라 라벨은 그대로 유효하다.
    """
    a = float(alpha)
    if a <= 0:
        return frame
    blur = cv2.GaussianBlur(frame, (0, 0), max(0.1, float(sigma)))
    out = cv2.addWeighted(frame, 1.0 + a, blur, -a, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def _enhance_lowlight(frame, resolved: dict):
    """야간 보정 1회. 가중치가 있으면 Zero-DCE++, 없으면 감마+CLAHE 폴백."""
    if _zdce.available():
        try:
            return _zdce.enhance(frame, brightness_gain=float(resolved.get("zerodce_gain", 1.0)))
        except Exception:
            # 가중치는 있으나 torch 미설치/로드 실패 등 → 폴백.
            pass
    return _enhance_lowlight_fallback(
        frame,
        gamma=resolved["night_gamma"],
        clahe_clip=resolved["night_clahe_clip"],
        clahe_grid=resolved["night_clahe_grid"],
    )


def lowlight_engine() -> str:
    """현재 야간 보정에 쓰일 엔진 이름('zero_dce++' | 'gamma_clahe')."""
    return "zero_dce++" if _zdce.available() else "gamma_clahe"


def resolve(cfg=None) -> dict:
    """전처리 설정을 서버 기본값 위에 병합해 완성한다.

    우선순위(호출자가 미리 합쳐 넘기는 것을 전제): 요청 > 영상별 저장값 > 서버 기본값.
    None/누락 필드는 서버 기본값으로 채운다.
    """
    from core.config import get_settings

    s = get_settings()
    cfg = dict(cfg or {})

    def _bool(key, default):
        v = cfg.get(key)
        return bool(default) if v is None else bool(v)

    def _num(key, default, cast):
        v = cfg.get(key)
        return cast(default) if v is None else cast(v)

    # 야간 판정 임계(밝기). lowlight_threshold 우선, 예전 이름(daynight_threshold)도 허용.
    thr = cfg.get("lowlight_threshold")
    if thr is None:
        thr = cfg.get("daynight_threshold")
    thr = s.daynight_threshold if thr is None else float(thr)

    return {
        # 무엇을 켤지
        "auto": _bool("auto", s.preprocess_auto),
        "lowlight": _bool("lowlight", s.preprocess_lowlight),
        "dehaze": _bool("dehaze", s.preprocess_dehaze),
        "clahe": _bool("clahe", s.preprocess_clahe),
        # auto 판정 임계(밝기). 나머지 임계(대비/채도/선명)는 엔진 기본값 사용.
        "dark_th": float(thr),
        # 저조도 폴백(감마+CLAHE) 파라미터
        "night_gamma": _num("night_gamma", s.night_gamma, float),
        "night_clahe_clip": _num("night_clahe_clip", s.night_clahe_clip, float),
        "night_clahe_grid": _num("night_clahe_grid", s.night_clahe_grid, int),
        # 안개 제거(DCP) 파라미터 — 추론 기본값과 동일
        "dehaze_omega": _num("dehaze_omega", s.dehaze_omega, float),
        "dehaze_t0": _num("dehaze_t0", s.dehaze_t0, float),
        "dehaze_wsz": _num("dehaze_wsz", s.dehaze_wsz, int),
        "dehaze_scale": _num("dehaze_scale", s.dehaze_scale, float),
        "dehaze_guide_r": _num("dehaze_guide_r", s.dehaze_guide_r, int),
        # 화질 향상(CLAHE, quality) 파라미터 — 추론 기본값과 동일
        "clahe_clip": _num("clahe_clip", s.quality_clahe_clip, float),
        "clahe_grid": _num("clahe_grid", s.quality_clahe_grid, int),
        # 표적 강조(Stage3, emphasis) — 언샤프 마스크. 기본 OFF(옵트인).
        "emphasis": _bool("emphasis", s.preprocess_emphasis),
        "emphasis_sigma": _num("emphasis_sigma", s.emphasis_sigma, float),
        "emphasis_alpha": _num("emphasis_alpha", s.emphasis_alpha, float),
        # 해상도 다운스케일(적용은 video.fit_frame)
        "resize": _bool("resize", s.frame_resize),
        "resize_width": _num("resize_width", s.frame_width, int),
        "resize_height": _num("resize_height", s.frame_height, int),
    }


def apply(frame, resolved: dict) -> Tuple[object, dict]:
    """전처리(야간 보정 → 안개 제거 → CLAHE)를 적용한다. (보정 프레임, 판정정보) 반환.

    판정정보 = {"lowlight": "night"|"day"|"off", "dehaze": bool, "clahe": bool}.
    프레임마다 무엇을 적용했는지 라벨 문서에 남겨 집계/디버깅에 쓴다. 리사이즈는 하지 않는다.
    """
    info = {"lowlight": "off", "dehaze": False, "clahe": False, "emphasis": False}
    out = frame

    want_low = bool(resolved.get("lowlight"))
    want_fog = bool(resolved.get("dehaze"))
    auto = bool(resolved.get("auto", True))

    do_low, do_fog = want_low, want_fog
    if auto and (want_low or want_fog):
        # 추론과 동일한 다중지표 판정. 밝기 임계만 사용자값으로 덮는다.
        cond = _classify.classify_conditions(frame, {"dark_th": resolved.get("dark_th", 60.0)})
        do_low = want_low and bool(cond["dark"])
        do_fog = want_fog and bool(cond["fog"])

    # 1) 야간 보정(dark)
    if want_low:
        if do_low:
            out = _enhance_lowlight(out, resolved)
            info["lowlight"] = "night"
        else:
            info["lowlight"] = "day"

    # 2) 안개 제거(fog)
    if do_fog:
        out = _dcp.preprocess(
            out,
            wsz=int(resolved.get("dehaze_wsz", 15)),
            t0=float(resolved.get("dehaze_t0", 0.4)),
            omega=float(resolved.get("dehaze_omega", 0.80)),
            scale=float(resolved.get("dehaze_scale", 0.25)),
            guide_r=int(resolved.get("dehaze_guide_r", 20)),
        )
        info["dehaze"] = True

    # 3) 화질 향상(CLAHE, quality) — 추론 Stage2 기본 ON
    if resolved.get("clahe"):
        out = _clahe.preprocess_with_params(
            out,
            clip_limit=float(resolved.get("clahe_clip", 2.0)),
            tile_grid_size=(int(resolved.get("clahe_grid", 8)), int(resolved.get("clahe_grid", 8))),
            mode="lab_l",
        )
        info["clahe"] = True

    # 4) 표적 강조(Stage3, emphasis) — 언샤프 마스크. 추론 체인과 동일하게 맨 마지막.
    if resolved.get("emphasis"):
        out = _emphasis_unsharp(
            out,
            sigma=float(resolved.get("emphasis_sigma", 1.0)),
            alpha=float(resolved.get("emphasis_alpha", 0.5)),
        )
        info["emphasis"] = True

    return out, info

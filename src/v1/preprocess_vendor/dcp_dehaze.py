"""
dcp_dehaze
==========

tracker_py `preprocess/modules/fog/dcp_dehaze.py` 를 복사(vendored).
Dark Channel Prior(DCP) 기반 안개 제거. cv2/numpy 만 사용(GPU/가중치 불필요).

추론 서비스는 기본으로 GPU 버전(`dcp_dehaze_gpu`)을 쓰지만 **같은 DCP 알고리즘**이라,
GPU 가 없는 학습-데이터 준비 환경에서 이 CPU 버전을 같은 파라미터로 돌리면 결과가 사실상
동일하다. 그래서 추론 기본값(wsz=15, omega=0.80, t0=0.4, scale=0.25, guide_r=20)을 그대로
넘겨 쓰면 학습==추론 전처리가 맞는다.

원본: https://github.com/He-Zhang/image_dehaze
논문: He et al., "Single Image Haze Removal Using Dark Channel Prior", CVPR 2009
      He et al., "Guided Image Filtering", ECCV 2010
"""

import cv2
import numpy as np


def _get_dark_channel(im, wsz):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (wsz, wsz))
    return cv2.erode(dc, kernel)


def _atm_light(im, dark):
    imsz = np.prod(im.shape[:2])
    darkvec = dark.flatten()
    indices = darkvec.argsort()[imsz - max(imsz // 1000, 1):]
    A = np.mean(im.reshape(-1, 3)[indices], axis=0, keepdims=True)
    return A


def _guided_filter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_32F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_32F, (r, r))
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_32F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(im * im, cv2.CV_32F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    return mean_a * im + mean_b


def preprocess(frame, wsz=15, t0=0.4, omega=0.80, scale=0.25, guide_r=20):
    """DCP dehaze 전처리(BGR uint8 입출력).

    Parameters
    ----------
    frame : np.ndarray  BGR uint8 입력.
    wsz   : int         dark channel 패치 크기(홀수 권장).
    t0    : float       최소 transmission(낮을수록 짙은 안개 강하게 제거).
    omega : float       dehaze 강도(0~1).
    scale : float       투과율 추정용 다운스케일 비율(0.25=1/4 해상도로 추정 → 빠름).
    guide_r : int       guided filter 반경.
    """
    h, w = frame.shape[:2]
    I = frame.astype(np.float32) / 255.0

    sh, sw = int(h * scale), int(w * scale)
    small = cv2.resize(I, (sw, sh))
    small_gray = cv2.cvtColor(
        cv2.resize(frame, (sw, sh)), cv2.COLOR_BGR2GRAY
    ).astype(np.float32) / 255.0

    swsz = max(int(wsz * scale), 3) | 1

    dark = _get_dark_channel(small, swsz)
    A = _atm_light(small, dark)
    te = 1.0 - omega * _get_dark_channel(small / A, swsz)
    tr = _guided_filter(small_gray, te, guide_r, 0.0004)

    t = cv2.resize(tr, (w, h))
    t = np.maximum(t, t0)

    res = (I - A) / t[:, :, np.newaxis] + A
    return np.clip(res * 255, 0, 255).astype(np.uint8)

"""
services/sam.py
===============

SAM(누끼) 세그멘테이션 래퍼.

첫 프레임에 박스만 주면 그 박스를 프롬프트로 객체 마스크를 따고(누끼), 마스크의
최소회전사각형(minAreaRect)으로 **타이트한 OBB** 를 만든다. 다음 프레임에서는
'이전 박스'를 다시 프롬프트로 써서 객체를 프레임 간에 이어 붙인다(propagate).

`ultralytics` 에 SAM/SAM2/MobileSAM 이 포함돼 있어 **새 의존성 없이** 가중치만 있으면
된다(최초 사용 시 자동 다운로드). 모델은 프로세스 전역에 한 번만 로드해 재사용한다.
"""

from __future__ import annotations

import threading
from typing import List, Optional

import cv2
import numpy as np

from core.config import get_settings

_model = None
_model_key = None
_lock = threading.Lock()


def _load():
    """SAM 모델을 로드(캐시)해 돌려준다. 실패하면 원인을 담아 올린다."""
    global _model, _model_key
    s = get_settings()
    key = (s.sam_model, s.device)
    with _lock:
        if _model is not None and _model_key == key:
            return _model
        try:
            from ultralytics import SAM
            model = SAM(s.sam_model)
        except Exception as e:  # noqa: BLE001 - 가중치 없음/다운로드 실패 등 원인을 그대로 노출
            raise RuntimeError(
                f"SAM 가중치를 불러오지 못했습니다: {s.sam_model!r}. 인터넷이 되면 최초 "
                f"사용 시 자동 다운로드되고, 폐쇄망이면 가중치 파일을 두고 "
                f"NIT_TRAIN_SAM_MODEL 에 경로를 지정하세요. (원인: {type(e).__name__}: {e})"
            ) from e
        _model = model
        _model_key = key
        print(f"[sam] 로드 완료: {s.sam_model} (device={s.device})", flush=True)
        return _model


def _extract_masks(res, width: int, height: int, n: int) -> List[Optional[np.ndarray]]:
    """ultralytics 결과 1건에서 마스크(uint8 HxW) 리스트를 뽑는다(요청 박스 순서 기준)."""
    out: List[Optional[np.ndarray]] = [None] * n
    if not res:
        return out
    r = res[0]
    masks = getattr(r, "masks", None)
    data = getattr(masks, "data", None) if masks is not None else None
    if data is None:
        return out
    try:
        arr = data.cpu().numpy()
    except AttributeError:
        arr = np.asarray(data)
    for i in range(min(int(arr.shape[0]), n)):
        m = (arr[i] > 0.5).astype("uint8")
        if m.shape[:2] != (height, width):
            m = cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST)
        out[i] = m
    return out


def segment_boxes(image_bgr, boxes_xyxy: List[List[float]]) -> List[Optional[np.ndarray]]:
    """여러 박스 프롬프트를 세그멘트. `boxes_xyxy` 순서에 맞춘 마스크 리스트를 돌려준다.

    빈 마스크/실패는 None. 한 번의 호출로 여러 박스를 넣으면 이미지 인코딩을 한 번만 하지만,
    반환 마스크 개수가 입력 박스와 어긋나면(모델/버전 차이) 정렬이 깨지므로 박스별로 다시 돌려
    정합성을 보장한다.
    """
    if not boxes_xyxy:
        return []
    model = _load()
    s = get_settings()
    h, w = image_bgr.shape[:2]
    boxes = [[float(v) for v in b] for b in boxes_xyxy]

    res = model(image_bgr, bboxes=boxes, device=s.device, verbose=False, retina_masks=True)
    masks = _extract_masks(res, w, h, len(boxes))
    if sum(m is not None for m in masks) == len(boxes):
        return masks

    # 개수/정렬이 어긋나면 박스별 단일 호출로 확실하게 맞춘다.
    out: List[Optional[np.ndarray]] = []
    for b in boxes:
        r = model(image_bgr, bboxes=[b], device=s.device, verbose=False, retina_masks=True)
        got = _extract_masks(r, w, h, 1)
        out.append(got[0])
    return out


def _component_at(mask, x: float, y: float):
    """클릭 지점에 연결된 성분만 남긴 마스크. 떨어져 있는 궤적/잡티 덩어리를 떼어낸다.

    지점이 마스크 밖이면(엇나간 클릭) 가장 큰 성분을 쓴다.
    """
    m = mask.astype("uint8")
    n, lbl = cv2.connectedComponents(m)
    if n <= 1:
        return None
    h, w = m.shape[:2]
    xi = min(max(int(round(x)), 0), w - 1)
    yi = min(max(int(round(y)), 0), h - 1)
    lab = int(lbl[yi, xi])
    if lab == 0:
        areas = [(int((lbl == l).sum()), l) for l in range(1, n)]
        if not areas:
            return None
        lab = max(areas)[1]
    return (lbl == lab).astype("uint8")


def _trim_trail(comp, x: float, y: float, shrink: float = 0.35):
    """움직이는 물체 뒤에 이어진 '가는 궤적/자국'을 잘라내고 물체 본체만 남긴다.

    거리변환으로 클릭 지점 주변의 두꺼운 '본체'만 남긴 뒤, 그 코어를 본체 반경만큼 다시
    부풀려(원래 마스크 안에서) 형태를 복원한다. 궤적은 본체보다 얇으므로 임계값 아래로 사라진다.

    shrink: 0에 가까울수록 원본에 가깝게(덜 자름), 1에 가까울수록 코어만 남김(많이 자름).
    """
    m = comp.astype("uint8")
    if m.sum() == 0:
        return m
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    peak = float(dt.max())
    if peak < 2.0:
        return m  # 이미 얇음 — 자를 게 없음
    h, w = m.shape[:2]
    xi = min(max(int(round(x)), 0), w - 1)
    yi = min(max(int(round(y)), 0), h - 1)
    d_click = float(dt[yi, xi])
    # 본체 반경 추정: 클릭 지점 두께와 전체 최대 두께 중 큰 값을 신뢰.
    body_r = max(d_click, peak)
    thr = max(shrink * body_r, 2.0)
    core = (dt >= thr).astype("uint8")
    core = _component_at(core, x, y)
    if core is None or core.sum() == 0:
        return m
    # 코어를 본체 반경만큼 부풀려 원래 마스크와 교집합 → 궤적 없는 본체 복원.
    k = max(int(round(thr)) * 2 + 1, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    grown = cv2.dilate(core, kernel, iterations=1)
    recon = cv2.bitwise_and(grown, m)
    recon = _component_at(recon, x, y)
    return recon if (recon is not None and recon.sum() > 0) else m


def _drop_shadow(image_bgr, comp, x: float, y: float, ratio: float = 0.6):
    """마스크 안에서 **물체 본체보다 뚜렷하게 어두운(그림자)** 픽셀을 색/밝기로 떼어낸다.

    클릭 지점 주변(물체 본체)의 perceptual 밝기(LAB L)를 기준으로, `ratio×본체밝기` 미만인
    어두운 픽셀을 그림자로 보고 제거한다. 그림자는 물체의 밝은 윗면보다 어둡다는 성질을 쓴다.
    본체까지 과하게 깎여 마스크가 크게 줄면(어두운 물체 등) 원본 성분을 그대로 유지한다.
    """
    m = comp.astype("uint8")
    area0 = int(m.sum())
    if area0 == 0 or image_bgr is None or ratio <= 0:
        return m
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype("float32")
    h, w = m.shape[:2]
    xi = min(max(int(round(x)), 0), w - 1)
    yi = min(max(int(round(y)), 0), h - 1)
    # 본체 밝기 표본: 클릭 지점 주변 반경 내 마스크 픽셀의 중앙값.
    r = max(3, int(round((area0 ** 0.5) / 4)))
    y0, y1 = max(0, yi - r), min(h, yi + r + 1)
    x0, x1 = max(0, xi - r), min(w, xi + r + 1)
    win_m = m[y0:y1, x0:x1]
    win_L = L[y0:y1, x0:x1]
    body_vals = win_L[win_m > 0]
    if body_vals.size < 5:
        body_vals = L[m > 0]
    if body_vals.size == 0:
        return m
    body_L = float(np.median(body_vals))
    thr = float(ratio) * body_L
    keep = ((L >= thr).astype("uint8")) & m
    keep = _component_at(keep, x, y)
    if keep is None:
        return m
    # 안전장치: 본체까지 과하게 깎이면(면적 10% 미만) 원본 유지.
    if int(keep.sum()) < 0.1 * area0:
        return m
    return keep


def tighten_mask(mask, x: float, y: float, shrink: float = 0.45,
                 image_bgr=None, shadow_ratio: float = 0.0):
    """마스크를 (x,y) 기준으로 조인다: 연결 성분만 남기고 그림자/궤적을 떼어낸다.

    처리 순서: 연결성분 → (색/밝기 그림자 제거) → (기하 오프닝으로 얇은 돌출부 제거).
    점 클릭 누끼와 중심점 전파가 같은 규칙으로 조여지도록 공용화한 헬퍼.
    실패 시 None. shadow_ratio<=0 이면 그림자 제거 생략, shrink<=0 이면 기하 오프닝 생략.
    """
    if mask is None:
        return None
    comp = _component_at(mask, x, y)
    if comp is None or comp.sum() == 0:
        return None
    if image_bgr is not None and shadow_ratio and shadow_ratio > 0:
        comp = _drop_shadow(image_bgr, comp, x, y, float(shadow_ratio))
    if shrink and shrink > 0:
        comp = _trim_trail(comp, x, y, float(shrink))
    return comp


def _clip_to_box(mask, clamp_box, w: int, h: int):
    """마스크를 사각 영역 안으로만 남긴다(밖은 0). 물체 예상 범위 밖으로 뻗는 그림자를 잘라낸다."""
    if clamp_box is None:
        return mask
    x1, y1, x2, y2 = [float(v) for v in clamp_box[:4]]
    x1 = int(max(0, min(w - 1, round(min(x1, x2)))))
    x2 = int(max(0, min(w - 1, round(max(x1, x2)))))
    y1 = int(max(0, min(h - 1, round(min(y1, y2)))))
    y2 = int(max(0, min(h - 1, round(max(y1, y2)))))
    region = np.zeros_like(mask)
    region[y1:y2 + 1, x1:x2 + 1] = 1
    return (mask & region).astype("uint8")


def segment_point(image_bgr, x: float, y: float, *, min_area: float = 16.0,
                  shrink: float = 0.45, shadow: bool = True,
                  shadow_ratio: float = 0.6, clamp_box=None,
                  neg_points=None) -> Optional[List[List[float]]]:
    """한 점(포그라운드)을 프롬프트로 그 지점 물체를 세그멘트 → OBB 4점.

    '마우스로 찍은 곳 주변'을 보고 누끼를 따는 대화형 라벨링용. 실패/빈 마스크는 None.

    범위 축소(그림자·궤적 제거):
    - `neg_points=[[x,y],...]`: **배경(그림자)로 제외**할 점들. SAM 에 label=0 으로 주면 그 지점을
      물체에서 빼고 다시 딴다. 밝기·크기와 무관하게 그림자를 확실히 떼는 가장 강력한 수단.
    - `clamp_box=[x1,y1,x2,y2]`: 마스크를 이 사각 범위 안으로만 남긴다. 이전 박스(또는 클릭 반경)
      밖으로 뻗는 그림자를 색과 무관하게 잘라낸다.
    - 클릭 지점에 **연결된 성분만** 남겨, 떨어져 있는 궤적/잡티 덩어리를 버린다.
    - `shadow=True`(기본): 본체보다 어두운 **그림자 픽셀**을 색/밝기로 떼어낸다(`shadow_ratio`).
    - 본체에 붙어 이어진 **얇은 궤적 돌출부**는 거리변환 오프닝으로 잘라낸다(`shrink`).
    """
    model = _load()
    s = get_settings()
    h, w = image_bgr.shape[:2]
    pts: List[List[float]] = [[float(x), float(y)]]
    lbls: List[int] = [1]
    for np_ in (neg_points or []):
        if np_ is None or len(np_) < 2:
            continue
        pts.append([float(np_[0]), float(np_[1])])
        lbls.append(0)
    res = model(image_bgr, points=pts, labels=lbls,
                device=s.device, verbose=False, retina_masks=True)
    masks = _extract_masks(res, w, h, 1)
    if not masks or masks[0] is None:
        return None
    m = _clip_to_box(masks[0], clamp_box, w, h)
    comp = tighten_mask(m, x, y, shrink,
                        image_bgr=image_bgr if shadow else None,
                        shadow_ratio=(shadow_ratio if shadow else 0.0))
    if comp is None or comp.sum() == 0:
        return None
    return mask_to_obb_poly(comp, min_area)


def segment_box(image_bgr, box_xyxy: List[float], *, min_area: float = 16.0,
                shrink: float = 0.45, shadow: bool = True,
                shadow_ratio: float = 0.6) -> Optional[List[List[float]]]:
    """**박스 프롬프트**로 그 영역 물체를 세그멘트 → OBB 4점.

    점 프롬프트(segment_point)와 달리 박스로 물체를 '락온'하므로, 프레임 사이 물체가 움직여도
    (박스가 새 위치를 덮는 한) 놓치지 않는다. 전파(이전 박스로 다음 프레임 잡기)에 적합.
    조이기(그림자/궤적)는 점 누끼와 동일 규칙을 박스 중심 기준으로 적용한다.
    """
    h, w = image_bgr.shape[:2]
    masks = segment_boxes(image_bgr, [box_xyxy])
    if not masks or masks[0] is None:
        return None
    cx = (float(box_xyxy[0]) + float(box_xyxy[2])) / 2.0
    cy = (float(box_xyxy[1]) + float(box_xyxy[3])) / 2.0
    comp = tighten_mask(masks[0], cx, cy, shrink,
                        image_bgr=image_bgr if shadow else None,
                        shadow_ratio=(shadow_ratio if shadow else 0.0))
    if comp is None or comp.sum() == 0:
        return None
    return mask_to_obb_poly(comp, min_area)


def mask_to_obb_poly(mask, min_area: float = 16.0) -> Optional[List[List[float]]]:
    """이진 마스크 → 최소회전사각형 4점(OBB). 너무 작은 마스크는 버린다."""
    if mask is None:
        return None
    cnts, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < float(min_area):
        return None
    box = cv2.boxPoints(cv2.minAreaRect(c))
    return [[float(x), float(y)] for x, y in box]


def available() -> bool:
    """ultralytics 에 SAM 이 있는지(가중치 다운로드 전에도 True). 라우트 사전검증용."""
    try:
        from ultralytics import SAM  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False

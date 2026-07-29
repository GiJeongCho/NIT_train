"""
utils/geometry.py
=================

라벨 좌표 변환. 이 서비스의 라벨은 **항상 4점 폴리곤(픽셀 좌표)** 하나로 통일한다.

이유: 축정렬 박스(detect)는 회전 박스(obb)의 특수한 경우다. 내부를 폴리곤으로 두면
같은 라벨 자산으로 detect 학습(`class cx cy w h`)과 OBB 학습(`class x1 y1 … y4`)을
모두 내보낼 수 있고, 나중에 태스크를 바꿔도 라벨을 다시 만들 필요가 없다.

폴리곤 점 순서는 ultralytics 규약과 같게 시계방향(좌상→우상→우하→좌하)을 쓴다.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

Point = Tuple[float, float]
Poly = List[List[float]]


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def xyxy_to_poly(x1: float, y1: float, x2: float, y2: float) -> Poly:
    """축정렬 박스 → 시계방향 4점 폴리곤."""
    return [[float(x1), float(y1)], [float(x2), float(y1)],
            [float(x2), float(y2)], [float(x1), float(y2)]]


def poly_to_xyxy(poly: Sequence[Sequence[float]]) -> List[float]:
    """폴리곤을 감싸는 축정렬 박스(AABB)."""
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]
    return [min(xs), min(ys), max(xs), max(ys)]


def poly_area(poly: Sequence[Sequence[float]]) -> float:
    """신끈 공식. 너무 작은(사실상 점) 라벨을 걸러내는 데 쓴다."""
    n = len(poly)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = float(poly[i][0]), float(poly[i][1])
        x2, y2 = float(poly[(i + 1) % n][0]), float(poly[(i + 1) % n][1])
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def normalize_poly(poly: Sequence[Sequence[float]], width: int, height: int) -> List[float]:
    """픽셀 폴리곤 → 0~1 정규화된 8개 값(x1 y1 … x4 y4)."""
    w = max(1, int(width))
    h = max(1, int(height))
    out: List[float] = []
    for p in poly:
        out.append(clamp(float(p[0]) / w, 0.0, 1.0))
        out.append(clamp(float(p[1]) / h, 0.0, 1.0))
    return out


def denormalize_poly(values: Sequence[float], width: int, height: int) -> Poly:
    """0~1 정규화된 8개 값 → 픽셀 폴리곤."""
    w = max(1, int(width))
    h = max(1, int(height))
    pts: Poly = []
    for i in range(0, len(values) - 1, 2):
        pts.append([float(values[i]) * w, float(values[i + 1]) * h])
    return pts


def sanitize_poly(raw, width: int, height: int) -> Poly:
    """외부(프런트) 입력 폴리곤을 검증해 화면 안으로 클램프한다.

    4점이 아니면 AABB 로 강제한다. 라벨 파일에 4점 아닌 도형이 섞이면
    데이터셋 내보내기에서 조용히 망가지므로 입구에서 막는다.
    """
    pts: Poly = []
    for p in (raw or []):
        if isinstance(p, dict):
            x, y = p.get("x"), p.get("y")
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            x, y = p[0], p[1]
        else:
            continue
        try:
            pts.append([clamp(float(x), 0.0, float(width)), clamp(float(y), 0.0, float(height))])
        except (TypeError, ValueError):
            continue
    if len(pts) == 4:
        return pts
    if len(pts) >= 2:
        x1, y1, x2, y2 = poly_to_xyxy(pts)
        return xyxy_to_poly(x1, y1, x2, y2)
    raise ValueError("폴리곤 좌표가 부족합니다 (최소 2점, 권장 4점)")


def to_obb_line(class_id: int, poly: Sequence[Sequence[float]], width: int, height: int) -> str:
    """YOLO OBB 라벨 한 줄: `class x1 y1 x2 y2 x3 y3 x4 y4` (정규화)."""
    vals = normalize_poly(poly, width, height)
    return " ".join([str(int(class_id))] + [f"{v:.6f}" for v in vals])


def to_detect_line(class_id: int, poly: Sequence[Sequence[float]], width: int, height: int) -> str:
    """YOLO detect 라벨 한 줄: `class cx cy w h` (정규화, 폴리곤의 AABB)."""
    x1, y1, x2, y2 = poly_to_xyxy(poly)
    w = max(1, int(width))
    h = max(1, int(height))
    cx = clamp((x1 + x2) / 2 / w, 0.0, 1.0)
    cy = clamp((y1 + y2) / 2 / h, 0.0, 1.0)
    bw = clamp((x2 - x1) / w, 0.0, 1.0)
    bh = clamp((y2 - y1) / h, 0.0, 1.0)
    return f"{int(class_id)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def to_label_line(task: str, class_id: int, poly: Sequence[Sequence[float]],
                  width: int, height: int) -> str:
    """태스크에 맞는 라벨 한 줄. `obb` 면 8좌표, 그 외(detect)는 cxcywh."""
    if str(task).lower() == "obb":
        return to_obb_line(class_id, poly, width, height)
    return to_detect_line(class_id, poly, width, height)

"""
services/propagate.py
=====================

첫(씨앗) 프레임의 박스로 이후 프레임을 **SAM 누끼**로 자동 라벨(전파)한다.

사람이 '3·라벨링'에서 씨앗 프레임에 박스를 잡아주면, '이전 중심 누끼 전파'(W 키와 동일
방식)로::

    이전 박스 중심점 → SAM 점-프롬프트 누끼 → minAreaRect OBB → 그 박스 중심을 다음 프레임 점으로 → …

식으로 **비정상 구간 끝까지** 라벨을 이어 붙인다. 궤적·그림자는 shrink(누끼 강도)로 조인다.
대상 프레임은 이미 추출·전처리돼
저장된 JPG 라, 학습에 들어갈 이미지와 100% 같은 그림에 라벨이 붙는다. 결과는 사람이
검수하도록 `pending` 으로 남긴다(정상 구간은 애초에 라벨링이 필요 없다).

수 초~수십 초 걸릴 수 있어 백그라운드 잡으로 돌리고 진행률만 폴링한다.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2

from core import store
from core.config import get_settings
from services import (
    annotations,
    jobs,
    sam as sam_svc,
    segments as segments_svc,
    video as video_svc,
)
from services.jobs import Job
from utils.geometry import clamp, poly_to_xyxy


def _expand(box: List[float], margin: float, w: int, h: int) -> List[float]:
    """박스를 상하좌우로 margin 비율만큼 넓혀 프롬프트 여유를 준다(프레임 간 이동 흡수)."""
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = (x2 - x1), (y2 - y1)
    dx, dy = bw * margin, bh * margin
    return [clamp(x1 - dx, 0, w - 1), clamp(y1 - dy, 0, h - 1),
            clamp(x2 + dx, 0, w - 1), clamp(y2 + dy, 0, h - 1)]


def _seed_tracks(seed: dict) -> List[dict]:
    """씨앗 프레임의 객체 → 전파용 트랙 상태. track_id 가 없으면 새로 부여한다."""
    objs = seed.get("objects") or []
    used = [int(o["track_id"]) for o in objs if o.get("track_id") is not None]
    nxt = (max(used) + 1) if used else 1
    out: List[dict] = []
    for i, o in enumerate(objs):
        box = o.get("bbox") or (poly_to_xyxy(o.get("poly")) if o.get("poly") else None)
        if not box:
            continue
        tid = o.get("track_id")
        if tid is None:
            tid = nxt
            nxt += 1
        out.append({
            "oid": str(o.get("id") or f"o{i + 1}"),
            "box": [float(v) for v in box],
            "class_name": o.get("class_name"),
            "class_id": int(o.get("class_id", -1)),
            "model_class_name": o.get("model_class_name") or o.get("class_name"),
            "track_id": int(tid),
            "miss": 0,
            "active": True,
        })
    return out


def _segment_end(video_id: str, seed_time: float) -> float:
    """씨앗 시각이 속한 비정상 구간의 끝(sec). 없으면 영상 끝까지."""
    segs = segments_svc.get(video_id).get("segments") or []
    for s in segs:
        if s.get("kind") != "abnormal":
            continue
        a, b = float(s["start_sec"]), float(s["end_sec"])
        if a - 1e-6 <= seed_time <= b + 1e-6:
            return b
    meta = video_svc.get_meta(video_id)
    dur = float(meta.get("duration_sec") or 0.0)
    return dur if dur > 0 else (seed_time + 1e9)


def _target_frames(video_id: str, seed_index: int, end_sec: float, max_frames: int
                   ) -> List[Tuple[str, int, float]]:
    """씨앗 다음부터 end_sec 까지의 **이미 추출된** 프레임을 프레임 인덱스 순으로."""
    out: List[Tuple[str, int, float]] = []
    for fid in store.list_frame_ids(video_id):
        doc = store.read_json(annotations.label_path(video_id, fid), None)
        if not isinstance(doc, dict):
            continue
        fidx = int(doc.get("frame_index") or 0)
        ftime = float(doc.get("time_sec") or 0.0)
        if fidx <= seed_index or ftime > end_sec + 1e-6:
            continue
        out.append((fid, fidx, ftime))
    out.sort(key=lambda t: t[1])
    return out[: max(0, int(max_frames))]


def segment_point(video_id: str, frame_id: str, x: float, y: float,
                  *, min_area: float = 16.0, shrink: float = 0.45,
                  shadow: bool = True, shadow_ratio: float = 0.6,
                  clamp_box=None, neg_points=None) -> dict:
    """저장된 프레임 이미지에서 한 점을 프롬프트로 누끼를 따 OBB 를 돌려준다(대화형).

    '3·라벨링'에서 Q(누끼 모드) 상태로 물체를 클릭하면 이 함수가 호출된다. 학습에 들어갈
    바로 그 저장 이미지(전처리 완료본)에서 세그멘트하므로 라벨과 이미지가 어긋나지 않는다.
    """
    annotations.load(video_id, frame_id)  # 존재 검증(없으면 KeyError)
    img_path = annotations.image_path(video_id, frame_id)
    img = cv2.imread(str(img_path)) if img_path.exists() else None
    if img is None:
        raise KeyError(f"프레임 이미지가 없습니다: {video_id}/{frame_id}")
    h, w = img.shape[:2]
    xx = clamp(float(x), 0, w - 1)
    yy = clamp(float(y), 0, h - 1)
    poly = sam_svc.segment_point(img, xx, yy, min_area=float(min_area), shrink=float(shrink),
                                 shadow=bool(shadow), shadow_ratio=float(shadow_ratio),
                                 clamp_box=clamp_box, neg_points=neg_points)
    if poly is None:
        return {"found": False, "poly": None, "bbox": None, "width": w, "height": h}
    poly = [[round(clamp(float(px), 0, w - 1), 2), round(clamp(float(py), 0, h - 1), 2)]
            for px, py in poly]
    return {"found": True, "poly": poly, "bbox": [round(v, 2) for v in poly_to_xyxy(poly)],
            "width": w, "height": h}


def segment_box(video_id: str, frame_id: str, box: List[float],
                *, min_area: float = 16.0, shrink: float = 0.45,
                shadow: bool = True, shadow_ratio: float = 0.6,
                margin: float = 0.2) -> dict:
    """저장된 프레임 이미지에서 **박스 프롬프트**로 누끼를 따 OBB 를 돌려준다(전파용).

    W(이전 중심 누끼) 전파에서, 이전 프레임 박스를 `margin`만큼 넓혀 이 프레임 물체를 락온한다.
    점 프롬프트보다 프레임 간 이동에 강하다(중심점이 물체를 벗어나도 박스가 새 위치를 덮음).
    """
    annotations.load(video_id, frame_id)  # 존재 검증(없으면 KeyError)
    img_path = annotations.image_path(video_id, frame_id)
    img = cv2.imread(str(img_path)) if img_path.exists() else None
    if img is None:
        raise KeyError(f"프레임 이미지가 없습니다: {video_id}/{frame_id}")
    h, w = img.shape[:2]
    if not box or len(box) < 4:
        return {"found": False, "poly": None, "bbox": None, "width": w, "height": h}
    prompt = _expand([float(v) for v in box[:4]], float(margin), w, h)
    poly = sam_svc.segment_box(img, prompt, min_area=float(min_area), shrink=float(shrink),
                               shadow=bool(shadow), shadow_ratio=float(shadow_ratio))
    if poly is None:
        return {"found": False, "poly": None, "bbox": None, "width": w, "height": h}
    poly = [[round(clamp(float(px), 0, w - 1), 2), round(clamp(float(py), 0, h - 1), 2)]
            for px, py in poly]
    return {"found": True, "poly": poly, "bbox": [round(v, 2) for v in poly_to_xyxy(poly)],
            "width": w, "height": h}


def start(video_id: str, params: Optional[dict] = None) -> Job:
    """씨앗 프레임 박스로 SAM 전파 잡을 시작한다."""
    params = dict(params or {})
    seed_id = str(params.get("seed_frame_id") or "").strip()
    if not seed_id:
        raise ValueError("seed_frame_id 가 필요합니다")

    seed = annotations.load(video_id, seed_id)  # 없으면 KeyError
    if not (seed.get("objects") or []):
        raise ValueError("씨앗 프레임에 박스가 없습니다. 먼저 첫 프레임에 박스를 잡아 저장하세요.")
    if not sam_svc.available():
        raise ValueError("SAM 을 쓸 수 없습니다(ultralytics 에 SAM 미포함). 설치를 확인하세요.")

    if jobs.active_for("propagate", video_id):
        raise ValueError("이 영상에 이미 실행 중인 전파 작업이 있습니다.")

    s = get_settings()
    seed_time = float(seed.get("time_sec") or 0.0)
    spec = {
        "seed_frame_id": seed_id,
        "margin": float(params.get("margin", 0.25)),
        "min_area": float(params.get("min_area", 16.0)),
        "shrink": float(params.get("shrink", 0.45)),
        "shadow": bool(params.get("shadow", True)),
        "shadow_ratio": float(params.get("shadow_ratio", 0.6)),
        "clamp_margin": float(params.get("clamp_margin", 0.4)),
        "max_miss": int(params.get("max_miss", 3)),
        "end_sec": round(_segment_end(video_id, seed_time), 3),
        "max_frames": int(params.get("max_frames") or s.extract_max_frames),
        "sam_model": s.sam_model,
    }
    return jobs.submit("propagate", lambda job: _run(job, video_id, spec),
                       target=video_id, params=spec)


def _run(job: Job, video_id: str, spec: dict) -> None:
    seed_id = spec["seed_frame_id"]
    seed = annotations.load(video_id, seed_id)
    seed_index = int(seed.get("frame_index") or 0)
    end_sec = float(spec["end_sec"])

    targets = _target_frames(video_id, seed_index, end_sec, spec["max_frames"])
    tracks = _seed_tracks(seed)
    job.set_total(len(targets))
    if not targets:
        job.result = {"video_id": video_id, "seed": seed_id, "frames": 0, "objects": 0,
                      "dropped": 0, "tracks": len(tracks), "end_sec": end_sec}
        job.message = "전파할 다음 프레임이 없습니다(구간 끝이거나 추출된 프레임이 없음)."
        return
    if not tracks:
        job.result = {"video_id": video_id, "seed": seed_id, "frames": 0, "objects": 0,
                      "dropped": 0, "tracks": 0, "end_sec": end_sec}
        job.message = "씨앗 프레임에 유효한 박스가 없습니다."
        return

    job.message = f"{len(targets)}개 프레임에 {len(tracks)}개 객체 전파 (SAM 로드 중…)"
    # SAM 을 미리 한 번 로드해 첫 프레임 지연/오류를 여기서 소화한다.
    sam_svc._load()

    written = 0
    total_objs = 0
    dropped = 0

    for fid, fidx, ftime in targets:
        job.check_canceled()
        img_path = annotations.image_path(video_id, fid)
        img = cv2.imread(str(img_path)) if img_path.exists() else None
        if img is None:
            job.advance(1)
            continue
        h, w = img.shape[:2]

        active = [t for t in tracks if t["active"]]
        if not active:
            # 모든 트랙이 소실됐으면 더 그릴 게 없다. 남은 프레임은 빠르게 넘긴다.
            job.advance(1)
            continue

        shrink = float(spec.get("shrink", 0.45))
        shadow = bool(spec.get("shadow", True))
        shadow_ratio = float(spec.get("shadow_ratio", 0.6))

        objs: List[dict] = []
        for t in active:
            # 이전 박스 **중심점**에 Q 를 찍는 것과 동일(점 프롬프트). clamp 은 걸지 않는다 — 이전 박스
            # 위치로 제한하면 물체가 움직였을 때 잘려서 놓치기 때문. 그림자는 색/궤적/E(네거티브)로 처리.
            bx = t["box"]
            cx = clamp((float(bx[0]) + float(bx[2])) / 2.0, 0, w - 1)
            cy = clamp((float(bx[1]) + float(bx[3])) / 2.0, 0, h - 1)
            poly = sam_svc.segment_point(img, cx, cy, min_area=spec["min_area"], shrink=shrink,
                                         shadow=shadow, shadow_ratio=shadow_ratio)
            if poly is None:
                t["miss"] += 1
                if t["miss"] > spec["max_miss"]:
                    t["active"] = False
                    dropped += 1
                continue
            poly = [[clamp(float(x), 0, w - 1), clamp(float(y), 0, h - 1)] for x, y in poly]
            t["box"] = poly_to_xyxy(poly)
            t["miss"] = 0
            objs.append({
                "id": t["oid"],
                "class_name": t["class_name"],
                "class_id": int(t["class_id"]),
                "model_class_name": t["model_class_name"],
                "score": None,
                "poly": [[round(x, 2), round(y, 2)] for x, y in poly],
                "bbox": [round(v, 2) for v in t["box"]],
                "track_id": t["track_id"],
                "is_predicted": False,
                "track_status": "sam",
                "source": "sam",
                "verified": False,
            })

        doc = store.read_json(annotations.label_path(video_id, fid), None)
        if not isinstance(doc, dict):
            doc = annotations.new_doc(video_id, fidx, time_sec=ftime, width=w, height=h,
                                      objects=[], segment_kind="abnormal", model="SAM")
        # SAM 전파 결과로 이 프레임의 초안을 교체한다(사람 검수 대기). 씨앗 프레임은 건드리지 않는다.
        doc["objects"] = objs
        doc["status"] = "pending"
        doc["source"] = "auto"          # 아직 사람이 안 본 초안
        doc["sam_propagated"] = True
        annotations.save(video_id, fid, doc)

        written += 1
        total_objs += len(objs)
        job.advance(1, f"{written}/{len(targets)} 프레임 · 객체 {total_objs}개 · 소실 {dropped}")

    job.result = {
        "video_id": video_id,
        "seed": seed_id,
        "frames": written,
        "objects": total_objs,
        "dropped": dropped,
        "tracks": len(tracks),
        "end_sec": end_sec,
        "sam_model": spec["sam_model"],
        "progress": annotations.progress(video_id),
    }
    job.message = (f"완료: {written}개 프레임 라벨(객체 {total_objs}개, 트랙 소실 {dropped}개). "
                   f"3·라벨링에서 검수하세요.")

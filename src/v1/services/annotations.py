"""
services/annotations.py
=======================

프레임 단위 라벨(자동 초안 + 사람 수정)의 저장/조회/검수.

라벨 1장 = `workspace/videos/<video_id>/labels/<frame_id>.json` 하나.
프레임 이미지 1장 = `.../frames/<frame_id>.jpg`.

라벨 문서::

    {
      "video_id": "...", "frame_id": "f000123",
      "frame_index": 123, "time_sec": 4.1,
      "width": 640, "height": 480,
      "segment_kind": "normal",          # 어느 구간에서 뽑혔는지
      "status": "pending",               # pending | approved | rejected
      "source": "auto",                  # auto | manual (사람이 손대면 manual)
      "model": "yolo26l.pt",             # 초안을 만든 가중치
      "objects": [
        {"id": "o1",
         "class_name": "VIDAR",          # ★ 라벨의 진짜 값(프로젝트 클래스 목록의 이름)
         "class_id": 6,                  # 목록 인덱스. -1 = 아직 클래스 미확정
         "model_class_name": "truck",    # 모델이 준 힌트(참고용)
         "score": 0.87,                  # 초안 신뢰도(사람이 만든 건 null)
         "poly": [[x, y] * 4],           # 픽셀 4점(회전 박스). 축정렬도 4점으로 저장
         "bbox": [x1, y1, x2, y2],
         "track_id": 3,                  # 같은 객체를 프레임 간 연결하는 id
         "source": "auto", "verified": false}
      ],
      "updated_at": "..."
    }

검수 규칙: `approved` 로 올리려면 모든 객체의 클래스가 확정돼야 한다(`class_id >= 0`).
"라벨링은 모든 부분을 해야 한다"는 요구를 데이터셋 입구가 아니라 **승인 시점에**
막아, 반쯤 라벨된 프레임이 학습에 섞이지 않게 한다.
객체가 0개인 프레임은 배경(negative) 샘플로 유효하므로 그대로 승인할 수 있다.
"""

from __future__ import annotations

import colorsys
from pathlib import Path
from typing import List, Optional

from core import store
from services import classes as classes_svc
from utils.geometry import poly_area, poly_to_xyxy, sanitize_poly

STATUSES = ("pending", "approved", "rejected")

# 이보다 작은(픽셀²) 박스는 학습에 도움이 안 되고 라벨 노이즈만 된다.
_MIN_OBJ_AREA = 4.0


def frame_id_for(frame_index: int) -> str:
    return f"f{int(frame_index):06d}"


def label_path(video_id: str, frame_id: str) -> Path:
    return store.frame_label_path(video_id, frame_id)


def image_path(video_id: str, frame_id: str) -> Path:
    return store.frame_image_path(video_id, frame_id)


def load(video_id: str, frame_id: str) -> dict:
    doc = store.read_json(label_path(video_id, frame_id), None)
    if not isinstance(doc, dict):
        raise KeyError(f"라벨을 찾을 수 없습니다: {video_id}/{frame_id}")
    return doc


def save(video_id: str, frame_id: str, doc: dict) -> dict:
    doc["updated_at"] = store.now_iso()
    store.write_json(label_path(video_id, frame_id), doc)
    return doc


def exists(video_id: str, frame_id: str) -> bool:
    return label_path(video_id, frame_id).exists()


def build_objects(raw_detections, width: int, height: int, *, source: str = "auto") -> List[dict]:
    """추론 결과(또는 프런트 입력) → 라벨 객체 리스트로 정규화.

    모델이 준 클래스명이 프로젝트 클래스 목록에 그대로 있으면 확정 라벨로,
    없으면 `class_id=-1`(사람이 골라야 하는 상태)로 둔다.
    """
    out: List[dict] = []
    for i, det in enumerate(raw_detections or []):
        poly = sanitize_poly(det.get("poly"), width, height)
        if poly_area(poly) < _MIN_OBJ_AREA:
            continue
        model_name = det.get("model_class_name") or det.get("class_name")
        # 프런트가 명시적으로 고른 클래스가 있으면 그것을 우선한다.
        chosen = det.get("class_name") if source == "manual" else None
        if chosen is None:
            chosen = model_name if classes_svc.index_of(model_name) >= 0 else None
        cid = classes_svc.index_of(chosen)
        out.append({
            "id": str(det.get("id") or f"o{i + 1}"),
            "class_name": chosen,
            "class_id": cid,
            "model_class_name": model_name,
            "score": det.get("score"),
            "poly": [[round(float(x), 2), round(float(y), 2)] for x, y in poly],
            "bbox": [round(v, 2) for v in poly_to_xyxy(poly)],
            "track_id": det.get("track_id"),
            "source": str(det.get("source") or source),
            "verified": bool(det.get("verified", False)),
        })
    return out


def new_doc(video_id: str, frame_index: int, *, time_sec: float, width: int, height: int,
            objects: List[dict], segment_kind: str = "normal",
            model: Optional[str] = None, daynight: Optional[str] = None,
            dehaze: Optional[bool] = None, clahe: Optional[bool] = None,
            preprocess: Optional[dict] = None) -> dict:
    doc = {
        "video_id": video_id,
        "frame_id": frame_id_for(frame_index),
        "frame_index": int(frame_index),
        "time_sec": round(float(time_sec), 3),
        "width": int(width),
        "height": int(height),
        "segment_kind": segment_kind,
        "status": "pending",
        "source": "auto",
        "model": model,
        # 이 프레임을 무엇으로 전처리했는지 남긴다(야간 보정 day/night/off, 안개 제거 여부,
        # 적용 설정). 학습 데이터가 어떻게 만들어졌는지 추적하고 비중을 집계하기 위함.
        "daynight": daynight,
        "dehaze": dehaze,
        "clahe": clahe,
        "preprocess": preprocess,
        "objects": objects,
        "updated_at": store.now_iso(),
    }
    return doc


def unresolved(doc: dict) -> List[str]:
    """클래스가 아직 정해지지 않은 객체 id 목록."""
    return [str(o.get("id")) for o in (doc.get("objects") or [])
            if int(o.get("class_id", -1)) < 0]


def update(video_id: str, frame_id: str, *, objects=None, status: Optional[str] = None,
           note: Optional[str] = None, force: bool = False) -> dict:
    """사람이 수정한 라벨을 저장한다. `objects` 를 보내면 전체 교체."""
    doc = load(video_id, frame_id)
    if objects is not None:
        doc["objects"] = build_objects(objects, doc["width"], doc["height"], source="manual")
        doc["source"] = "manual"
    if note is not None:
        doc["note"] = str(note)
    if status is not None:
        _apply_status(doc, status, force=force)
    return save(video_id, frame_id, doc)


def _apply_status(doc: dict, status: str, *, force: bool = False) -> None:
    st = str(status).strip().lower()
    if st not in STATUSES:
        raise ValueError(f"status 는 {list(STATUSES)} 중 하나여야 합니다: {status!r}")
    if st == "approved" and not force:
        missing = unresolved(doc)
        if missing:
            raise ValueError(
                f"클래스가 정해지지 않은 객체가 있어 승인할 수 없습니다: {missing}. "
                "모든 객체에 클래스를 지정하거나 해당 객체를 삭제하세요."
            )
    doc["status"] = st
    if st == "approved":
        for o in doc.get("objects") or []:
            o["verified"] = True


def set_status(video_id: str, frame_id: str, status: str, *, force: bool = False) -> dict:
    doc = load(video_id, frame_id)
    _apply_status(doc, status, force=force)
    return save(video_id, frame_id, doc)


def bulk_status(video_id: str, status: str, *, frame_ids=None, force: bool = False) -> dict:
    """여러 프레임의 상태를 한 번에 바꾼다(전체 승인 버튼용).

    실패한 프레임은 건너뛰고 이유를 모아 돌려준다. 하나 때문에 전체가 실패하면
    수천 장 검수 흐름이 막힌다.
    """
    targets = list(frame_ids) if frame_ids else store.list_frame_ids(video_id)
    ok, failed = [], []
    for fid in targets:
        try:
            set_status(video_id, fid, status, force=force)
            ok.append(fid)
        except (KeyError, ValueError) as e:
            failed.append({"frame_id": fid, "error": str(e)})
    return {"updated": len(ok), "failed": failed, "status": status}


def propagate_class(video_id: str, *, track_id: int, class_name: str,
                    frame_ids=None) -> dict:
    """같은 `track_id` 를 가진 객체의 클래스를 모든 프레임에 일괄 적용한다.

    자동 라벨이 붙여둔 track_id 덕분에, 사용자는 객체 하나당 한 번만 클래스를
    고르면 그 객체가 등장하는 수백 프레임이 함께 채워진다. 라벨링 공수를
    프레임 수가 아니라 객체 수에 비례하게 만드는 핵심 기능이다.
    """
    cid = classes_svc.index_of(class_name)
    if cid < 0:
        raise ValueError(f"클래스 목록에 없는 이름입니다: {class_name!r}")
    targets = list(frame_ids) if frame_ids else store.list_frame_ids(video_id)
    changed_frames = 0
    changed_objects = 0
    for fid in targets:
        try:
            doc = load(video_id, fid)
        except KeyError:
            continue
        hit = False
        for o in doc.get("objects") or []:
            if o.get("track_id") is not None and int(o["track_id"]) == int(track_id):
                o["class_name"] = class_name
                o["class_id"] = cid
                changed_objects += 1
                hit = True
        if hit:
            changed_frames += 1
            save(video_id, fid, doc)
    return {"track_id": int(track_id), "class_name": class_name,
            "frames": changed_frames, "objects": changed_objects}


def tracks(video_id: str) -> List[dict]:
    """영상 안의 track_id 요약. 프런트의 "객체별 라벨링" 패널용."""
    acc: dict = {}
    for fid in store.list_frame_ids(video_id):
        doc = store.read_json(label_path(video_id, fid), None)
        if not isinstance(doc, dict):
            continue
        for o in doc.get("objects") or []:
            tid = o.get("track_id")
            if tid is None:
                continue
            key = int(tid)
            entry = acc.setdefault(key, {
                "track_id": key, "frames": 0, "first_frame": fid, "last_frame": fid,
                "class_name": o.get("class_name"), "model_class_name": o.get("model_class_name"),
                "resolved": False, "avg_score": 0.0, "_score_sum": 0.0,
            })
            entry["frames"] += 1
            entry["last_frame"] = fid
            if o.get("class_name"):
                entry["class_name"] = o.get("class_name")
            entry["_score_sum"] += float(o.get("score") or 0.0)
    out = []
    for entry in acc.values():
        entry["avg_score"] = round(entry.pop("_score_sum") / max(1, entry["frames"]), 3)
        entry["resolved"] = classes_svc.index_of(entry.get("class_name")) >= 0
        out.append(entry)
    out.sort(key=lambda e: (-e["frames"], e["track_id"]))
    return out


def summarize(doc: dict) -> dict:
    """목록 응답용 경량 요약(폴리곤 좌표 제외)."""
    objs = doc.get("objects") or []
    return {
        "frame_id": doc.get("frame_id"),
        "frame_index": doc.get("frame_index"),
        "time_sec": doc.get("time_sec"),
        "status": doc.get("status"),
        "source": doc.get("source"),
        "segment_kind": doc.get("segment_kind"),
        "daynight": doc.get("daynight"),
        "dehaze": doc.get("dehaze"),
        "clahe": doc.get("clahe"),
        "n_objects": len(objs),
        "n_unresolved": len(unresolved(doc)),
        "class_names": sorted({o["class_name"] for o in objs if o.get("class_name")}),
        "track_ids": sorted({int(o["track_id"]) for o in objs if o.get("track_id") is not None}),
    }


def list_frames(video_id: str, *, status: Optional[str] = None, offset: int = 0,
                limit: int = 100) -> dict:
    fids = store.list_frame_ids(video_id)
    items: List[dict] = []
    for fid in fids:
        doc = store.read_json(label_path(video_id, fid), None)
        if not isinstance(doc, dict):
            continue
        if status and doc.get("status") != status:
            continue
        items.append(summarize(doc))
    total = len(items)
    start = max(0, int(offset))
    end = start + max(1, int(limit))
    return {"video_id": video_id, "total": total, "offset": start,
            "limit": int(limit), "items": items[start:end]}


def progress(video_id: str) -> dict:
    """라벨링 진행률. 프런트 진행바 + 데이터셋 빌드 가능 여부 판단용."""
    counts = {s: 0 for s in STATUSES}
    total = 0
    objects = 0
    unresolved_objects = 0
    empty = 0
    for fid in store.list_frame_ids(video_id):
        doc = store.read_json(label_path(video_id, fid), None)
        if not isinstance(doc, dict):
            continue
        total += 1
        counts[doc.get("status", "pending")] = counts.get(doc.get("status", "pending"), 0) + 1
        objs = doc.get("objects") or []
        objects += len(objs)
        unresolved_objects += len(unresolved(doc))
        if not objs:
            empty += 1
    return {
        "frames": total,
        **counts,
        "empty_frames": empty,
        "objects": objects,
        "unresolved_objects": unresolved_objects,
        "ready": total > 0 and counts["approved"] > 0,
        "approved_ratio": round(counts["approved"] / total, 4) if total else 0.0,
    }


def class_stats(video_id: Optional[str] = None) -> dict:
    """클래스별 객체 수. 데이터 불균형을 학습 전에 눈으로 확인하기 위한 지표."""
    vids = [video_id] if video_id else store.list_video_ids()
    acc: dict = {}
    for vid in vids:
        for fid in store.list_frame_ids(vid):
            doc = store.read_json(label_path(vid, fid), None)
            if not isinstance(doc, dict) or doc.get("status") == "rejected":
                continue
            for o in doc.get("objects") or []:
                key = o.get("class_name") or "(미확정)"
                acc[key] = acc.get(key, 0) + 1
    return dict(sorted(acc.items(), key=lambda kv: -kv[1]))


def delete_frame(video_id: str, frame_id: str) -> dict:
    """프레임과 라벨을 완전히 제거한다. (검수에서 '이 장은 못 쓴다' 처리)

    보통은 `status=rejected` 로 남겨 이력을 보존하는 편이 낫다. 이 함수는 잘못
    추출된 프레임을 정리할 때만 쓴다.
    """
    label_path(video_id, frame_id).unlink(missing_ok=True)
    image_path(video_id, frame_id).unlink(missing_ok=True)
    return {"video_id": video_id, "frame_id": frame_id, "deleted": True}


# ── 오버레이(검수 화면용) ──────────────────────────────────────────────
_GOLDEN_RATIO_CONJUGATE = 0.61803398875


def color_for(index: int) -> tuple:
    """클래스 인덱스 → BGR. 인접 인덱스끼리 색상이 최대한 멀어지게 황금각 분할."""
    h = ((int(index) + 1) * _GOLDEN_RATIO_CONJUGATE) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.98)
    return (int(b * 255), int(g * 255), int(r * 255))


def draw_overlay(frame, doc: dict):
    """라벨을 프레임에 그린다(회전 박스 폴리곤 + 클래스/트랙 표시)."""
    import cv2

    vis = frame.copy()
    for o in doc.get("objects") or []:
        cid = int(o.get("class_id", -1))
        # 미확정 객체는 회색 점선 느낌으로 눈에 띄게 구분한다(검수 대상).
        color = (160, 160, 160) if cid < 0 else color_for(cid)
        pts = [[int(round(x)), int(round(y))] for x, y in (o.get("poly") or [])]
        if len(pts) >= 3:
            import numpy as np
            cv2.polylines(vis, [np.array(pts, dtype=np.int32).reshape(-1, 1, 2)],
                          True, color, 2)
        label = o.get("class_name") or "?"
        if o.get("track_id") is not None:
            label = f"#{o['track_id']} {label}"
        if o.get("score") is not None:
            label = f"{label} {float(o['score']):.2f}"
        x1, y1 = (pts[0] if pts else [0, 0])
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = y1 - 4 if y1 - 4 - th >= 0 else y1 + th + 4
        cv2.rectangle(vis, (x1, ty - th - 2), (x1 + tw + 2, ty + bl + 2), color, -1)
        cv2.putText(vis, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def overlay_jpeg(video_id: str, frame_id: str) -> bytes:
    import cv2

    from services import video as video_svc

    p = image_path(video_id, frame_id)
    if not p.exists():
        raise KeyError(f"프레임 이미지가 없습니다: {video_id}/{frame_id}")
    frame = cv2.imread(str(p))
    if frame is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {p.name}")
    return video_svc.encode_jpeg(draw_overlay(frame, load(video_id, frame_id)))

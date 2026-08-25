"""
services/autolabel.py
=====================

정상 구간 → 프레임 추출 → YOLO 자동 라벨 초안 생성 (백그라운드 잡).

파이프라인 한 단계::

    segments(정상 - 비정상) → 샘플링(초당 N장) → 640x480 정규화 → JPEG 저장
                            → YOLO 추론(+트래킹) → labels/<frame_id>.json

설계 근거
- **샘플링**: 30fps 영상을 전부 뽑으면 1분에 1800장이고, 인접 프레임은 거의 같은
  그림이라 학습 가치가 낮으면서 검수 부담만 커진다. 기본 초당 2장.
- **트래킹**: `track_id` 를 붙여두면 사용자가 객체당 한 번만 클래스를 고르고
  전체 프레임에 전파할 수 있다(`annotations.propagate_class`).
- **사람 작업 보호**: 다시 돌려도 사람이 수정/승인한 프레임은 기본적으로 건드리지
  않는다. 자동 초안을 새 모델로 갱신하려면 `overwrite="auto"` 를 쓴다.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import cv2

from core.config import get_settings
from core import store
from services import (
    annotations,
    jobs,
    preprocess as pp,
    segments as segments_svc,
    video as video_svc,
)
from services.detector import get_detector, resolve_model_path
from services.jobs import Job

OVERWRITE_MODES = ("skip", "auto", "all")


def _plan(fps: float, ranges: Sequence[Tuple[float, float, str]], stride: int,
          max_frames: int) -> List[Tuple[int, int, str]]:
    """뽑을 프레임 인덱스 계획. (frame_idx, frame_idx, kind) 목록을 만든다.

    `ranges` 는 (start_sec, end_sec, kind) 로, 프레임마다 정상/비정상을 그대로 물려
    저장한다(정상은 자동 승인, 비정상은 검수 대기). 미리 목록을 만드는 이유: 진행률
    (total)을 정확히 보여줄 수 있고, 상한(max_frames)에 걸릴 때 어디까지 뽑았는지가
    결정적(deterministic)이 된다.

    같은 프레임 인덱스가 정상·비정상 양쪽에 걸리면 비정상을 남긴다(사람이 봐야 하므로).
    """
    picked: dict = {}
    for start_sec, end_sec, kind in ranges:
        start_idx = int(math.floor(start_sec * fps))
        end_idx = int(math.ceil(end_sec * fps))
        # 전 구간 공통 그리드(stride 배수)에 스냅한다. 구간 경계가 바뀌어도 프레임 인덱스가
        # 고정돼, 재추출(overwrite=auto) 시 같은 frame_id 를 갱신한다 → 중복·고아 프레임 방지.
        first = ((start_idx + stride - 1) // stride) * stride
        for idx in range(first, end_idx + 1, stride):
            prev = picked.get(idx)
            if prev == "abnormal":
                continue                       # 비정상이 이긴다 — 유지
            if prev is None:
                if len(picked) >= max_frames:
                    break                      # 상한 도달 → 새 프레임은 그만
                picked[idx] = kind
            elif kind == "abnormal":
                picked[idx] = "abnormal"       # 정상 → 비정상 승격(개수 증가 없음)
    # 오름차순 인덱스로 반환 → 트래커가 순차 처리해 track_id 가 이어진다.
    return [(idx, idx, picked[idx]) for idx in sorted(picked)]


def start(video_id: str, params: Optional[dict] = None) -> Job:
    """프레임 추출 + 자동 라벨 잡을 시작한다."""
    params = dict(params or {})
    s = get_settings()

    # 잡을 띄우기 전에 검증해 사용자가 즉시 에러를 받게 한다(잡 안에서 죽으면 늦다).
    video_svc.get_meta(video_id)
    kinds = params.get("kinds") or ["normal"]
    segments_svc.selection_plan(video_id, kinds)
    model_path = resolve_model_path(params.get("model"))
    if not model_path.exists():
        raise ValueError(f"모델 파일을 찾을 수 없습니다: {params.get('model') or model_path}")
    mode = str(params.get("overwrite") or "skip").lower()
    if mode not in OVERWRITE_MODES:
        raise ValueError(f"overwrite 는 {list(OVERWRITE_MODES)} 중 하나여야 합니다: {mode!r}")

    running = jobs.active_for("extract", video_id)
    if running:
        raise ValueError(f"이 영상에 이미 실행 중인 작업이 있습니다: job {running['id']}")

    # 전처리 설정 우선순위: 추출 요청 > 영상별 저장값('구간' 단계 지정) > 서버 기본값.
    # 요청에 들어온 전처리 키만 골라 영상별 저장값 위에 덮는다.
    saved = video_svc.get_preprocess(video_id)
    override = {k: params[k] for k in (
        "auto", "lowlight", "dehaze", "clahe", "emphasis", "lowlight_threshold",
        "night_clahe_clip", "night_clahe_grid", "night_gamma",
        "dehaze_omega", "dehaze_t0", "dehaze_wsz", "dehaze_scale", "dehaze_guide_r",
        "clahe_clip", "clahe_grid", "emphasis_sigma", "emphasis_alpha",
        "resize", "resize_width", "resize_height",
    ) if params.get(k) is not None}
    preprocess = pp.resolve({**saved, **override})

    spec = {
        "kinds": list(kinds),
        "fps": float(params.get("fps") if params.get("fps") is not None else s.extract_fps),
        "conf": float(params.get("conf") if params.get("conf") is not None else s.autolabel_conf),
        "iou": float(params.get("iou") if params.get("iou") is not None else s.autolabel_iou),
        "imgsz": int(params.get("imgsz") or s.autolabel_imgsz),
        "max_det": int(params.get("max_det") or s.autolabel_max_det),
        "max_frames": int(params.get("max_frames") or s.extract_max_frames),
        "track": bool(params.get("track", s.autolabel_track)),
        "model": str(model_path),
        "overwrite": mode,
        "preprocess": preprocess,
    }
    return jobs.submit("extract", lambda job: _run(job, video_id, spec),
                       target=video_id, params=spec)


def _should_write(video_id: str, frame_id: str, mode: str) -> bool:
    """사람 작업을 지우지 않기 위한 재실행 정책."""
    if not annotations.exists(video_id, frame_id):
        return True
    if mode == "all":
        return True
    if mode == "auto":
        doc = store.read_json(annotations.label_path(video_id, frame_id), {}) or {}
        # 자동 초안이고 아직 검수 전인 것만 갱신한다.
        return doc.get("source") == "auto" and doc.get("status") == "pending"
    return False


def _run(job: Job, video_id: str, spec: dict) -> None:
    meta = video_svc.get_meta(video_id)
    fps = float(meta.get("fps") or 30.0)
    ranges = segments_svc.selection_plan(video_id, spec["kinds"])
    if not ranges:
        raise ValueError("선택된 구간이 없습니다. 정상 구간을 지정하거나 비정상 구간을 줄이세요.")

    stride = max(1, int(round(fps / spec["fps"]))) if spec["fps"] > 0 else 1
    plan = _plan(fps, ranges, stride, spec["max_frames"])
    if not plan:
        raise ValueError("추출할 프레임이 없습니다(구간이 너무 짧습니다).")
    n_abnormal_ranges = sum(1 for _, _, k in ranges if k == "abnormal")
    # 구간이 바뀌어 어긋난 이전 자동·검수전 프레임(고아/중복)을 먼저 정리한다.
    # 사람이 손댄(manual/approved/rejected) 프레임은 건드리지 않는다.
    pruned = 0
    if spec["overwrite"] in ("auto", "all"):
        planned_ids = {annotations.frame_id_for(idx) for idx, _, _ in plan}
        pruned = annotations.prune_auto_pending(video_id, planned_ids)
    job.set_total(len(plan))
    job.message = (f"{len(plan)}장 추출 예정 (stride={stride}, 구간 {len(ranges)}개"
                   f", 비정상 {n_abnormal_ranges}개, 고아 정리 {pruned}장)")

    detector = get_detector(spec["model"])
    # 영상이 바뀌면 트랙 상태를 비운다. 안 비우면 이전 영상의 track_id 가 이어져
    # 클래스 전파가 엉뚱한 영상까지 번진다.
    detector.reset_tracker()

    path = video_svc.source_path(video_id)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"영상을 열 수 없습니다: {path.name}")

    frames_dir = store.frames_dir(video_id)
    frames_dir.mkdir(parents=True, exist_ok=True)
    store.labels_dir(video_id).mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    auto_approved = 0
    n_objects = 0
    daynight_counts = {"day": 0, "night": 0, "off": 0}
    dehazed = 0
    clahed = 0
    model_name = resolve_model_path(spec["model"]).name
    prep = spec.get("preprocess") or pp.resolve(None)
    quality = [int(cv2.IMWRITE_JPEG_QUALITY), get_settings().frame_jpeg_quality]

    try:
        for target_idx, _, kind in plan:
            job.check_canceled()
            frame_id = annotations.frame_id_for(target_idx)

            if not _should_write(video_id, frame_id, spec["overwrite"]):
                skipped += 1
                job.advance(1)
                continue

            # 랜덤 seek 은 키프레임 정렬 때문에 정확하지 않을 수 있지만, 학습 프레임은
            # "정확히 그 인덱스"일 필요가 없다(구간 안이면 충분). 순차 디코딩보다
            # 훨씬 빠르므로 seek 을 쓴다.
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                job.advance(1)
                continue

            # 전처리: 안개 제거 → 야간 보정 → (선택적) 다운스케일. 저장 이미지와 추론
            # 입력이 같도록 이 순서로 한 번만 적용한다(라벨과 이미지가 어긋나지 않게).
            frame, info = pp.apply(frame, prep)
            frame = video_svc.fit_frame(frame, prep["resize_width"], prep["resize_height"],
                                        resize=prep["resize"])
            daynight_counts[info["lowlight"]] = daynight_counts.get(info["lowlight"], 0) + 1
            if info["dehaze"]:
                dehazed += 1
            if info.get("clahe"):
                clahed += 1
            h, w = frame.shape[:2]

            dets = detector.infer(
                frame, conf=spec["conf"], iou=spec["iou"], imgsz=spec["imgsz"],
                max_det=spec["max_det"], track=spec["track"],
            )
            objects = annotations.build_objects(dets, w, h, source="auto")

            cv2.imwrite(str(frames_dir / f"{frame_id}.jpg"), frame, quality)
            doc = annotations.new_doc(
                video_id, target_idx,
                time_sec=target_idx / fps if fps > 0 else 0.0,
                width=w, height=h, objects=objects,
                segment_kind=kind, model=model_name,
                daynight=info["lowlight"], dehaze=info["dehaze"],
                clahe=info.get("clahe", False), preprocess=prep,
            )
            # 정책(사용자 지정): 정상 구간은 사람 검수가 필요 없다 → 자동 라벨을 그대로
            # 승인 상태로 저장해 바로 학습에 들어가게 한다. 비정상 구간은 pending 으로 남겨
            # 3·라벨링에서 사람이 수정한다. 단, 정상이라도 클래스 미확정 객체가 있으면
            # (모델 클래스와 어긋난 경우) 승인하지 않고 pending 으로 두어 검수에 노출한다.
            if kind == "normal" and not annotations.unresolved(doc):
                doc["status"] = "approved"
                for o in doc.get("objects") or []:
                    o["verified"] = True
                auto_approved += 1
            annotations.save(video_id, frame_id, doc)

            written += 1
            n_objects += len(objects)
            job.advance(1, f"{written}장 라벨링 (객체 {n_objects}개, 자동승인 {auto_approved}장, 야간 {daynight_counts['night']}장)")
    finally:
        cap.release()

    job.result = {
        "video_id": video_id,
        "frames_written": written,
        "frames_skipped": skipped,
        "frames_pruned": pruned,
        "frames_auto_approved": auto_approved,
        "objects": n_objects,
        "stride": stride,
        "ranges": [[round(a, 3), round(b, 3), k] for a, b, k in ranges],
        "model": model_name,
        "tracked": spec["track"],
        "preprocess": prep,
        "lowlight_engine": pp.lowlight_engine(),
        "daynight_counts": daynight_counts,
        "dehaze_frames": dehazed,
        "clahe_frames": clahed,
        "progress": annotations.progress(video_id),
        "tracks": len(annotations.tracks(video_id)),
    }
    _lowlight = "" if not prep.get("lowlight") else (
        f", 야간보정 {daynight_counts['night']}장(주간 {daynight_counts['day']})")
    _dehaze = f", 안개제거 {dehazed}장" if prep.get("dehaze") else ""
    _clahe = f", CLAHE {clahed}장" if prep.get("clahe") else ""
    _approved = f", 정상 자동승인 {auto_approved}장(비정상은 검수 대기)" if auto_approved else ""
    job.message = f"완료: {written}장 저장, {skipped}장 건너뜀, 객체 {n_objects}개{_approved}{_lowlight}{_dehaze}{_clahe}"

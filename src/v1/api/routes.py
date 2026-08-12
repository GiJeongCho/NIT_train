"""
api/routes.py
=============

FastAPI 라우트. HTTP 입출력과 검증만 하고 로직은 services 로 위임한다(얇게 유지).

MLOps 파이프라인 순서대로 그룹이 나뉘어 있다.

  1. 영상    : 업로드/등록 → 재생/썸네일 (프런트가 구간을 찍을 재료)
  2. 구간    : 정상/비정상 구간 저장 (학습에 쓸 부분 선별)
  3. 라벨링  : 프레임 추출 + 자동 라벨 초안 → 사람 검수/수정/승인
  4. 데이터셋: 승인된 라벨만 모아 YOLO 데이터셋 스냅샷 생성
  5. 학습    : 데이터셋 + 가중치 선택 → 학습 run (진행률/중단/이어서)
  6. 모델    : 학습 결과 승격 · 추론 서비스로 배포

예외는 app.py 의 핸들러가 HTTP 로 변환한다(ValueError→400, KeyError→404 …).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Body, File, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from core.config import get_settings
from core import store
from services import (
    annotations,
    autolabel,
    classes as classes_svc,
    dataset as dataset_svc,
    detector as detector_svc,
    jobs,
    registry,
    segments as segments_svc,
    trainer,
    video as video_svc,
)

router = APIRouter()

_TAG_MISC = "0. 기타"
_TAG_VIDEO = "1. 영상 (Videos)"
_TAG_SEGMENT = "2. 구간 (Segments)"
_TAG_LABEL = "3. 라벨링 (Labeling)"
_TAG_DATASET = "4. 데이터셋 (Datasets)"
_TAG_TRAIN = "5. 학습 (Training)"
_TAG_MODEL = "6. 모델 (Models)"
_TAG_JOB = "7. 작업 (Jobs)"

_CHUNK = 1024 * 1024


def _pp_lowlight_engine() -> str:
    """현재 야간 보정 엔진 이름('zero_dce++' 가중치 있음 | 'gamma_clahe' 폴백)."""
    from services import preprocess as pp_svc
    return pp_svc.lowlight_engine()


# ══════════════════════════════════════════════════════════════════════
# 0. 기타
# ══════════════════════════════════════════════════════════════════════
@router.get("/healthz", tags=[_TAG_MISC], summary="헬스체크")
async def healthz() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@router.get(
    "/api/meta",
    tags=[_TAG_MISC],
    summary="서버 설정/기본값 조회",
    description=(
        "프런트가 첫 로딩에 필요한 값(기본 가중치, 클래스 목록, 추출/학습 기본값, "
        "허용 확장자)을 한 번에 준다. 화면에 기본값을 하드코딩하지 않기 위한 엔드포인트."
    ),
)
async def meta() -> JSONResponse:
    s = get_settings()
    return JSONResponse({
        "version": "1.0.0",
        "workspace": str(s.workspace_dir),
        "device": s.device,
        "default_model": store.rel_to_workspace(s.base_model) or str(s.base_model),
        "default_model_exists": s.base_model.exists(),
        "class_names": classes_svc.names(),
        "segment_kinds": list(segments_svc.KINDS),
        "label_statuses": list(annotations.STATUSES),
        "dataset_tasks": list(dataset_svc.TASKS),
        "default_task": s.dataset_task,
        "split_modes": list(dataset_svc.SPLIT_MODES),
        "defaults": {
            "extract": {"fps": s.extract_fps, "conf": s.autolabel_conf,
                        "iou": s.autolabel_iou, "imgsz": s.autolabel_imgsz,
                        "track": s.autolabel_track, "max_frames": s.extract_max_frames},
            "frame": {"resize": s.frame_resize, "width": s.frame_width,
                      "height": s.frame_height},
            "preprocess": {"auto": s.preprocess_auto,
                           "lowlight": s.preprocess_lowlight,
                           "dehaze": s.preprocess_dehaze,
                           "clahe": s.preprocess_clahe,
                           "lowlight_engine": _pp_lowlight_engine(),
                           "lowlight_threshold": s.daynight_threshold,
                           "night_clahe_clip": s.night_clahe_clip,
                           "night_clahe_grid": s.night_clahe_grid,
                           "night_gamma": s.night_gamma,
                           "dehaze_omega": s.dehaze_omega,
                           "dehaze_t0": s.dehaze_t0,
                           "dehaze_wsz": s.dehaze_wsz,
                           "dehaze_scale": s.dehaze_scale,
                           "dehaze_guide_r": s.dehaze_guide_r,
                           "clahe_clip": s.quality_clahe_clip,
                           "clahe_grid": s.quality_clahe_grid,
                           "resize": s.frame_resize,
                           "resize_width": s.frame_width,
                           "resize_height": s.frame_height},
            "dataset": {"splits": s.splits(), "split_mode": s.split_mode,
                        "chunk_size": s.split_chunk_size,
                        "task": s.dataset_task,
                        "only_approved": s.dataset_only_approved},
            "train": {"epochs": s.train_epochs, "imgsz": s.train_imgsz,
                      "batch": s.train_batch, "workers": s.train_workers,
                      "patience": s.train_patience},
        },
    })


class ClassesRequest(BaseModel):
    class_names: List[str] = Field(..., description="클래스 목록. 기존 순서 유지 + 뒤에 추가만 허용")


@router.get(
    "/api/classes",
    tags=[_TAG_MISC],
    summary="클래스 목록 조회",
    description="라벨의 클래스 공간. **목록의 순서가 곧 학습 클래스 id** 다.",
)
async def get_classes() -> JSONResponse:
    return JSONResponse(classes_svc.get())


@router.put(
    "/api/classes",
    tags=[_TAG_MISC],
    summary="클래스 목록 갱신",
    description=(
        "클래스를 추가하거나 오타를 고친다.\n\n"
        "**순서 변경/삭제는 거부한다.** 이미 저장된 라벨과 데이터셋이 인덱스를 클래스 "
        "id 로 쓰므로, 순서가 바뀌면 과거 데이터의 정답이 조용히 뒤바뀐다."
    ),
)
async def put_classes(req: ClassesRequest) -> JSONResponse:
    return JSONResponse(classes_svc.put(req.class_names))


# ══════════════════════════════════════════════════════════════════════
# 1. 영상
# ══════════════════════════════════════════════════════════════════════
class PathRequest(BaseModel):
    path: str = Field(..., description="서버에 있는 영상 파일 경로(절대 또는 워크스페이스 상대)")


@router.post(
    "/api/videos",
    tags=[_TAG_VIDEO],
    summary="영상 업로드",
    description=(
        "학습 소재 영상을 업로드해 등록한다. 업로드가 끝나면 fps/해상도/길이를 뽑아 "
        "`video_id` 와 함께 돌려준다. 이 id 로 이후 구간 지정·프레임 추출을 한다."
    ),
)
async def upload_video(file: UploadFile = File(...)) -> JSONResponse:
    video_svc.check_ext(file.filename or "")
    # 임시파일을 uploads 안에 만든다. 같은 볼륨이라 등록이 rename 한 번으로 끝나고,
    # 수 GB 영상을 임시 디렉터리에서 다시 복사하지 않는다.
    upload_dir = store.uploads_root()
    upload_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(upload_dir), prefix=".upload-", suffix=".part")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        # 큰 파일이 이벤트 루프를 막지 않도록 청크 단위로 await 하며 기록한다.
        with tmp.open("wb") as f:
            while True:
                chunk = await file.read(_CHUNK)
                if not chunk:
                    break
                f.write(chunk)
        await file.close()
        meta = video_svc.register_upload(tmp, file.filename or "video.mp4")
    finally:
        await file.close()
        tmp.unlink(missing_ok=True)
    return JSONResponse({"ok": True, **meta})


@router.post(
    "/api/videos/path",
    tags=[_TAG_VIDEO],
    summary="서버 경로 영상 등록 (복사 없음)",
    description=(
        "이미 서버(또는 마운트된 NAS)에 있는 영상을 **복사하지 않고** 등록한다. "
        "수십 GB 를 중복 저장하지 않기 위한 경로. 원본이 지워지면 프레임 재추출이 "
        "불가능하므로 meta 에 `managed=false` 로 남는다."
    ),
)
async def register_video_path(req: PathRequest) -> JSONResponse:
    return JSONResponse({"ok": True, **video_svc.register_path(req.path)})


@router.get("/api/videos", tags=[_TAG_VIDEO], summary="영상 목록 (+구간/라벨링 진행률)")
async def list_videos() -> JSONResponse:
    return JSONResponse({"items": video_svc.list_videos()})


@router.get("/api/videos/{video_id}", tags=[_TAG_VIDEO], summary="영상 상세")
async def get_video(video_id: str) -> JSONResponse:
    return JSONResponse({
        **video_svc.get_meta(video_id),
        "segments": segments_svc.get(video_id).get("segments"),
        "segment_summary": segments_svc.summary(video_id),
        "labeling": annotations.progress(video_id),
        "class_stats": annotations.class_stats(video_id),
    })


@router.delete(
    "/api/videos/{video_id}",
    tags=[_TAG_VIDEO],
    summary="영상 삭제 (프레임/라벨 포함)",
    description="이미 만들어진 데이터셋은 이미지를 자체 보관하므로 영향받지 않는다.",
)
async def delete_video(video_id: str) -> JSONResponse:
    return JSONResponse(video_svc.delete_video(video_id))


@router.get(
    "/api/videos/{video_id}/stream",
    tags=[_TAG_VIDEO],
    summary="영상 재생 (Range 지원)",
    description=(
        "`<video src>` 로 바로 붙일 수 있는 원본 스트림. HTTP Range 를 처리하므로 "
        "타임라인 탐색(seek)이 된다 — 구간 지정 UI 의 전제 조건."
    ),
)
async def stream_video(video_id: str, request: Request):
    path = video_svc.source_path(video_id)
    return _ranged_file(path, request)


@router.get(
    "/api/videos/{video_id}/frame",
    tags=[_TAG_VIDEO],
    summary="특정 시각 프레임(JPEG)",
    description=(
        "구간을 찍을 때 쓰는 미리보기/썸네일. `t` 는 초 단위.\n\n"
        "`preprocess=1` 이면 저장 때와 같은 전처리(추론 tracker_py 와 동일: 야간 보정 → "
        "안개 제거 → CLAHE → 선택적 리사이즈)를 적용해 돌려준다. 판정 결과를 헤더로 준다: "
        "야간 `X-Daynight`(day/night/off), 안개 제거 `X-Dehaze`(0/1), CLAHE `X-Clahe`(0/1). "
        "쿼리(`auto`, `lowlight`, `dehaze`, `clahe`, `lowlight_threshold`, `night_gamma`, "
        "`resize` …)로 설정을 덮어써 '보정 전/후'를 비교한다."
    ),
)
async def video_frame(
    video_id: str,
    t: float = Query(0.0, ge=0.0, description="시각(초)"),
    preprocess: bool = Query(False, description="전처리 적용 미리보기"),
    auto: Optional[bool] = Query(None, description="프레임별 자동 판정(끄면 강제 적용)"),
    lowlight: Optional[bool] = Query(None, description="야간 보정"),
    dehaze: Optional[bool] = Query(None, description="안개 제거"),
    clahe: Optional[bool] = Query(None, description="화질 향상 CLAHE"),
    lowlight_threshold: Optional[float] = Query(None, ge=0, le=255),
    night_gamma: Optional[float] = Query(None, ge=0.1),
    resize: Optional[bool] = Query(None),
    resize_width: Optional[int] = Query(None, ge=32),
    resize_height: Optional[int] = Query(None, ge=32),
) -> Response:
    if not preprocess:
        frame = video_svc.grab_at(video_id, t)
        return Response(content=video_svc.encode_jpeg(frame), media_type="image/jpeg")
    saved = video_svc.get_preprocess(video_id)
    override = {k: v for k, v in {
        "auto": auto, "lowlight": lowlight, "dehaze": dehaze, "clahe": clahe,
        "lowlight_threshold": lowlight_threshold,
        "night_gamma": night_gamma, "resize": resize,
        "resize_width": resize_width, "resize_height": resize_height,
    }.items() if v is not None}
    frame, info = video_svc.grab_at(video_id, t, preprocess={**saved, **override})
    return Response(content=video_svc.encode_jpeg(frame), media_type="image/jpeg",
                    headers={"X-Daynight": info["lowlight"],
                             "X-Dehaze": "1" if info["dehaze"] else "0",
                             "X-Clahe": "1" if info.get("clahe") else "0"})


class PreprocessRequest(BaseModel):
    auto: Optional[bool] = Field(None, description="프레임별 자동 판정(끄면 켠 전처리를 전 프레임 강제 적용)")
    lowlight: Optional[bool] = Field(None, description="야간 보정(저조도 밝기↑)")
    dehaze: Optional[bool] = Field(None, description="안개 제거(dehaze)")
    clahe: Optional[bool] = Field(None, description="화질 향상 CLAHE(추론 Stage2)")
    lowlight_threshold: Optional[float] = Field(None, ge=0, le=255)
    night_clahe_clip: Optional[float] = Field(None, ge=0.1)
    night_clahe_grid: Optional[int] = Field(None, ge=1)
    night_gamma: Optional[float] = Field(None, ge=0.1, description="야간 밝기 감마(폴백 엔진에서만 사용, 클수록 밝음)")
    dehaze_omega: Optional[float] = Field(None, ge=0, le=1)
    dehaze_t0: Optional[float] = Field(None, ge=0, le=1)
    dehaze_wsz: Optional[int] = Field(None, ge=3)
    dehaze_scale: Optional[float] = Field(None, gt=0, le=1)
    dehaze_guide_r: Optional[int] = Field(None, ge=1)
    clahe_clip: Optional[float] = Field(None, ge=0.1)
    clahe_grid: Optional[int] = Field(None, ge=1)
    resize: Optional[bool] = Field(None, description="해상도 다운스케일 여부")
    resize_width: Optional[int] = Field(None, ge=32)
    resize_height: Optional[int] = Field(None, ge=32)


@router.get(
    "/api/videos/{video_id}/preprocess",
    tags=[_TAG_VIDEO],
    summary="영상별 전처리 설정 조회",
    description="저장된 영상별 전처리 설정과, 서버 기본값을 합쳐 실제로 적용될 값을 함께 준다.",
)
async def get_preprocess(video_id: str) -> JSONResponse:
    from services import preprocess as pp_svc
    saved = video_svc.get_preprocess(video_id)
    return JSONResponse({
        "video_id": video_id,
        "saved": saved,
        "effective": pp_svc.resolve(saved),
    })


@router.put(
    "/api/videos/{video_id}/preprocess",
    tags=[_TAG_VIDEO],
    summary="영상별 전처리 설정 저장 (학습 전 지정)",
    description=(
        "안개 제거/야간 보정을 **영상 단위로** 켜고 끄거나 임계치·감마를 저장한다. 자동 야간 "
        "판정이 틀리는 영상을 위한 경로다. 저장값은 추출 시 기본값으로 쓰이고, 추출 요청이 "
        "값을 명시하면 요청이 우선한다."
    ),
)
async def put_preprocess(video_id: str, req: PreprocessRequest) -> JSONResponse:
    from services import preprocess as pp_svc
    saved = video_svc.set_preprocess(video_id, req.model_dump(exclude_none=True))
    return JSONResponse({"ok": True, "video_id": video_id, "saved": saved,
                         "effective": pp_svc.resolve(saved)})


def _ranged_file(path: Path, request: Request):
    """HTTP Range 부분 응답. 브라우저 `<video>` 탐색을 위해 직접 처리한다."""
    size = path.stat().st_size
    range_header = request.headers.get("range") or request.headers.get("Range")
    media_type = "video/mp4" if path.suffix.lower() in (".mp4", ".m4v") else "application/octet-stream"
    if not range_header or not range_header.startswith("bytes="):
        return FileResponse(path, media_type=media_type,
                            headers={"Accept-Ranges": "bytes"})

    spec = range_header.removeprefix("bytes=").split(",")[0].strip()
    start_s, _, end_s = spec.partition("-")
    try:
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else size - 1
    except ValueError:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
    start = max(0, start)
    end = min(size - 1, end)
    if start > end:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})

    def body():
        remaining = end - start + 1
        with path.open("rb") as f:
            f.seek(start)
            while remaining > 0:
                chunk = f.read(min(_CHUNK, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    return StreamingResponse(body(), status_code=206, media_type=media_type, headers={
        "Content-Range": f"bytes {start}-{end}/{size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(end - start + 1),
    })


# ══════════════════════════════════════════════════════════════════════
# 2. 구간
# ══════════════════════════════════════════════════════════════════════
class SegmentIn(BaseModel):
    kind: str = Field("normal", description="normal | abnormal")
    start_sec: float = Field(..., ge=0)
    end_sec: float = Field(..., ge=0)
    note: str = ""


class SegmentsRequest(BaseModel):
    segments: List[SegmentIn] = Field(default_factory=list)


@router.get("/api/videos/{video_id}/segments", tags=[_TAG_SEGMENT], summary="구간 조회")
async def get_segments(video_id: str) -> JSONResponse:
    return JSONResponse({**segments_svc.get(video_id),
                         "summary": segments_svc.summary(video_id)})


@router.put(
    "/api/videos/{video_id}/segments",
    tags=[_TAG_SEGMENT],
    summary="정상/비정상 구간 저장 (전체 교체)",
    description=(
        "프런트 타임라인에서 찍은 구간을 통째로 저장한다. 초(sec) 단위 실수.\n\n"
        "- `normal` : 학습에 쓸 구간 (여러 개 가능)\n"
        "- `abnormal` : 못 쓸 구간 (흔들림/노출 폭주/렌즈 오염 등, 여러 개 가능)\n\n"
        "겹치는 같은 종류 구간은 자동 병합된다. **정상 ∩ 비정상은 비정상이 이긴다.**\n"
        "정상 구간을 하나도 지정하지 않으면 영상 전체를 정상으로 본다."
    ),
)
async def put_segments(video_id: str, req: SegmentsRequest) -> JSONResponse:
    doc = segments_svc.put(video_id, [s.model_dump() for s in req.segments])
    return JSONResponse({
        **doc,
        "summary": segments_svc.summary(video_id),
        "selection_ranges": [[round(a, 3), round(b, 3)]
                             for a, b in segments_svc.selection_ranges(video_id)],
    })


@router.get(
    "/api/videos/{video_id}/selection",
    tags=[_TAG_SEGMENT],
    summary="실제 추출 대상 구간 미리보기",
    description="`정상 - 비정상` 계산 결과. 프레임을 뽑기 전에 무엇이 대상인지 확인용.",
)
async def get_selection(
    video_id: str,
    kinds: List[str] = Query(default=["normal"], description="합칠 구간 종류"),
) -> JSONResponse:
    ranges = segments_svc.selection_ranges(video_id, kinds)
    total = sum(b - a for a, b in ranges)
    return JSONResponse({
        "video_id": video_id,
        "kinds": kinds,
        "ranges": [[round(a, 3), round(b, 3)] for a, b in ranges],
        "total_sec": round(total, 3),
    })


@router.get(
    "/api/videos/{video_id}/detect_preview",
    tags=[_TAG_SEGMENT],
    summary="추출 전 모델 탐지 미리보기 (단일 프레임)",
    description=(
        "본격 추출(수천 장) 전에, 현재 지점 프레임 **한 장**에 대해 고른 가중치로 자동 라벨 "
        "결과를 미리 본다. 추출과 **똑같은 전처리**를 적용한 뒤 탐지하므로, 실제 초안이 어떻게 "
        "나올지 그대로 확인할 수 있다.\n\n"
        "- `model` 로 가중치를 고른다(생략 시 서버 기본값). `conf`/`iou`/`imgsz` 로 임계값 조정.\n"
        "- 전처리 쿼리(`auto`,`lowlight`,`dehaze`,`clahe`,`lowlight_threshold`,`night_gamma`,"
        "`resize` …)는 프레임 미리보기와 동일하게 덮어쓸 수 있다.\n"
        "- 반환은 회전박스 꼭짓점(`poly`, 전처리 후 프레임 픽셀 좌표) 목록이라, 프런트가 "
        "미리보기 이미지 위에 그대로 겹쳐 그릴 수 있다. 초안이 0개면 이 가중치가 이 영상의 "
        "표적을 모른다는 뜻이다(라벨링 화면에서 직접 그리거나 승격 모델로 다시 시도)."
    ),
)
async def detect_preview(
    video_id: str,
    t: float = Query(0.0, ge=0.0, description="시각(초)"),
    model: Optional[str] = Query(None, description="자동 라벨 가중치(생략 시 서버 기본값)"),
    conf: Optional[float] = Query(None, ge=0, le=1),
    iou: Optional[float] = Query(None, ge=0, le=1),
    imgsz: Optional[int] = Query(None, ge=32),
    max_det: Optional[int] = Query(None, ge=1),
    auto: Optional[bool] = Query(None),
    lowlight: Optional[bool] = Query(None),
    dehaze: Optional[bool] = Query(None),
    clahe: Optional[bool] = Query(None),
    lowlight_threshold: Optional[float] = Query(None, ge=0, le=255),
    night_gamma: Optional[float] = Query(None, ge=0.1),
    resize: Optional[bool] = Query(None),
    resize_width: Optional[int] = Query(None, ge=32),
    resize_height: Optional[int] = Query(None, ge=32),
) -> JSONResponse:
    saved = video_svc.get_preprocess(video_id)
    override = {k: v for k, v in {
        "auto": auto, "lowlight": lowlight, "dehaze": dehaze, "clahe": clahe,
        "lowlight_threshold": lowlight_threshold, "night_gamma": night_gamma,
        "resize": resize, "resize_width": resize_width, "resize_height": resize_height,
    }.items() if v is not None}
    frame, info = video_svc.grab_at(video_id, t, preprocess={**saved, **override})
    det = detector_svc.get_detector(model)
    objs = det.infer(frame, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, track=False)
    h, w = frame.shape[:2]
    di = det.info()
    return JSONResponse({
        "video_id": video_id,
        "t": round(float(t), 3),
        "width": int(w),
        "height": int(h),
        "model": di["name"],
        "model_path": di["path"],
        "task": det.task,
        "preprocess": info,
        "count": len(objs),
        "detections": [{
            "class_name": o["class_name"],
            "score": o["score"],
            "track_id": o.get("track_id"),
            "poly": o["poly"],
            "bbox": o["bbox"],
        } for o in objs],
    })


# ══════════════════════════════════════════════════════════════════════
# 3. 라벨링
# ══════════════════════════════════════════════════════════════════════
class ExtractRequest(BaseModel):
    kinds: List[str] = Field(default_factory=lambda: ["normal"], description="추출할 구간 종류")
    fps: Optional[float] = Field(None, description="초당 추출 장수 (0=모든 프레임)")
    model: Optional[str] = Field(None, description="자동 라벨에 쓸 가중치 (기본 test_model/yolo26l-obb.pt)")
    conf: Optional[float] = Field(None, ge=0, le=1)
    iou: Optional[float] = Field(None, ge=0, le=1)
    imgsz: Optional[int] = Field(None, ge=32)
    max_det: Optional[int] = Field(None, ge=1)
    max_frames: Optional[int] = Field(None, ge=1)
    track: Optional[bool] = Field(None, description="track_id 부여 (클래스 일괄 전파용)")
    overwrite: str = Field("skip", description="skip | auto | all")
    # ── 전처리(생략 시 영상별 저장값 → 서버 기본값 순으로 채워진다) ──
    auto: Optional[bool] = Field(None, description="프레임별 자동 판정(끄면 켠 전처리를 전 프레임 강제 적용)")
    lowlight: Optional[bool] = Field(None, description="야간 보정(저조도 밝기↑). auto 면 어두운 프레임만 적용")
    dehaze: Optional[bool] = Field(None, description="안개 제거(dehaze). auto 면 안개 프레임만 적용")
    clahe: Optional[bool] = Field(None, description="화질 향상 CLAHE(추론 Stage2)")
    lowlight_threshold: Optional[float] = Field(None, ge=0, le=255,
                                                description="야간 자동 판정 밝기 임계치(0~255)")
    night_clahe_clip: Optional[float] = Field(None, ge=0.1)
    night_clahe_grid: Optional[int] = Field(None, ge=1)
    night_gamma: Optional[float] = Field(None, ge=0.1, description="야간 밝기 감마(폴백 엔진에서만 사용)")
    dehaze_omega: Optional[float] = Field(None, ge=0, le=1)
    dehaze_t0: Optional[float] = Field(None, ge=0, le=1)
    dehaze_wsz: Optional[int] = Field(None, ge=3)
    dehaze_scale: Optional[float] = Field(None, gt=0, le=1)
    dehaze_guide_r: Optional[int] = Field(None, ge=1)
    clahe_clip: Optional[float] = Field(None, ge=0.1)
    clahe_grid: Optional[int] = Field(None, ge=1)
    resize: Optional[bool] = Field(None, description="해상도 다운스케일 여부(선택)")
    resize_width: Optional[int] = Field(None, ge=32)
    resize_height: Optional[int] = Field(None, ge=32)


@router.post(
    "/api/videos/{video_id}/extract",
    tags=[_TAG_LABEL],
    summary="프레임 추출 + 자동 라벨 초안 생성 (작업)",
    description=(
        "정상 구간(비정상 제외)에서 프레임을 뽑아 저장하고, YOLO 로 자동 라벨 초안을 만든다.\n\n"
        "- 기본 초당 2장. 30fps 를 전부 뽑으면 1분에 1800장이고 인접 프레임은 거의 같은 "
        "그림이라 검수 부담만 커진다.\n"
        "- `track=true` 면 객체마다 `track_id` 가 붙는다. 프런트에서 객체당 한 번만 "
        "클래스를 고르고 `POST .../propagate` 로 전체 프레임에 전파할 수 있다.\n"
        "- 재실행 정책(`overwrite`): `skip`=기존 라벨 보존(기본), "
        "`auto`=검수 전 자동 초안만 갱신, `all`=전부 재생성(**사람 수정 소실**).\n\n"
        "반환된 `job_id` 로 `GET /api/jobs/{job_id}` 진행률을 폴링한다."
    ),
)
async def extract_frames(video_id: str, req: ExtractRequest) -> JSONResponse:
    job = autolabel.start(video_id, req.model_dump(exclude_none=True))
    return JSONResponse({"ok": True, "job_id": job.id, **job.to_dict()})


@router.get(
    "/api/videos/{video_id}/frames",
    tags=[_TAG_LABEL],
    summary="프레임 목록 (라벨 요약)",
    description="폴리곤 좌표를 뺀 경량 요약. 검수 화면의 썸네일 그리드용.",
)
async def list_frames(
    video_id: str,
    status: Optional[str] = Query(None, description="pending | approved | rejected"),
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
) -> JSONResponse:
    return JSONResponse(annotations.list_frames(video_id, status=status,
                                                offset=offset, limit=limit))


@router.get("/api/videos/{video_id}/progress", tags=[_TAG_LABEL], summary="라벨링 진행률")
async def labeling_progress(video_id: str) -> JSONResponse:
    return JSONResponse({
        "video_id": video_id,
        **annotations.progress(video_id),
        "class_stats": annotations.class_stats(video_id),
    })


@router.get(
    "/api/videos/{video_id}/tracks",
    tags=[_TAG_LABEL],
    summary="객체(track) 목록",
    description=(
        "자동 라벨이 붙인 track_id 요약. 객체 하나당 등장 프레임 수와 현재 클래스를 준다.\n"
        "프런트는 이 목록을 띄워 **객체 단위로** 클래스를 지정하게 하면 된다 "
        "(프레임 수백 장을 한 장씩 고르지 않아도 된다)."
    ),
)
async def list_tracks(video_id: str) -> JSONResponse:
    return JSONResponse({"video_id": video_id, "items": annotations.tracks(video_id)})


@router.get("/api/videos/{video_id}/frames/{frame_id}", tags=[_TAG_LABEL], summary="프레임 라벨 조회")
async def get_frame(video_id: str, frame_id: str) -> JSONResponse:
    return JSONResponse(annotations.load(video_id, frame_id))


@router.get(
    "/api/videos/{video_id}/frames/{frame_id}/image",
    tags=[_TAG_LABEL],
    summary="프레임 이미지 (JPEG)",
    description="`overlay=1` 이면 현재 라벨(회전 박스 + 클래스/트랙)을 그려서 준다.",
)
async def get_frame_image(video_id: str, frame_id: str,
                          overlay: bool = Query(False)) -> Response:
    if overlay:
        return Response(content=annotations.overlay_jpeg(video_id, frame_id),
                        media_type="image/jpeg")
    path = annotations.image_path(video_id, frame_id)
    if not path.exists():
        raise KeyError(f"프레임 이미지가 없습니다: {video_id}/{frame_id}")
    return FileResponse(path, media_type="image/jpeg")


class ObjectIn(BaseModel):
    id: Optional[str] = None
    class_name: Optional[str] = Field(None, description="클래스 목록에 있는 이름. null=미확정")
    poly: List[List[float]] = Field(..., description="픽셀 좌표 4점 [[x,y]*4]. 2점이면 AABB 로 변환")
    track_id: Optional[int] = None
    score: Optional[float] = None
    model_class_name: Optional[str] = None


class FrameUpdateRequest(BaseModel):
    objects: Optional[List[ObjectIn]] = Field(None, description="보내면 전체 교체")
    status: Optional[str] = Field(None, description="pending | approved | rejected")
    note: Optional[str] = None
    force: bool = Field(False, description="미확정 클래스가 있어도 승인 강행")


@router.put(
    "/api/videos/{video_id}/frames/{frame_id}",
    tags=[_TAG_LABEL],
    summary="프레임 라벨 저장 (사람 수정)",
    description=(
        "박스 추가/이동/삭제와 클래스 지정 결과를 저장한다. `objects` 를 보내면 전체 교체다.\n\n"
        "`status=approved` 로 함께 보내면 검수 완료 처리된다. 단 **클래스가 정해지지 않은 "
        "객체가 하나라도 있으면 거부**한다(일부만 라벨된 이미지는 모델을 망친다). "
        "객체가 0개인 프레임은 배경 샘플로 유효하므로 승인된다."
    ),
)
async def update_frame(video_id: str, frame_id: str, req: FrameUpdateRequest) -> JSONResponse:
    objects = ([o.model_dump() for o in req.objects] if req.objects is not None else None)
    return JSONResponse(annotations.update(video_id, frame_id, objects=objects,
                                           status=req.status, note=req.note,
                                           force=req.force))


class StatusRequest(BaseModel):
    status: str = Field(..., description="pending | approved | rejected")
    frame_ids: Optional[List[str]] = Field(None, description="생략 시 영상의 전체 프레임")
    force: bool = False


@router.post(
    "/api/videos/{video_id}/frames/{frame_id}/status",
    tags=[_TAG_LABEL],
    summary="프레임 검수 상태 변경",
)
async def set_frame_status(video_id: str, frame_id: str,
                           req: StatusRequest) -> JSONResponse:
    return JSONResponse(annotations.set_status(video_id, frame_id, req.status,
                                               force=req.force))


@router.post(
    "/api/videos/{video_id}/frames/status",
    tags=[_TAG_LABEL],
    summary="검수 상태 일괄 변경",
    description=(
        "전체 승인 버튼용. 승인 조건을 못 채운 프레임은 건너뛰고 이유를 모아 돌려준다"
        "(하나 때문에 수천 장 검수가 막히지 않게)."
    ),
)
async def bulk_frame_status(video_id: str, req: StatusRequest) -> JSONResponse:
    return JSONResponse(annotations.bulk_status(video_id, req.status,
                                                frame_ids=req.frame_ids,
                                                force=req.force))


class PropagateRequest(BaseModel):
    track_id: int = Field(..., description="클래스를 적용할 객체의 track_id")
    class_name: str = Field(..., description="클래스 목록에 있는 이름")
    frame_ids: Optional[List[str]] = Field(None, description="생략 시 전체 프레임")


@router.post(
    "/api/videos/{video_id}/propagate",
    tags=[_TAG_LABEL],
    summary="객체 클래스 일괄 전파 (track_id 기준)",
    description=(
        "같은 `track_id` 를 가진 객체의 클래스를 모든 프레임에 한 번에 적용한다.\n\n"
        "라벨링 공수를 **프레임 수가 아니라 객체 수**에 비례하게 만드는 핵심 기능이다. "
        "300 프레임에 등장하는 전차 1대는 클릭 한 번으로 끝난다."
    ),
)
async def propagate_class(video_id: str, req: PropagateRequest) -> JSONResponse:
    return JSONResponse(annotations.propagate_class(
        video_id, track_id=req.track_id, class_name=req.class_name,
        frame_ids=req.frame_ids))


@router.delete(
    "/api/videos/{video_id}/frames/{frame_id}",
    tags=[_TAG_LABEL],
    summary="프레임 삭제 (이미지+라벨)",
    description="보통은 `status=rejected` 로 이력을 남기는 편이 낫다. 잘못 추출된 프레임 정리용.",
)
async def delete_frame(video_id: str, frame_id: str) -> JSONResponse:
    return JSONResponse(annotations.delete_frame(video_id, frame_id))


# ══════════════════════════════════════════════════════════════════════
# 4. 데이터셋
# ══════════════════════════════════════════════════════════════════════
class DatasetRequest(BaseModel):
    name: Optional[str] = Field(None, description="사람이 알아볼 이름")
    video_ids: Optional[List[str]] = Field(None, description="학습에 쓸 영상들. 생략 시 전체, []=영상 없이 병합만")
    task: Optional[str] = Field(None, description="obb | detect — 생략 시 서버 기본값(obb)")
    base_datasets: Optional[List[str]] = Field(
        None,
        description=(
            "같이 넣을 기존 데이터셋. dataset_id 또는 서버 폴더 경로"
            "(예: C:/project/tracker_py/train_data/preprocessed_obb). 원래 분할 유지."
        ),
    )
    class_names: Optional[List[str]] = Field(None, description="생략 시 프로젝트 클래스 목록")
    include_kinds: Optional[List[str]] = Field(None, description="기본 ['normal']")
    only_approved: Optional[bool] = Field(None, description="기본 true (승인된 프레임만)")
    splits: Optional[dict] = Field(None, description='{"train":0.8,"valid":0.15,"test":0.05}')
    split_mode: Optional[str] = Field(None, description="chunk | random | video")
    chunk_size: Optional[int] = Field(None, ge=1)
    seed: Optional[int] = None
    link_images: Optional[bool] = Field(None, description="하드링크로 디스크 절약(기본 true)")


@router.post(
    "/api/datasets",
    tags=[_TAG_DATASET],
    summary="데이터셋 빌드 (작업)",
    description=(
        "선택한 영상들의 **승인된** 라벨을 모아 ultralytics 학습용 데이터셋 스냅샷을 만든다.\n\n"
        "- 이미지를 데이터셋 폴더로 링크/복사해 고정한다 → 나중에 라벨을 고쳐도 이미 "
        "학습한 데이터셋은 변하지 않는다(실험 재현성).\n"
        "- 분할 기본값 `chunk`: 연속 프레임을 블록으로 묶어 배분한다. 무작위로 나누면 "
        "train/valid 에 사실상 같은 장면이 들어가 검증 점수가 부풀려진다.\n"
        "- 클래스가 미확정인 객체가 있는 프레임은 통째로 제외되고 `excluded` 에 집계된다.\n"
        "- `base_datasets` 로 기존 데이터셋(또는 `train_data/preprocessed_obb` 같은 폴더)을 "
        "같이 넣을 수 있다. 기존 자산 + 새로 라벨한 프레임을 합쳐 학습하는 실제 운용 흐름이다. "
        "클래스는 **이름으로** 다시 매핑되고, 기존 데이터셋의 train/valid/test 분할은 그대로 유지된다.\n\n"
        "출력은 `tracker_py/train_data/preprocessed_obb` 와 같은 구조다"
        "(`data.yaml` + `{train,valid,test}/{images,labels}`).\n"
        "`task` 는 학습에 쓸 가중치와 일치해야 한다(`obb`=8좌표, `detect`=cxcywh)."
    ),
)
async def create_dataset(req: DatasetRequest) -> JSONResponse:
    job = dataset_svc.start(req.model_dump(exclude_none=True))
    return JSONResponse({"ok": True, "job_id": job.id, "dataset_id": job.target,
                         **job.to_dict()})


class DatasetImportRequest(BaseModel):
    # 필드 이름은 `snapshot` 이지만 요청 본문의 키는 `copy` 다.
    # pydantic BaseModel 에 `copy` 메서드가 있어 그대로 쓰면 이름이 가려진다(경고).
    model_config = ConfigDict(populate_by_name=True)

    path: str = Field(..., description="YOLO 데이터셋 폴더(절대 또는 워크스페이스 상대)")
    name: Optional[str] = Field(None, description="생략 시 폴더 이름")
    task: Optional[str] = Field(None, description="생략 시 data.yaml 의 task 또는 라벨 토큰 수로 추정")
    class_names: Optional[List[str]] = Field(None, description="data.yaml 에 names 가 없을 때만 필요")
    snapshot: bool = Field(False, alias="copy",
                           description="true=워크스페이스로 복사해 스냅샷 고정")
    link_images: Optional[bool] = Field(None, description="copy 시 하드링크 사용(기본 true)")


@router.post(
    "/api/datasets/import",
    tags=[_TAG_DATASET],
    summary="기존 YOLO 데이터셋 폴더 등록",
    description=(
        "이미 완성된 학습 전 구조(`tracker_py/train_data/preprocessed_obb` 등)를 그대로 "
        "데이터셋 목록에 올린다. 다시 만들지 않고 바로 학습에 쓸 수 있다.\n\n"
        "- `copy=false`(기본): **복사하지 않고** `data.yaml` 만 만들어 원본을 가리킨다. "
        "수천 장을 중복 저장하지 않는다. 대신 원본이 바뀌면 데이터셋도 바뀐다.\n"
        "- `copy=true`: 워크스페이스로 하드링크/복사해 스냅샷으로 고정하고, 클래스 id 를 "
        "현재 프로젝트 목록 기준으로 다시 쓴다.\n\n"
        "`data.yaml` 이 없어도 폴더 구조와 라벨 토큰 수(5=detect, 9=obb)로 태스크를 추정한다."
    ),
)
async def import_dataset(req: DatasetImportRequest) -> JSONResponse:
    spec = req.model_dump(exclude_none=True, by_alias=True)
    return JSONResponse({"ok": True, **dataset_svc.import_dir(spec)})


@router.get(
    "/api/datasets/inspect",
    tags=[_TAG_DATASET],
    summary="폴더 미리보기 (등록 전 확인)",
    description="등록하기 전에 그 폴더가 어떤 태스크/클래스/장수인지 읽어만 본다.",
)
async def inspect_dataset_dir(
    path: str = Query(..., description="YOLO 데이터셋 폴더 경로"),
) -> JSONResponse:
    return JSONResponse(dataset_svc.inspect(path))


@router.get("/api/datasets", tags=[_TAG_DATASET], summary="데이터셋 목록")
async def list_datasets() -> JSONResponse:
    return JSONResponse({"items": dataset_svc.list_datasets()})


@router.get(
    "/api/datasets/{dataset_id}",
    tags=[_TAG_DATASET],
    summary="데이터셋 상세 (manifest)",
    description="클래스 분포·영상별 프레임 수·제외 사유까지 포함. 학습 전 데이터 점검용.",
)
async def get_dataset(dataset_id: str) -> JSONResponse:
    return JSONResponse(dataset_svc.get(dataset_id))


@router.get("/api/datasets/{dataset_id}/data.yaml", tags=[_TAG_DATASET], summary="data.yaml 조회")
async def get_dataset_yaml(dataset_id: str) -> Response:
    path = store.dataset_yaml_path(dataset_id)
    if not path.exists():
        raise KeyError(f"data.yaml 이 없습니다: {dataset_id}")
    return Response(content=path.read_text(encoding="utf-8"), media_type="text/plain")


@router.delete("/api/datasets/{dataset_id}", tags=[_TAG_DATASET], summary="데이터셋 삭제")
async def delete_dataset(dataset_id: str) -> JSONResponse:
    return JSONResponse(dataset_svc.delete(dataset_id))


# ══════════════════════════════════════════════════════════════════════
# 5. 학습
# ══════════════════════════════════════════════════════════════════════
class TrainRequest(BaseModel):
    dataset_id: str = Field(..., description="POST /api/datasets 로 만든 데이터셋")
    model: Optional[str] = Field(None, description="사전학습 가중치. 기본 test_model/yolo26l-obb.pt")
    epochs: Optional[int] = Field(None, ge=1)
    imgsz: Optional[int] = Field(None, ge=32)
    batch: Optional[int] = Field(None, ge=1)
    device: Optional[str] = Field(None, description='CUDA 인덱스("0") 또는 "cpu"')
    workers: Optional[int] = Field(None, ge=0, description="DataLoader 워커. Windows 는 4 이하 권장")
    patience: Optional[int] = Field(None, ge=1)
    note: Optional[str] = None
    extra: Optional[dict] = Field(None, description="ultralytics train() 에 그대로 전달할 추가 인자")


@router.post(
    "/api/train",
    tags=[_TAG_TRAIN],
    summary="학습 시작",
    description=(
        "데이터셋과 가중치를 골라 학습을 시작한다. 기본값은 `test/yolo.py` 에서 검증된 "
        "epochs=100 / imgsz=640 / batch=16 / workers=4 이고, 기본 가중치는 "
        "`test_model/yolo26l-obb.pt` 다(회전박스 학습이 기본).\n\n"
        "학습은 **API 와 분리된 자식 프로세스**에서 돈다. GPU 사고가 서비스를 같이 "
        "죽이지 않고, 중단도 프로세스 종료로 확실하게 끝난다.\n\n"
        "`workers` 를 크게 잡으면(Windows spawn) 워커마다 RAM 을 물어 후반 epoch 에서 "
        "DataLoader 가 죽는다. RAM 부족이 재발하면 4 → 2 로 낮춘다."
    ),
)
async def start_train(req: TrainRequest) -> JSONResponse:
    return JSONResponse({"ok": True, **trainer.start(req.model_dump(exclude_none=True))})


@router.get("/api/train", tags=[_TAG_TRAIN], summary="학습 run 목록")
async def list_runs() -> JSONResponse:
    return JSONResponse({"items": trainer.list_runs()})


@router.get(
    "/api/train/{run_id}",
    tags=[_TAG_TRAIN],
    summary="학습 진행률/지표/로그",
    description="epoch 진행률, 최근 metrics, results.csv 마지막 행, 로그 tail 을 함께 준다.",
)
async def train_status(run_id: str,
                       log_lines: int = Query(200, ge=0, le=5000)) -> JSONResponse:
    return JSONResponse(trainer.status(run_id, log_lines=log_lines))


@router.post(
    "/api/train/{run_id}/stop",
    tags=[_TAG_TRAIN],
    summary="학습 중단",
    description="자식 프로세스와 DataLoader 워커 트리까지 종료한다(GPU 메모리 회수).",
)
async def stop_train(run_id: str) -> JSONResponse:
    return JSONResponse(trainer.stop(run_id))


@router.post(
    "/api/train/{run_id}/resume",
    tags=[_TAG_TRAIN],
    summary="중단된 학습 이어서",
    description="`train/weights/last.pt` 부터 이어서 학습한다(test/yolo.py 의 resume 과 동일).",
)
async def resume_train(run_id: str) -> JSONResponse:
    return JSONResponse(trainer.resume(run_id))


@router.get("/api/train/{run_id}/log", tags=[_TAG_TRAIN], summary="학습 로그 tail")
async def train_log(run_id: str, lines: int = Query(500, ge=1, le=20000)) -> Response:
    return Response(content=store.tail_text(store.run_log_path(run_id), lines),
                    media_type="text/plain")


@router.get(
    "/api/train/{run_id}/weights/{which}",
    tags=[_TAG_TRAIN],
    summary="학습 가중치 다운로드 (best|last)",
)
async def download_weights(run_id: str, which: str) -> FileResponse:
    path = trainer.weights_path(run_id, which)
    return FileResponse(path, media_type="application/octet-stream",
                        filename=f"{run_id}_{which}.pt")


@router.delete("/api/train/{run_id}", tags=[_TAG_TRAIN], summary="학습 run 삭제")
async def delete_run(run_id: str) -> JSONResponse:
    return JSONResponse(trainer.delete(run_id))


# ══════════════════════════════════════════════════════════════════════
# 6. 모델
# ══════════════════════════════════════════════════════════════════════
@router.get(
    "/api/models",
    tags=[_TAG_MODEL],
    summary="사용 가능한 가중치 + 승격 이력",
    description=(
        "`test_model/`(사전학습), `workspace/models/`(승격), `runs/*/train/weights/`(학습 산출물) "
        "를 모두 모아 준다. 자동 라벨(`extract.model`)과 학습(`train.model`)의 선택 목록."
    ),
)
async def list_models() -> JSONResponse:
    return JSONResponse(registry.list_models())


class PromoteRequest(BaseModel):
    run_id: str
    alias: str = Field(..., description="배포 후보 이름 (영문/숫자/.-_)")
    which: str = Field("best", description="best | last")
    note: str = ""
    deploy: bool = Field(False, description="추론 서비스 models 폴더로 복사까지 수행")


@router.post(
    "/api/models/promote",
    tags=[_TAG_MODEL],
    summary="학습 결과 승격 (+선택적 배포)",
    description=(
        "학습 가중치를 `workspace/models/<alias>.pt` 로 복사해 고정하고, 어떤 데이터셋·"
        "어떤 지표에서 나왔는지 함께 기록한다(`registry.json`). run 을 지워도 남는다.\n\n"
        "`deploy=true` 면 `NIT_TRAIN_TRACKER_MODELS_DIR` 로 복사한다. 그 뒤 추론 서비스의 "
        "`POST /api/detector/model` 을 호출하면 재시작 없이 새 모델이 적용된다."
    ),
)
async def promote_model(req: PromoteRequest) -> JSONResponse:
    return JSONResponse({"ok": True, **registry.promote(
        req.run_id, alias=req.alias, which=req.which, note=req.note, deploy=req.deploy)})


@router.delete("/api/models/{alias}", tags=[_TAG_MODEL], summary="승격 취소(파일+이력 삭제)")
async def unpromote_model(alias: str) -> JSONResponse:
    return JSONResponse(registry.unpromote(alias))


# ══════════════════════════════════════════════════════════════════════
# 7. 작업
# ══════════════════════════════════════════════════════════════════════
@router.get(
    "/api/jobs",
    tags=[_TAG_JOB],
    summary="작업 목록",
    description="실행 중 + 최근 완료된 추출/데이터셋 작업(최신순).",
)
async def list_jobs(kind: Optional[str] = Query(None, description="extract | dataset"),
                    target: Optional[str] = Query(None, description="video_id / dataset_id"),
                    limit: int = Query(50, ge=1, le=500)) -> JSONResponse:
    return JSONResponse({"items": jobs.list_jobs(kind=kind, target=target, limit=limit)})


@router.get(
    "/api/jobs/{job_id}",
    tags=[_TAG_JOB],
    summary="작업 진행률",
    description="`status`: queued | running | done | error | canceled, `progress`: 0~1.",
)
async def get_job(job_id: str) -> JSONResponse:
    doc = jobs.get(job_id)
    if doc is None:
        raise KeyError(f"작업을 찾을 수 없습니다: {job_id}")
    return JSONResponse(doc)


@router.post(
    "/api/jobs/{job_id}/cancel",
    tags=[_TAG_JOB],
    summary="작업 취소",
    description="협조적 취소. 진행 중인 프레임까지 저장한 뒤 멈춘다(반쯤 쓰인 라벨 방지).",
)
async def cancel_job(job_id: str) -> JSONResponse:
    return JSONResponse({"job_id": job_id, "canceled": jobs.cancel(job_id)})


@router.get(
    "/api/detector/weights",
    tags=[_TAG_MODEL],
    summary="가중치 목록 (경량)",
    description="자동 라벨 모델 선택 드롭다운용. 승격 이력 없이 파일 목록만 준다.",
)
async def list_weights() -> JSONResponse:
    return JSONResponse({"items": detector_svc.list_weights()})

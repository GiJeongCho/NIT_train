"""
services/video.py
=================

학습 소재가 되는 영상의 등록·메타 조회·썸네일.

드론/CCTV 영상은 파일이 크고 개수가 많다. 그래서 업로드본은 워크스페이스 밖으로
복사하지 않고 `workspace/uploads/` 에 한 번만 두고, 이후 모든 단계(구간 지정,
프레임 추출, 데이터셋 빌드)는 이 파일을 참조만 한다.

서버에 이미 있는 파일은 `register_path()` 로 **복사 없이** 등록한다(수십 GB 를
중복 저장하지 않기 위함). 이 경우 원본이 지워지면 프레임 재추출이 불가능하므로
meta 에 `managed=false` 로 남긴다.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import cv2

from core.config import VIDEO_EXTS, get_settings
from core import store


def probe(path: Path) -> dict:
    """OpenCV 로 영상 메타 추출. 열 수 없으면 예외."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"영상을 열 수 없습니다: {path.name}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        # 손상/가변 프레임레이트 파일은 fps 를 0 이나 비정상 값으로 준다.
        # 구간(초) ↔ 프레임 인덱스 변환의 기준이므로 상식적인 값으로 보정한다.
        if not (0 < fps < 1000):
            fps = 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    return {
        "fps": round(fps, 3),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": round(frame_count / fps, 3) if frame_count > 0 else 0.0,
    }


def check_ext(filename: str) -> str:
    ext = Path(filename or "").suffix.lower()
    if ext not in VIDEO_EXTS:
        raise ValueError(f"지원하지 않는 형식: {ext or '(없음)'} (허용: {sorted(VIDEO_EXTS)})")
    return ext


def _write_meta(video_id: str, source_path: Path, original_name: str,
                managed: bool) -> dict:
    info = probe(source_path)
    meta = {
        "video_id": video_id,
        "original_name": original_name,
        "path": str(source_path),
        "managed": managed,          # True 면 이 서비스가 파일 수명을 소유(삭제 시 함께 삭제)
        "size_mb": round(source_path.stat().st_size / 1e6, 2),
        "created_at": store.now_iso(),
        **info,
    }
    store.write_json(store.video_meta_path(video_id), meta)
    store.frames_dir(video_id).mkdir(parents=True, exist_ok=True)
    store.labels_dir(video_id).mkdir(parents=True, exist_ok=True)
    return meta


def register_upload(tmp_path: Path, original_name: str) -> dict:
    """업로드가 끝난 임시 파일을 영상으로 등록한다(uploads 로 이동)."""
    ext = check_ext(original_name)
    video_id = store.new_id()
    dest = store.uploads_root() / f"{video_id}{ext}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp_path), str(dest))
    try:
        return _write_meta(video_id, dest, original_name, managed=True)
    except Exception:
        # 메타를 못 만들면(=열 수 없는 파일) 반쯤 등록된 영상을 남기지 않는다.
        dest.unlink(missing_ok=True)
        shutil.rmtree(store.video_dir(video_id), ignore_errors=True)
        raise


def register_path(raw_path: str) -> dict:
    """서버에 이미 있는 영상 파일을 복사 없이 등록한다."""
    src = Path(str(raw_path).strip())
    if not src.is_absolute():
        src = (store.workspace() / src).resolve()
    if not src.exists() or not src.is_file():
        raise ValueError(f"파일을 찾을 수 없습니다: {raw_path}")
    check_ext(src.name)
    video_id = store.new_id()
    try:
        return _write_meta(video_id, src, src.name, managed=False)
    except Exception:
        shutil.rmtree(store.video_dir(video_id), ignore_errors=True)
        raise


def get_meta(video_id: str) -> dict:
    meta = store.read_json(store.video_meta_path(video_id), None)
    if not isinstance(meta, dict):
        raise KeyError(f"등록되지 않은 영상: {video_id}")
    return meta


def source_path(video_id: str) -> Path:
    p = Path(get_meta(video_id).get("path", ""))
    if not p.exists():
        raise FileNotFoundError(f"원본 영상 파일이 사라졌습니다: {p}")
    return p


def list_videos() -> list:
    """영상 목록 + 라벨링 진행 요약. 프런트 첫 화면이 이것만으로 그려지게 한다."""
    from services import annotations, segments

    out = []
    for vid in store.list_video_ids():
        meta = store.read_json(store.video_meta_path(vid), None)
        if not isinstance(meta, dict):
            continue
        out.append({
            **meta,
            "segments": segments.summary(vid),
            "labeling": annotations.progress(vid),
        })
    out.sort(key=lambda m: str(m.get("created_at") or ""), reverse=True)
    return out


def delete_video(video_id: str) -> dict:
    """영상과 그 파생물(프레임/라벨)을 지운다.

    이미 만들어진 데이터셋은 이미지를 자기 폴더로 복사/링크해 갖고 있으므로 영향받지
    않는다(학습 재현성 보장). 대신 manifest 의 출처 추적이 끊긴다.
    """
    meta = get_meta(video_id)
    if meta.get("managed"):
        Path(meta.get("path", "")).unlink(missing_ok=True)
    shutil.rmtree(store.video_dir(video_id), ignore_errors=True)
    return {"video_id": video_id, "deleted": True}


def fit_frame(frame, width: Optional[int] = None, height: Optional[int] = None):
    """프레임을 운용 스펙 해상도로 정규화한다.

    운영 추론(tracker_py)이 640x480 으로 정규화해 처리하므로, 학습 이미지도 같은
    해상도 분포여야 한다. 다른 해상도로 학습하면 실전에서 객체 크기(픽셀) 분포가
    어긋나 작은 객체 성능이 떨어진다.
    """
    s = get_settings()
    if not s.frame_resize:
        return frame
    w = int(width or s.frame_width)
    h = int(height or s.frame_height)
    fh, fw = frame.shape[:2]
    if (fw, fh) == (w, h):
        return frame
    interp = cv2.INTER_AREA if (fw * fh) > (w * h) else cv2.INTER_LINEAR
    return cv2.resize(frame, (w, h), interpolation=interp)


def grab_at(video_id: str, time_sec: float, *, fit: bool = True):
    """특정 시각의 프레임 1장. 구간 지정 UI 의 미리보기/썸네일용."""
    path = source_path(video_id)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"영상을 열 수 없습니다: {path.name}")
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(time_sec)) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            # 끝부분/키프레임 문제로 seek 이 실패하면 첫 프레임으로 대체한다.
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError("프레임을 읽을 수 없습니다")
    finally:
        cap.release()
    return fit_frame(frame) if fit else frame


def encode_jpeg(frame, quality: Optional[int] = None) -> bytes:
    q = int(quality if quality is not None else get_settings().frame_jpeg_quality)
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        raise ValueError("JPEG 인코딩 실패")
    return buf.tobytes()

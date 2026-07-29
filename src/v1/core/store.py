"""
core/store.py
=============

워크스페이스 레이아웃과 JSON 저장/조회. 이 서비스의 "DB" 역할.

온프레미스(폐쇄망) 배포를 전제로 별도 DB 를 두지 않는다. 상태 전부가 파일이라
`workspace/` 하나만 백업/마운트하면 이관이 끝나고, 라벨은 사람이 열어볼 수 있는
JSON 이라 파이프라인이 깨져도 손으로 복구할 수 있다.

레이아웃::

    workspace/
    ├── uploads/<video_id><ext>          # 업로드 원본 영상
    ├── videos/<video_id>/
    │   ├── meta.json                    # 원본/fps/해상도/프레임수
    │   ├── segments.json                # 정상/비정상 구간
    │   ├── frames/<frame_id>.jpg        # 추출된 학습 후보 이미지
    │   └── labels/<frame_id>.json       # 프레임 라벨(자동 초안 + 사람 수정)
    ├── datasets/<dataset_id>/
    │   ├── data.yaml                    # ultralytics 학습 입력
    │   ├── manifest.json                # 어떤 영상/프레임으로 만들었는지 추적
    │   └── {train,valid,test}/{images,labels}/
    ├── runs/<run_id>/
    │   ├── spec.json                    # 학습 요청 원본(재현용)
    │   ├── train.log                    # 자식 프로세스 stdout/stderr
    │   └── train/                       # ultralytics 산출물(weights/results.csv/plots)
    ├── models/<alias>.pt                # 승격(promote)된 배포 후보 가중치
    │   └── registry.json
    └── jobs/<job_id>.json               # 완료된 잡 스냅샷(재시작 후에도 조회 가능)
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from core.config import get_settings

# 외부 입력으로 들어온 id 를 경로에 쓰기 전에 검사한다(경로 탈출 방지).
_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def new_id(prefix: str = "") -> str:
    """시간 정렬이 가능한 짧은 id. 파일 목록을 그냥 정렬하면 생성순이 된다."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{prefix}{stamp}-{uuid.uuid4().hex[:6]}" if prefix else f"{stamp}-{uuid.uuid4().hex[:6]}"


def safe_id(value: str) -> str:
    """경로 조각으로 쓸 수 있는 id 인지 검증하고 그대로 반환."""
    v = str(value or "").strip()
    if not _ID_RE.match(v):
        raise ValueError(f"잘못된 id: {value!r}")
    return v


# ── 루트 ──────────────────────────────────────────────────────────────
def workspace() -> Path:
    return get_settings().workspace_dir


def uploads_root() -> Path:
    return get_settings().upload_dir


def videos_root() -> Path:
    return workspace() / "videos"


def datasets_root() -> Path:
    return workspace() / "datasets"


def runs_root() -> Path:
    return workspace() / "runs"


def models_root() -> Path:
    return workspace() / "models"


def jobs_root() -> Path:
    return workspace() / "jobs"


def ensure_workspace() -> None:
    """startup 에서 1회. 쓰기 가능한지도 여기서 드러난다(늦게 터지지 않게)."""
    for p in (uploads_root(), videos_root(), datasets_root(), runs_root(),
              models_root(), jobs_root()):
        p.mkdir(parents=True, exist_ok=True)


# ── 영상 ──────────────────────────────────────────────────────────────
def video_dir(video_id: str) -> Path:
    return videos_root() / safe_id(video_id)


def video_meta_path(video_id: str) -> Path:
    return video_dir(video_id) / "meta.json"


def video_segments_path(video_id: str) -> Path:
    return video_dir(video_id) / "segments.json"


def frames_dir(video_id: str) -> Path:
    return video_dir(video_id) / "frames"


def labels_dir(video_id: str) -> Path:
    return video_dir(video_id) / "labels"


def frame_image_path(video_id: str, frame_id: str) -> Path:
    return frames_dir(video_id) / f"{safe_id(frame_id)}.jpg"


def frame_label_path(video_id: str, frame_id: str) -> Path:
    return labels_dir(video_id) / f"{safe_id(frame_id)}.json"


def list_video_ids() -> list:
    root = videos_root()
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "meta.json").exists())


def list_frame_ids(video_id: str) -> list:
    d = labels_dir(video_id)
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.json"))


# ── 데이터셋 / 학습 / 모델 / 잡 ─────────────────────────────────────────
def dataset_dir(dataset_id: str) -> Path:
    return datasets_root() / safe_id(dataset_id)


def dataset_manifest_path(dataset_id: str) -> Path:
    return dataset_dir(dataset_id) / "manifest.json"


def dataset_yaml_path(dataset_id: str) -> Path:
    return dataset_dir(dataset_id) / "data.yaml"


def list_dataset_ids() -> list:
    root = datasets_root()
    if not root.exists():
        return []
    return sorted((p.name for p in root.iterdir()
                   if p.is_dir() and (p / "manifest.json").exists()), reverse=True)


def run_dir(run_id: str) -> Path:
    return runs_root() / safe_id(run_id)


def run_spec_path(run_id: str) -> Path:
    return run_dir(run_id) / "spec.json"


def run_state_path(run_id: str) -> Path:
    return run_dir(run_id) / "state.json"


def run_log_path(run_id: str) -> Path:
    return run_dir(run_id) / "train.log"


def run_output_dir(run_id: str) -> Path:
    """ultralytics 가 project/name 으로 산출물을 쓰는 위치."""
    return run_dir(run_id) / "train"


def list_run_ids() -> list:
    root = runs_root()
    if not root.exists():
        return []
    return sorted((p.name for p in root.iterdir()
                   if p.is_dir() and (p / "spec.json").exists()), reverse=True)


def registry_path() -> Path:
    return models_root() / "registry.json"


def job_path(job_id: str) -> Path:
    return jobs_root() / f"{safe_id(job_id)}.json"


# ── JSON IO ───────────────────────────────────────────────────────────
def read_json(path: Path, default: Any = None) -> Any:
    try:
        with Path(path).open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_json(path: Path, obj: Any) -> None:
    """같은 디렉터리 임시파일에 쓰고 교체한다.

    라벨링 중 서버가 죽어도 반쯽 쓰인 JSON 이 남지 않게 한다(작업 손실 방지).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=f".{p.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def rel_to_workspace(path: Optional[Path]) -> Optional[str]:
    """응답에 절대경로(서버 내부 구조)를 노출하지 않기 위한 상대화."""
    if path is None:
        return None
    p = Path(path)
    try:
        return p.relative_to(workspace()).as_posix()
    except ValueError:
        return p.as_posix()


def tail_text(path: Path, lines: int) -> str:
    """로그 tail. 수백 MB 로그도 끝부분만 읽는다."""
    p = Path(path)
    if not p.exists():
        return ""
    block = 8192
    try:
        with p.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            buf = b""
            while size > 0 and buf.count(b"\n") <= lines:
                step = min(block, size)
                size -= step
                f.seek(size)
                buf = f.read(step) + buf
        text = buf.decode("utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-lines:])

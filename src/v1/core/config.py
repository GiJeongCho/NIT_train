"""
core/config.py
==============

환경변수 기반 설정. 컨테이너/온프레미스 배포에서 모든 동작을 env 로 조정한다.

설정은 프로세스 시작 시 한 번 읽어 `get_settings()` 싱글톤으로 공유한다.

접두사는 `NIT_TRAIN_` 이다. 추론 서비스(tracker_py)가 `NIT_` 를 쓰므로 같은 호스트/
컨테이너 네트워크에서 두 서비스를 함께 띄울 때 `NIT_PORT` 같은 변수가 충돌하지 않도록
학습 서비스는 별도 접두사를 쓴다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

# 이 패키지(src/v1) 의 절대 경로. 모든 상대 경로의 기준.
BASE_DIR = Path(__file__).resolve().parents[1]
# 프로젝트 루트(NIT_train/). test_model/ 등 앱 밖의 자원 기준.
PROJECT_ROOT = BASE_DIR.parents[1]


def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return v.strip() if v and v.strip() else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on", "y")


def _env_path(name: str, default: Path) -> Path:
    v = os.getenv(name)
    if not v or not v.strip():
        return default
    p = Path(v.strip())
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def _env_list(name: str, default: list) -> list:
    """쉼표 구분 문자열 → 리스트. 빈 값이면 default."""
    v = os.getenv(name)
    if not v or not v.strip():
        return list(default)
    return [x.strip() for x in v.split(",") if x.strip()]


# 자동 라벨 초안의 기본 클래스 집합.
# tracker_py 의 OBB 학습 데이터셋(train_data/preprocessed_obb/data.yaml)과 동일한 순서를
# 기본값으로 둔다. 클래스 인덱스가 어긋나면 기존 데이터셋과 합칠 수 없기 때문이다.
DEFAULT_CLASS_NAMES = [
    "Ikv_91_105",
    "Jagdtiger",
    "Panther_II",
    "Strv_101",
    "Strv_103A",
    "Tiger_II_10.5_",
    "VIDAR",
]

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}


@dataclass
class Settings:
    # ── 서버 ──────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8888

    # ── 저장 루트 ─────────────────────────────────────────────────────
    # 워크스페이스 하나에 영상·프레임·라벨·데이터셋·학습결과를 모두 담는다.
    # 도커에서는 이 경로 하나만 볼륨으로 마운트하면 전체 상태가 보존된다.
    workspace_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "workspace")
    # 업로드 원본 영상(워크스페이스 안). 대용량이므로 분리 마운트도 가능.
    upload_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "workspace" / "uploads")
    # 사전학습 가중치 보관 폴더(기본 모델이 여기 있다).
    model_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "test_model")
    # 학습/자동라벨의 기본 가중치. 사용자가 별도 지정하지 않으면 항상 이 파일.
    #
    # OBB 가중치를 기본으로 쓴다. 목표 산출물이 `train_data/preprocessed_obb` 와 같은
    # 회전박스 데이터셋이고, 축정렬 모델(yolo26l.pt)로는 회전 초안을 만들 수 없다.
    # 사전학습 클래스도 DOTA 항공영상 계열이라 드론 영상에 훨씬 잘 맞는다.
    # 축정렬로 가려면 NIT_TRAIN_BASE_MODEL=yolo26l.pt + NIT_TRAIN_TASK=detect.
    base_model: Path = field(default_factory=lambda: PROJECT_ROOT / "test_model" / "yolo26l-obb.pt")

    # ── 추론(자동 라벨 초안) ───────────────────────────────────────────
    device: str = "0"                 # CUDA 인덱스 또는 "cpu"
    autolabel_conf: float = 0.25      # 초안은 사람이 지우는 게 추가보다 싸므로 낮게(재현율 우선)
    autolabel_iou: float = 0.7
    autolabel_imgsz: int = 640
    autolabel_max_det: int = 300
    # 트래킹으로 track_id 를 붙인다. 같은 객체가 프레임마다 같은 id 를 받으므로
    # 프런트에서 "이 객체의 클래스"를 한 번 고치면 전체 프레임에 전파할 수 있다.
    autolabel_track: bool = True
    autolabel_tracker: str = "botsort.yaml"
    # startup 에서 기본 가중치를 미리 로드/워밍업할지. 끄면(기본) 첫 추출 작업에서
    # 로드한다. 학습만 쓰는 세션에서 수백 MB 가중치를 헛되게 올리지 않기 위함.
    preload_model: bool = False

    # ── 프레임 추출 ────────────────────────────────────────────────────
    extract_fps: float = 2.0          # 구간에서 초당 몇 장 뽑을지 (0=모든 프레임)
    extract_max_frames: int = 20000    # 안전장치(디스크/라벨링 부하 폭주 방지)
    frame_jpeg_quality: int = 92
    # 운용 스펙(드론 입력 640x480)에 맞춰 프레임을 정규화해 저장한다.
    # 추론 때와 같은 해상도 분포로 학습해야 실전 성능이 맞는다.
    frame_resize: bool = True
    frame_width: int = 640
    frame_height: int = 480

    # ── 데이터셋 ───────────────────────────────────────────────────────
    class_names: list = field(default_factory=lambda: list(DEFAULT_CLASS_NAMES))
    # 기본 태스크. obb = 회전박스(8좌표), detect = 축정렬(cxcywh).
    # `tracker_py/train_data/preprocessed_obb` 와 같은 구조가 목표 산출물이므로 obb.
    dataset_task: str = "obb"
    split_train: float = 0.8
    split_valid: float = 0.15
    split_test: float = 0.05
    # 인접 프레임은 거의 같은 그림이다. 무작위로 나누면 train/valid 에 사실상 같은
    # 이미지가 들어가 검증 점수가 부풀려진다(leakage). 기본은 연속 프레임을
    # 블록 단위로 묶어 배분한다.
    split_mode: str = "chunk"         # chunk | random | video
    split_chunk_size: int = 30
    split_seed: int = 0
    # 데이터셋에 넣을 라벨 상태. 기본은 사람이 승인한 프레임만.
    dataset_only_approved: bool = True
    # 데이터셋 이미지를 복사하지 않고 하드링크로 연결(디스크 절약). 실패 시 복사 폴백.
    dataset_link_images: bool = True

    # ── 학습 기본값 (test/yolo.py 의 검증된 값) ─────────────────────────
    train_epochs: int = 100
    train_imgsz: int = 640
    train_batch: int = 16
    # Windows spawn 은 워커마다 RAM 을 크게 물어 후반 epoch 에서 DataLoader 가 죽는다.
    # 8 이상은 쓰지 않는다(test/yolo.py 주석의 실측 경험).
    train_workers: int = 4
    train_patience: int = 50
    # 학습은 API 프로세스와 분리된 자식 프로세스에서 돌린다. GPU 메모리 누수·크래시가
    # API 를 같이 죽이지 않고, 중단(stop)도 프로세스 종료로 확실하게 된다.
    train_python: str = ""            # "" = 현재 인터프리터(sys.executable)

    # ── tracker_py(추론 서비스) 재사용 ─────────────────────────────────
    # 자동 라벨은 기본적으로 이 프로세스 안에서 직접 추론한다(프레임 수천 장을
    # HTTP 로 왕복하면 느리다). 다만 학습 결과를 운영 추론 서비스에 반영하거나
    # 최신 탐지 결과를 가져올 때 이 주소를 쓴다.
    tracker_api_url: str = "http://127.0.0.1:8886"
    # 승격(promote) 시 가중치를 복사해 넣을 추론 서비스의 models 폴더. 비어 있으면 승격만 하고 배포 안 함.
    tracker_models_dir: Path = field(default_factory=lambda: Path(""))

    # ── 로깅 ──────────────────────────────────────────────────────────
    job_log_tail_lines: int = 200      # 로그 조회 시 기본 tail 줄 수

    @classmethod
    def from_env(cls) -> "Settings":
        workspace = _env_path("NIT_TRAIN_WORKSPACE", PROJECT_ROOT / "workspace")
        model_dir = _env_path("NIT_TRAIN_MODEL_DIR", PROJECT_ROOT / "test_model")
        return cls(
            host=_env_str("NIT_TRAIN_HOST", "0.0.0.0"),
            port=_env_int("NIT_TRAIN_PORT", 8888),
            workspace_dir=workspace,
            upload_dir=_env_path("NIT_TRAIN_UPLOAD_DIR", workspace / "uploads"),
            model_dir=model_dir,
            base_model=_env_path("NIT_TRAIN_BASE_MODEL", model_dir / "yolo26l-obb.pt"),
            device=_env_str("NIT_TRAIN_DEVICE", "0"),
            autolabel_conf=_env_float("NIT_TRAIN_AUTOLABEL_CONF", 0.25),
            autolabel_iou=_env_float("NIT_TRAIN_AUTOLABEL_IOU", 0.7),
            autolabel_imgsz=_env_int("NIT_TRAIN_AUTOLABEL_IMGSZ", 640),
            autolabel_max_det=_env_int("NIT_TRAIN_AUTOLABEL_MAX_DET", 300),
            autolabel_track=_env_bool("NIT_TRAIN_AUTOLABEL_TRACK", True),
            autolabel_tracker=_env_str("NIT_TRAIN_AUTOLABEL_TRACKER", "botsort.yaml"),
            preload_model=_env_bool("NIT_TRAIN_PRELOAD", False),
            extract_fps=_env_float("NIT_TRAIN_EXTRACT_FPS", 2.0),
            extract_max_frames=_env_int("NIT_TRAIN_EXTRACT_MAX_FRAMES", 20000),
            frame_jpeg_quality=max(50, min(100, _env_int("NIT_TRAIN_FRAME_JPEG_QUALITY", 92))),
            frame_resize=_env_bool("NIT_TRAIN_FRAME_RESIZE", True),
            frame_width=_env_int("NIT_TRAIN_FRAME_WIDTH", 640),
            frame_height=_env_int("NIT_TRAIN_FRAME_HEIGHT", 480),
            class_names=_env_list("NIT_TRAIN_CLASS_NAMES", DEFAULT_CLASS_NAMES),
            dataset_task=_env_str("NIT_TRAIN_TASK", "obb").lower(),
            split_train=_env_float("NIT_TRAIN_SPLIT_TRAIN", 0.8),
            split_valid=_env_float("NIT_TRAIN_SPLIT_VALID", 0.15),
            split_test=_env_float("NIT_TRAIN_SPLIT_TEST", 0.05),
            split_mode=_env_str("NIT_TRAIN_SPLIT_MODE", "chunk"),
            split_chunk_size=max(1, _env_int("NIT_TRAIN_SPLIT_CHUNK_SIZE", 30)),
            split_seed=_env_int("NIT_TRAIN_SPLIT_SEED", 0),
            dataset_only_approved=_env_bool("NIT_TRAIN_ONLY_APPROVED", True),
            dataset_link_images=_env_bool("NIT_TRAIN_LINK_IMAGES", True),
            train_epochs=_env_int("NIT_TRAIN_EPOCHS", 100),
            train_imgsz=_env_int("NIT_TRAIN_IMGSZ", 640),
            train_batch=_env_int("NIT_TRAIN_BATCH", 16),
            train_workers=_env_int("NIT_TRAIN_WORKERS", 4),
            train_patience=_env_int("NIT_TRAIN_PATIENCE", 50),
            train_python=_env_str("NIT_TRAIN_PYTHON", ""),
            tracker_api_url=_env_str("NIT_TRAIN_TRACKER_API", "http://127.0.0.1:8886"),
            tracker_models_dir=_env_path("NIT_TRAIN_TRACKER_MODELS_DIR", Path("")),
            job_log_tail_lines=_env_int("NIT_TRAIN_LOG_TAIL", 200),
        )

    def splits(self) -> dict:
        """train/valid/test 비율을 합이 1 이 되도록 정규화해 반환."""
        raw = {
            "train": max(0.0, self.split_train),
            "valid": max(0.0, self.split_valid),
            "test": max(0.0, self.split_test),
        }
        total = sum(raw.values())
        if total <= 0:
            return {"train": 1.0, "valid": 0.0, "test": 0.0}
        return {k: v / total for k, v in raw.items()}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()

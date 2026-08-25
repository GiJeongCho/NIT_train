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
    #
    # tracker_py 프로덕션 OBB 모델(7종 전차 학습 완료)을 기본으로 쓴다. 일반 사전학습
    # (yolo26l-obb.pt / DOTA)은 전차 클래스를 모르므로 자동 라벨이 초반만 잡히거나
    # 비어 있게 된다. 이 모델은 tracker_py 와 동일한 추론 결과를 재사용하기 위한 것이며,
    # 같은 도메인 파인튜닝의 출발점으로도 이상적이다.
    # 일반 사전학습으로 되돌리려면 NIT_TRAIN_BASE_MODEL=yolo26l-obb.pt (축정렬은 yolo26l.pt).
    base_model: Path = field(default_factory=lambda: PROJECT_ROOT / "test_model" / "tracker_obb_best.pt")

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
    # 트래킹 엔진 선택. True(기본)면 추론 서비스(tracker_py)에서 이식한 커스텀 칼만
    # 트래커(services/kalman_tracker.py)로 track_id 를 붙인다 — 운영 추론과 같은 규칙.
    # False 면 ultralytics 내장 트래커(autolabel_tracker=botsort.yaml)로 폴백한다.
    autolabel_custom_tracker: bool = True
    # 커스텀 트래커가 track_id 를 '확정'해 내보내기까지 필요한 연속 탐지 프레임 수.
    # 자동 라벨/미리보기는 초당 몇 장만 뽑는(저 fps) 샘플링이라, 기본 2 로 두면 확정 전에
    # 창이 끝나 ID 가 안 붙어 보인다. 1 로 낮춰 두 번째 프레임부터 ID 가 보이게 한다.
    autolabel_track_min_hits: int = 1
    # 커스텀 트래커의 '고스트(예측) 트랙' 을 라벨로도 내보낼지. True 면 이번 프레임에 YOLO
    # 탐지가 빠져도 칼만 예측 박스를 track_id 와 함께 그대로 내보낸다 → 깜빡임(보였다 안 보였다)
    # 이 사라지고 트래킹이 연속돼 보인다. 예측 박스는 objects[].is_predicted=True 로 표시한다.
    autolabel_emit_ghost: bool = True
    # 고스트(예측만으로 유지)로 보여줄 최대 프레임 수. 트래커 기본값(100)은 30fps 연속 영상
    # 기준이라 저 fps 샘플링(2fps)에서는 사라진 표적이 수십 초 남는다. 저 fps 에 맞춰 짧게 둔다.
    autolabel_track_max_age: int = 3
    # startup 에서 기본 가중치를 미리 로드/워밍업할지. 끄면(기본) 첫 추출 작업에서
    # 로드한다. 학습만 쓰는 세션에서 수백 MB 가중치를 헛되게 올리지 않기 위함.
    preload_model: bool = False

    # ── 프레임 추출 ────────────────────────────────────────────────────
    extract_fps: float = 2.0          # 구간에서 초당 몇 장 뽑을지 (0=모든 프레임)
    extract_max_frames: int = 20000    # 안전장치(디스크/라벨링 부하 폭주 방지)
    frame_jpeg_quality: int = 92
    # 운용 스펙(드론 입력 640x480)에 맞춰 프레임을 정규화해 저장한다.
    # 추론 때와 같은 해상도 분포로 학습해야 실전 성능이 맞는다.
    # 다운스케일이 싫은 경우 추출 요청/영상별 설정으로 끌 수 있다(선택).
    frame_resize: bool = True
    frame_width: int = 640
    frame_height: int = 480

    # ── 전처리 (추론 서비스 tracker_py 와 동일한 엔진/기본값) ─────────────
    # 학습 이미지는 추론 입력과 똑같이 전처리돼야 분포가 맞으므로, tracker_py 의 전처리를
    # 그대로 떼어 온 preprocess_vendor 엔진을 쓴다(services/preprocess.py).
    #
    # 세 전처리(야간 보정/안개 제거/CLAHE)를 서로 독립적으로 켠다. `auto` 를 켜면
    # 프레임마다 tracker_py 와 동일한 다중지표(밝기·대비·채도·선명도)로 저조도/안개를
    # 판정해 해당 프레임에만 적용한다. 끄면 켜 둔 전처리를 모든 프레임에 강제 적용한다.
    preprocess_auto: bool = True        # 프레임별 자동 판정(끄면 강제 적용)
    preprocess_lowlight: bool = True    # 야간 보정(저조도) 기본 ON (tracker_py dark_enabled=True)
    preprocess_dehaze: bool = True      # 안개 제거(dehaze) 기본 ON (tracker_py fog_enabled=True)
    preprocess_clahe: bool = True       # 화질 향상 CLAHE(추론 Stage2) 기본 ON
    # auto 판정의 밝기 임계(0~255). tracker_py dark_th 기본값과 동일.
    daynight_threshold: float = 60.0
    # 야간 보정 '폴백'(Zero-DCE++ 가중치가 없을 때) 파라미터. 감마가 '밝기' 축이다.
    night_clahe_clip: float = 3.0
    night_clahe_grid: int = 8
    night_gamma: float = 1.6
    # 안개 제거(DCP) 파라미터 — tracker_py config_yolo.yaml 의 fog 기본값과 동일.
    dehaze_omega: float = 0.80
    dehaze_t0: float = 0.4
    dehaze_wsz: int = 15
    dehaze_scale: float = 0.25
    dehaze_guide_r: int = 20
    # 화질 향상 CLAHE(quality) 파라미터 — tracker_py 의 clahe_lab 기본값과 동일.
    quality_clahe_clip: float = 2.0
    quality_clahe_grid: int = 8
    # 표적 강조(Stage3, tracker_py emphasis). 소형 표적의 고주파(윤곽)를 언샤프 마스크로 증폭.
    # 주의: 이 전처리는 '전처리 정합 학습'(학습·추론 동일 체인)을 전제로 한다. 기존 가중치가
    # 표적강조 없이 학습됐다면 켜는 순간 도메인 격차가 생겨 검출이 흔들릴 수 있으니 기본 OFF(옵트인).
    preprocess_emphasis: bool = False   # 표적 강조(Stage3) 기본 OFF
    emphasis_sigma: float = 1.0         # 언샤프 가우시안 시그마(클수록 넓은 윤곽 강조)
    emphasis_alpha: float = 0.5         # 강조 세기(클수록 또렷/과하면 노이즈·링잉)

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
            base_model=_env_path("NIT_TRAIN_BASE_MODEL", model_dir / "tracker_obb_best.pt"),
            device=_env_str("NIT_TRAIN_DEVICE", "0"),
            autolabel_conf=_env_float("NIT_TRAIN_AUTOLABEL_CONF", 0.25),
            autolabel_iou=_env_float("NIT_TRAIN_AUTOLABEL_IOU", 0.7),
            autolabel_imgsz=_env_int("NIT_TRAIN_AUTOLABEL_IMGSZ", 640),
            autolabel_max_det=_env_int("NIT_TRAIN_AUTOLABEL_MAX_DET", 300),
            autolabel_track=_env_bool("NIT_TRAIN_AUTOLABEL_TRACK", True),
            autolabel_tracker=_env_str("NIT_TRAIN_AUTOLABEL_TRACKER", "botsort.yaml"),
            autolabel_custom_tracker=_env_bool("NIT_TRAIN_CUSTOM_TRACKER", True),
            autolabel_track_min_hits=max(1, _env_int("NIT_TRAIN_TRACK_MIN_HITS", 1)),
            autolabel_emit_ghost=_env_bool("NIT_TRAIN_EMIT_GHOST", True),
            autolabel_track_max_age=max(1, _env_int("NIT_TRAIN_TRACK_MAX_AGE", 3)),
            preload_model=_env_bool("NIT_TRAIN_PRELOAD", False),
            extract_fps=_env_float("NIT_TRAIN_EXTRACT_FPS", 2.0),
            extract_max_frames=_env_int("NIT_TRAIN_EXTRACT_MAX_FRAMES", 20000),
            frame_jpeg_quality=max(50, min(100, _env_int("NIT_TRAIN_FRAME_JPEG_QUALITY", 92))),
            frame_resize=_env_bool("NIT_TRAIN_FRAME_RESIZE", True),
            frame_width=_env_int("NIT_TRAIN_FRAME_WIDTH", 640),
            frame_height=_env_int("NIT_TRAIN_FRAME_HEIGHT", 480),
            preprocess_auto=_env_bool("NIT_TRAIN_PREPROCESS_AUTO", True),
            preprocess_lowlight=_env_bool("NIT_TRAIN_LOWLIGHT", True),
            preprocess_dehaze=_env_bool("NIT_TRAIN_DEHAZE", True),
            preprocess_clahe=_env_bool("NIT_TRAIN_CLAHE", True),
            daynight_threshold=_env_float("NIT_TRAIN_DAYNIGHT_THRESHOLD", 60.0),
            night_clahe_clip=_env_float("NIT_TRAIN_NIGHT_CLAHE_CLIP", 3.0),
            night_clahe_grid=_env_int("NIT_TRAIN_NIGHT_CLAHE_GRID", 8),
            night_gamma=_env_float("NIT_TRAIN_NIGHT_GAMMA", 1.6),
            dehaze_omega=_env_float("NIT_TRAIN_DEHAZE_OMEGA", 0.80),
            dehaze_t0=_env_float("NIT_TRAIN_DEHAZE_T0", 0.4),
            dehaze_wsz=_env_int("NIT_TRAIN_DEHAZE_WSZ", 15),
            dehaze_scale=_env_float("NIT_TRAIN_DEHAZE_SCALE", 0.25),
            dehaze_guide_r=_env_int("NIT_TRAIN_DEHAZE_GUIDE_R", 20),
            quality_clahe_clip=_env_float("NIT_TRAIN_QUALITY_CLAHE_CLIP", 2.0),
            quality_clahe_grid=_env_int("NIT_TRAIN_QUALITY_CLAHE_GRID", 8),
            preprocess_emphasis=_env_bool("NIT_TRAIN_EMPHASIS", False),
            emphasis_sigma=_env_float("NIT_TRAIN_EMPHASIS_SIGMA", 1.0),
            emphasis_alpha=_env_float("NIT_TRAIN_EMPHASIS_ALPHA", 0.5),
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

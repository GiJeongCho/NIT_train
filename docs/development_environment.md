# 개발환경 (Development Environment)

> 대상: `NIT_train/` (FastAPI 학습·라벨링 서비스)
> 검증 핀: Python 3.10.20 / torch 2.11.0(cu128) / ultralytics 8.4.37 / fastapi 0.119 / pydantic 2.13
> 작성일: 2026-07-29

---

## 1. 사전 요구사항

| 항목 | 권장 |
|---|---|
| OS | Windows 10/11 (개발) · Linux x86_64 (배포) |
| GPU | NVIDIA CUDA 지원 GPU (Blackwell 계열은 cu128) |
| 패키지 매니저 | Anaconda / Miniconda |
| Docker (선택) | NVIDIA Container Toolkit (`--gpus all`) |

GPU가 없으면 `NIT_TRAIN_DEVICE=cpu` 로 동작은 하지만 자동 라벨과 학습이 실용 속도가
아니다(개발/디버깅용).

---

## 2. 로컬(conda) 셋업

현재는 추론 서비스와 같은 `NIT` 환경을 그대로 쓴다. 이미 `torch`/`ultralytics`/`opencv` 가
들어 있어 추가 설치가 거의 없다.

```powershell
conda activate NIT
pip install -r requirements.txt
```

> **정리 계획**: 이 서비스는 전처리(Zero-DCE / AOD-Net / wavelet)와 TensorRT가 필요 없다.
> 배포 시에는 `requirements.txt` 만으로 만든 가벼운 전용 환경(또는 도커 이미지)을 쓴다.
> `requirements.txt` 는 그 최소 목록으로 유지한다.

### 실행

```powershell
conda activate NIT
cd C:\project\NIT_train\src\v1     # ★ 반드시 src/v1 에서 실행
python main.py
```

또는 uvicorn 직접:

```powershell
uvicorn app:app --host 0.0.0.0 --port 8888 --reload
```

CWD와 무관하게 띄우려면:

```powershell
python C:\project\NIT_train\src\api.py
```

핫리로드는 `NIT_TRAIN_RELOAD=1` 또는 `--reload`.

### 빠른 점검

```powershell
curl http://localhost:8888/healthz     # {"status":"ok"}
curl http://localhost:8888/api/meta    # 기본값/클래스 목록
# 브라우저: http://localhost:8888/docs  (Swagger)
```

---

## 3. 스모크 테스트

파이프라인 전 구간(업로드 → 구간 → 자동 라벨 → 검수 → 데이터셋 → 학습)을 임시
워크스페이스에서 한 번 태워본다. 라벨 좌표 변환·분할·파일 배치처럼 조용히 틀리기 쉬운
부분을 배포 전에 잡는 것이 목적이다.

```powershell
conda activate NIT
cd C:\project\NIT_train\test
$env:PYTHONIOENCODING="utf-8"
python smoke_api.py            # 학습 제외 (약 10초)
python smoke_api.py --train    # 1 epoch 학습 + 모델 승격까지 (GPU 필요, 약 30초)
```

실제 워크스페이스를 건드리지 않고 임시 디렉터리에서 돌며, 끝나면 지운다(`--keep` 로 보존).

---

## 4. 의존성

`requirements.txt` — 이 앱이 직접 쓰는 것만.

```
fastapi / uvicorn[standard] / python-multipart / pydantic
ultralytics / opencv-python / numpy / lap / psutil
```

| 패키지 | 왜 필요한가 |
|---|---|
| `ultralytics` | YOLO 추론(자동 라벨) + 학습 |
| `opencv-python` | 영상 디코딩, 프레임 추출/리사이즈, JPEG 인코딩 |
| `lap` | ultralytics 트래커(BoT-SORT)의 선형 할당. **없으면 런타임에 pip 설치를 시도해 폐쇄망에서 실패한다** |
| `psutil` | 학습 자식 프로세스 생존 확인 |
| `python-multipart` | 영상 업로드(multipart) |

`torch` / `torchvision` 은 GPU 스택이라 여기에 넣지 않는다. conda 환경 또는 도커 base
이미지가 제공한다(추론 서비스 `setup_env.sh` 와 같은 정책).

테스트 전용: `httpx` (FastAPI `TestClient` 가 요구).

---

## 5. Docker

```bash
cd NIT_train
docker build -t nit-train .

docker run --gpus all -p 8888:8888 \
  -v /data/nit-train-workspace:/app/workspace \
  -v /data/models:/app/test_model \
  nit-train
```

compose:

```bash
docker compose up --build
```

- base 이미지 `ultralytics/ultralytics:latest` 가 torch/CUDA/opencv/ultralytics 를 제공하므로
  빌드는 앱 추가 의존성만 설치한다.
- **`workspace/` 볼륨 하나에 모든 상태(영상·라벨·데이터셋·학습결과)가 들어간다.**
  이것만 마운트하면 컨테이너를 갈아끼워도 작업이 보존된다.
- 학습 산출물이 커지므로 볼륨 용량을 넉넉히 잡는다(영상 1편당 프레임 수천 장 + 가중치 수십 MB).

---

## 6. 환경변수

전체는 `src/v1/core/config.py` 참고. 접두사는 `NIT_TRAIN_` 이다.

> 추론 서비스(tracker_py)가 `NIT_` 를 쓰므로 같은 호스트에서 함께 띄울 때
> `NIT_PORT` 같은 변수가 충돌하지 않도록 접두사를 분리했다.

### 서버 · 저장소

| 변수 | 기본값 | 설명 |
|---|---|---|
| `NIT_TRAIN_HOST` / `NIT_TRAIN_PORT` | `0.0.0.0` / `8888` | 바인드 주소 |
| `NIT_TRAIN_RELOAD` | `0` | uvicorn 핫리로드 |
| `NIT_TRAIN_WORKSPACE` | `<프로젝트>/workspace` | **모든 상태의 루트.** 도커 볼륨 대상 |
| `NIT_TRAIN_UPLOAD_DIR` | `<workspace>/uploads` | 업로드 원본. 대용량이면 별도 마운트 |
| `NIT_TRAIN_MODEL_DIR` | `<프로젝트>/test_model` | 사전학습 가중치 폴더 |
| `NIT_TRAIN_BASE_MODEL` | `<model_dir>/yolo26l-obb.pt` | 자동 라벨·학습 기본 가중치(회전박스) |

### 추론 (자동 라벨)

| 변수 | 기본값 | 설명 |
|---|---|---|
| `NIT_TRAIN_DEVICE` | `0` | CUDA 인덱스 또는 `cpu` |
| `NIT_TRAIN_AUTOLABEL_CONF` | `0.25` | 초안 신뢰도 임계. **낮을수록 많이 제안**(재현율 우선) |
| `NIT_TRAIN_AUTOLABEL_IOU` | `0.7` | NMS 임계 |
| `NIT_TRAIN_AUTOLABEL_IMGSZ` | `640` | 추론 해상도 |
| `NIT_TRAIN_AUTOLABEL_MAX_DET` | `300` | 프레임당 최대 탐지 수 |
| `NIT_TRAIN_AUTOLABEL_TRACK` | `1` | `track_id` 부여(클래스 일괄 전파용) |
| `NIT_TRAIN_AUTOLABEL_TRACKER` | `botsort.yaml` | ultralytics 트래커 설정 |
| `NIT_TRAIN_PRELOAD` | `0` | startup에서 기본 가중치 미리 로드. 학습만 쓰면 끄는 게 이득 |

### 프레임 추출

| 변수 | 기본값 | 설명 |
|---|---|---|
| `NIT_TRAIN_EXTRACT_FPS` | `2.0` | 초당 추출 장수 (0=모든 프레임) |
| `NIT_TRAIN_EXTRACT_MAX_FRAMES` | `20000` | 1회 추출 상한(디스크/검수 폭주 방지) |
| `NIT_TRAIN_FRAME_RESIZE` | `1` | 운용 스펙 해상도로 정규화(0=원본 유지). 추출 요청/영상별로도 끌 수 있음 |
| `NIT_TRAIN_FRAME_WIDTH` / `_HEIGHT` | `640` / `480` | 정규화 해상도(추론 서비스와 동일) |
| `NIT_TRAIN_FRAME_JPEG_QUALITY` | `92` | 저장 화질. 낮추면 디스크는 줄지만 작은 표적이 뭉갠다 |

### 전처리 (추론 tracker_py 와 동일 엔진)

학습 이미지는 **추론 입력과 똑같이** 전처리돼야 분포가 맞으므로, 추론 서비스(tracker_py)의
전처리를 그대로 떼어 온 `preprocess_vendor` 엔진을 쓴다(`src/v1/preprocess_vendor/`). 추론과
같은 순서로 적용한다: **야간 보정(dark) → 안개 제거(fog) → CLAHE(quality)**.

세 전처리는 **서로 독립** 스위치다. `NIT_TRAIN_PREPROCESS_AUTO=1`(자동)이면 프레임마다
tracker_py 와 동일한 다중지표(밝기·대비·채도·선명도)로 저조도/안개를 판정해 해당 프레임에만
적용하고, `0`(강제)이면 켠 전처리를 모든 프레임에 적용한다. 자동/설정이 맞지 않는 영상은
'구간' 화면에서 영상별로 조정해 저장한다(`PUT /api/videos/{id}/preprocess`). 추출 요청이 값을
명시하면 그것이 최우선이다.

야간 보정(dark)은 **Zero-DCE++ 가중치가 있으면**(추론과 동일) 그것을, **없으면 감마+CLAHE
폴백**을 쓴다. 완전 일치가 필요하면 `src/v1/preprocess_vendor/zero_dce_weights/Epoch99.pth` 를
넣거나 `NIT_TRAIN_ZERODCE_WEIGHTS` 로 절대경로를 지정한다.

| 변수 | 기본값 | 설명 |
|---|---|---|
| `NIT_TRAIN_PREPROCESS_AUTO` | `1` | 프레임별 자동 판정(0=켠 전처리를 전 프레임 강제 적용) |
| `NIT_TRAIN_LOWLIGHT` | `1` | 야간 보정(저조도) 적용 여부 (tracker_py `dark_enabled` 기본과 동일) |
| `NIT_TRAIN_DEHAZE` | `1` | 안개 제거(dehaze, DCP) 적용 여부 (tracker_py `fog_enabled` 기본과 동일) |
| `NIT_TRAIN_CLAHE` | `1` | 화질 향상 CLAHE(추론 Stage2) 적용 여부 |
| `NIT_TRAIN_DAYNIGHT_THRESHOLD` | `60.0` | auto 판정 밝기 임계(tracker_py `dark_th` 와 동일) |
| `NIT_TRAIN_ZERODCE_WEIGHTS` | 번들 `Epoch99.pth` | Zero-DCE++ 가중치 경로. 기본은 `preprocess_vendor/zero_dce_weights/Epoch99.pth`(리포 포함) → 야간 보정을 추론과 동일하게 수행. CPU 로도 동작 |
| `NIT_TRAIN_NIGHT_GAMMA` | `1.6` | 야간 밝기 감마(**가중치 없을 때 폴백에서만** 사용, >1 이면 밝게) |
| `NIT_TRAIN_NIGHT_CLAHE_CLIP` | `3.0` | 폴백 야간 CLAHE 국소 대비 한계 |
| `NIT_TRAIN_NIGHT_CLAHE_GRID` | `8` | 폴백 야간 CLAHE 타일 그리드(NxN) |
| `NIT_TRAIN_DEHAZE_OMEGA` | `0.80` | 안개 제거 강도(tracker_py fog 기본값과 동일) |
| `NIT_TRAIN_DEHAZE_T0` | `0.4` | 투과율 하한(하늘 영역 과보정 방지) |
| `NIT_TRAIN_DEHAZE_WSZ` | `15` | dark channel 패치 크기 |
| `NIT_TRAIN_DEHAZE_SCALE` | `0.25` | 투과율 추정 다운스케일 비율 |
| `NIT_TRAIN_DEHAZE_GUIDE_R` | `20` | guided filter 반경 |
| `NIT_TRAIN_QUALITY_CLAHE_CLIP` | `2.0` | CLAHE(quality) 대비 한계(tracker_py 와 동일) |
| `NIT_TRAIN_QUALITY_CLAHE_GRID` | `8` | CLAHE(quality) 타일 그리드(NxN) |

### 데이터셋

| 변수 | 기본값 | 설명 |
|---|---|---|
| `NIT_TRAIN_CLASS_NAMES` | tracker_py OBB 7종 | 초기 클래스 목록(쉼표 구분). 이후는 `workspace/classes.json` 이 우선 |
| `NIT_TRAIN_TASK` | `obb` | 데이터셋 기본 태스크. `detect` 로 바꿀 때는 `BASE_MODEL` 도 함께 바꾼다 |
| `NIT_TRAIN_SPLIT_TRAIN` / `_VALID` / `_TEST` | `0.8` / `0.15` / `0.05` | 분할 비율 |
| `NIT_TRAIN_SPLIT_MODE` | `chunk` | `chunk`\|`random`\|`video` (누수 방지) |
| `NIT_TRAIN_SPLIT_CHUNK_SIZE` | `30` | chunk 모드 블록 크기 |
| `NIT_TRAIN_SPLIT_SEED` | `0` | 분할 재현용 |
| `NIT_TRAIN_ONLY_APPROVED` | `1` | 승인된 프레임만 데이터셋에 포함 |
| `NIT_TRAIN_LINK_IMAGES` | `1` | 하드링크로 디스크 절약(실패 시 복사 폴백) |

### 학습

| 변수 | 기본값 | 설명 |
|---|---|---|
| `NIT_TRAIN_EPOCHS` | `100` | |
| `NIT_TRAIN_IMGSZ` | `640` | |
| `NIT_TRAIN_BATCH` | `16` | |
| `NIT_TRAIN_WORKERS` | `4` | **Windows는 4 이하.** 워커마다 RAM을 크게 물어 후반 epoch에서 DataLoader가 죽는다(ArrayMemoryError). 재발하면 2 |
| `NIT_TRAIN_PATIENCE` | `50` | 조기 종료 |
| `NIT_TRAIN_PYTHON` | `""` | 학습에 쓸 인터프리터. 비우면 API와 같은 환경 |

### 연동

| 변수 | 기본값 | 설명 |
|---|---|---|
| `NIT_TRAIN_TRACKER_API` | `http://127.0.0.1:8886` | 추론 서비스 주소 |
| `NIT_TRAIN_TRACKER_MODELS_DIR` | `""` | 승격 모델을 배포할 추론 서비스 `models/` 폴더. 비우면 배포 비활성 |
| `NIT_TRAIN_LOG_TAIL` | `200` | 로그 조회 기본 줄 수 |

---

## 7. 두 서비스를 함께 띄울 때

| 서비스 | 포트 | 역할 |
|---|---|---|
| tracker_py | 8886 | 운영 실시간 추론 |
| NIT_train | 8888 | 학습·라벨링 |
| NIT_train_front | (별도) | 라벨링 UI |

같은 GPU를 쓰면 **학습과 실시간 추론이 경합**한다. 실시간 처리 fps가 떨어지므로 운영 중에는
학습을 돌리지 않거나, `NIT_TRAIN_DEVICE` 로 다른 GPU를 지정한다.

모델 배포 흐름:

```bash
# 1) 승격 + 추론 서비스 models 폴더로 복사
curl -X POST http://localhost:8888/api/models/promote \
  -d '{"run_id":"...","alias":"drone-v3","deploy":true}'

# 2) 추론 서비스에서 재시작 없이 교체
curl -X POST http://localhost:8886/api/detector/model \
  -d '{"path":"models/drone-v3.pt"}'
```

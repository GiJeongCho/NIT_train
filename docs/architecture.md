# 아키텍처 (Architecture)

> 대상: `NIT_train/src/v1/` 학습·라벨링 API
> 작성일: 2026-07-29

드론(운용상 CCTV 계열) 영상 한 편을 넣으면, 사람이 최소한만 손대고 **학습된 모델**까지
나오는 온프레미스 MLOps 파이프라인이다. 폐쇄망 배포를 전제로 외부 서비스·별도 DB에
의존하지 않는다.

---

## 1. 전체 흐름

```
[NIT_train_front]  브라우저 UI
        │  HTTP (CORS)
        ▼
┌─────────────────────────── NIT_train API (:8888) ───────────────────────────┐
│                                                                             │
│  1. 영상 등록      POST /api/videos            영상 + fps/해상도/길이         │
│         │                                                                   │
│  2. 구간 지정      PUT  /api/videos/{id}/segments                            │
│         │          정상 구간(여러 개) / 비정상 구간(여러 개)                  │
│         │          → 실제 대상 = (정상 합집합) − (비정상 합집합)              │
│         ▼                                                                   │
│  3. 추출+자동라벨  POST /api/videos/{id}/extract        [백그라운드 잡]       │
│         │          구간에서 초당 N장 → 640x480 정규화 → JPEG 저장            │
│         │          → YOLO 추론(+트래킹) → 라벨 초안 JSON                     │
│         ▼                                                                   │
│  4. 사람 검수      GET/PUT /api/videos/{id}/frames/{fid}                     │
│         │          POST .../propagate  ← track_id 로 클래스 일괄 적용        │
│         │          POST .../frames/status ← 일괄 승인                        │
│         ▼                                                                   │
│  5. 데이터셋       POST /api/datasets                   [백그라운드 잡]       │
│         │          승인된 라벨만 모아 YOLO 데이터셋 스냅샷 + data.yaml        │
│         │          POST /api/datasets/import ← 기존 train_data 폴더 등록      │
│         │          base_datasets ← 기존 데이터셋과 병합(클래스 이름 매핑)      │
│         ▼                                                                   │
│  6. 학습           POST /api/train                      [자식 프로세스]       │
│         │          ultralytics train → runs/<run_id>/train/weights/best.pt   │
│         ▼                                                                   │
│  7. 승격/배포      POST /api/models/promote                                  │
│                    workspace/models/<alias>.pt (+ 추론 서비스로 복사)         │
└─────────────────────────────────────────────────────────────────────────────┘
        │  (선택) 가중치 복사 + POST /api/detector/model
        ▼
[tracker_py 추론 서비스 :8886]  운영 실시간 탐지
```

---

## 2. 계층 구조

```
src/v1/
├── main.py            # 엔트리포인트 (uvicorn 구동)
├── app.py             # FastAPI 앱 + lifespan + 예외→HTTP 변환
├── core/
│   ├── config.py      # 환경변수 기반 Settings (설정의 단일 소스)
│   └── store.py       # 워크스페이스 레이아웃 + 원자적 JSON IO ("DB" 역할)
├── api/
│   └── routes.py      # HTTP 입출력/검증만. 로직은 services 위임
├── services/
│   ├── classes.py     # 프로젝트 클래스 목록(= 학습 클래스 id 순서)
│   ├── video.py       # 영상 등록/메타/썸네일/해상도 정규화
│   ├── segments.py    # 정상·비정상 구간, 구간 연산(합집합/차집합)
│   ├── detector.py    # YOLO 추론 (tracker_py 로직 이식 + poly/track_id)
│   ├── autolabel.py   # 구간 → 프레임 추출 → 자동 라벨 초안 [잡]
│   ├── annotations.py # 라벨 CRUD·검수·전파·오버레이
│   ├── dataset.py     # YOLO 데이터셋 빌드/등록/병합 [잡]
│   ├── trainer.py     # 학습 run 관리 (기동/중단/재개/상태)
│   ├── train_runner.py# 학습 자식 프로세스 엔트리 (앱 import 안 함)
│   ├── registry.py    # 모델 승격/배포 이력
│   └── jobs.py        # 공통 백그라운드 잡 레지스트리
└── utils/
    └── geometry.py    # 폴리곤 ↔ 정규화 좌표, 태스크별 라벨 한 줄 생성
```

계층 책임은 tracker_py 개발표준과 같다. **api 는 얇게**, 로직은 services, 설정은 core.

---

## 3. 핵심 설계 결정과 근거

### 3.1 라벨은 항상 "4점 폴리곤"으로 저장한다

축정렬 박스(detect)는 회전 박스(OBB)의 특수한 경우다. 내부 표현을 폴리곤 하나로 두면
같은 라벨 자산에서 두 가지 학습 포맷을 모두 내보낼 수 있다.

| 태스크 | 내보내는 라벨 한 줄 |
|---|---|
| `obb` (기본) | `class x1 y1 x2 y2 x3 y3 x4 y4` (정규화) |
| `detect` | `class cx cy w h` (폴리곤의 AABB, 정규화) |

태스크를 갈아타도 **라벨을 다시 만들 필요가 없다.** 데이터셋만 다시 빌드한다.

기본값은 `obb` 다. 드론 상공 촬영은 표적이 임의 방향으로 놓여 축정렬 박스가 배경을 크게
포함하고, 운영 추론(`tracker_py`)과 기존 학습 데이터(`train_data/preprocessed_obb`)가
이미 OBB이기 때문이다.

### 3.2 "모델이 준 클래스"와 "우리 클래스"를 분리한다

기본 가중치 `yolo26l-obb.pt` 는 DOTA 15 클래스(plane/ship/large-vehicle…)다.
우리가 학습시킬 대상은
표적 종류(Panther_II, VIDAR…)라 **클래스 공간이 다르다.** 그래서

- 모델이 준 이름은 `model_class_name` 으로 참고용만 보관하고,
- 라벨의 진짜 값은 프로젝트 클래스 목록(`services/classes.py`)의 `class_name` 이다.

자동 라벨은 **"여기에 물체가 있다"는 박스 제안**으로 쓰고, 종류는 사람이 정한다.
클래스가 미확정(`class_id = -1`)인 객체가 남아 있으면 그 프레임은 승인되지 않는다.

### 3.3 트래킹으로 라벨링 공수를 객체 수에 비례시킨다

자동 라벨 단계에서 `model.track()` 으로 `track_id` 를 붙인다. 300 프레임에 등장하는
전차 1대는 `POST /api/videos/{id}/propagate` 한 번이면 300개 객체의 클래스가 채워진다.
라벨링 시간이 **프레임 수가 아니라 객체 수**에 비례하게 되는 것이 이 파이프라인의
실효 생산성을 결정한다.

### 3.4 데이터셋은 스냅샷이다

빌드 시점에 이미지를 데이터셋 폴더로 하드링크(실패 시 복사)하고 라벨 txt를 굽는다.
나중에 라벨을 고치거나 영상을 지워도 **이미 학습한 데이터셋은 변하지 않는다.**
`manifest.json` 에 어떤 영상·어떤 설정·클래스 분포·제외 사유까지 남겨 실험을 재현한다.

### 3.5 기존 `train_data` 구조를 입출력 양쪽으로 쓴다

빌드 산출물이 `tracker_py/train_data/preprocessed_obb` 와 같은 구조
(`data.yaml` + `{train,valid,test}/{images,labels}`)다. 그래서 반대 방향도 성립한다 —
이미 라벨된 폴더를 `POST /api/datasets/import` 로 등록하고, 새 라벨과 함께
`base_datasets` 로 병합할 수 있다.

이것이 **부트스트랩의 전제**다. 사전학습 가중치는 이 도메인에서 아무것도 못 잡으므로
(§6), 기존 데이터셋으로 먼저 학습해 승격한 모델을 자동 라벨에 투입한다.

병합 규칙은 셋이다. 클래스는 **id가 아니라 이름으로 매핑**하고(순서가 다른 데이터셋을
합칠 때 정답이 뒤바뀌는 것을 막는다), 병합된 데이터는 **원래 train/valid 분할을 유지**하고
(과거 실험과 점수를 비교할 수 있게), **태스크가 다르면 거부**한다. 자세한 내용은
[`dataset_format.md`](dataset_format.md) §6.

### 3.6 분할은 기본 `chunk` (누수 방지)

인접 프레임은 사실상 같은 그림이다. 무작위로 나누면 train과 valid에 같은 장면이 들어가
검증 mAP가 부풀려진다. 기본값은 연속 프레임을 블록으로 묶어 통째로 배분한다.

| 모드 | 묶는 단위 | 쓸 때 |
|---|---|---|
| `chunk` (기본) | 연속 프레임 N장 | 일반적인 경우 |
| `video` | 영상 전체 | 영상이 충분히 많고 일반화를 엄격히 볼 때 |
| `random` | 프레임 1장 | 프레임 간 상관이 없을 때만 |

### 3.7 학습은 자식 프로세스에서 돌린다

`trainer.py` 는 `train_runner.py` 를 `subprocess` 로 띄운다.

- GPU OOM·드라이버 크래시가 API를 같이 죽이지 않는다.
- Windows spawn에서 DataLoader 워커가 FastAPI 앱을 재import하는 부작용이 없다.
- 중단이 프로세스 트리 종료로 확실하게 끝난다(GPU 메모리 회수).
- API를 재시작해도 학습은 계속 돌고, `state.json` 으로 상태를 이어서 읽는다.

### 3.8 상태는 전부 파일이다 (DB 없음)

폐쇄망 온프레미스에서 DB 하나를 더 운영하는 비용이 이득보다 크다.
`workspace/` 한 디렉터리만 백업/마운트하면 이관이 끝나고, 라벨은 사람이 열어볼 수 있는
JSON이라 파이프라인이 깨져도 손으로 복구할 수 있다. 쓰기는 임시파일+`os.replace`로
원자적이라 중간에 죽어도 반쯤 쓰인 라벨이 남지 않는다.

---

## 4. tracker_py 재사용 관계

| 재사용 대상 | 방식 | 이유 |
|---|---|---|
| YOLO 결과 파싱(OBB/detect 분기, `xyxyxyxy`) | `services/detector.py` 에 **로직 이식** | 라벨을 만든 기준과 운영 탐지 기준이 같아야 함 |
| 클래스 순서 | `preprocessed_obb/data.yaml` 과 동일 순서를 기본값으로 | 기존 데이터셋과 합칠 때 클래스 id가 어긋나지 않게 |
| 입력 정규화 640×480 | 동일 스펙으로 프레임 저장 | 학습/추론의 객체 픽셀 크기 분포를 맞춤 |
| 학습 하이퍼파라미터 | `test/yolo.py` 의 검증값을 기본값으로 | Windows RAM 이슈까지 반영된 실측치 |
| 모델 배포 | 가중치 복사 → tracker_py `POST /api/detector/model` | 추론 서비스 재시작 없이 교체 |

프레임 수천 장을 HTTP로 왕복하면 느리므로 **자동 라벨 추론은 이 프로세스 안에서** 한다.
tracker_py HTTP API는 배포/연동 용도로만 쓴다.

---

## 5. 워크스페이스 레이아웃

```
workspace/
├── classes.json                    # 프로젝트 클래스 목록 (순서 = 클래스 id)
├── uploads/<video_id>.mp4          # 업로드 원본
├── videos/<video_id>/
│   ├── meta.json                   # fps/해상도/길이/원본 경로
│   ├── segments.json               # 정상/비정상 구간
│   ├── frames/<frame_id>.jpg       # 추출된 학습 후보 이미지
│   └── labels/<frame_id>.json      # 라벨(자동 초안 + 사람 수정)
├── datasets/<dataset_id>/
│   ├── data.yaml  manifest.json
│   └── {train,valid,test}/{images,labels}/
├── runs/<run_id>/
│   ├── spec.json  state.json  train.log
│   └── train/weights/{best,last}.pt, results.csv
├── models/<alias>.pt + registry.json
└── jobs/<job_id>.json              # 완료된 잡 스냅샷
```

도커에서는 `workspace/` 하나만 볼륨으로 마운트하면 전체 상태가 보존된다.

---

## 6. 알려진 제약

- **사전학습 가중치는 이 도메인에서 아무것도 못 잡는다.** `yolo26l-obb.pt`(DOTA 15)와
  `yolo26l.pt`(COCO 80) 모두 탑다운 드론 영상의 10px 표적을 `conf=0.05` 에서도 탐지하지
  못했다(실측). 따라서 첫 자동 라벨은 객체 0개가 정상이고, 기존 데이터셋으로 한 번 학습해
  승격한 모델을 초안 생성에 투입해야 한다(§3.5, `labeling_workflow.md` 부트스트랩).
- 자동 라벨의 클래스는 힌트일 뿐이라 **최초 1회는 사람이 클래스를 지정해야 한다.**
  1차 학습 모델을 승격해 다음 라벨링의 자동 라벨 모델로 쓰면(`extract.model`) 그 뒤부터는
  클래스까지 맞아 들어간다 — 이 되먹임이 파이프라인의 본체다.
- 데이터셋 `task` 와 가중치 태스크가 어긋나면 학습 러너가 시작 직후 막는다. 태스크를 바꿀 때는
  `NIT_TRAIN_TASK` 와 `NIT_TRAIN_BASE_MODEL` 을 함께 바꾼다.
- 참조 등록(`import` 의 `copy=false`)한 데이터셋은 원본 폴더가 지워지면 학습이 깨진다.
  고정이 필요하면 `copy=true` 로 스냅샷을 뜬다.
- 잡 진행률은 프로세스 메모리에 있으므로 uvicorn 워커는 1개로 고정한다.

# NIT_train — 드론 영상 라벨링 + YOLO 학습 API (MLOps)

드론/CCTV 영상 한 편을 넣으면, 사람이 최소한만 손대고 **학습된 모델**까지 나오는
온프레미스 파이프라인이다. 폐쇄망 배포를 전제로 외부 서비스와 별도 DB에 의존하지 않는다.

```
영상 업로드 → 정상/비정상 구간 지정 → 자동 라벨 초안(YOLO) → 사람 검수
           → YOLO 데이터셋 생성 → 학습 → 모델 승격/배포 → (다시 자동 라벨에 투입)
```

기본 태스크는 **회전 박스(OBB)** 다. 데이터셋 구조는
`tracker_py/train_data/preprocessed_obb` 와 동일하므로, 이미 라벨된 폴더를 그대로 등록해
학습에 쓰거나 새로 라벨한 프레임과 병합할 수 있다.

웹 콘솔은 [`NIT_train_front`](../NIT_train_front/README.md) 에 있다.

---

## 빠른 시작

```powershell
conda activate NIT
cd C:\project\NIT_train\src\v1
python main.py
```

- Swagger: <http://localhost:8888/docs>
- 헬스체크: `curl http://localhost:8888/healthz`
- 기본값 확인: `curl http://localhost:8888/api/meta`

파이프라인 전 구간 검증:

```powershell
cd C:\project\NIT_train\test
$env:PYTHONIOENCODING="utf-8"
python smoke_api.py            # 영상 → 라벨 → 데이터셋 (학습 제외)
python smoke_api.py --train    # 1 epoch 학습 + 승격까지 (GPU)
python smoke_obb.py            # 기존 OBB 폴더 등록/병합
python smoke_obb.py --train    # 등록 → 학습 → 승격 → 승격 모델로 추론 (부트스트랩 전 구간)
```

웹 콘솔까지 함께 확인하려면:

```powershell
python C:\project\NIT_train_front\test\api_contract.py   # 프런트가 읽는 응답 필드 계약
python C:\project\NIT_train_front\server.py              # http://127.0.0.1:8890
```

Docker:

```bash
docker compose up --build      # workspace/ 볼륨에 모든 상태 보존
```

---

## 5분 사용 예시

```bash
API=http://localhost:8888

# 0. (권장) 이미 라벨된 OBB 폴더를 등록해 시작 모델을 만든다 — 부트스트랩
#    사전학습 가중치는 이 도메인(10px 표적)에서 아무것도 못 잡는다.
curl -G $API/api/datasets/inspect \
  --data-urlencode "path=C:/project/tracker_py/train_data/preprocessed_obb"
curl -X POST $API/api/datasets/import -H "Content-Type: application/json" \
  -d '{"path":"C:/project/tracker_py/train_data/preprocessed_obb","copy":false}'
#    → POST /api/train → POST /api/models/promote {"alias":"drone-v1"}
#    이후 3번 extract 에서 model="models/drone-v1.pt" 로 초안을 받는다.

# 1. 영상 등록
VID=$(curl -s -F "file=@drone_01.mp4" $API/api/videos | jq -r .video_id)

# 2. 쓸 구간 지정 (정상에서 비정상을 뺀 부분만 학습에 쓰인다)
curl -X POST $API/api/videos/$VID/segments -H "Content-Type: application/json" -d '{
  "segments": [
    {"kind":"normal",   "start_sec": 12, "end_sec": 260},
    {"kind":"abnormal", "start_sec": 45, "end_sec": 52, "note":"급기동 흔들림"}
  ]}' -X PUT

# 3. 프레임 추출 + 자동 라벨 (백그라운드 잡)
JOB=$(curl -s -X POST $API/api/videos/$VID/extract \
  -H "Content-Type: application/json" -d '{"fps":2,"track":true}' | jq -r .job_id)
curl -s $API/api/jobs/$JOB    # 진행률 폴링

# 4. 객체별로 클래스 한 번씩 지정 → 등장하는 모든 프레임에 전파
curl -s $API/api/videos/$VID/tracks
curl -X POST $API/api/videos/$VID/propagate \
  -H "Content-Type: application/json" -d '{"track_id":3,"class_name":"Panther_II"}'

# 5. 검수 완료 처리
curl -X POST $API/api/videos/$VID/frames/status \
  -H "Content-Type: application/json" -d '{"status":"approved"}'

# 6. 데이터셋 → 학습 (task 생략 = obb. 기존 데이터셋을 함께 넣을 수 있다)
DS=$(curl -s -X POST $API/api/datasets -H "Content-Type: application/json" \
  -d "{\"name\":\"drone-v1\",\"video_ids\":[\"$VID\"],
       \"base_datasets\":[\"C:/project/tracker_py/train_data/preprocessed_obb\"]}" | jq -r .dataset_id)
RUN=$(curl -s -X POST $API/api/train -H "Content-Type: application/json" \
  -d "{\"dataset_id\":\"$DS\",\"epochs\":100}" | jq -r .run_id)
curl -s $API/api/train/$RUN   # epoch/mAP/로그

# 7. 승격
curl -X POST $API/api/models/promote -H "Content-Type: application/json" \
  -d "{\"run_id\":\"$RUN\",\"alias\":\"drone-v1\"}"
```

---

## 문서

| 문서 | 내용 |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | 전체 파이프라인, 계층 구조, **설계 결정과 근거** |
| [`docs/api.md`](docs/api.md) | API 레퍼런스 + 프런트 연동 순서 |
| [`docs/labeling_workflow.md`](docs/labeling_workflow.md) | 사용자 관점 워크플로(화면 설계 근거) |
| [`docs/dataset_format.md`](docs/dataset_format.md) | 라벨/데이터셋 포맷, 클래스 id 규약, 기존 폴더 등록·병합, OBB vs detect |
| [`docs/development_environment.md`](docs/development_environment.md) | 셋업/실행/**환경변수 전체 표** |
| [`docs/development_standards.md`](docs/development_standards.md) | 코드·설정·데이터 안전·잡·문서 표준 |
| [`../NIT_train_front/README.md`](../NIT_train_front/README.md) | 웹 콘솔 실행/단축키 · `docs/ui.md` 는 화면 설계 근거 |

---

## 핵심 개념 3가지

### 1. 라벨링 공수는 프레임 수가 아니라 **객체 수**에 비례한다

자동 라벨 단계에서 트래킹으로 `track_id` 를 붙인다. 300 프레임에 등장하는 전차 1대는
`POST /api/videos/{id}/propagate` 한 번으로 300개 라벨이 채워진다.
영상 한 편에 표적이 5대면 기본 작업량은 클릭 5번이다.

### 2. 자동 라벨은 "박스 제안"이고, 종류는 사람이 정한다

기본 가중치 `yolo26l-obb.pt` 는 DOTA 15 클래스라 우리 표적 이름을 모른다. 모델이 준 이름은
`model_class_name`(힌트)로만 남기고, 라벨의 정답은 프로젝트 클래스 목록의 `class_name` 이다.
클래스가 미확정인 객체가 남아 있으면 그 프레임은 **승인되지 않는다** —
일부만 라벨된 이미지는 모델을 적극적으로 망가뜨리기 때문이다.

### 3. 되먹임: 회차가 쌓일수록 사람 개입이 줄어든다

```
0차: 기존 OBB 폴더 등록 → 학습 → drone-v1 승격        (시작 모델 만들기)
1차: extract 시 model="models/drone-v1.pt" → 박스 + 클래스까지 제안
     (overwrite="auto" 로 검수 전 초안만 갱신, 사람 작업은 보존)
2차: 새 라벨 + 기존 데이터셋 병합 학습 → drone-v2 → 반복 …
```

---

## 상태 저장 위치

모든 상태가 `workspace/` 하나에 모인다. 도커에서는 이 디렉터리만 볼륨으로 마운트한다.

```
workspace/
├── classes.json          # 클래스 목록 (순서 = 학습 클래스 id)
├── uploads/              # 업로드 원본 영상
├── videos/<id>/          # meta / segments / frames / labels
├── datasets/<id>/        # data.yaml + train,valid,test
├── runs/<id>/            # spec / state / train.log / weights
├── models/               # 승격된 배포 후보 + registry.json
└── jobs/                 # 완료된 작업 스냅샷
```

DB를 쓰지 않는 이유와 원자적 쓰기 규약은 [`docs/architecture.md`](docs/architecture.md#37-상태는-전부-파일이다-db-없음) 참고.

---

## 알아둘 제약

- **사전학습 가중치로는 첫 초안이 0개다.** `yolo26l-obb.pt`(DOTA)·`yolo26l.pt`(COCO) 모두
  탑다운 드론 영상의 10px 표적을 `conf=0.05` 에서도 못 잡는다(실측). 위 0단계 부트스트랩으로
  자체 모델을 먼저 만든다 → [`labeling_workflow.md`](docs/labeling_workflow.md)
- 데이터셋 `task` 와 가중치 태스크가 어긋나면 학습이 시작 직후 막힌다
  → [`dataset_format.md`](docs/dataset_format.md)
- 학습은 API와 분리된 **자식 프로세스**에서 돈다. API를 재시작해도 학습은 계속된다.
- uvicorn 워커는 **1개 고정**(잡 상태가 프로세스 메모리에 있음).
- Windows에서 `workers` 를 크게 잡으면 후반 epoch에서 DataLoader가 죽는다. 4 이하 권장.
- 같은 GPU에서 운영 추론(tracker_py:8886)과 학습을 동시에 돌리면 실시간 fps가 떨어진다.

---

## 관련 서비스

| 서비스 | 포트 | 역할 |
|---|---|---|
| `tracker_py` | 8886 | 운영 실시간 추론(전처리 + YOLO + 트래킹) |
| **`NIT_train`** | **8888** | **학습·라벨링 (이 저장소)** |
| `NIT_train_front` | 8890 | 라벨링·학습 웹 콘솔 (빌드 없는 정적 SPA) |

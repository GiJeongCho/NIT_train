# API 레퍼런스

> 대상: `NIT_train/src/v1/api/routes.py`
> Swagger UI: 서버 실행 후 `http://<host>:8888/docs`
> 작성일: 2026-07-29

파이프라인 순서대로 그룹이 나뉜다. 프런트(`NIT_train_front`)는 이 순서대로 화면을 만들면 된다.

| # | 그룹 | 하는 일 |
|---|---|---|
| 0 | 기타 | 헬스체크, 서버 기본값, 클래스 목록 |
| 1 | 영상 | 업로드/등록, 재생(Range), 썸네일 |
| 2 | 구간 | 정상/비정상 구간 저장, 추출 대상 미리보기 |
| 3 | 라벨링 | 프레임 추출+자동 라벨, 검수/수정/전파/승인 |
| 4 | 데이터셋 | 승인 라벨 → YOLO 데이터셋 스냅샷 |
| 5 | 학습 | 학습 시작/진행률/중단/재개/가중치 |
| 6 | 모델 | 가중치 목록, 승격, 배포 |
| 7 | 작업 | 백그라운드 잡 진행률/취소 |

오래 걸리는 작업(추출, 데이터셋)은 **잡 방식**이다. 요청 → `job_id` 즉시 반환 →
`GET /api/jobs/{job_id}` 폴링(1초 주기 권장).

에러는 `{"detail": "..."}` 형태다. `400`=입력 오류, `404`=대상 없음, `410`=원본 파일 소실.

---

## 0. 기타

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/healthz` | 헬스체크 (`{"status":"ok"}`) |
| GET | `/api/meta` | 서버 기본값 일괄 조회 |
| GET | `/api/classes` | 클래스 목록 |
| PUT | `/api/classes` | 클래스 추가/이름 수정 |

### `GET /api/meta`

프런트 첫 로딩용. 기본값을 화면에 하드코딩하지 않기 위한 엔드포인트다.

```json
{
  "default_model": "C:/project/NIT_train/test_model/yolo26l-obb.pt",
  "default_model_exists": true,
  "class_names": ["Ikv_91_105", "...", "VIDAR"],
  "segment_kinds": ["normal", "abnormal"],
  "label_statuses": ["pending", "approved", "rejected"],
  "dataset_tasks": ["obb", "detect"],
  "default_task": "obb",
  "split_modes": ["chunk", "random", "video"],
  "defaults": {
    "extract": {"fps": 2.0, "conf": 0.25, "iou": 0.7, "imgsz": 640, "track": true},
    "dataset": {"splits": {"train": 0.8, "valid": 0.15, "test": 0.05},
                "split_mode": "chunk", "task": "obb"},
    "train": {"epochs": 100, "imgsz": 640, "batch": 16, "workers": 4}
  }
}
```

### `PUT /api/classes`

```bash
curl -X PUT http://localhost:8888/api/classes \
  -H "Content-Type: application/json" \
  -d '{"class_names": ["Ikv_91_105","Jagdtiger","Panther_II","Strv_101","Strv_103A","Tiger_II_10.5_","VIDAR","T-90"]}'
```

> ⚠️ **목록의 순서가 곧 학습 클래스 id 다.** 순서 변경과 삭제는 400으로 거부한다.
> 이미 저장된 라벨과 데이터셋이 인덱스를 정답으로 쓰고 있어서, 순서가 바뀌면 과거
> 데이터의 정답이 조용히 뒤바뀐다. 추가는 항상 **뒤에만** 한다.

---

## 1. 영상

| 메서드 | 경로 | 설명 |
|---|---|---|
| POST | `/api/videos` | 영상 업로드 (multipart `file`) |
| POST | `/api/videos/path` | 서버 경로 등록 (복사 없음) |
| GET | `/api/videos` | 목록 (+구간/라벨링 진행률) |
| GET | `/api/videos/{video_id}` | 상세 |
| DELETE | `/api/videos/{video_id}` | 삭제 (프레임/라벨 포함) |
| GET | `/api/videos/{video_id}/stream` | 재생 (HTTP Range 지원) |
| GET | `/api/videos/{video_id}/frame?t=` | 특정 시각 프레임 JPEG |

```bash
curl -F "file=@drone_01.mp4" http://localhost:8888/api/videos
# → {"ok":true,"video_id":"20260729-150904-049310","fps":30.0,
#    "width":1920,"height":1080,"duration_sec":312.4,"frame_count":9372, ...}
```

수십 GB 파일을 중복 저장하지 않으려면 업로드 대신 경로 등록을 쓴다.

```bash
curl -X POST http://localhost:8888/api/videos/path \
  -H "Content-Type: application/json" -d '{"path": "/mnt/nas/drone/2026-07-29.mp4"}'
# → managed=false (원본이 지워지면 프레임 재추출 불가)
```

`/stream` 은 Range 요청을 직접 처리한다. `<video src>` 로 붙이면 타임라인 탐색(seek)이
되고, 이것이 구간 지정 UI의 전제 조건이다.

---

## 2. 구간

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/api/videos/{video_id}/segments` | 구간 조회 |
| PUT | `/api/videos/{video_id}/segments` | 정상/비정상 구간 저장 (전체 교체) |
| GET | `/api/videos/{video_id}/selection` | 실제 추출 대상 구간 미리보기 |

```bash
curl -X PUT http://localhost:8888/api/videos/$VID/segments \
  -H "Content-Type: application/json" -d '{
    "segments": [
      {"kind": "normal",   "start_sec": 12.0, "end_sec": 95.0},
      {"kind": "normal",   "start_sec": 140.0, "end_sec": 260.0},
      {"kind": "abnormal", "start_sec": 45.0, "end_sec": 52.0, "note": "급기동 흔들림"}
    ]}'
```

응답의 `selection_ranges` 가 실제로 프레임을 뽑을 구간이다.

```json
{"selection_ranges": [[12.0, 45.0], [52.0, 95.0], [140.0, 260.0]]}
```

규칙:

- 같은 종류끼리 겹치면 **자동 병합**된다(드래그 중복으로 같은 프레임을 두 번 처리하지 않게).
- **정상 ∩ 비정상 = 비정상.** 사람이 "쓰지 말라"고 한 쪽이 항상 이긴다.
- 정상 구간을 하나도 안 찍으면 **영상 전체를 정상**으로 본다(짧은 클립 편의).
- 0.05초 미만 구간은 실수 클릭으로 보고 버린다.

---

## 3. 라벨링

| 메서드 | 경로 | 설명 |
|---|---|---|
| POST | `/api/videos/{video_id}/extract` | 프레임 추출 + 자동 라벨 초안 **[잡]** |
| GET | `/api/videos/{video_id}/frames` | 프레임 목록(경량 요약) |
| GET | `/api/videos/{video_id}/progress` | 라벨링 진행률 + 클래스 분포 |
| GET | `/api/videos/{video_id}/tracks` | 객체(track) 목록 |
| GET | `/api/videos/{video_id}/frames/{frame_id}` | 프레임 라벨 |
| GET | `.../frames/{frame_id}/image?overlay=` | 프레임 이미지(JPEG) |
| PUT | `/api/videos/{video_id}/frames/{frame_id}` | 라벨 저장(사람 수정) |
| POST | `.../frames/{frame_id}/status` | 개별 검수 상태 변경 |
| POST | `/api/videos/{video_id}/frames/status` | 일괄 상태 변경 |
| POST | `/api/videos/{video_id}/propagate` | track_id 기준 클래스 일괄 적용 |
| DELETE | `.../frames/{frame_id}` | 프레임 삭제 |

### `POST .../extract`

```bash
curl -X POST http://localhost:8888/api/videos/$VID/extract \
  -H "Content-Type: application/json" \
  -d '{"fps": 2, "conf": 0.25, "track": true, "overwrite": "skip"}'
# → {"ok":true,"job_id":"20260729-150904-8ceafa", ...}
```

| 필드 | 기본 | 의미 |
|---|---|---|
| `kinds` | `["normal"]` | 어떤 구간을 합칠지 |
| `fps` | `2.0` | 초당 추출 장수. `0`=모든 프레임 |
| `model` | `test_model/yolo26l-obb.pt` | 자동 라벨에 쓸 가중치 |
| `conf` | `0.25` | 낮을수록 많이 제안(재현율 우선) |
| `iou` / `imgsz` / `max_det` | `0.7` / `640` / `300` | 추론 파라미터 |
| `max_frames` | `20000` | 안전장치 |
| `track` | `true` | `track_id` 부여 |
| `overwrite` | `"skip"` | 재실행 정책 |

`conf` 기본값이 운영 추론보다 낮은 이유: 초안은 **사람이 지우는 게 추가하는 것보다 싸다.**
빠뜨린 객체를 새로 그리는 비용이 오탐을 지우는 비용보다 훨씬 크다.

> ⚠️ **첫 추출은 객체 0개가 정상이다.** 사전학습 가중치(COCO 80 / DOTA 15)는 탑다운 드론
> 영상의 10px 표적을 `conf=0.05` 에서도 잡지 못한다(실측). 기존 라벨 데이터셋으로 한 번
> 학습해 승격한 모델을 `model` 로 지정해야 쓸만한 초안이 나온다.
> 부트스트랩 절차는 `labeling_workflow.md` 참고.

`overwrite` 재실행 정책:

| 값 | 동작 |
|---|---|
| `skip` (기본) | 이미 라벨이 있으면 건드리지 않음 |
| `auto` | 검수 전(`status=pending` + `source=auto`)인 자동 초안만 새 모델로 갱신 |
| `all` | 전부 재생성 — **사람 수정이 소실된다** |

### `GET .../frames`

```json
{"total": 620, "offset": 0, "limit": 100, "items": [
  {"frame_id": "f000000", "frame_index": 0, "time_sec": 0.0,
   "status": "pending", "source": "auto", "segment_kind": "normal",
   "n_objects": 8, "n_unresolved": 8,
   "class_names": [], "track_ids": [2, 5]}
]}
```

`n_unresolved` 는 클래스가 아직 안 정해진 객체 수다. 검수 화면에서 이 값이 0이 아닌
프레임만 필터링하면 남은 일이 바로 보인다.

### `GET .../frames/{frame_id}`

```json
{
  "video_id": "...", "frame_id": "f000123", "frame_index": 123, "time_sec": 4.1,
  "width": 640, "height": 480, "segment_kind": "normal",
  "status": "pending", "source": "auto", "model": "yolo26l-obb.pt",
  "objects": [{
    "id": "o1",
    "class_name": null,            // ← 라벨의 진짜 값. null = 미확정
    "class_id": -1,                // ← 클래스 목록 인덱스. -1 = 미확정
    "model_class_name": "truck",   // ← 모델이 준 힌트(참고용)
    "score": 0.87,
    "poly": [[210.0, 180.0], [301.0, 180.0], [301.0, 262.0], [210.0, 262.0]],
    "bbox": [210.0, 180.0, 301.0, 262.0],
    "track_id": 3,
    "source": "auto", "verified": false
  }]
}
```

`poly` 는 **픽셀 좌표 4점**이다(시계방향). OBB 모델이면 실제 회전 꼭짓점, detect 모델이면
축정렬 박스의 네 꼭짓점이 들어간다.

`image?overlay=1` 은 이 라벨을 그려서 준다. 미확정 객체는 회색으로 그려 눈에 띈다.

### `POST .../propagate` — 이 파이프라인의 핵심

```bash
curl -X POST http://localhost:8888/api/videos/$VID/propagate \
  -H "Content-Type: application/json" \
  -d '{"track_id": 3, "class_name": "Panther_II"}'
# → {"track_id":3,"class_name":"Panther_II","frames":287,"objects":287}
```

같은 `track_id` 객체의 클래스를 모든 프레임에 한 번에 적용한다. **라벨링 공수를 프레임 수가
아니라 객체 수에 비례하게** 만든다. 프런트는 `GET .../tracks` 로 객체 목록을 띄우고
객체별로 클래스를 고르게 하는 UI를 우선 제공하는 것이 좋다.

### `PUT .../frames/{frame_id}`

```bash
curl -X PUT http://localhost:8888/api/videos/$VID/frames/f000123 \
  -H "Content-Type: application/json" -d '{
    "objects": [
      {"class_name": "Panther_II", "poly": [[210,180],[301,180],[301,262],[210,262]], "track_id": 3}
    ],
    "status": "approved"}'
```

`objects` 를 보내면 **전체 교체**다(추가/이동/삭제 결과를 통째로 보낸다).
`poly` 는 4점 권장, 2점이면 AABB로 변환한다.

### 승인 규칙

`status=approved` 는 **클래스가 정해지지 않은 객체가 하나라도 있으면 400으로 거부**한다.

```json
{"detail": "클래스가 정해지지 않은 객체가 있어 승인할 수 없습니다: ['o1','o2']. 모든 객체에 클래스를 지정하거나 해당 객체를 삭제하세요."}
```

"라벨링은 모든 부분을 해야 한다"는 요구를 데이터셋 입구가 아니라 **승인 시점에** 막는다.
일부만 라벨된 이미지는 라벨 없는 객체를 배경으로 학습시켜 모델을 망가뜨린다.

- 객체가 **0개**인 프레임은 배경(negative) 샘플로 유효하므로 그대로 승인된다.
- `force=true` 로 강행할 수 있으나, 미확정 객체가 있는 프레임은 데이터셋 빌드에서 제외된다.
- 일괄 승인(`POST .../frames/status`)은 실패한 프레임을 건너뛰고 이유를 모아 돌려준다
  (하나 때문에 수천 장 검수가 막히지 않게).

---

## 4. 데이터셋

| 메서드 | 경로 | 설명 |
|---|---|---|
| POST | `/api/datasets` | 데이터셋 빌드(+기존 데이터셋 병합) **[잡]** |
| GET | `/api/datasets/inspect?path=` | 기존 YOLO 폴더 미리보기 |
| POST | `/api/datasets/import` | 기존 YOLO 폴더 등록 |
| GET | `/api/datasets` | 목록 |
| GET | `/api/datasets/{dataset_id}` | 상세 manifest |
| GET | `/api/datasets/{dataset_id}/data.yaml` | data.yaml 원문 |
| DELETE | `/api/datasets/{dataset_id}` | 삭제 |

```bash
curl -X POST http://localhost:8888/api/datasets \
  -H "Content-Type: application/json" -d '{
    "name": "drone-2026-07",
    "video_ids": ["20260729-150904-049310", "20260729-151233-77ab21"],
    "base_datasets": ["C:/project/tracker_py/train_data/preprocessed_obb"],
    "splits": {"train": 0.8, "valid": 0.15, "test": 0.05},
    "split_mode": "chunk"}'
# → {"ok":true,"job_id":"...","dataset_id":"20260729-150934-04eca4"}
```

| 필드 | 기본 | 의미 |
|---|---|---|
| `video_ids` | 전체 | 학습에 쓸 영상 선택. `[]` = 영상 없이 병합만 |
| `base_datasets` | — | 같이 넣을 기존 데이터셋(id 또는 서버 폴더 경로). 원래 분할 유지 |
| `task` | `obb` | `obb`(8좌표) 또는 `detect`(cxcywh). **학습할 모델과 일치해야 함** |
| `only_approved` | `true` | 승인된 프레임만 |
| `include_kinds` | `["normal"]` | 포함할 구간 종류 |
| `splits` | `0.8/0.15/0.05` | 분할 비율 |
| `split_mode` | `chunk` | 누수 방지 분할 방식 |
| `class_names` | 프로젝트 목록 | 데이터셋 고유 클래스를 쓰고 싶을 때 |

manifest 응답 일부:

```json
{
  "task": "obb",
  "counts": {"train": 496, "valid": 93, "test": 31},
  "total_objects": 3120,
  "class_histogram": {"Panther_II": 1802, "VIDAR": 890, "Jagdtiger": 428},
  "per_video_frames": {"2026...049310": 410, "2026...77ab21": 210},
  "sources": [{"name": "preprocessed_obb", "prefix": "preprocessed_obb",
               "added": {"train": 388, "valid": 39, "test": 0},
               "class_map": {"Panther_II": 2, "VIDAR": 6}}],
  "excluded": {"미승인(pending)": 88, "미확정/목록 외 클래스 포함": 12},
  "warnings": ["샘플이 10개 미만인 클래스: ['Strv_101']"]
}
```

`excluded` 와 `warnings` 를 프런트에 그대로 보여주면 "왜 프레임이 이것밖에 안 들어갔지"를
사용자가 스스로 해결할 수 있다. `sources` 는 병합 내역이다(어느 폴더가 어떤 접두사로
들어갔고 클래스가 어떻게 매핑됐는지).

### `GET /api/datasets/inspect` — 등록 전 확인

```bash
curl -G http://localhost:8888/api/datasets/inspect \
  --data-urlencode "path=C:/project/tracker_py/train_data/preprocessed_obb"
# → {"dir":"C:/project/tracker_py/train_data/preprocessed_obb","yaml":"data.yaml",
#    "task":"obb","names":["Ikv_91_105", ...],
#    "splits":{"train":{"dir":"train","images":388,"labels":388},
#              "valid":{"dir":"valid","images":39,"labels":39}},
#    "total_images":427}
```

아무것도 바꾸지 않는다. `data.yaml` 이 없으면 라벨 토큰 수로 태스크를 추정한다(9=obb, 5=detect).

### `POST /api/datasets/import` — 기존 폴더 등록

```bash
curl -X POST http://localhost:8888/api/datasets/import \
  -H "Content-Type: application/json" -d '{
    "path": "C:/project/tracker_py/train_data/preprocessed_obb",
    "copy": false}'
```

| 필드 | 기본 | 의미 |
|---|---|---|
| `path` | — | YOLO 데이터셋 폴더(절대 또는 워크스페이스 상대) |
| `copy` | `false` | `false`=원본 참조(디스크 절약) / `true`=워크스페이스로 복사해 스냅샷 고정 |
| `task` | 자동 | 생략 시 `data.yaml` 또는 라벨 토큰 수로 추정 |
| `class_names` | `data.yaml` | `names` 가 없는 폴더일 때만 필요 |
| `link_images` | `true` | `copy=true` 일 때 하드링크 사용 |

참조 등록(`copy=false`)은 원본이 지워지면 학습이 깨진다. 대신 수천 장을 중복 저장하지 않는다.
자세한 규칙(클래스 이름 매핑, 분할 유지)은 `dataset_format.md` §6.

---

## 5. 학습

| 메서드 | 경로 | 설명 |
|---|---|---|
| POST | `/api/train` | 학습 시작 |
| GET | `/api/train` | run 목록 |
| GET | `/api/train/{run_id}` | 진행률/지표/로그 |
| POST | `/api/train/{run_id}/stop` | 중단 |
| POST | `/api/train/{run_id}/resume` | `last.pt` 부터 이어서 |
| GET | `/api/train/{run_id}/log` | 로그 tail |
| GET | `/api/train/{run_id}/weights/{best\|last}` | 가중치 다운로드 |
| DELETE | `/api/train/{run_id}` | run 삭제 |

```bash
curl -X POST http://localhost:8888/api/train \
  -H "Content-Type: application/json" \
  -d '{"dataset_id": "20260729-150934-04eca4", "epochs": 100, "batch": 16}'
# → {"ok":true,"run_id":"20260729-150935-d51e4f","spec":{...},"warnings":[]}
```

기본값은 `test/yolo.py` 에서 검증된 값이고, 기본 가중치는 `test_model/yolo26l-obb.pt` 다.

| 필드 | 기본 | 비고 |
|---|---|---|
| `model` | `test_model/yolo26l-obb.pt` | `GET /api/models` 의 `path` 를 넣는다 |
| `epochs` / `imgsz` / `batch` | `100` / `640` / `16` | |
| `device` | `0` | CUDA 인덱스 또는 `cpu` |
| `workers` | `4` | **Windows 는 4 이하.** 워커마다 RAM을 크게 물어 후반 epoch에서 DataLoader가 죽는다. 재발하면 2로 |
| `patience` | `50` | 조기 종료 |
| `extra` | — | ultralytics `train()` 인자를 그대로 전달 |

학습은 **API와 분리된 자식 프로세스**에서 돈다. API를 재시작해도 학습은 계속되고,
`GET /api/train/{run_id}` 로 상태를 이어서 읽는다.

### `GET /api/train/{run_id}`

```json
{
  "status": "running",          // starting|running|done|error|stopped|unknown
  "epoch": 37, "epochs": 100, "progress": 0.37,
  "metrics": {"metrics/mAP50(B)": 0.812, "metrics/mAP50-95(B)": 0.564},
  "best_fitness": 0.591,
  "results_csv": {"epochs_done": 37, "last_row": {...}},
  "weights": {"best": "runs/2026.../train/weights/best.pt", "last": "..."},
  "log_tail": "..."
}
```

`unknown` 은 프로세스가 사라졌는데 상태가 `running` 으로 굳은 경우다(강제 종료/재부팅).
`resume` 으로 이어서 하면 된다.

> `stop` 은 `taskkill /T`(Windows) / `killpg`(POSIX)로 **DataLoader 워커 트리까지** 종료한다.
> 워커가 남으면 GPU 메모리를 계속 잡고 있어 다음 학습이 OOM 난다.

---

## 6. 모델

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/api/models` | 가중치 목록 + 승격 이력 |
| GET | `/api/detector/weights` | 가중치 목록만(경량, 드롭다운용) |
| POST | `/api/models/promote` | 학습 결과 승격 (+선택적 배포) |
| DELETE | `/api/models/{alias}` | 승격 취소 |

`GET /api/models` 는 세 곳의 가중치를 모두 모아 준다.

| `origin` | 위치 |
|---|---|
| `pretrained` | `test_model/` (기본 `yolo26l-obb.pt`) |
| `promoted` | `workspace/models/` |
| `run:<run_id>` | `workspace/runs/<run_id>/train/weights/` |

```bash
curl -X POST http://localhost:8888/api/models/promote \
  -H "Content-Type: application/json" \
  -d '{"run_id": "20260729-150935-d51e4f", "alias": "drone-v3", "which": "best", "deploy": false}'
```

승격은 가중치를 `workspace/models/<alias>.pt` 로 복사해 고정하고, 어떤 데이터셋·어떤
지표에서 나왔는지 `registry.json` 에 남긴다. run을 지워도 남으므로 "모델 파일만 있고 출처를
아무도 모르는" 상황을 막는다.

`deploy=true` 면 `NIT_TRAIN_TRACKER_MODELS_DIR` 로 복사한다. 그 뒤 추론 서비스의
`POST /api/detector/model` 을 호출하면 재시작 없이 새 모델이 적용된다.

---

## 7. 작업 (Jobs)

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/api/jobs` | 목록 (`kind`=`extract`\|`dataset`, `target` 필터) |
| GET | `/api/jobs/{job_id}` | 진행률 |
| POST | `/api/jobs/{job_id}/cancel` | 취소 |

```json
{
  "id": "20260729-150904-8ceafa", "kind": "extract",
  "target": "20260729-150904-049310",
  "status": "running",           // queued|running|done|error|canceled
  "total": 620, "done": 147, "progress": 0.2371,
  "rate_per_sec": 4.8, "elapsed_sec": 30.6,
  "message": "147장 라벨링 (객체 981개)",
  "error": null, "result": {}
}
```

취소는 협조적이다. 진행 중인 프레임까지 저장한 뒤 멈춘다(반쯤 쓰인 라벨을 남기지 않기 위해).
완료된 잡은 `workspace/jobs/<id>.json` 에 남아 API 재시작 후에도 조회된다.

---

## 8. 프런트 연동 요약 (NIT_train_front)

구현된 콘솔이 `NIT_train_front/` 에 있다(빌드 없는 ES 모듈 SPA). 화면 설계 근거는
`NIT_train_front/docs/ui.md`, 응답 필드 계약 테스트는 `NIT_train_front/test/api_contract.py` 다.

전형적인 화면 흐름과 호출 순서:

```
[부트스트랩]     GET  /api/datasets/inspect?path=...      기존 train_data 폴더 확인
                 POST /api/datasets/import                등록 → 바로 학습 → 승격
[영상 목록]      GET  /api/videos
[업로드]         POST /api/videos                        (진행률: XHR upload progress)
[구간 지정]      GET  /api/videos/{id}/stream            <video> + 타임라인
                 PUT  /api/videos/{id}/segments
                 GET  /api/videos/{id}/selection         대상 구간 확인
[자동 라벨]      POST /api/videos/{id}/extract           → job 폴링
[객체 라벨링]    GET  /api/videos/{id}/tracks            객체 목록
                 POST /api/videos/{id}/propagate         객체당 클래스 1회 지정
[프레임 검수]    GET  /api/videos/{id}/frames?status=pending
                 GET  .../frames/{fid}/image             (캔버스 위에 poly 그리기)
                 PUT  .../frames/{fid}                   박스 수정 + approved
                 POST .../frames/status                  일괄 승인
[데이터셋]       POST /api/datasets                      → job 폴링
                 GET  /api/datasets/{id}                 클래스 분포/제외 사유 확인
[학습]           GET  /api/models                        가중치 선택
                 POST /api/train                         → GET /api/train/{run_id} 폴링
[배포]           POST /api/models/promote
```

- CORS는 모든 출처를 허용한다(`allow_origins=["*"]`). 프런트는 별도 포트에서 서빙해도 된다.
- 폴링 주기는 잡 1초, 학습 3~5초를 권장한다(학습은 epoch 단위라 더 자주 볼 필요가 없다).
- 이미지 캐시: 프레임 이미지는 내용이 바뀌지 않으므로 `frame_id` 를 캐시 키로 써도 된다.
  단 `overlay=1` 은 라벨을 고칠 때마다 바뀌므로 캐시하지 말 것.

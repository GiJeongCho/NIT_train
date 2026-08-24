# 데이터셋·라벨 포맷

> 대상: `NIT_train/src/v1/services/dataset.py`, `utils/geometry.py`
> 작성일: 2026-07-29

---

## 1. 내부 라벨 포맷 (원본 자산)

라벨 1장 = `workspace/videos/<video_id>/labels/<frame_id>.json` 하나.
이미지 1장 = `workspace/videos/<video_id>/frames/<frame_id>.jpg`.

`frame_id` 는 원본 영상의 프레임 인덱스로 만든다: `f{index:06d}` (예: `f000123`).

```json
{
  "video_id": "20260729-150904-049310",
  "frame_id": "f000123",
  "frame_index": 123,
  "time_sec": 4.1,
  "width": 640, "height": 480,
  "segment_kind": "normal",
  "status": "pending",
  "source": "auto",
  "model": "yolo26l-obb.pt",
  "objects": [
    {
      "id": "o1",
      "class_name": "Panther_II",
      "class_id": 2,
      "model_class_name": "truck",
      "score": 0.87,
      "poly": [[210.0,180.0],[301.0,180.0],[301.0,262.0],[210.0,262.0]],
      "bbox": [210.0, 180.0, 301.0, 262.0],
      "track_id": 3,
      "source": "auto",
      "verified": false
    }
  ],
  "updated_at": "2026-07-29T15:09:04"
}
```

| 필드 | 의미 |
|---|---|
| `poly` | **픽셀 좌표 4점**(시계방향 좌상→우상→우하→좌하). 라벨의 원본 |
| `bbox` | `poly` 의 AABB. 조회 편의용 파생값 |
| `class_name` | 라벨의 진짜 값. 프로젝트 클래스 목록의 이름. `null`=미확정 |
| `class_id` | 클래스 목록 인덱스. `-1`=미확정 |
| `model_class_name` | 자동 라벨 모델이 준 힌트(COCO 등). 참고용 |
| `track_id` | 프레임 간 같은 객체를 잇는 id. 클래스 일괄 전파의 키 |
| `status` | `pending`(검수 전) / `approved`(검수 완료) / `rejected`(제외) |
| `source` | `auto`(초안) / `manual`(사람이 손댐) |
| `verified` | 승인 시 `true` 로 표시 |

### 왜 폴리곤 하나로 통일하는가

축정렬 박스는 회전 박스의 특수한 경우다. 내부를 폴리곤으로 두면 같은 라벨 자산에서
`detect` 와 `obb` 두 가지 학습 포맷을 모두 내보낼 수 있다. **태스크를 바꿔도 라벨을 다시
만들 필요가 없다.**

---

## 2. 내보내는 YOLO 라벨 포맷

`POST /api/datasets` 의 `task` 에 따라 라벨 txt 한 줄의 형태가 달라진다.
생략하면 서버 기본값인 `obb` 다(`NIT_TRAIN_TASK`).
좌표는 모두 이미지 크기로 나눈 **0~1 정규화** 값이다.

### `task: "obb"` (기본)

```
class x1 y1 x2 y2 x3 y3 x4 y4
```

```
6 0.485938 0.475000 0.459375 0.475000 0.459375 0.518750 0.485938 0.518750
```

`tracker_py/train_data/preprocessed_obb/` 와 **완전히 같은 포맷**이라 기존 데이터셋과
그대로 합칠 수 있다(§6).

### `task: "detect"`

```
class cx cy w h
```

```
0 0.858031 0.239479 0.034281 0.082792
0 0.920367 0.342823 0.040578 0.090354
```

폴리곤의 AABB를 중심점+크기로 바꿔 쓴다. 회전 정보는 이 시점에 버려진다.

> 객체가 없는 프레임은 **빈 txt 파일**을 만든다. YOLO는 이를 배경(negative) 샘플로 쓰며,
> 오탐을 줄이는 데 실제로 효과가 있다. 라벨 파일 자체가 없으면 ultralytics가 경고를 낸다.

---

## 3. 데이터셋 디렉터리

```
workspace/datasets/<dataset_id>/
├── data.yaml
├── manifest.json
├── train/
│   ├── images/<video_id>_<frame_id>.jpg
│   └── labels/<video_id>_<frame_id>.txt
├── valid/{images,labels}/
└── test/{images,labels}/
```

파일 이름에 `video_id` 를 접두사로 붙인다. 영상이 달라도 프레임 번호는 겹치기 때문이다.

이미지는 기본적으로 **하드링크**로 연결한다(디스크 절약). 도커 바인드 마운트처럼 볼륨이
다르면 하드링크가 실패하므로 자동으로 복사로 폴백한다(`link_images: false` 로 강제 가능).

### `data.yaml`

```yaml
# NIT_train 자동 생성 (2026-07-29T15:09:08)
# dataset: drone-2026-07  task: obb
# frames: train=496 valid=93 test=31

path: C:/project/NIT_train/workspace/datasets/20260729-150908-3ae7c1
train: train/images
val: valid/images
test: test/images

task: obb

names:
  0: Ikv_91_105
  1: Jagdtiger
  2: Panther_II
  3: Strv_101
  4: Strv_103A
  5: Tiger_II_10.5_
  6: VIDAR
```

`path` 를 절대경로로 박는다. 학습 자식 프로세스의 CWD가 무엇이든 데이터셋을 찾게 하기 위함이다.
`task` 키는 ultralytics가 무시하지만, 학습 러너가 **모델 태스크와 일치하는지 검증**하는 데 쓴다.

### `manifest.json`

무엇으로 어떻게 만들었는지 전부 남긴다. 실험 재현과 데이터 점검의 근거다.

```json
{
  "dataset_id": "20260729-150908-3ae7c1",
  "task": "obb",
  "spec": { "video_ids": ["..."], "splits": {...}, "split_mode": "chunk", "seed": 0, ... },
  "counts": {"train": 496, "valid": 93, "test": 31},
  "objects": {"train": 2540, "valid": 470, "test": 110},
  "class_histogram": {"Panther_II": 1802, "VIDAR": 890},
  "per_video_frames": {"2026...049310": 410},
  "sources": [
    {"name": "preprocessed_obb", "prefix": "preprocessed_obb",
     "ref": "C:/project/tracker_py/train_data/preprocessed_obb",
     "dir": "C:/project/tracker_py/train_data/preprocessed_obb",
     "added": {"train": 388, "valid": 39, "test": 0},
     "class_map": {"Panther_II": 2, "VIDAR": 6}}
  ],
  "excluded": {"미승인(pending)": 88, "미확정/목록 외 클래스 포함": 12},
  "warnings": ["샘플이 10개 미만인 클래스: ['Strv_101']"]
}
```

---

## 4. 클래스 id 규약

**클래스 목록(`workspace/classes.json`)의 순서가 곧 클래스 id 다.**

기본값은 `tracker_py/train_data/preprocessed_obb/data.yaml` 과 같은 순서로 둔다.
기존 데이터셋과 섞을 때 id가 어긋나면 정답이 통째로 뒤바뀌기 때문이다.

```
0 Ikv_91_105   1 Jagdtiger   2 Panther_II   3 Strv_101
4 Strv_103A    5 Tiger_II_10.5_             6 VIDAR
```

`PUT /api/classes` 는 **추가와 이름 수정만** 허용하고 순서 변경·삭제를 거부한다.
클래스를 늘려야 하면 항상 목록 **뒤에** 붙인다.

라벨 JSON은 `class_name`(문자열)을 정답으로 들고 있고 `class_id` 는 캐시된 인덱스다.
데이터셋 빌드 시점에 이름 → 인덱스를 다시 계산하므로, 이름만 맞으면 안전하다.

---

## 5. 분할(split) 방식

인접 프레임은 사실상 같은 그림이다. 무작위로 나누면 train과 valid에 같은 장면이 들어가
검증 mAP가 부풀려지고, 실전 성능과 괴리가 생긴다.

| `split_mode` | 묶는 단위 | 특징 |
|---|---|---|
| `chunk` (기본) | 연속 프레임 `chunk_size`(기본 30)장 | 누수를 크게 줄이면서 비율도 맞음 |
| `video` | 영상 전체 | 가장 엄격. 영상이 10편 이상일 때 권장 |
| `random` | 프레임 1장 | 프레임 간 상관이 없을 때만 |

배분은 그룹을 `seed` 로 섞은 뒤, **목표 비율 대비 가장 뒤처진 split에 그룹을 주는** 방식이다.
그룹 크기가 제각각이어도(영상 단위 분할 등) 목표 비율에 가깝게 수렴한다.

> 프레임 수가 적어 그룹이 1~2개뿐이면 한 split으로 몰린다. 그래서 `chunk` 모드는
> 최소 10개 그룹이 나오도록 블록 크기를 자동으로 줄인다.

---

## 6. 기존 데이터셋 등록과 병합

`tracker_py/train_data/preprocessed_obb` 가 이 파이프라인의 **학습 직전 표준 구조**다.
빌드 산출물(§3)이 그 구조와 같으므로, 반대 방향(기존 폴더 → 파이프라인)도 성립한다.

```
tracker_py/train_data/preprocessed_obb/
├── data.yaml                 # names + task
├── train/{images,labels}/    # labels: class x1 y1 … x4 y4 (정규화)
└── valid/{images,labels}/
```

`train/`, `valid`(=`val`), `test/` 별칭을 모두 인식하고, `data.yaml` 이 없어도 폴더 구조와
**라벨 토큰 수(9=obb, 5=detect)** 로 태스크를 추정한다.

### 미리 보기 → 등록

```http
GET  /api/datasets/inspect?path=C:/project/tracker_py/train_data/preprocessed_obb
POST /api/datasets/import   {"path": "...", "copy": false}
```

`inspect` 는 아무것도 바꾸지 않고 태스크·클래스·split 별 장수만 돌려준다. 등록 전에
"이 폴더가 정말 내가 생각한 그것인지" 를 확인하는 용도다.

| `copy` | 동작 | 쓰는 상황 |
|---|---|---|
| `false` (기본) | 원본 폴더를 **참조만** 한다. `manifest.source_dir` 에 경로를 남긴다 | 수천 장을 중복 저장하지 않는다 |
| `true` | 워크스페이스로 복사(가능하면 하드링크)해 **스냅샷**으로 고정 | 원본이 바뀔 수 있거나 실험을 재현해야 할 때 |

참조 등록은 원본이 삭제·수정되면 학습이 깨진다. 그 대가로 디스크를 쓰지 않는다.
어느 쪽이든 학습에는 똑같이 쓸 수 있다.

### 새 라벨과 함께 병합

```http
POST /api/datasets
{
  "video_ids": ["20260729-150904-049310"],
  "base_datasets": ["C:/project/tracker_py/train_data/preprocessed_obb", "<dataset_id>"]
}
```

`base_datasets` 항목은 **데이터셋 id 또는 서버 폴더 경로**다. `video_ids: []` 로 두면
영상 없이 기존 데이터셋만 합친다.

규칙은 셋이다.

1. **클래스는 이름으로 매핑한다.** 소스의 `names` 를 현재 클래스 목록 인덱스로 다시 쓴다
   (`manifest.sources[].class_map`). id를 그대로 믿으면 순서가 다른 두 데이터셋을 합쳤을 때
   정답이 통째로 뒤바뀐다. 목록에 없는 이름이 있으면 등록을 거부하고, `PUT /api/classes` 로
   먼저 추가하게 한다.
2. **원래 분할을 유지한다.** 병합된 train은 train으로, valid는 valid로 들어간다.
   남의 valid를 내 train으로 섞으면 과거 실험과 점수를 비교할 수 없다.
3. **태스크가 다르면 거부한다.** 회전박스 라벨(9토큰)과 축정렬 라벨(5토큰)은 한 데이터셋에
   섞을 수 없다.

파일명은 소스별 접두사(`manifest.sources[].prefix`)를 붙여 stem 충돌을 막는다.

---

## 7. 태스크 선택: obb(기본) vs detect

기본값은 **`obb`** 다(`NIT_TRAIN_TASK`). 기본 가중치도 회전박스 모델
`test_model/yolo26l-obb.pt`(head = `OBB26`)다.

드론 상공 촬영은 표적이 임의 방향으로 놓인다. 축정렬 박스는 비스듬한 차량을 감쌀 때 배경을
크게 포함해 IoU 학습 신호가 흐려진다. 운영 추론(`tracker_py`)도 이미 OBB 모델을 쓰고 있고
기존 학습 데이터(`train_data/preprocessed_obb`)도 OBB다. 그래서 OBB를 기본으로 둔다.

| 하려는 것 | 데이터셋 `task` | 가중치 |
|---|---|---|
| 회전 박스(OBB) 학습 — 기본 | `obb` | `test_model/yolo26l-obb.pt` (기본값 그대로) |
| 축정렬 박스 학습 | `detect` | `test_model/yolo26l.pt` 등 detect 가중치 |

둘이 어긋나면 학습 러너가 **시작 직후** 다음 오류로 막는다(ultralytics가 라벨 컬럼 수
오류로 한참 뒤에 죽는 것을 방지).

```
모델 태스크(detect)와 데이터셋 태스크(obb)가 다릅니다.
데이터셋을 task=detect 로 다시 빌드하거나, task=obb 용 가중치를 쓰세요.
```

가중치의 태스크는 파일을 열어 head 를 보고 판단하고, 파일명에 `obb` 가 있으면 그것도 참고한다.
detect로 가려면 `NIT_TRAIN_TASK=detect` 와 `NIT_TRAIN_BASE_MODEL` 을 함께 바꾸거나,
요청마다 `task` 와 `model` 을 지정한다.

라벨 원본은 폴리곤이므로(§1) **태스크를 바꿔도 라벨을 다시 만들 필요는 없다.**
데이터셋만 다시 빌드하면 된다.

---

## 8. 학습 전 전처리 (추론 tracker_py 와 동일 엔진)

프레임을 **디스크에 저장하기 전에** 다듬는 단계다. 핵심 원칙은 **학습 이미지 = 추론 입력**
이다. 그래서 추론 서비스(tracker_py)가 추론 직전에 돌리는 전처리를 **그대로 떼어 온**
`preprocess_vendor` 패키지(`src/v1/preprocess_vendor/`)를 사용한다. 자체 구현이 아니라 tracker_py
의 실제 모듈(DCP 안개 제거, CLAHE, 다중지표 판정)을 복사한 것이라, 추론과 알고리즘·기본값이
같다. 오케스트레이션은 `services/preprocess.py`, 적용 지점은 `services/autolabel.py` 의 추출 루프다.

추론 `run_all` 과 동일한 순서로 **한 번만** 적용한다:
`야간 보정(dark) → 안개 제거(fog) → CLAHE(quality) → (선택적) 다운스케일`.

세 전처리는 **서로 독립 스위치**다(자유롭게 조합). `auto` 를 켜면 프레임마다 tracker_py 와
동일한 **다중지표(밝기·대비·채도·선명도)** 로 저조도/안개를 판정해 해당 프레임에만, 끄면 켠
전처리를 **모든 프레임**에 강제 적용한다. 모든 값의 우선순위는 **추출 요청 > 영상별 저장값
('구간' 단계 지정) > 서버 기본값** 이다.

적용된 설정과 프레임별 판정 결과는 라벨 문서에 남는다.

```json
{ "daynight": "night", "dehaze": true, "clahe": true,
  "preprocess": {"auto": true, "lowlight": true, "dehaze": true, "clahe": true,
                 "dark_th": 60.0, "dehaze_omega": 0.8, "clahe_clip": 2.0, ...} }
```

추출 잡 결과에는 집계와 사용된 야간 엔진이 담긴다:
`"daynight_counts": {"day": 320, "night": 80, "off": 0}`, `"dehaze_frames": 400`,
`"clahe_frames": 400`, `"lowlight_engine": "zero_dce++"`.

### 8.1 야간 보정 (low-light / dark)

어두운 프레임을 그대로 학습에 넣으면 저조도 구간에서 표적 특징이 뭉개진다. `lowlight` 를
켜고 `auto` 면 tracker_py 와 동일한 다중지표 판정으로 **저조도로 분류된 프레임에만** 보정을
건다(밝기 임계 `dark_th` 기본 60). 자동 판정은 프레임 단위라 한 영상 안 낮→밤 전환도 따라간다.

보정 엔진은 두 가지다:

- **Zero-DCE++ (추론과 동일, 기본)** — `preprocess_vendor/zero_dce_weights/Epoch99.pth`(약 52KB,
  리포에 포함)를 로드해 추론 서비스와 **완전히 동일한** 저조도 보정을 한다. 초경량 CNN 이라
  GPU 없이 **CPU 로도** 오프라인 프레임 추출에 충분하다(감마 파라미터는 무시됨).
  `NIT_TRAIN_ZERODCE_WEIGHTS` 로 다른 가중치를 지정할 수도 있다.
- **감마 + CLAHE 폴백** — 가중치가 없을 때만. 감마(전역 밝기) + LAB L 채널 CLAHE(국소 대비).
  단, 아주 어두운 야간에서는 **과노출(하얗게 뜸)** 되기 쉬워 Zero-DCE++ 와 결과가 다르다.
  가급적 위 가중치를 유지해 추론과 일치시킨다.

> 현재 어떤 엔진이 걸렸는지는 `GET /api/meta` 의 `defaults.preprocess.lowlight_engine`
> (`zero_dce++` | `gamma_clahe`)로 확인한다.

자동이 틀리는 영상은 **학습 전 '구간' 단계에서** 끄거나 `auto=false`(강제)로 저장한다
(`PUT /api/videos/{id}/preprocess`).

### 8.2 안개 제거 (dehaze / fog)

원거리 드론/CCTV 영상은 안개·연무·미세먼지로 대비가 죽는다. `dehaze` 를 켜면 tracker_py 와
동일한 **Dark Channel Prior(DCP)** 로 대기산란(airlight)을 추정해 걷어낸다. 기본 파라미터도
추론과 같다(`dehaze_omega=0.80`, `dehaze_t0=0.4`, `dehaze_wsz=15`, `dehaze_scale=0.25`,
`dehaze_guide_r=20`). 추론 기본은 GPU DCP 지만 같은 알고리즘이라 CPU 로 돌려도 결과가 동일하다.

### 8.3 화질 향상 (CLAHE / quality)

추론 파이프라인의 Stage2(기본 ON)에 해당한다. LAB 의 L 채널에 CLAHE(clip 2.0, 8×8)를 걸어
국소 대비를 살린다. 추론이 전 프레임에 적용하므로 학습도 기본 ON 이며, 저조도/안개와 달리
자동 판정 없이 켜져 있으면 모든 프레임에 적용한다.

'구간' 화면의 미리보기로 보정 전/후를 눈으로 비교할 수 있다
(`GET /api/videos/{id}/frame?preprocess=1&auto=0&lowlight=1&dehaze=1&clahe=1`, 결과는
`X-Daynight`/`X-Dehaze`/`X-Clahe` 헤더).

> 이 보정은 학습 이미지에 굽히므로, 실운영 추론(tracker_py)도 같은 전처리를 입력에
> 적용해야 분포가 맞는다. 그래서 tracker_py 의 전처리를 복사해 재사용한다.

### 8.4 해상도 다운스케일 (선택)

추출 프레임은 기본적으로 **640×480** 으로 정규화해 저장한다. 운영 추론(tracker_py)이
입력을 640×480으로 정규화하기 때문이다. 학습 이미지가 다른 해상도면 객체의 픽셀 크기
분포가 어긋나 특히 작은 표적의 성능이 떨어진다(`imgsz=640` 학습과도 맞는다).

원본 해상도를 유지하고 싶으면 **선택으로 끌 수 있다** — 서버 기본값 `NIT_TRAIN_FRAME_RESIZE=0`,
또는 추출 요청/영상별 설정의 `resize:false`. 크기를 바꾸려면 `resize_width/height` 를 준다.

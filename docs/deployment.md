# 배포 & 버전관리 (Docker)

NIT_train(학습/라벨링 API)와 NIT_train_front(정적 화면)를 Docker 로 함께 띄우고,
이미지·코드 버전을 일관되게 추적하는 방법을 정리한다.

## 구성

| 서비스 | 이미지 | 포트 | 역할 |
|---|---|---|---|
| `nit-train` | `nit-train:<버전>` | 8888 | FastAPI 학습/라벨링 API (+ Swagger `/docs`) |
| `nit-train-front` | `nit-train-front:<버전>` | 8890 | 프레임워크 없는 정적 SPA |

- 백엔드 base 이미지는 `ultralytics/ultralytics:latest` (torch/CUDA/opencv/ultralytics 포함).
- 프런트는 `python:3.12-slim` + 표준 라이브러리 `server.py` 만으로 서빙(의존성 0, 폐쇄망 OK).
- compose 파일은 `NIT_train/docker-compose.yml`. 프런트는 `../NIT_train_front` 를 빌드 컨텍스트로 쓴다.

## 빠른 시작

```powershell
cd C:\project\NIT_train
copy .env.example .env          # 필요시 값 수정
.\scripts\release.ps1 -Up       # 버전 스탬프 빌드 + 기동
```

Linux/macOS:

```bash
cd NIT_train
cp .env.example .env
UP=1 ./scripts/release.sh
```

수동(스크립트 없이):

```bash
docker compose up -d --build
```

- 화면: `http://<호스트>:8890`
- API/Swagger: `http://<호스트>:8888/docs`
- GPU 없는 호스트: `-f docker-compose.yml -f docker-compose.cpu.yml` (또는 스크립트 `-Cpu` / `CPU=1`).

## 영속 데이터 (볼륨)

이미지는 언제든 갈아끼운다. **상태는 전부 볼륨에** 둔다.

| 호스트 경로 | 컨테이너 | 내용 |
|---|---|---|
| `./workspace` | `/app/workspace` | 영상·프레임·라벨·데이터셋·학습결과 (유일한 보존 대상) |
| `./test_model` | `/app/test_model` | 사전학습 가중치(`*.pt`). 이미지에 굽지 않고 마운트 |
| `./external_datasets` *(선택)* | `/app/external_datasets:ro` | 외부 데이터 재임포트용 원본. 평소엔 불필요 |

> 임포트된 외부 데이터의 학습 이미지는 이미 `workspace/datasets` 로 복사돼 있으므로,
> 재임포트가 필요 없다면 `external_datasets` 볼륨은 열지 않아도 학습·추론이 전부 동작한다.

## 버전관리

버전은 **코드·이미지·배포**에서 한 값으로 흐른다.

1. **단일 출처 `VERSION` 파일** — SemVer(`MAJOR.MINOR.PATCH`). 릴리스마다 이 값을 올린다.
2. **이미지에 각인** — `scripts/release.*` 가 빌드 시 아래를 build-arg 로 주입:
   - `VERSION` (VERSION 파일)
   - `GIT_SHA` (`git rev-parse --short=12 HEAD`)
   - `BUILD_DATE` (UTC ISO8601)

   이 값들은 이미지 ENV(`NIT_TRAIN_VERSION`/`_GIT_SHA`/`_BUILD_DATE`)와
   OCI 라벨(`org.opencontainers.image.version/revision/created`)에 동시에 박힌다.
3. **런타임 노출** — 떠 있는 게 정확히 무엇인지 확인:
   ```bash
   curl http://<호스트>:8888/api/version
   # {"version":"1.0.0","git_sha":"...","build_date":"..."}
   ```
   Swagger 제목 버전과 `/api/meta` 의 `build` 필드에도 같은 값이 나온다.
   이미지 라벨 확인: `docker inspect --format '{{json .Config.Labels}}' nit-train:1.0.0`.
4. **이미지 태그** — 스크립트가 `nit-train:<버전>` 과 `nit-train:latest` 둘 다 태깅.
   롤백은 `NIT_TRAIN_TAG=<이전버전> docker compose up -d` 로 즉시.
5. **깃 태그(권장)** — 릴리스 커밋에 맞춰:
   ```bash
   git tag -a v1.0.0 -m "release 1.0.0" && git push origin v1.0.0
   ```

### 릴리스 체크리스트

1. `VERSION` 을 올린다 (예: `1.0.0` → `1.1.0`).
2. 커밋: `git commit -am "release: v1.1.0"`.
3. 빌드: `.\scripts\release.ps1 -Up` (또는 `UP=1 ./scripts/release.sh`).
4. `curl .../api/version` 로 버전 확인.
5. 깃 태그 푸시: `git tag -a v1.1.0 -m ... && git push --tags`.

## 커밋하면 안 되는 것 (이미 `.gitignore`/`.dockerignore` 처리됨)

- `workspace/`, `runs/`, `outputs/` — 대용량 작업 산출물(재생성 가능).
- `*.pt`, `*.engine`, `*.onnx` — 가중치/엔진 캐시.
- `external_datasets/` 의 이미지·라벨(약 2.4만 장) — `README.md` 만 추적.
- `.env` — 로컬 배포 설정. `.env.example` 만 커밋.
- `__pycache__/`, `*.pyc` — 컴파일 캐시(과거 추적분은 언트랙 완료).

## 주요 환경변수

| 변수 | 기본 | 설명 |
|---|---|---|
| `NIT_TRAIN_DEVICE` | `0` | GPU 인덱스 또는 `cpu` |
| `NIT_TRAIN_WORKERS` | `4` | 학습 DataLoader 워커(RAM 에 맞춰 조정) |
| `NIT_TRAIN_API` *(프런트)* | 빈값 | 비우면 브라우저가 `현재 호스트:8888` 사용. 백엔드가 다른 호스트면 `http://IP:8888` |
| `NIT_TRAIN_TAG` | `latest` | compose 가 쓸 이미지 태그(= 배포 버전) |

전체 목록은 `src/v1/core/config.py` 참고.

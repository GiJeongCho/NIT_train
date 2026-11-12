# 윈도우에서 Docker 로 마는 법

NIT_train(학습/라벨링 API) + NIT_train_front(화면)를 **Windows + Docker Desktop**에서 빌드·실행하는 순서.
리눅스 서버 배포는 `docs/deployment.md` 와 동일하고, 여기서는 윈도우 특유의 함정만 더 짚는다.

## 0. 사전 준비 (한 번만)

1. **Docker Desktop** 설치(완료). 실행 후 우하단 고래 아이콘이 초록이면 준비됨.
2. **WSL2 백엔드** 사용 확인: Docker Desktop → Settings → General → *Use the WSL 2 based engine* 체크.
3. **(GPU 학습을 쓸 때만)** NVIDIA GPU + 최신 드라이버. Docker Desktop(WSL2)은 GPU 를 자동 중계한다.
   PowerShell 에서 아래가 `nvidia-smi` 표를 출력하면 GPU 통과 성공:
   ```powershell
   docker run --rm --gpus all ultralytics/ultralytics nvidia-smi
   ```
   표가 안 뜨면 → 이 문서 맨 아래 "GPU 가 안 잡힐 때" 로. GPU 없이도 CPU 로 돌릴 수 있다(느림).

## 1. 구성

| 서비스 | 포트 | 접속 |
|---|---|---|
| `nit-train` (백엔드 API) | 8888 | http://localhost:8888/docs |
| `nit-train-front` (화면) | 8890 | http://localhost:8890 |

- compose: `C:\project\NIT_train\docker-compose.yml` (프런트는 `..\NIT_train_front` 를 빌드).
- 백엔드 base 이미지 `ultralytics/ultralytics` 에 torch/CUDA/opencv 가 들어 있어 따로 안 깐다.

## 2. 빌드 & 실행 (PowerShell)

```powershell
cd C:\project\NIT_train
copy .env.example .env        # 처음 한 번. 필요하면 값 수정
```

**GPU 가 있으면:**
```powershell
.\scripts\release.ps1 -Up
```

**GPU 가 없으면(CPU):**
```powershell
.\scripts\release.ps1 -Cpu -Up
```

스크립트가 싫으면 compose 직접:
```powershell
docker compose up -d --build                                   # GPU
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d --build   # CPU
```

빌드는 최초 1회 ultralytics 이미지(수 GB)를 받느라 오래 걸린다. 두 번째부터는 캐시로 빠르다.

## 3. 떴는지 확인

```powershell
docker compose ps                     # State 가 running / healthy 인지
docker compose logs -f nit-train      # 백엔드 로그 (Ctrl+C 로 빠져나오기)
```

- 화면: 브라우저로 http://localhost:8890
- API 문서(Swagger): http://localhost:8888/docs
- 버전 확인: http://localhost:8888/api/version  → `{"version":"1.0.0","git_sha":"...","build_date":"..."}`

## 4. 데이터는 어디에 남나 (볼륨)

컨테이너를 지워도 **작업은 호스트 폴더에 남는다**. compose 가 아래를 바인드 마운트한다.

| 윈도우 폴더 | 컨테이너 | 내용 |
|---|---|---|
| `C:\project\NIT_train\workspace` | `/app/workspace` | 영상·프레임·라벨·데이터셋·학습결과 (유일한 보존 대상) |
| `C:\project\NIT_train\test_model` | `/app/test_model` | 사전학습 가중치(`*.pt`) |

> Docker Desktop(WSL2)은 `C:` 드라이브를 자동 공유하므로 추가 설정이 필요 없다.
> 혹시 "파일 공유" 오류가 나면 Settings → Resources → File Sharing 에 `C:\project` 를 추가.

## 5. 중지 / 삭제 / 롤백

```powershell
docker compose stop                   # 잠깐 멈춤(데이터·컨테이너 유지)
docker compose down                   # 컨테이너 제거(볼륨=호스트 폴더는 그대로)
# 특정 버전으로 되돌리기(이미지가 있으면):
$env:NIT_TRAIN_TAG="1.0.0"; docker compose up -d
```

## 6. 윈도우에서 자주 막히는 것

**GPU 가 안 잡힐 때**
- Docker Desktop → Settings → Resources → WSL Integration 켜져 있는지.
- `docker run --rm --gpus all ultralytics/ultralytics nvidia-smi` 로 단독 확인.
- 그래도 안 되면 CPU 로: `.\scripts\release.ps1 -Cpu -Up` (학습은 많이 느리다).

**`exec /bin/sh: ... : no such file` 또는 스크립트가 CRLF 라 깨질 때**
- 셸 스크립트는 LF 여야 한다. 레포에 `.gitattributes` 로 `*.sh=LF` 를 강제해 뒀다.
  직접 받은 파일이 CRLF 면 메모장++ 등에서 LF 로 바꿔 저장.

**포트 충돌 (8888/8890 이미 사용 중)**
- `docker-compose.yml` 의 `ports` 를 `"18888:8888"` 처럼 호스트 쪽만 바꾼다.

**학습 후반 epoch 에서 컨테이너가 죽을 때(공유메모리 부족)**
- compose 에 `shm_size: "8gb"` 가 이미 있다. 더 필요하면 값을 키운다.
- WSL2 전체 메모리는 `C:\Users\<나>\.wslconfig` 의 `[wsl2] memory=` 로 조절.

**빌드가 너무 느리거나 디스크를 많이 먹을 때**
- 최초 1회 ultralytics base(수 GB) 다운로드 때문. 이후엔 캐시. `docker system df` 로 용량 확인, `docker system prune` 로 정리.

## 7. 버전 각인(선택)

`scripts\release.ps1` 로 빌드하면 `VERSION` 파일 값 + git 커밋 + 빌드시각이 이미지에 박히고
`http://localhost:8888/api/version` 으로 확인된다. 릴리스 규칙·태깅은 `docs/version_management.md` 참고.

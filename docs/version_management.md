# 버전관리 가이드

NIT_train 저장소의 깃 사용 규칙. "무엇을 올리고, 무엇을 올리지 않고, 어떻게 릴리스하나"를 한 곳에 정리한다.

## 원격 저장소

| 이름 | 주소 | 용도 |
|---|---|---|
| `origin` | `https://github.com/GiJeongCho/NIT_train.git` | 개인/백업 |
| `gitlab` | `https://git.biz.ppsystem.co.kr/NIT/NIT_train_fastapi.git` | 회사(사내망) |

- 기본 작업 브랜치는 `main`.
- 푸시 대상은 목적에 맞게 고른다: 사내 공유는 `gitlab`, 개인 백업은 `origin`.
  ```bash
  git push gitlab main      # 회사
  git push origin main      # 개인
  ```

## 브랜치 & 커밋

- 작은 변경은 `main`에 직접, 큰 작업은 `feat/<이름>` 브랜치에서 하고 병합한다.
- 커밋 메시지는 "무엇을 왜"가 한 줄에 드러나게. 접두어 예: `feat:`, `fix:`, `docs:`, `refactor:`, `chore:`.
  ```
  feat: 데이터셋 zip 내보내기 추가
  fix: COCO 임포트 시 클래스 인덱스 어긋남 수정
  ```

## 푸시 전 습관 — 먼저 당기고(rebase) 올린다

원격에 먼저 들어간 커밋이 있으면 `push`가 거부된다(`rejected … fetch first`).
웹에서 파일을 바로 고쳤거나 다른 PC에서 올린 경우가 흔하다. 항상 아래 순서로:

```bash
git pull --rebase origin main    # (또는 gitlab)
git push origin main
```

`--rebase`를 쓰면 불필요한 머지 커밋 없이 내 커밋이 원격 커밋 위에 깔끔하게 쌓인다.

## 올리면 안 되는 것 (`.gitignore`로 처리됨)

대용량·재생성 가능·비밀 정보는 커밋하지 않는다.

| 대상 | 이유 |
|---|---|
| `workspace/`, `runs/`, `outputs/` | 영상·프레임·라벨·데이터셋·학습결과. 수 GB, 재생성 가능 |
| `external_datasets/`의 이미지·라벨(약 2.4만 장) | 외부 데이터 원본. `README.md`만 추적해 "무엇이 있어야 하는지"만 남긴다 |
| `*.pt`, `*.engine`, `*.onnx` | 가중치/엔진 캐시(수십~수백 MB) |
| `.env` | 로컬 설정/비밀값 |
| `__pycache__/`, `*.pyc` | 파이썬 컴파일 캐시 |

> 과거에 실수로 추적되던 `*.pyc`는 `git rm --cached`로 제거했다. 새로 생긴 건 `.gitignore`가 막는다.
> 데이터를 꼭 공유해야 하면 깃에 넣지 말고 별도 스토리지(사내 공유 폴더/오브젝트 스토리지)로 전달한다.

## 줄바꿈 정책 (`.gitattributes`)

윈도에서 개발하고 리눅스 환경에서 실행하므로 줄바꿈을 통일한다.

- 텍스트 기본 `LF`, 셸 스크립트(`*.sh`) `LF`, PowerShell(`*.ps1`) `CRLF`.
- `*.pt`/`*.png`/`*.zip` 등은 바이너리로 지정해 변환·diff를 막는다.

## 릴리스 & 버전 태깅

버전은 [SemVer](https://semver.org/lang/ko/) `MAJOR.MINOR.PATCH`를 쓴다.

- **MAJOR**: 호환이 깨지는 API/데이터 형식 변경
- **MINOR**: 하위호환 기능 추가(예: 새 엔드포인트, 새 화면)
- **PATCH**: 버그 수정

릴리스할 때 커밋에 깃 태그를 달아 "이 시점"을 고정한다.

```bash
git tag -a v1.1.0 -m "데이터 페이지 + zip 내보내기"
git push origin v1.1.0      # 필요하면 gitlab 에도
# 전체 태그 한 번에: git push origin --tags
```

지난 태그 이후 바뀐 것 보기:

```bash
git log --oneline v1.0.0..HEAD
```

## 되돌리기 빠른 참조

```bash
git restore <파일>                 # 워킹트리 수정 취소(아직 커밋 전)
git restore --staged <파일>        # add 취소(스테이징 해제)
git revert <커밋>                  # 특정 커밋을 되돌리는 새 커밋(공유된 커밋에 안전)
git reset --hard <커밋>            # 로컬 전용. 그 커밋으로 강제 복귀(주의: 작업 소실)
```

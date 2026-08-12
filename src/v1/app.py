"""
app.py
======

FastAPI 앱 팩토리.

lifespan 에서:
  startup  → 워크스페이스 생성/쓰기 확인 → 클래스 목록 초기화 → (선택) 가중치 워밍업
  shutdown → 없음(학습은 자식 프로세스라 앱 종료와 독립적으로 계속 돈다)

무거운 모델 로드는 기본적으로 startup 에서 하지 않는다. 이 서비스의 주 작업은 학습이고,
학습은 자식 프로세스가 자기 모델을 따로 올린다. 자동 라벨을 자주 쓰는 배치라면
`NIT_TRAIN_PRELOAD=1` 로 미리 올려 첫 추출의 로딩 지연을 없앤다.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import router
from core.config import get_settings
from core import store


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    store.ensure_workspace()
    print(f"[init] workspace={s.workspace_dir}", flush=True)

    # 클래스 목록 파일을 미리 만들어 둔다(프런트가 첫 요청부터 목록을 받게).
    from services import classes as classes_svc
    names = classes_svc.names()
    print(f"[init] 클래스 {len(names)}개: {names}", flush=True)

    if not s.base_model.exists():
        # 기본 가중치가 없어도 학습/데이터셋 기능은 동작하므로 죽이지 않고 경고만 한다.
        print(f"[warn] 기본 가중치 없음: {s.base_model} "
              f"(자동 라벨을 쓰려면 모델을 넣거나 NIT_TRAIN_BASE_MODEL 로 지정)", flush=True)
    elif s.preload_model:
        from services.detector import get_detector
        print("[init] 기본 가중치 로드 + 워밍업...", flush=True)
        get_detector().warmup()

    print(f"[init] 준비 완료 → http://{s.host}:{s.port}/docs", flush=True)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="NIT 학습 API (MLOps)",
        version="1.0.0",
        description=(
            "드론/CCTV 영상 → 구간 선별 → 자동 라벨 초안 → 사람 검수 → 데이터셋 → "
            "YOLO 학습 → 모델 승격까지의 파이프라인 API."
        ),
        lifespan=lifespan,
    )
    app.include_router(router)

    # 프런트(NIT_train_front)는 별도 서버에서 서빙되고 이 API 를 교차 출처로 호출한다.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        # 전처리 미리보기가 판정 결과를 헤더로 읽는다(교차 출처라 노출 필요).
        expose_headers=["X-Daynight", "X-Dehaze", "X-Clahe"],
    )

    # 라우트를 얇게 유지하기 위해 서비스 계층이 올리는 표준 예외를 여기서 HTTP 로 번역한다.
    # (services 는 HTTP 를 모르고, routes 는 try/except 로 덮이지 않는다.)
    @app.exception_handler(ValueError)
    async def _bad_request(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(KeyError)
    async def _not_found(request: Request, exc: KeyError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc).strip("'\"")})

    @app.exception_handler(FileNotFoundError)
    async def _gone(request: Request, exc: FileNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=410, content={"detail": str(exc)})

    return app


app = create_app()

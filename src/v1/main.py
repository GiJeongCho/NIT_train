"""
main.py
=======

엔트리포인트. uvicorn 으로 FastAPI 앱(app.py:app) 을 구동한다.

실행 (둘 다 동일):
    cd src/v1
    python main.py
또는:
    cd src/v1
    uvicorn app:app --host 0.0.0.0 --port 8888 --reload

핫리로드: 환경변수 NIT_TRAIN_RELOAD=1 로 켜거나, 위 uvicorn 명령에 --reload 사용.
모든 동작은 환경변수로 조정 (core/config.py 참고).
"""

from __future__ import annotations

import os

import uvicorn

from core.config import get_settings


def main() -> None:
    s = get_settings()
    reload = os.getenv("NIT_TRAIN_RELOAD", "0").strip().lower() in ("1", "true", "yes", "on")
    uvicorn.run(
        "app:app",
        host=s.host,
        port=s.port,
        reload=reload,
        # 단일 워커. 잡 레지스트리와 로드된 가중치가 프로세스 메모리에 있으므로
        # 워커를 늘리면 잡 진행률 조회가 다른 워커로 가서 404 가 된다.
        workers=1,
        log_level="info",
    )


if __name__ == "__main__":
    main()

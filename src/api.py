"""
api.py
======

편의 런처. 실제 앱은 `src/v1/` 에 있다(개발표준: 앱 루트는 `src/v1/`).

    python src/api.py

내부적으로 `src/v1` 을 sys.path 에 올린 뒤 `v1/main.py` 와 같은 방식으로 구동한다.
CWD 와 무관하게 뜨므로 배치 스크립트/서비스 등록에 쓰기 편하다.
정석 실행은 `cd src/v1 && python main.py` 다.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent / "v1"


def main() -> None:
    sys.path.insert(0, str(APP_DIR))
    # uvicorn 이 "app:app" 문자열을 import 할 수 있어야 하므로 CWD 도 맞춘다(reload 대응).
    os.chdir(APP_DIR)
    from main import main as run  # noqa: PLC0415 - sys.path 설정 후에 import 해야 한다

    run()


if __name__ == "__main__":
    main()

"""
services/train_runner.py
========================

학습 자식 프로세스의 엔트리포인트. **앱 모듈을 import 하지 않는** 독립 스크립트다.

    python train_runner.py <run_dir>

왜 별도 프로세스인가
- ultralytics 학습은 GPU 메모리를 오래 점유하고, DataLoader 워커가 죽으면
  프로세스 전체가 함께 넘어간다. API 프로세스 안에서 돌리면 학습 사고가 그대로
  서비스 장애가 된다.
- Windows 는 멀티프로세싱이 spawn 이라, 워커가 부모 모듈을 다시 import 한다.
  FastAPI 앱을 부모로 두면 앱이 워커마다 재초기화되는 부작용이 생긴다.
- 중단(stop)이 프로세스 종료로 확실하게 끝난다.

진행 상황은 `<run_dir>/state.json` 에 쓰고, stdout/stderr 는 부모가 `train.log` 로
리다이렉트한다. 부모(API)는 이 두 파일만 읽어 상태를 보고한다.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class State:
    """state.json 갱신기. 부모가 언제 읽어도 깨지지 않게 원자적으로 쓴다."""

    def __init__(self, path: Path, initial: dict) -> None:
        self.path = path
        self.data = dict(initial)
        self.flush()

    def update(self, **kw) -> None:
        self.data.update(kw)
        self.flush()

    def flush(self) -> None:
        self.data["heartbeat"] = _now()
        tmp = self.path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as e:
            print(f"[runner] state 저장 실패: {e}", flush=True)


def _read_yaml_task(data_yaml: Path) -> str:
    """data.yaml 의 `task:` 값만 가볍게 읽는다(PyYAML 없이도 동작하게)."""
    try:
        for line in data_yaml.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("task:"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python train_runner.py <run_dir>", flush=True)
        return 2
    run_dir = Path(sys.argv[1]).resolve()
    spec = json.loads((run_dir / "spec.json").read_text(encoding="utf-8"))

    state = State(run_dir / "state.json", {
        "run_id": run_dir.name,
        "status": "running",
        "pid": os.getpid(),
        "epoch": 0,
        "epochs": int(spec.get("epochs") or 0),
        "metrics": {},
        "started_at": _now(),
        "finished_at": None,
        "error": None,
        "weights": {},
    })

    try:
        from ultralytics import YOLO

        resume = bool(spec.get("resume"))
        out_dir = run_dir / spec.get("name", "train")
        if resume:
            last = out_dir / "weights" / "last.pt"
            if not last.exists():
                raise FileNotFoundError(f"이어서 학습할 체크포인트가 없습니다: {last}")
            print(f"[runner] resume: {last}", flush=True)
            model = YOLO(str(last))
        else:
            model = YOLO(str(spec["model"]))

        model_task = str(getattr(model, "task", "") or "")
        data_task = _read_yaml_task(Path(spec["data"]))
        if not resume and data_task and model_task and data_task != model_task:
            # 여기서 막지 않으면 ultralytics 가 라벨 컬럼 수 오류로 한참 뒤에 죽는다.
            raise ValueError(
                f"모델 태스크({model_task})와 데이터셋 태스크({data_task})가 다릅니다. "
                f"데이터셋을 task={model_task} 로 다시 빌드하거나, "
                f"task={data_task} 용 가중치(예: 이름에 'obb' 가 든 모델)를 쓰세요."
            )

        def on_epoch_end(trainer) -> None:
            metrics = {}
            for k, v in (getattr(trainer, "metrics", None) or {}).items():
                try:
                    metrics[str(k)] = round(float(v), 6)
                except (TypeError, ValueError):
                    continue
            state.update(
                epoch=int(getattr(trainer, "epoch", 0)) + 1,
                epochs=int(getattr(trainer, "epochs", 0) or state.data.get("epochs") or 0),
                metrics=metrics,
                best_fitness=(float(trainer.best_fitness)
                              if getattr(trainer, "best_fitness", None) is not None else None),
            )

        model.add_callback("on_fit_epoch_end", on_epoch_end)

        if resume:
            model.train(resume=True)
        else:
            model.train(
                data=spec["data"],
                epochs=int(spec["epochs"]),
                imgsz=int(spec["imgsz"]),
                batch=int(spec["batch"]),
                device=spec["device"],
                workers=int(spec["workers"]),
                patience=int(spec.get("patience", 50)),
                project=str(run_dir),
                name=str(spec.get("name", "train")),
                exist_ok=True,
                **(spec.get("extra") or {}),
            )

        weights = {}
        for tag in ("best", "last"):
            p = out_dir / "weights" / f"{tag}.pt"
            if p.exists():
                weights[tag] = str(p)
        state.update(status="done", finished_at=_now(), weights=weights)
        print(f"[runner] 완료: {weights}", flush=True)
        return 0

    except BaseException as e:  # noqa: BLE001 - 실패 이유를 반드시 state 에 남긴다
        state.update(status="error", finished_at=_now(),
                     error=f"{type(e).__name__}: {e}")
        print(f"[runner] 실패: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        return 1


# Windows 멀티프로세싱(spawn) 대응: DataLoader 워커가 이 모듈을 다시 import 하므로
# 진입점은 반드시 이 가드 안에 둔다.
if __name__ == "__main__":
    sys.exit(main())

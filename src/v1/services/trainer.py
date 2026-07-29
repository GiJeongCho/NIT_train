"""
services/trainer.py
===================

학습 실행(run) 관리. 실제 학습은 `train_runner.py` 자식 프로세스가 하고, 여기서는
요청 검증 · 프로세스 기동/중단 · 상태 집계만 한다.

`test/yolo.py` 의 검증된 하이퍼파라미터(epochs=100 / imgsz=640 / batch=16 /
workers=4)를 기본값으로 쓰고, 기본 가중치는 `test_model/yolo26l.pt` 다.

run 폴더::

    runs/<run_id>/
    ├── spec.json      # 요청 원본 + 해석된 절대경로 (재현용)
    ├── state.json     # 자식 프로세스가 갱신 (status/epoch/metrics/heartbeat)
    ├── train.log      # 자식 stdout/stderr
    └── train/         # ultralytics 산출물 (weights/best.pt, results.csv, plots)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional

from core.config import get_settings
from core import store
from services import dataset as dataset_svc
from services.detector import infer_task, resolve_model_path

_RUNNER = Path(__file__).resolve().parent / "train_runner.py"

# 살아있는 자식 프로세스 핸들. API 를 재시작하면 비므로 pid 로 폴백 확인한다.
_procs: Dict[str, subprocess.Popen] = {}
_lock = threading.Lock()

# heartbeat 가 이 시간 이상 멈춘 running 은 죽은 것으로 본다(프로세스 강제종료/전원 차단).
_STALE_SEC = 900


def _python() -> str:
    """학습에 쓸 인터프리터. 기본은 API 를 돌리는 것과 같은 환경(NIT)."""
    cfg = get_settings().train_python
    return cfg if cfg else sys.executable


def _is_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        import psutil
        return psutil.pid_exists(int(pid))
    except ImportError:
        pass
    if os.name == "nt":
        try:
            out = subprocess.run(["tasklist", "/FI", f"PID eq {int(pid)}"],
                                 capture_output=True, text=True, timeout=5)
            return str(pid) in (out.stdout or "")
        except (OSError, subprocess.SubprocessError):
            return True  # 확인 실패 시 살아있다고 보수적으로 판단
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def start(raw_spec: Optional[dict] = None) -> dict:
    """학습을 시작하고 run 요약을 반환한다."""
    raw = dict(raw_spec or {})
    s = get_settings()

    dataset_id = str(raw.get("dataset_id") or "").strip()
    if not dataset_id:
        raise ValueError("dataset_id 가 필요합니다. 먼저 POST /api/datasets 로 데이터셋을 만드세요.")
    manifest = dataset_svc.get(dataset_id)
    data_yaml = store.dataset_yaml_path(dataset_id)
    if not data_yaml.exists():
        raise ValueError(f"data.yaml 이 없습니다(빌드 미완료?): {dataset_id}")

    model_path = resolve_model_path(raw.get("model"))
    if not model_path.exists():
        raise ValueError(f"모델 파일을 찾을 수 없습니다: {raw.get('model') or model_path}")

    ds_task = str(manifest.get("task") or "detect")
    hint = infer_task(model_path) or "detect"
    warnings = []
    if hint != ds_task:
        # 이름 기반 추정이라 확정은 아니다. 자식 프로세스가 실제 태스크로 다시 검증한다.
        warnings.append(
            f"모델 이름으로 추정한 태스크({hint})가 데이터셋 태스크({ds_task})와 다릅니다. "
            f"실제 태스크가 다르면 학습 시작 직후 실패합니다."
        )

    run_id = store.new_id()
    run_dir = store.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    spec = {
        "run_id": run_id,
        "name": "train",
        "dataset_id": dataset_id,
        "dataset_name": manifest.get("name"),
        "dataset_task": ds_task,
        "data": str(data_yaml),
        "model": str(model_path),
        "model_name": model_path.name,
        "epochs": int(raw.get("epochs") or s.train_epochs),
        "imgsz": int(raw.get("imgsz") or s.train_imgsz),
        "batch": int(raw.get("batch") or s.train_batch),
        "device": raw.get("device") if raw.get("device") is not None else s.device,
        "workers": int(raw.get("workers") if raw.get("workers") is not None else s.train_workers),
        "patience": int(raw.get("patience") or s.train_patience),
        "class_names": manifest.get("class_names"),
        "frames": manifest.get("counts"),
        "extra": dict(raw.get("extra") or {}),
        "resume": False,
        "note": str(raw.get("note") or ""),
        "created_at": store.now_iso(),
    }
    store.write_json(store.run_spec_path(run_id), spec)

    _spawn(run_id)
    return {"run_id": run_id, "spec": spec, "warnings": warnings}


def resume(run_id: str) -> dict:
    """중단된 학습을 `last.pt` 부터 이어서 한다 (test/yolo.py 의 resume 과 같은 동작)."""
    spec = store.read_json(store.run_spec_path(run_id), None)
    if not isinstance(spec, dict):
        raise KeyError(f"학습 run 을 찾을 수 없습니다: {run_id}")
    state = store.read_json(store.run_state_path(run_id), {}) or {}
    if state.get("status") == "running" and _is_alive(state.get("pid")):
        raise ValueError(f"이미 실행 중입니다: {run_id}")
    last = store.run_output_dir(run_id) / "weights" / "last.pt"
    if not last.exists():
        raise ValueError(f"이어서 학습할 체크포인트가 없습니다: {store.rel_to_workspace(last)}")

    spec["resume"] = True
    spec["resumed_at"] = store.now_iso()
    store.write_json(store.run_spec_path(run_id), spec)
    _spawn(run_id)
    return {"run_id": run_id, "resumed": True}


def _spawn(run_id: str) -> None:
    run_dir = store.run_dir(run_id)
    log_path = store.run_log_path(run_id)
    log = log_path.open("ab")
    log.write(f"\n===== {store.now_iso()} 학습 시작 (run {run_id}) =====\n".encode("utf-8"))
    log.flush()

    env = dict(os.environ)
    # 자식은 앱 모듈을 쓰지 않지만, 로그가 버퍼에 갇히면 진행 상황이 안 보인다.
    env["PYTHONUNBUFFERED"] = "1"

    creationflags = 0
    start_new_session = False
    if os.name == "nt":
        # 자식과 그 워커들을 하나의 프로세스 그룹으로 묶어 중단 시 통째로 정리한다.
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        start_new_session = True

    proc = subprocess.Popen(
        [_python(), str(_RUNNER), str(run_dir)],
        stdout=log, stderr=subprocess.STDOUT,
        cwd=str(_RUNNER.parent),
        env=env,
        creationflags=creationflags,
        start_new_session=start_new_session,
    )
    with _lock:
        _procs[run_id] = proc
    print(f"[trainer] run={run_id} pid={proc.pid} 시작", flush=True)


def stop(run_id: str) -> dict:
    """학습을 중단한다. 자식과 DataLoader 워커까지 함께 정리한다."""
    state = store.read_json(store.run_state_path(run_id), {}) or {}
    with _lock:
        proc = _procs.get(run_id)
    pid = proc.pid if proc is not None else state.get("pid")
    if not pid or not _is_alive(pid):
        return {"run_id": run_id, "stopped": False, "reason": "실행 중이 아닙니다"}

    if os.name == "nt":
        # taskkill /T 로 자식 트리(DataLoader 워커)까지 종료. terminate() 만으로는
        # 워커가 남아 GPU 메모리를 계속 잡는다.
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                       capture_output=True, check=False)
    else:
        try:
            os.killpg(os.getpgid(int(pid)), 15)
        except (OSError, ProcessLookupError):
            if proc is not None:
                proc.terminate()

    # 자식이 state 를 못 남기고 죽으므로 여기서 중단 상태를 기록한다.
    state.update({"status": "stopped", "finished_at": store.now_iso()})
    store.write_json(store.run_state_path(run_id), state)
    return {"run_id": run_id, "stopped": True, "pid": pid}


def _parse_results_csv(path: Path) -> dict:
    """ultralytics results.csv 의 헤더 + 마지막 행. state.json 이 없을 때의 근거 자료."""
    if not path.exists():
        return {}
    try:
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except OSError:
        return {}
    if len(lines) < 2:
        return {}
    header = [h.strip() for h in lines[0].split(",")]
    last = [v.strip() for v in lines[-1].split(",")]
    row = {}
    for k, v in zip(header, last):
        try:
            row[k] = float(v)
        except ValueError:
            row[k] = v
    return {"epochs_done": len(lines) - 1, "last_row": row}


def status(run_id: str, *, log_lines: Optional[int] = None) -> dict:
    spec = store.read_json(store.run_spec_path(run_id), None)
    if not isinstance(spec, dict):
        raise KeyError(f"학습 run 을 찾을 수 없습니다: {run_id}")
    state = store.read_json(store.run_state_path(run_id), {}) or {}
    csv = _parse_results_csv(store.run_output_dir(run_id) / "results.csv")

    st = str(state.get("status") or "starting")
    epochs = int(state.get("epochs") or spec.get("epochs") or 0)
    epoch = int(state.get("epoch") or csv.get("epochs_done") or 0)

    if st == "running":
        alive = _is_alive(state.get("pid"))
        stale = _heartbeat_age(state.get("heartbeat")) > _STALE_SEC
        if not alive and stale:
            # 프로세스가 사라졌는데 state 가 running 으로 굳은 경우(강제종료/재부팅).
            st = "unknown"

    weights = {}
    for tag in ("best", "last"):
        p = store.run_output_dir(run_id) / "weights" / f"{tag}.pt"
        if p.exists():
            weights[tag] = store.rel_to_workspace(p)

    return {
        "run_id": run_id,
        "status": st,
        "epoch": epoch,
        "epochs": epochs,
        "progress": round(min(1.0, epoch / epochs), 4) if epochs > 0 else 0.0,
        "metrics": state.get("metrics") or {},
        "best_fitness": state.get("best_fitness"),
        "results_csv": csv,
        "weights": weights,
        "pid": state.get("pid"),
        "error": state.get("error"),
        "started_at": state.get("started_at"),
        "finished_at": state.get("finished_at"),
        "heartbeat": state.get("heartbeat"),
        "spec": spec,
        "log_tail": store.tail_text(store.run_log_path(run_id),
                                    int(log_lines or get_settings().job_log_tail_lines)),
    }


def _heartbeat_age(stamp: Optional[str]) -> float:
    if not stamp:
        return float("inf")
    try:
        t = time.mktime(time.strptime(str(stamp), "%Y-%m-%dT%H:%M:%S"))
    except ValueError:
        return float("inf")
    return max(0.0, time.time() - t)


def list_runs() -> list:
    out = []
    for run_id in store.list_run_ids():
        spec = store.read_json(store.run_spec_path(run_id), {}) or {}
        state = store.read_json(store.run_state_path(run_id), {}) or {}
        epochs = int(state.get("epochs") or spec.get("epochs") or 0)
        epoch = int(state.get("epoch") or 0)
        out.append({
            "run_id": run_id,
            "status": state.get("status") or "starting",
            "dataset_id": spec.get("dataset_id"),
            "dataset_name": spec.get("dataset_name"),
            "model_name": spec.get("model_name"),
            "task": spec.get("dataset_task"),
            "epoch": epoch,
            "epochs": epochs,
            "progress": round(min(1.0, epoch / epochs), 4) if epochs > 0 else 0.0,
            "metrics": state.get("metrics") or {},
            "created_at": spec.get("created_at"),
            "finished_at": state.get("finished_at"),
            "error": state.get("error"),
        })
    return out


def weights_path(run_id: str, which: str = "best") -> Path:
    tag = str(which).lower()
    if tag not in ("best", "last"):
        raise ValueError("which 는 best 또는 last 여야 합니다")
    p = store.run_output_dir(run_id) / "weights" / f"{tag}.pt"
    if not p.exists():
        raise KeyError(f"가중치가 아직 없습니다: {run_id}/{tag}.pt")
    return p


def delete(run_id: str) -> dict:
    state = store.read_json(store.run_state_path(run_id), {}) or {}
    if state.get("status") == "running" and _is_alive(state.get("pid")):
        raise ValueError("실행 중인 학습은 지울 수 없습니다. 먼저 중단하세요.")
    shutil.rmtree(store.run_dir(run_id), ignore_errors=True)
    return {"run_id": run_id, "deleted": True}

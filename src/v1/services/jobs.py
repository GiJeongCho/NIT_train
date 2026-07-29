"""
services/jobs.py
================

백그라운드 잡 레지스트리. 프레임 추출/자동 라벨링, 데이터셋 빌드처럼 수 분씩
걸리는 작업을 HTTP 요청 밖에서 돌리고 진행률만 폴링하게 한다.

- 잡은 데몬 스레드에서 돈다(GPU 추론은 GIL 을 놓는 C 확장에서 대기하므로
  이벤트 루프를 막지 않는다). 학습처럼 프로세스 격리가 필요한 작업은
  `services/trainer.py` 가 별도 자식 프로세스로 돌린다.
- 진행률은 메모리에서 읽고, 종료 시점의 스냅샷은 `workspace/jobs/<id>.json` 에
  남긴다. API 를 재시작해도 "그 작업이 끝났는지" 조회가 된다.
- 취소는 협조적(cooperative)이다. 잡 본문이 루프마다 `job.check_canceled()` 를
  호출해 스스로 빠져나온다. 강제 종료는 하지 않는다(반쯤 쓰인 라벨 방지).
"""

from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

from core import store


class JobCanceled(Exception):
    """잡 본문이 취소 요청을 감지했을 때 올린다(에러가 아니라 정상 종료 경로)."""


@dataclass
class Job:
    id: str
    kind: str                       # extract | dataset
    target: str = ""                # video_id / dataset_id 등 대상 식별자
    status: str = "queued"          # queued | running | done | error | canceled
    total: int = 0
    done: int = 0
    message: str = ""
    error: Optional[str] = None
    result: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    created_at: str = field(default_factory=store.now_iso)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    _t0: float = 0.0
    _elapsed: float = 0.0
    _cancel: threading.Event = field(default_factory=threading.Event)

    # ── 잡 본문에서 쓰는 헬퍼 ────────────────────────────────────────
    def set_total(self, n: int) -> None:
        self.total = max(0, int(n))

    def advance(self, n: int = 1, message: str = "") -> None:
        self.done += int(n)
        if message:
            self.message = message

    def cancel(self) -> None:
        self._cancel.set()

    def canceled(self) -> bool:
        return self._cancel.is_set()

    def check_canceled(self) -> None:
        if self._cancel.is_set():
            raise JobCanceled()

    def elapsed_sec(self) -> float:
        if self.status == "running" and self._t0 > 0:
            return time.perf_counter() - self._t0
        return self._elapsed

    def to_dict(self) -> dict:
        if self.status in ("done",):
            progress = 1.0
        elif self.total > 0:
            progress = min(1.0, max(0.0, self.done / self.total))
        else:
            progress = 0.0
        elapsed = self.elapsed_sec()
        return {
            "id": self.id,
            "kind": self.kind,
            "target": self.target,
            "status": self.status,
            "total": self.total,
            "done": self.done,
            "progress": round(progress, 4),
            "rate_per_sec": round(self.done / elapsed, 2) if elapsed > 0 else 0.0,
            "elapsed_sec": round(elapsed, 2),
            "message": self.message,
            "error": self.error,
            "result": self.result,
            "params": self.params,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


_jobs: Dict[str, Job] = {}
_lock = threading.Lock()


def submit(kind: str, fn: Callable[[Job], None], target: str = "",
           params: Optional[dict] = None) -> Job:
    """잡을 등록하고 즉시 백그라운드 실행을 시작한다."""
    job = Job(id=store.new_id(), kind=kind, target=str(target or ""), params=params or {})
    with _lock:
        _jobs[job.id] = job
    threading.Thread(target=_run, args=(job, fn), name=f"{kind}-{job.id[-6:]}",
                     daemon=True).start()
    return job


def _run(job: Job, fn: Callable[[Job], None]) -> None:
    job.status = "running"
    job.started_at = store.now_iso()
    job._t0 = time.perf_counter()
    try:
        fn(job)
        job.status = "done"
        job.message = job.message or "완료"
    except JobCanceled:
        job.status = "canceled"
        job.message = "사용자 취소"
    except Exception as e:  # noqa: BLE001 - 잡 스레드에서 예외가 새면 상태가 영구 running 이 된다
        job.status = "error"
        job.error = f"{type(e).__name__}: {e}"
        job.message = "실패"
        print(f"[job] 실패 {job.kind} id={job.id}: {job.error}", flush=True)
        print(traceback.format_exc(), flush=True)
    finally:
        job._elapsed = time.perf_counter() - job._t0
        job.finished_at = store.now_iso()
        _persist(job)
        print(f"[job] {job.kind} id={job.id} status={job.status} "
              f"done={job.done}/{job.total} elapsed={job._elapsed:.1f}s", flush=True)


def _persist(job: Job) -> None:
    try:
        store.write_json(store.job_path(job.id), job.to_dict())
    except OSError as e:
        print(f"[job] 스냅샷 저장 실패 id={job.id}: {e}", flush=True)


def get(job_id: str) -> Optional[dict]:
    """메모리에 없으면(재시작 후) 디스크 스냅샷에서 찾는다."""
    with _lock:
        job = _jobs.get(job_id)
    if job is not None:
        return job.to_dict()
    try:
        return store.read_json(store.job_path(job_id), None)
    except ValueError:
        return None


def cancel(job_id: str) -> bool:
    with _lock:
        job = _jobs.get(job_id)
    if job is None or job.status not in ("queued", "running"):
        return False
    job.cancel()
    job.message = "취소 요청됨"
    return True


def list_jobs(kind: Optional[str] = None, target: Optional[str] = None,
              limit: int = 50) -> list:
    """실행 중 + 최근 완료 잡 목록(최신순)."""
    with _lock:
        live = [j.to_dict() for j in _jobs.values()]
    seen = {j["id"] for j in live}
    saved: list = []
    root = store.jobs_root()
    if root.exists():
        for p in sorted(root.glob("*.json"), key=lambda x: x.name, reverse=True)[:limit * 3]:
            if p.stem in seen:
                continue
            d = store.read_json(p, None)
            if isinstance(d, dict):
                saved.append(d)
    items = live + saved
    if kind:
        items = [j for j in items if j.get("kind") == kind]
    if target:
        items = [j for j in items if j.get("target") == target]
    items.sort(key=lambda j: str(j.get("created_at") or ""), reverse=True)
    return items[:limit]


def active_for(kind: str, target: str) -> Optional[dict]:
    """같은 대상에 대해 이미 돌고 있는 잡. 중복 실행을 막는 데 쓴다."""
    with _lock:
        for j in _jobs.values():
            if j.kind == kind and j.target == target and j.status in ("queued", "running"):
                return j.to_dict()
    return None

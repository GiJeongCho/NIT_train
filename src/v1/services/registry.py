"""
services/registry.py
====================

모델 레지스트리. 학습 산출물 중 "배포 후보"를 골라 이름을 붙여 보관하고, 필요하면
추론 서비스(tracker_py)의 `models/` 로 내보낸다.

`runs/<run_id>/train/weights/best.pt` 는 실험 산출물이고 run 을 지우면 사라진다.
승격(promote)은 그 가중치를 `workspace/models/<alias>.pt` 로 **복사**해 고정하고,
어떤 데이터셋·어떤 지표로 나온 모델인지 `registry.json` 에 함께 남긴다.
"모델 파일만 남고 출처를 아무도 모르는" 상황을 막는 게 목적이다.

배포는 파일 복사까지만 한다. tracker_py 는 `POST /api/detector/model` 로 런타임
교체를 지원하므로, 복사 후 그 API 를 호출하면 재시작 없이 새 모델이 적용된다.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Optional

from core.config import get_settings
from core import store
from services import detector as detector_svc, trainer as trainer_svc

_ALIAS_RE = re.compile(r"^[A-Za-z0-9._-]{1,48}$")


def _load() -> dict:
    doc = store.read_json(store.registry_path(), None)
    if not isinstance(doc, dict) or not isinstance(doc.get("models"), list):
        return {"models": [], "updated_at": None}
    return doc


def _save(doc: dict) -> dict:
    doc["updated_at"] = store.now_iso()
    store.write_json(store.registry_path(), doc)
    return doc


def list_models() -> dict:
    """쓸 수 있는 가중치 전체 + 승격 이력."""
    return {
        "default_model": store.rel_to_workspace(get_settings().base_model)
                         or str(get_settings().base_model),
        "weights": detector_svc.list_weights(),
        "promoted": _load()["models"],
    }


def model_file(alias: str) -> Path:
    """승격 모델(`workspace/models/<alias>.pt`) 파일 경로. 내보내기(다운로드)용.

    학습 서버에 가져다 설치할 수 있도록 그대로 파일을 내려준다. alias 는 파일명이
    되므로 경로 탈출(`../`)을 막기 위해 승격과 동일한 화이트리스트로 검증한다.
    """
    name = str(alias or "").strip()
    if not _ALIAS_RE.match(name):
        raise ValueError("alias 는 영문/숫자/.-_ 조합 1~48자여야 합니다")
    path = store.models_root() / f"{name}.pt"
    if not path.exists():
        raise KeyError(f"승격된 모델이 없습니다: {name}")
    return path


def promote(run_id: str, *, alias: str, which: str = "best",
            note: str = "", deploy: bool = False) -> dict:
    """학습 결과를 배포 후보로 승격한다."""
    name = str(alias or "").strip()
    if not _ALIAS_RE.match(name):
        raise ValueError("alias 는 영문/숫자/.-_ 조합 1~48자여야 합니다")

    src = trainer_svc.weights_path(run_id, which)
    st = trainer_svc.status(run_id, log_lines=0)
    spec = st.get("spec") or {}

    dest = store.models_root() / f"{name}.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)

    entry = {
        "alias": name,
        "path": store.rel_to_workspace(dest),
        "abs_path": dest.as_posix(),
        "run_id": run_id,
        "which": which,
        "dataset_id": spec.get("dataset_id"),
        "dataset_name": spec.get("dataset_name"),
        "task": spec.get("dataset_task"),
        "class_names": spec.get("class_names"),
        "base_model": spec.get("model_name"),
        "epochs": st.get("epochs"),
        "metrics": st.get("metrics"),
        "best_fitness": st.get("best_fitness"),
        "size_mb": round(dest.stat().st_size / 1e6, 1),
        "note": str(note or ""),
        "promoted_at": store.now_iso(),
        "deployed_to": None,
    }

    if deploy:
        entry["deployed_to"] = _deploy(dest, name)

    doc = _load()
    doc["models"] = [m for m in doc["models"] if m.get("alias") != name] + [entry]
    doc["models"].sort(key=lambda m: str(m.get("promoted_at") or ""), reverse=True)
    _save(doc)
    return entry


def _deploy(src: Path, alias: str) -> Optional[str]:
    """추론 서비스의 models 폴더로 복사한다(설정된 경우에만)."""
    target_dir = get_settings().tracker_models_dir
    if not target_dir or str(target_dir) in ("", "."):
        raise ValueError(
            "배포 경로가 설정되지 않았습니다. NIT_TRAIN_TRACKER_MODELS_DIR 에 "
            "추론 서비스의 models 폴더를 지정하세요."
        )
    target_dir = Path(target_dir)
    if not target_dir.exists():
        raise ValueError(f"배포 경로가 없습니다: {target_dir}")
    dest = target_dir / f"{alias}.pt"
    shutil.copy2(src, dest)
    print(f"[registry] 배포 완료: {dest}", flush=True)
    return dest.as_posix()


def unpromote(alias: str) -> dict:
    doc = _load()
    before = len(doc["models"])
    doc["models"] = [m for m in doc["models"] if m.get("alias") != alias]
    (store.models_root() / f"{alias}.pt").unlink(missing_ok=True)
    _save(doc)
    return {"alias": alias, "removed": before - len(doc["models"])}

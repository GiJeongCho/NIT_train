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

from core.config import PROJECT_ROOT, get_settings
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


def inference_weights(alias: str) -> Path:
    """**추론 전용**으로 학습 부산물을 떼어낸 가중치 경로를 만든다.

    ultralytics 체크포인트(`.pt`)에는 옵티마이저·EMA·학습 인자 등 **재학습에만 필요한**
    데이터가 함께 들어있다. 배포/추론에는 모델 가중치만 있으면 되므로 `strip_optimizer`
    로 그 부분을 떼고(반정밀도로 저장) 파일 크기도 줄인다. 실패하면(라이브러리/포맷 문제)
    원본을 그대로 쓴다 — 원본도 추론은 가능하기 때문이다.
    """
    src = model_file(alias)
    out = store.models_root() / f"{alias}.infer.pt"
    try:
        from ultralytics.utils.torch_utils import strip_optimizer
        shutil.copy2(src, out)
        strip_optimizer(str(out))       # 제자리(out)에서 학습용 키 제거 + half 저장
    except Exception:                    # noqa: BLE001 — 스트립 실패 시 원본으로 폴백
        shutil.copy2(src, out)
    return out


def bundle_path(alias: str) -> Path:
    """승격 모델을 **추론 전용 가중치(.pt) + 메타(metadata.yaml)** 한 zip 으로 묶은 경로.

    내보내기 한 번으로 배포에 필요한 두 파일을 같이 받게 한다. 요청마다 새로 굽는다.
    """
    import zipfile

    src = inference_weights(alias)      # 학습기 뗀 추론 전용 가중치
    yaml_text = metadata_yaml(alias)
    out = store.models_root() / f"{alias}_bundle.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(src, arcname=f"{alias}.pt")
        zf.writestr("metadata.yaml", yaml_text)
    return out


def _entry(alias: str) -> dict:
    name = str(alias or "").strip()
    for m in _load()["models"]:
        if m.get("alias") == name:
            return m
    raise KeyError(f"승격된 모델이 없습니다: {name}")


def names_from_weights(path: Path) -> Optional[list]:
    """`.pt` 에 내장된 클래스 이름을 인덱스 순 리스트로 읽는다(가중치가 곧 진실).

    ultralytics 체크포인트는 `model.names`(=`{0:'a',1:'b'}`)에 클래스 맵을 담는다.
    실패하면 None 을 돌려주고, 호출자가 레지스트리 기록으로 폴백한다.
    """
    try:
        from ultralytics import YOLO
        names = YOLO(str(path)).names
    except Exception as e:  # noqa: BLE001 - 가중치 파싱 실패 시 폴백을 쓰게 한다
        print(f"[registry] 가중치에서 names 읽기 실패({path.name}): {e}", flush=True)
        return None
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
    if isinstance(names, (list, tuple)):
        return [str(n) for n in names]
    return None


def metadata_yaml(alias: str) -> str:
    """승격 모델과 **함께 내보낼 메타 YAML** 텍스트.

    클래스 이름/순서는 레지스트리 기록이 아니라 **실제 `.pt` 내장값**에서 읽어, 배포한
    가중치와 절대 어긋나지 않게 한다(가중치를 못 읽으면 레지스트리 기록으로 폴백).
    task/nc/names 는 ultralytics/`data.yaml` 서식과 같게 두어 재학습에도 바로 쓸 수 있다.
    """
    entry = _entry(alias)
    path = model_file(alias)  # 경로 검증 + 존재 확인
    names = names_from_weights(path) or [str(n) for n in (entry.get("class_names") or [])]
    task = str(entry.get("task") or ("obb" if "obb" in path.name.lower() else "detect"))
    map50 = (entry.get("metrics") or {}).get("metrics/mAP50(B)")

    lines = [
        f"# NIT_train 모델 메타 (내보내기 동봉용) — 생성 {store.now_iso()}",
        f"# alias: {alias}",
        f"# 학습 run: {entry.get('run_id')} · 가중치: {entry.get('which')}",
        f"# 데이터셋: {entry.get('dataset_name') or ''} ({entry.get('dataset_id') or ''})",
        f"# 기반 가중치: {entry.get('base_model') or ''}"
        + (f" · mAP50: {round(float(map50), 4)}" if map50 is not None else ""),
        f"# 승격 시각: {entry.get('promoted_at')}",
        "#",
        "# names 는 이 폴더의 .pt 에 내장된 클래스 맵에서 읽었습니다(가중치와 일치).",
        "",
        f"task: {task}",
        f"nc: {len(names)}",
        "names:",
    ]
    if names:
        lines += [f"  {i}: {n}" for i, n in enumerate(names)]
    else:
        lines.append("  {}  # 클래스 이름을 확인할 수 없습니다")
    return "\n".join(lines) + "\n"


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

    # 배포는 '있으면 좋은' 부가 단계다. 배포 경로 미설정 등으로 실패해도 **승격(등록)
    # 자체는 성공**시킨다. 그래야 사용자가 버튼만 눌러도 6·모델에 모델이 남는다.
    if deploy:
        try:
            entry["deployed_to"] = _deploy(dest, name)
        except Exception as e:  # noqa: BLE001 — 배포 실패를 승격 실패로 번지지 않게
            entry["deploy_warning"] = str(e)

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


def _is_within(path: Path, root: Path) -> bool:
    """`path` 가 `root` 아래(또는 같음)인지. 경로 탈출(../)로 임의 파일 삭제를 막는다."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def delete_weight(target: str) -> dict:
    """'사용 가능한 가중치' 목록의 파일 하나를 실제로 지운다.

    프런트가 `path`(워크스페이스 상대) 또는 절대경로를 그대로 넘겨도 되도록 여러 기준으로
    해석한다. 안전을 위해 **허용 폴더 안**의 파일만 지운다.

    허용 위치
    - `test_model/`          : 사전학습 가중치
    - `workspace/models/`    : 승격된 배포 후보(이력도 함께 정리)
    - `workspace/runs/`      : 학습 산출물(best/last)

    보호 규칙
    - 현재 **기본 모델(base_model)** 은 지우지 못한다(자동 라벨·학습이 깨진다).
    - 허용 폴더 밖 경로는 거부한다(경로 탈출·임의 삭제 방지).
    """
    s = get_settings()
    raw = str(target or "").strip()
    if not raw:
        raise ValueError("삭제할 가중치 경로가 필요합니다")

    p = Path(raw)
    candidates = ([p] if p.is_absolute()
                  else [store.workspace() / p, PROJECT_ROOT / p,
                        s.model_dir / p, store.models_root() / p])
    path = next((c for c in candidates if c.exists()), None)
    if path is None:
        raise KeyError(f"가중치 파일을 찾을 수 없습니다: {raw}")
    rp = path.resolve()

    allowed = [s.model_dir.resolve(), store.models_root().resolve(),
               (store.workspace() / "runs").resolve()]
    if not any(_is_within(rp, root) for root in allowed):
        raise ValueError("가중치 폴더 밖의 파일은 지울 수 없습니다.")

    if s.base_model.exists() and rp == s.base_model.resolve():
        raise ValueError("기본 모델은 삭제할 수 없습니다. 먼저 다른 모델을 기본값으로 지정하세요.")

    rp.unlink()

    # workspace/models/ 파일이면 승격 이력(registry.json)도 함께 지운다.
    removed_promotion = 0
    if _is_within(rp, store.models_root().resolve()):
        doc = _load()
        before = len(doc["models"])
        doc["models"] = [m for m in doc["models"] if m.get("alias") != rp.stem]
        removed_promotion = before - len(doc["models"])
        if removed_promotion:
            _save(doc)

    return {"deleted": store.rel_to_workspace(path) or rp.as_posix(),
            "removed_promotion": removed_promotion}

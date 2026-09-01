"""
services/dataset.py
===================

검수된 라벨 → ultralytics 학습용 YOLO 데이터셋 빌드 (백그라운드 잡).

산출물은 `test/yolo.py` 가 쓰던 `tracker_py/train_data/preprocessed_obb/` 와 **같은
구조**다. 그 폴더를 그대로 다시 만들 수 있고, 반대로 그 폴더를 그대로 읽어들일 수도
있어야 한다는 요구를 이 모듈이 담당한다::

    datasets/<dataset_id>/
    ├── data.yaml              # train/val/test 경로 + names (train_data 와 동일 서식)
    ├── manifest.json          # 어떤 영상/프레임/설정으로 만들었는지(재현·감사용)
    ├── train/{images,labels}/
    ├── valid/{images,labels}/
    └── test/{images,labels}/

세 가지 경로로 데이터셋이 만들어진다.

1. **빌드**(`start`)      : 검수 완료된 프레임 라벨을 모아 새로 만든다.
2. **등록**(`import_dir`) : 이미 있는 YOLO 폴더(`train_data/preprocessed_obb` 등)를
   복사 없이 데이터셋 목록에 올린다. 그대로 학습에 쓸 수 있다.
3. **병합**(`base_datasets`): 빌드할 때 기존 데이터셋을 같이 넣는다. 기존 5,834장에
   새로 라벨한 프레임을 더해 학습하는 것이 실제 운용 흐름이다.

핵심 설계
- **스냅샷**: 이미지를 데이터셋 폴더로 링크/복사해 고정한다. 나중에 원본 영상이나
  라벨을 고쳐도 이미 학습한 데이터셋은 변하지 않는다(실험 재현성).
- **누수 방지 분할**: 인접 프레임은 사실상 같은 그림이다. 무작위 분할하면
  train 과 valid 에 같은 장면이 들어가 검증 mAP 가 부풀려진다. 기본 `chunk` 모드는
  연속 프레임을 블록으로 묶아 통째로 배분한다. 병합해 들어온 기존 데이터셋은
  **원래 분할을 그대로 유지**한다(train 이던 것은 train 으로).
- **태스크별 라벨**: 같은 라벨 자산에서 `obb`(8좌표) 또는 `detect`(cxcywh) 로 쓴다.
  학습에 쓸 모델의 태스크와 반드시 일치해야 한다(`trainer` 가 검증).
- **클래스 id 는 이름으로 맞춘다**: 병합/등록 시 원본의 클래스 인덱스를 그대로 믿지
  않고 이름 → 현재 목록 인덱스로 다시 매핑한다. 인덱스만 맞추면 정답이 조용히
  뒤바뀐다.
"""

from __future__ import annotations

import os
import random
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from core.config import get_settings
from core import store
from services import annotations, classes as classes_svc, jobs
from services.jobs import Job
from utils.geometry import to_label_line

SPLITS = ("train", "valid", "test")
SPLIT_MODES = ("chunk", "random", "video")
TASKS = ("detect", "obb")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ultralytics 관례상 valid 폴더 이름은 valid 또는 val 둘 다 쓰인다.
_SPLIT_DIRS = {"train": ("train",), "valid": ("valid", "val"), "test": ("test",)}

_TAG_RE = re.compile(r"[^A-Za-z0-9]+")


def _resolve_spec(raw: Optional[dict]) -> dict:
    s = get_settings()
    raw = dict(raw or {})

    # video_ids 를 명시적으로 [] 로 보내면 "영상 없이 기존 데이터셋만" 이라는 뜻이다.
    video_ids = [str(v) for v in (raw.get("video_ids") or [])]
    if not video_ids and "video_ids" not in raw:
        video_ids = store.list_video_ids()
    for vid in video_ids:
        if not store.video_meta_path(vid).exists():
            raise ValueError(f"등록되지 않은 영상: {vid}")

    task = str(raw.get("task") or s.dataset_task).lower()
    if task not in TASKS:
        raise ValueError(f"task 는 {list(TASKS)} 중 하나여야 합니다: {task!r}")

    mode = str(raw.get("split_mode") or s.split_mode).lower()
    if mode not in SPLIT_MODES:
        raise ValueError(f"split_mode 는 {list(SPLIT_MODES)} 중 하나여야 합니다: {mode!r}")

    ratios = raw.get("splits") or s.splits()
    try:
        ratios = {k: max(0.0, float(ratios.get(k, 0.0))) for k in SPLITS}
    except (TypeError, ValueError, AttributeError):
        raise ValueError("splits 는 {'train':0.8,'valid':0.15,'test':0.05} 형태여야 합니다")
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("splits 합이 0 입니다")
    ratios = {k: v / total for k, v in ratios.items()}

    names = [str(x) for x in (raw.get("class_names") or classes_svc.names())]
    if not names:
        raise ValueError("클래스 목록이 비어 있습니다")

    # 정상(자동 승인)과 비정상(사람 수정) 프레임을 모두 학습에 넣는다. only_approved 가
    # 켜져 있으면 비정상은 사람이 승인해야 들어가고, 정상은 추출 때 자동 승인돼 바로 들어간다.
    kinds = [str(k) for k in (raw.get("include_kinds") or ["normal", "abnormal"])]

    sources = [_resolve_source(ref, task) for ref in (raw.get("base_datasets") or [])]
    if not video_ids and not sources:
        raise ValueError(
            "데이터셋에 넣을 것이 없습니다. 영상을 등록·라벨링하거나 "
            "base_datasets 로 기존 데이터셋을 지정하세요."
        )

    return {
        "name": str(raw.get("name") or "").strip() or "dataset",
        "video_ids": video_ids,
        "task": task,
        "class_names": names,
        "include_kinds": kinds,
        "only_approved": bool(raw.get("only_approved", s.dataset_only_approved)),
        "splits": {k: round(v, 4) for k, v in ratios.items()},
        "split_mode": mode,
        "chunk_size": int(raw.get("chunk_size") or s.split_chunk_size),
        "seed": int(raw.get("seed") if raw.get("seed") is not None else s.split_seed),
        "link_images": bool(raw.get("link_images", s.dataset_link_images)),
        "base_datasets": sources,
    }


# ══════════════════════════════════════════════════════════════════════
# 기존 YOLO 폴더 읽기 (train_data/preprocessed_obb 호환)
# ══════════════════════════════════════════════════════════════════════
def _norm_names(raw_names) -> List[str]:
    """data.yaml 의 names 를 인덱스 순 리스트로. dict(0:..)/list 둘 다 받는다."""
    if isinstance(raw_names, dict):
        try:
            keys = sorted(raw_names.keys(), key=lambda k: int(k))
        except (TypeError, ValueError):
            keys = list(raw_names.keys())
        return [str(raw_names[k]) for k in keys]
    if isinstance(raw_names, (list, tuple)):
        return [str(n) for n in raw_names]
    return []


def _guess_task(labels_dir: Path) -> Optional[str]:
    """라벨 파일의 토큰 수로 태스크를 추정한다. detect=5, obb=9."""
    for p in sorted(labels_dir.glob("*.txt"))[:50]:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            n = len(line.split())
            if n == 9:
                return "obb"
            if n == 5:
                return "detect"
    return None


def _abs_dir(path) -> Path:
    p = Path(str(path).strip().strip('"')).expanduser()
    if not p.is_absolute():
        p = store.workspace() / p
    return p


def read_yolo_dir(path) -> dict:
    """YOLO 데이터셋 폴더의 구조를 읽는다(등록·병합의 공통 입구).

    `train_data/preprocessed_obb` 처럼 `{train,valid,test}/{images,labels}` +
    `data.yaml` 로 된 폴더를 대상으로 한다. data.yaml 이 없어도 폴더 구조와 라벨
    토큰 수로 태스크를 추정한다(names 는 이 경우 알 수 없어 호출자가 줘야 한다).
    """
    root = _abs_dir(path)
    if not root.is_dir():
        raise ValueError(f"폴더를 찾을 수 없습니다: {root}")

    names: List[str] = []
    task: Optional[str] = None
    yaml_name: Optional[str] = None
    for cand in ("data.yaml", "data.yml", "dataset.yaml"):
        if (root / cand).is_file():
            yaml_name = cand
            break
    if yaml_name:
        import yaml
        doc = yaml.safe_load((root / yaml_name).read_text(encoding="utf-8")) or {}
        if isinstance(doc, dict):
            names = _norm_names(doc.get("names"))
            task = str(doc.get("task") or "").strip().lower() or None

    splits: Dict[str, dict] = {}
    for split, aliases in _SPLIT_DIRS.items():
        for alias in aliases:
            images = root / alias / "images"
            labels = root / alias / "labels"
            if not images.is_dir():
                continue
            files = sorted(p.name for p in images.iterdir()
                           if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
            splits[split] = {
                "dir": alias,
                "images": len(files),
                "labels": len(list(labels.glob("*.txt"))) if labels.is_dir() else 0,
            }
            if task is None and labels.is_dir():
                task = _guess_task(labels)
            break
    if not splits:
        raise ValueError(
            f"YOLO 데이터셋 폴더가 아닙니다(train/valid/test 하위 images 없음): {root}"
        )
    return {
        "dir": root,
        "yaml": yaml_name,
        "names": names,
        "task": task,
        "splits": splits,
        "total_images": sum(v["images"] for v in splits.values()),
    }


def inspect(path) -> dict:
    """등록 전에 폴더를 미리 보여주기 위한 조회(프런트 확인용)."""
    info = read_yolo_dir(path)
    return {**info, "dir": info["dir"].as_posix()}


def _resolve_source(ref: str, task: str) -> dict:
    """`base_datasets` 항목(데이터셋 id 또는 서버 경로) → 병합 소스 정보."""
    ref = str(ref or "").strip()
    if not ref:
        raise ValueError("base_datasets 항목이 비어 있습니다")

    root_hint: Optional[str] = None
    label = ref
    try:
        manifest = store.read_json(store.dataset_manifest_path(ref), None)
    except ValueError:
        manifest = None  # id 형식이 아니면 경로로 취급
    if isinstance(manifest, dict):
        root_hint = manifest.get("source_dir") or str(store.dataset_dir(ref))
        label = str(manifest.get("name") or ref)

    info = read_yolo_dir(root_hint or ref)
    src_task = info["task"] or task
    if src_task != task:
        raise ValueError(
            f"합칠 데이터셋의 태스크가 다릅니다 ({label}: {src_task} ≠ {task}). "
            "회전박스(obb) 라벨과 축정렬(detect) 라벨은 한 데이터셋에 섞을 수 없습니다."
        )
    if not info["names"]:
        raise ValueError(
            f"클래스 목록을 알 수 없습니다 ({label}). data.yaml 의 names 를 채우세요."
        )
    return {
        "ref": ref,
        "name": label,
        "dir": info["dir"].as_posix(),
        "task": src_task,
        "names": info["names"],
        "splits": info["splits"],
        "total_images": info["total_images"],
    }


def _class_map(src_names: List[str], target_names: List[str], label: str) -> List[int]:
    """소스 클래스 인덱스 → 대상 인덱스. 이름으로 맞춘다."""
    index = {n: i for i, n in enumerate(target_names)}
    missing = [n for n in src_names if n not in index]
    if missing:
        raise ValueError(
            f"'{label}' 의 클래스가 현재 목록에 없습니다: {missing}. "
            "PUT /api/classes 로 먼저 추가하세요(순서 변경 없이 뒤에 추가)."
        )
    return [index[n] for n in src_names]


def _tag_for(text: str, used: set) -> str:
    """파일명 접두사. 소스가 여러 개여도 stem 이 겹치지 않게 한다."""
    base = _TAG_RE.sub("-", str(text)).strip("-")[:24] or "src"
    tag = base
    i = 2
    while tag in used:
        tag = f"{base}-{i}"
        i += 1
    used.add(tag)
    return tag


# ══════════════════════════════════════════════════════════════════════
# 프레임 수집 / 분할
# ══════════════════════════════════════════════════════════════════════
def _collect(spec: dict) -> tuple:
    """데이터셋에 넣을 프레임 목록과 제외 사유를 모은다."""
    name_to_id = {n: i for i, n in enumerate(spec["class_names"])}
    picked: List[dict] = []
    excluded: Dict[str, int] = {}

    def drop(reason: str) -> None:
        excluded[reason] = excluded.get(reason, 0) + 1

    for vid in spec["video_ids"]:
        for fid in store.list_frame_ids(vid):
            doc = store.read_json(annotations.label_path(vid, fid), None)
            if not isinstance(doc, dict):
                drop("라벨 파일 손상")
                continue
            if doc.get("status") == "rejected":
                drop("검수 제외(rejected)")
                continue
            if spec["only_approved"] and doc.get("status") != "approved":
                drop("미승인(pending)")
                continue
            if doc.get("segment_kind") and doc["segment_kind"] not in spec["include_kinds"]:
                drop(f"제외 구간({doc['segment_kind']})")
                continue
            if not store.frame_image_path(vid, fid).exists():
                drop("이미지 없음")
                continue

            lines: List[str] = []
            bad = False
            for o in doc.get("objects") or []:
                cname = o.get("class_name")
                if cname not in name_to_id:
                    bad = True
                    break
                lines.append(to_label_line(spec["task"], name_to_id[cname], o.get("poly") or [],
                                           int(doc.get("width") or 0), int(doc.get("height") or 0)))
            if bad:
                # 클래스가 미확정이거나 데이터셋 클래스 목록에 없는 객체가 있으면
                # 그 프레임은 통째로 뺀다. 일부만 라벨된 이미지는 "없는 객체"를
                # 배경으로 학습시켜 모델을 망친다.
                drop("미확정/목록 외 클래스 포함")
                continue

            picked.append({
                "video_id": vid,
                "frame_id": fid,
                "frame_index": int(doc.get("frame_index") or 0),
                "n_objects": len(lines),
                "lines": lines,
            })
    return picked, excluded


def _assign_splits(items: List[dict], spec: dict) -> Dict[str, List[dict]]:
    """분할. 모드별로 '무엇을 통째로 묶을지'만 다르고 배분 로직은 같다."""
    rng = random.Random(spec["seed"])
    items = sorted(items, key=lambda it: (it["video_id"], it["frame_index"]))

    groups: List[List[dict]]
    if spec["split_mode"] == "random":
        groups = [[it] for it in items]
    elif spec["split_mode"] == "video":
        by_video: Dict[str, List[dict]] = {}
        for it in items:
            by_video.setdefault(it["video_id"], []).append(it)
        groups = list(by_video.values())
    else:  # chunk
        # 블록이 너무 크면 그룹 수가 적어 비율을 못 맞춘다(프레임 20장 + 블록 30 이면
        # 그룹이 1개라 전부 train 으로 몰린다). 최소 10 그룹은 나오도록 줄인다.
        size = max(1, min(spec["chunk_size"], len(items) // 10 or 1))
        groups = []
        cur_video = None
        for it in items:
            if it["video_id"] != cur_video or len(groups[-1]) >= size:
                groups.append([])
                cur_video = it["video_id"]
            groups[-1].append(it)

    rng.shuffle(groups)

    out: Dict[str, List[dict]] = {k: [] for k in SPLITS}
    total = sum(len(g) for g in groups)
    # 그룹을 순서대로 넣으면서, 현재 채운 비율이 목표보다 가장 뒤처진 split 에 준다.
    # 그룹 크기가 달라도(영상 분할 등) 목표 비율에 가장 가깝게 수렴한다.
    wanted = {k: spec["splits"][k] * total for k in SPLITS}
    for g in groups:
        deficit = {k: wanted[k] - len(out[k]) for k in SPLITS if spec["splits"][k] > 0}
        target = max(deficit, key=deficit.get) if deficit else "train"
        out[target].extend(g)
    return out


# ══════════════════════════════════════════════════════════════════════
# 파일 배치 / data.yaml
# ══════════════════════════════════════════════════════════════════════
def _place_image(src: Path, dest: Path, link: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    if link:
        try:
            os.link(src, dest)
            return
        except OSError:
            # 다른 볼륨/파일시스템(도커 바인드 마운트 등)에서는 하드링크가 안 된다.
            pass
    shutil.copy2(src, dest)


def _write_yaml(path: Path, root: Path, spec: dict, counts: Dict[str, int],
                dirs: Optional[Dict[str, str]] = None) -> None:
    """`train_data/preprocessed_obb/data.yaml` 과 같은 서식으로 쓴다.

    `path:` 를 함께 넣어 데이터셋 폴더를 옮겨도 상대 경로가 깨지지 않게 한다.
    """
    dirs = dirs or {k: k for k in SPLITS}
    lines = [
        f"# NIT_train 자동 생성 ({store.now_iso()})",
        f"# dataset: {spec['name']}  task: {spec['task']}",
        f"# frames: train={counts['train']} valid={counts['valid']} test={counts['test']}",
        "",
        f"path: {root.as_posix()}",
        f"train: {dirs['train']}/images",
        f"val: {dirs['valid']}/images",
    ]
    if counts["test"] > 0:
        lines.append(f"test: {dirs['test']}/images")
    lines += ["", f"task: {spec['task']}", "", "names:"]
    for i, n in enumerate(spec["class_names"]):
        lines.append(f"  {i}: {n}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _remap_label(text: str, cmap: List[int], class_names: List[str],
                 hist: Dict[str, int]) -> List[str]:
    """라벨 txt 의 클래스 id 를 현재 목록 기준으로 바꿔 쓴다."""
    out: List[str] = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            cid = int(float(parts[0]))
        except ValueError:
            continue
        if cid < 0 or cid >= len(cmap):
            continue  # 소스 names 범위를 벗어난 id 는 신뢰할 수 없으므로 버린다
        mapped = cmap[cid]
        parts[0] = str(mapped)
        out.append(" ".join(parts))
        name = class_names[mapped]
        hist[name] = hist.get(name, 0) + 1
    return out


def _copy_split(src_root: Path, src_dir: str, dest_root: Path, split: str, *,
                tag: str, cmap: List[int], spec: dict, counts: Dict[str, int],
                objects: Dict[str, int], hist: Dict[str, int],
                job: Optional[Job] = None) -> None:
    images = src_root / src_dir / "images"
    labels = src_root / src_dir / "labels"
    if not images.is_dir():
        return
    for img in sorted(p for p in images.iterdir()
                      if p.is_file() and p.suffix.lower() in IMAGE_EXTS):
        if job is not None:
            job.check_canceled()
        stem = f"{tag}_{img.stem}"
        _place_image(img, dest_root / split / "images" / f"{stem}{img.suffix.lower()}",
                     spec["link_images"])
        src_label = labels / f"{img.stem}.txt"
        lines: List[str] = []
        if src_label.is_file():
            try:
                lines = _remap_label(src_label.read_text(encoding="utf-8", errors="replace"),
                                     cmap, spec["class_names"], hist)
            except OSError:
                lines = []
        (dest_root / split / "labels" / f"{stem}.txt").write_text(
            ("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
        counts[split] += 1
        objects[split] += len(lines)
        if job is not None:
            job.advance(1)


# ══════════════════════════════════════════════════════════════════════
# 빌드
# ══════════════════════════════════════════════════════════════════════
def start(raw_spec: Optional[dict] = None) -> Job:
    """데이터셋 빌드 잡을 시작한다."""
    spec = _resolve_spec(raw_spec)
    dataset_id = store.new_id()
    return jobs.submit("dataset", lambda job: _run(job, dataset_id, spec),
                       target=dataset_id, params=spec)


def _run(job: Job, dataset_id: str, spec: dict) -> None:
    picked, excluded = _collect(spec)
    sources = spec["base_datasets"]
    if not picked and not sources:
        raise ValueError(
            "데이터셋에 넣을 프레임이 없습니다. 제외 사유: "
            + (", ".join(f"{k}×{v}" for k, v in excluded.items()) or "(대상 프레임 없음)")
        )

    # 병합 소스는 클래스 매핑을 먼저 검증한다. 수천 장을 복사한 뒤 터지지 않게.
    used_tags: set = set()
    plans = []
    for src in sources:
        plans.append({
            "src": src,
            "tag": _tag_for(src["name"] or src["ref"], used_tags),
            "cmap": _class_map(src["names"], spec["class_names"], src["name"] or src["ref"]),
        })

    assigned = _assign_splits(picked, spec)
    merged_total = sum(p["src"]["total_images"] for p in plans)
    job.set_total(len(picked) + merged_total)
    job.message = (f"신규 {len(picked)}장 (train={len(assigned['train'])} "
                   f"valid={len(assigned['valid'])} test={len(assigned['test'])})"
                   + (f" + 기존 {merged_total}장 병합" if merged_total else ""))

    ddir = store.dataset_dir(dataset_id)
    for split in SPLITS:
        (ddir / split / "images").mkdir(parents=True, exist_ok=True)
        (ddir / split / "labels").mkdir(parents=True, exist_ok=True)

    counts = {k: 0 for k in SPLITS}
    objects = {k: 0 for k in SPLITS}
    class_hist: Dict[str, int] = {}
    per_video: Dict[str, int] = {}

    for split in SPLITS:
        for it in assigned[split]:
            job.check_canceled()
            # 영상 간 프레임 번호가 겹치므로 video_id 를 접두사로 붙여 유일하게 만든다.
            stem = f"{it['video_id']}_{it['frame_id']}"
            _place_image(store.frame_image_path(it["video_id"], it["frame_id"]),
                         ddir / split / "images" / f"{stem}.jpg", spec["link_images"])
            (ddir / split / "labels" / f"{stem}.txt").write_text(
                ("\n".join(it["lines"]) + "\n") if it["lines"] else "", encoding="utf-8")
            counts[split] += 1
            objects[split] += it["n_objects"]
            per_video[it["video_id"]] = per_video.get(it["video_id"], 0) + 1
            for line in it["lines"]:
                cid = int(line.split()[0])
                cname = spec["class_names"][cid]
                class_hist[cname] = class_hist.get(cname, 0) + 1
            job.advance(1)

    # 기존 데이터셋 병합. 원래 분할(train/valid/test)을 그대로 유지한다 —
    # 남의 valid 를 내 train 으로 섞으면 과거 실험과 점수를 비교할 수 없게 된다.
    source_report = []
    for plan in plans:
        src, tag = plan["src"], plan["tag"]
        before = dict(counts)
        job.message = f"기존 데이터셋 병합: {src['name']} ({src['total_images']}장)"
        for split, meta in src["splits"].items():
            _copy_split(Path(src["dir"]), meta["dir"], ddir, split, tag=tag,
                        cmap=plan["cmap"], spec=spec, counts=counts,
                        objects=objects, hist=class_hist, job=job)
        source_report.append({
            "ref": src["ref"],
            "name": src["name"],
            "dir": src["dir"],
            "prefix": tag,
            "added": {k: counts[k] - before[k] for k in SPLITS},
            "class_map": {n: plan["cmap"][i] for i, n in enumerate(src["names"])},
        })

    warnings: List[str] = []
    if counts["valid"] == 0 and spec["splits"]["valid"] > 0:
        # val 세트가 없으면 ultralytics 가 학습 중 검증을 못 해 best 선택과 조기종료가
        # 무의미해진다. 프레임이 너무 적을 때 조용히 넘어가지 않도록 남긴다.
        warnings.append("valid 세트가 비어 있습니다. 프레임을 더 모으거나 splits 를 조정하세요.")
    if sum(objects.values()) == 0:
        warnings.append("객체 라벨이 하나도 없습니다(전부 배경 이미지). 학습이 수렴하지 않습니다.")
    thin = [c for c, n in class_hist.items() if n < 10]
    if thin:
        warnings.append(f"샘플이 10개 미만인 클래스: {thin}")

    _write_yaml(store.dataset_yaml_path(dataset_id), ddir, spec, counts)

    manifest = {
        "dataset_id": dataset_id,
        "name": spec["name"],
        "task": spec["task"],
        "created_at": store.now_iso(),
        "imported": False,
        "spec": spec,
        "counts": counts,
        "objects": objects,
        "total_frames": sum(counts.values()),
        "total_objects": sum(objects.values()),
        "class_names": spec["class_names"],
        "class_histogram": dict(sorted(class_hist.items(), key=lambda kv: -kv[1])),
        "per_video_frames": per_video,
        "sources": source_report,
        "excluded": excluded,
        "warnings": warnings,
        "data_yaml": store.rel_to_workspace(store.dataset_yaml_path(dataset_id)),
    }
    store.write_json(store.dataset_manifest_path(dataset_id), manifest)

    job.result = manifest
    job.message = (f"완료: {sum(counts.values())}장 / 객체 {sum(objects.values())}개 "
                   f"(train={counts['train']} valid={counts['valid']} test={counts['test']})")


# ══════════════════════════════════════════════════════════════════════
# 등록 (기존 폴더 → 데이터셋 목록)
# ══════════════════════════════════════════════════════════════════════
def _scan_objects(root: Path, splits: Dict[str, dict], names: List[str]) -> tuple:
    """라벨을 읽어 split 별 객체 수와 클래스 분포를 센다(등록 시 현황 파악용)."""
    objects = {k: 0 for k in SPLITS}
    hist: Dict[str, int] = {}
    for split, meta in splits.items():
        labels = root / meta["dir"] / "labels"
        if not labels.is_dir():
            continue
        for p in labels.glob("*.txt"):
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for line in text.splitlines():
                parts = line.split()
                if not parts:
                    continue
                try:
                    cid = int(float(parts[0]))
                except ValueError:
                    continue
                objects[split] += 1
                key = names[cid] if 0 <= cid < len(names) else f"(id {cid})"
                hist[key] = hist.get(key, 0) + 1
    return objects, hist


def import_dir(raw: Optional[dict] = None) -> dict:
    """이미 있는 YOLO 폴더를 데이터셋으로 등록한다.

    `tracker_py/train_data/preprocessed_obb` 처럼 이미 완성된 학습 전 구조를
    다시 만들지 않고 그대로 학습에 쓰기 위한 경로다.

    - `copy=false`(기본): 복사하지 않고 `data.yaml` 만 만들어 원본을 가리킨다.
      수천 장을 중복 저장하지 않는다. 대신 원본이 바뀌면 데이터셋도 바뀐다.
    - `copy=true`: 워크스페이스로 하드링크/복사해 스냅샷으로 고정한다.
      클래스 id 도 현재 목록 기준으로 다시 써서 병합 가능한 상태가 된다.
    """
    s = get_settings()
    raw = dict(raw or {})
    info = read_yolo_dir(raw.get("path") or "")
    root = info["dir"]

    task = str(raw.get("task") or info["task"] or s.dataset_task).lower()
    if task not in TASKS:
        raise ValueError(f"task 는 {list(TASKS)} 중 하나여야 합니다: {task!r}")

    src_names = [str(x) for x in (raw.get("class_names") or info["names"])]
    if not src_names:
        raise ValueError(
            f"클래스 목록을 알 수 없습니다: {root}. data.yaml 에 names 를 넣거나 "
            "class_names 를 함께 보내세요."
        )

    copy = bool(raw.get("copy", False))
    spec = {
        "name": str(raw.get("name") or "").strip() or root.name,
        "task": task,
        "class_names": src_names,
        "link_images": bool(raw.get("link_images", s.dataset_link_images)),
    }

    dataset_id = store.new_id()
    ddir = store.dataset_dir(dataset_id)
    ddir.mkdir(parents=True, exist_ok=True)

    counts = {k: 0 for k in SPLITS}
    objects = {k: 0 for k in SPLITS}
    class_hist: Dict[str, int] = {}

    if copy:
        # 등록하면서 스냅샷으로 고정한다. 클래스는 현재 프로젝트 목록에 맞춰 다시 쓴다.
        spec["class_names"] = classes_svc.names() or src_names
        cmap = _class_map(src_names, spec["class_names"], spec["name"])
        for split in SPLITS:
            (ddir / split / "images").mkdir(parents=True, exist_ok=True)
            (ddir / split / "labels").mkdir(parents=True, exist_ok=True)
        tag = _tag_for(spec["name"], set())
        for split, meta in info["splits"].items():
            _copy_split(root, meta["dir"], ddir, split, tag=tag, cmap=cmap, spec=spec,
                        counts=counts, objects=objects, hist=class_hist)
        yaml_root, dirs, source_dir = ddir, {k: k for k in SPLITS}, None
    else:
        counts.update({k: v["images"] for k, v in info["splits"].items()})
        objects, class_hist = _scan_objects(root, info["splits"], src_names)
        yaml_root = root
        dirs = {k: (info["splits"].get(k, {}).get("dir") or k) for k in SPLITS}
        source_dir = root.as_posix()

    _write_yaml(store.dataset_yaml_path(dataset_id), yaml_root, spec, counts, dirs)

    warnings: List[str] = []
    if source_dir:
        warnings.append(
            "원본 폴더를 그대로 참조합니다(복사 안 함). 원본이 바뀌면 이 데이터셋도 "
            "바뀝니다. 고정이 필요하면 copy=true 로 다시 등록하세요."
        )
    for split, meta in info["splits"].items():
        if meta["images"] != meta["labels"]:
            warnings.append(
                f"{split}: 이미지 {meta['images']}장 / 라벨 {meta['labels']}개 — "
                "짝이 맞지 않습니다(라벨 없는 이미지는 배경으로 학습됩니다)."
            )
    if counts["valid"] == 0:
        warnings.append("valid 세트가 없습니다. 학습 중 검증과 best 선택이 무의미해집니다.")

    manifest = {
        "dataset_id": dataset_id,
        "name": spec["name"],
        "task": task,
        "created_at": store.now_iso(),
        "imported": True,
        "copied": copy,
        "source_dir": source_dir or root.as_posix(),
        "source_yaml": info["yaml"],
        "spec": {**spec, "path": root.as_posix(), "copy": copy},
        "counts": counts,
        "objects": objects,
        "total_frames": sum(counts.values()),
        "total_objects": sum(objects.values()),
        "class_names": spec["class_names"],
        "class_histogram": dict(sorted(class_hist.items(), key=lambda kv: -kv[1])),
        "per_video_frames": {},
        "sources": [],
        "excluded": {},
        "warnings": warnings,
        "data_yaml": store.rel_to_workspace(store.dataset_yaml_path(dataset_id)),
    }
    store.write_json(store.dataset_manifest_path(dataset_id), manifest)
    return manifest


# ══════════════════════════════════════════════════════════════════════
# 조회 / 삭제
# ══════════════════════════════════════════════════════════════════════
def get(dataset_id: str) -> dict:
    doc = store.read_json(store.dataset_manifest_path(dataset_id), None)
    if not isinstance(doc, dict):
        raise KeyError(f"데이터셋을 찾을 수 없습니다: {dataset_id}")
    return doc


# ── 라벨 미리보기(학습데이터 뷰어) ────────────────────────────────────────
def _yaml_layout(dataset_id: str) -> dict:
    """data.yaml 을 읽어 split 별 이미지 폴더 절대경로 + names/task 를 돌려준다.

    빌드/등록/병합 어느 경로로 만든 데이터셋이든 data.yaml 하나로 위치를 파악한다
    (`path:` + `train|val|test: <dir>/images`). 원본을 참조만 하는 등록 데이터셋도
    같은 방식으로 실제 이미지 폴더를 찾는다.
    """
    import yaml

    yp = store.dataset_yaml_path(dataset_id)
    if not yp.exists():
        raise KeyError(f"data.yaml 이 없습니다: {dataset_id}")
    doc = yaml.safe_load(yp.read_text(encoding="utf-8")) or {}
    if not isinstance(doc, dict):
        raise ValueError(f"data.yaml 형식이 올바르지 않습니다: {dataset_id}")
    root_raw = str(doc.get("path") or "").strip()
    root = _abs_dir(root_raw) if root_raw else store.dataset_dir(dataset_id)
    splits: Dict[str, Path] = {}
    for key in ("train", "val", "test"):
        rel = doc.get(key)
        if not rel:
            continue
        p = Path(str(rel))
        splits[key] = p if p.is_absolute() else (root / p)
    return {
        "root": root,
        "names": _norm_names(doc.get("names")),
        "task": str(doc.get("task") or "").strip().lower() or None,
        "splits": splits,
    }


def _norm_split(split) -> str:
    s = str(split or "train").strip().lower()
    if s in ("valid", "validation"):
        return "val"
    if s not in ("train", "val", "test"):
        raise ValueError("split 은 train|val|test 중 하나여야 합니다")
    return s


def _parse_label_objects(text: str, names: List[str]) -> List[dict]:
    """YOLO 라벨 텍스트 → 그리기용 객체(정규화 폴리곤/박스) 리스트.

    obb(`cls x1 y1 … x4 y4`)와 detect(`cls cx cy w h`)를 모두 4점 폴리곤으로 통일한다.
    좌표는 0~1 정규화 그대로 둔다(프런트가 표시 크기에 곱해 그린다).
    """
    objs: List[dict] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
            vals = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        if len(vals) >= 8:  # obb: 8좌표
            poly = [[vals[i], vals[i + 1]] for i in range(0, 8, 2)]
        else:               # detect: cxcywh → AABB 4점
            cx, cy, bw, bh = vals[:4]
            x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
            poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        objs.append({
            "class_id": cid,
            "class_name": names[cid] if 0 <= cid < len(names) else str(cid),
            "poly": [[round(x, 6), round(y, 6)] for x, y in poly],
            "bbox": [round(min(xs), 6), round(min(ys), 6), round(max(xs), 6), round(max(ys), 6)],
        })
    return objs


def samples(dataset_id: str, split="train", offset: int = 0, limit: int = 24) -> dict:
    """데이터셋 이미지 + 라벨(정규화 좌표)을 페이지 단위로 돌려준다(라벨 미리보기용)."""
    manifest = get(dataset_id)  # 존재 검증
    layout = _yaml_layout(dataset_id)
    names = layout["names"] or list(manifest.get("class_names") or [])
    sp = _norm_split(split)
    img_dir = layout["splits"].get(sp)
    out = {
        "dataset_id": dataset_id, "split": sp, "task": layout["task"],
        "class_names": names, "offset": int(offset), "limit": int(limit),
        "total": 0, "items": [],
    }
    if img_dir is None or not img_dir.is_dir():
        return out
    files = sorted(p.name for p in img_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    out["total"] = len(files)
    lbl_dir = img_dir.parent / "labels"
    for name in files[int(offset):int(offset) + int(limit)]:
        lbl = lbl_dir / (Path(name).stem + ".txt")
        objs: List[dict] = []
        if lbl.is_file():
            try:
                objs = _parse_label_objects(lbl.read_text(encoding="utf-8"), names)
            except OSError:
                objs = []
        out["items"].append({"name": name, "objects": objs, "n": len(objs)})
    return out


def sample_image_path(dataset_id: str, split, name: str) -> Path:
    """split 의 특정 이미지 절대경로. 경로 탈출을 막고 이미지 폴더 안만 허용한다."""
    layout = _yaml_layout(dataset_id)
    sp = _norm_split(split)
    img_dir = layout["splits"].get(sp)
    if img_dir is None or not img_dir.is_dir():
        raise KeyError(f"{dataset_id}/{sp}: 이미지 폴더가 없습니다")
    safe = Path(str(name)).name  # 디렉터리 성분 제거(traversal 방지)
    p = img_dir / safe
    if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
        raise KeyError(f"이미지를 찾을 수 없습니다: {sp}/{safe}")
    return p


def list_datasets() -> list:
    out = []
    for did in store.list_dataset_ids():
        doc = store.read_json(store.dataset_manifest_path(did), None)
        if isinstance(doc, dict):
            out.append({
                "dataset_id": did,
                "name": doc.get("name"),
                "task": doc.get("task"),
                "created_at": doc.get("created_at"),
                "counts": doc.get("counts"),
                "total_frames": doc.get("total_frames"),
                "total_objects": doc.get("total_objects"),
                "class_names": doc.get("class_names"),
                "imported": bool(doc.get("imported")),
                "source_dir": doc.get("source_dir") if doc.get("imported") else None,
                "n_sources": len(doc.get("sources") or []),
                "warnings": doc.get("warnings") or [],
            })
    return out


def delete(dataset_id: str) -> dict:
    """워크스페이스의 데이터셋 폴더만 지운다.

    복사 없이 등록한 데이터셋(`imported` + `copy=false`)은 워크스페이스에
    data.yaml/manifest 만 있으므로 **원본 폴더는 건드리지 않는다.**
    """
    doc = store.read_json(store.dataset_manifest_path(dataset_id), None) or {}
    shutil.rmtree(store.dataset_dir(dataset_id), ignore_errors=True)
    return {"dataset_id": dataset_id, "deleted": True,
            "source_kept": doc.get("source_dir") if doc.get("imported") else None}

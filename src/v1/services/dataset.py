"""
services/dataset.py
===================

검수된 라벨 → ultralytics 학습용 YOLO 데이터셋 빌드 (백그라운드 잡).

산출물은 `test/yolo.py` 가 쓰던 `train_data/preprocessed_obb/` 와 같은 구조다::

    datasets/<dataset_id>/
    ├── data.yaml
    ├── manifest.json          # 어떤 영상/프레임/설정으로 만들었는지(재현·감사용)
    ├── train/{images,labels}/
    ├── valid/{images,labels}/
    └── test/{images,labels}/

핵심 설계
- **스냅샷**: 이미지를 데이터셋 폴더로 링크/복사해 고정한다. 나중에 원본 영상이나
  라벨을 고쳐도 이미 학습한 데이터셋은 변하지 않는다(실험 재현성).
- **누수 방지 분할**: 인접 프레임은 사실상 같은 그림이다. 무작위 분할하면
  train 과 valid 에 같은 장면이 들어가 검증 mAP 가 부풀려진다. 기본 `chunk` 모드는
  연속 프레임을 블록으로 묶어 통째로 배분한다.
- **태스크별 라벨**: 같은 라벨 자산에서 `obb`(8좌표) 또는 `detect`(cxcywh) 로 쓴다.
  학습에 쓸 모델의 태스크와 반드시 일치해야 한다(`trainer` 가 검증).
"""

from __future__ import annotations

import os
import random
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


def _resolve_spec(raw: Optional[dict]) -> dict:
    s = get_settings()
    raw = dict(raw or {})

    video_ids = [str(v) for v in (raw.get("video_ids") or [])]
    if not video_ids:
        video_ids = store.list_video_ids()
    if not video_ids:
        raise ValueError("데이터셋에 넣을 영상이 없습니다. 먼저 영상을 등록·라벨링하세요.")
    for vid in video_ids:
        if not store.video_meta_path(vid).exists():
            raise ValueError(f"등록되지 않은 영상: {vid}")

    task = str(raw.get("task") or "detect").lower()
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

    kinds = [str(k) for k in (raw.get("include_kinds") or ["normal"])]

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
    }


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


def _write_yaml(path: Path, dataset_dir: Path, spec: dict, counts: Dict[str, int]) -> None:
    lines = [
        f"# NIT_train 자동 생성 ({store.now_iso()})",
        f"# dataset: {spec['name']}  task: {spec['task']}",
        f"# frames: train={counts['train']} valid={counts['valid']} test={counts['test']}",
        "",
        f"path: {dataset_dir.as_posix()}",
        "train: train/images",
        "val: valid/images",
    ]
    if counts["test"] > 0:
        lines.append("test: test/images")
    lines += ["", f"task: {spec['task']}", "", "names:"]
    for i, n in enumerate(spec["class_names"]):
        lines.append(f"  {i}: {n}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def start(raw_spec: Optional[dict] = None) -> Job:
    """데이터셋 빌드 잡을 시작한다."""
    spec = _resolve_spec(raw_spec)
    dataset_id = store.new_id()
    return jobs.submit("dataset", lambda job: _run(job, dataset_id, spec),
                       target=dataset_id, params=spec)


def _run(job: Job, dataset_id: str, spec: dict) -> None:
    picked, excluded = _collect(spec)
    if not picked:
        raise ValueError(
            "데이터셋에 넣을 프레임이 없습니다. 제외 사유: "
            + (", ".join(f"{k}×{v}" for k, v in excluded.items()) or "(대상 프레임 없음)")
        )

    assigned = _assign_splits(picked, spec)
    job.set_total(len(picked))
    job.message = (f"{len(picked)}장 배치 (train={len(assigned['train'])} "
                   f"valid={len(assigned['valid'])} test={len(assigned['test'])})")

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
        "spec": spec,
        "counts": counts,
        "objects": objects,
        "total_frames": sum(counts.values()),
        "total_objects": sum(objects.values()),
        "class_names": spec["class_names"],
        "class_histogram": dict(sorted(class_hist.items(), key=lambda kv: -kv[1])),
        "per_video_frames": per_video,
        "excluded": excluded,
        "warnings": warnings,
        "data_yaml": store.rel_to_workspace(store.dataset_yaml_path(dataset_id)),
    }
    store.write_json(store.dataset_manifest_path(dataset_id), manifest)

    job.result = manifest
    job.message = (f"완료: {sum(counts.values())}장 / 객체 {sum(objects.values())}개 "
                   f"(train={counts['train']} valid={counts['valid']} test={counts['test']})")


def get(dataset_id: str) -> dict:
    doc = store.read_json(store.dataset_manifest_path(dataset_id), None)
    if not isinstance(doc, dict):
        raise KeyError(f"데이터셋을 찾을 수 없습니다: {dataset_id}")
    return doc


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
            })
    return out


def delete(dataset_id: str) -> dict:
    shutil.rmtree(store.dataset_dir(dataset_id), ignore_errors=True)
    return {"dataset_id": dataset_id, "deleted": True}

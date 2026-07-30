"""
test/smoke_obb.py
=================

OBB 데이터셋 저장/관리(등록·병합) 검증 스모크.

`tracker_py/train_data/preprocessed_obb` 를 대상으로

1. 폴더 미리보기(inspect)       — task/클래스/장수를 제대로 읽는지
2. 참조 등록(copy=false)        — 복사 없이 학습 가능한 data.yaml 이 나오는지
3. 스냅샷 등록(copy=true)       — train_data 와 같은 구조로 재생성되는지
4. 병합 빌드(base_datasets)     — 기존 분할 유지 + 클래스 이름 매핑
5. data.yaml 서식 확인          — ultralytics 가 읽는 형태인지

서버 없이 서비스 계층을 직접 호출한다(GPU/학습 불필요).

실행::

    conda activate NIT
    python NIT_train/test/smoke_obb.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "v1"))

SOURCE = Path(r"C:\project\tracker_py\train_data\preprocessed_obb")

_ok = 0
_fail = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global _ok, _fail
    if cond:
        _ok += 1
        print(f"  [OK]   {label}" + (f" — {detail}" if detail else ""))
    else:
        _fail += 1
        print(f"  [FAIL] {label}" + (f" — {detail}" if detail else ""))


def main() -> int:
    from core import store
    from services import classes as classes_svc, dataset as dataset_svc

    if not SOURCE.is_dir():
        print(f"원본 데이터셋이 없습니다: {SOURCE}")
        return 2

    store.ensure_workspace()
    created: list = []

    print(f"\n[1] 폴더 미리보기: {SOURCE}")
    info = dataset_svc.inspect(SOURCE)
    print(f"      task={info['task']} names={len(info['names'])}개 total={info['total_images']}장")
    print(f"      splits={ {k: v['images'] for k, v in info['splits'].items()} }")
    check("task 를 obb 로 인식", info["task"] == "obb", str(info["task"]))
    check("클래스 7개 인식", len(info["names"]) == 7, str(info["names"]))
    check("train/valid/test 전부 인식", set(info["splits"]) == {"train", "valid", "test"})
    check("총 장수 일치", info["total_images"] == 4084 + 1166 + 584, str(info["total_images"]))

    print("\n[2] 참조 등록 (copy=false)")
    ref = dataset_svc.import_dir({"path": str(SOURCE), "name": "preprocessed_obb(참조)"})
    created.append(ref["dataset_id"])
    print(f"      dataset_id={ref['dataset_id']} frames={ref['total_frames']} "
          f"objects={ref['total_objects']}")
    print(f"      class_histogram={ref['class_histogram']}")
    check("task=obb", ref["task"] == "obb")
    check("imported 표시", ref["imported"] is True and ref["copied"] is False)
    check("원본 폴더를 그대로 가리킴", ref["source_dir"] == SOURCE.as_posix())
    check("분할 장수 유지", ref["counts"] == {"train": 4084, "valid": 1166, "test": 584},
          str(ref["counts"]))
    check("객체 라벨 존재", ref["total_objects"] > 0, f"{ref['total_objects']}개")
    yaml_text = store.dataset_yaml_path(ref["dataset_id"]).read_text(encoding="utf-8")
    print("      --- data.yaml ---")
    for line in yaml_text.splitlines():
        print(f"      {line}")
    check("data.yaml 이 원본 경로를 가리킴", SOURCE.as_posix() in yaml_text)
    check("train/val 경로 존재", "train: train/images" in yaml_text and "val: valid/images" in yaml_text)
    check("names 7줄", sum(1 for ln in yaml_text.splitlines() if ln.startswith("  ")) == 7)
    # 등록만으로 학습이 가능해야 한다 → data.yaml 의 실제 경로가 존재해야 한다.
    check("data.yaml 이 가리키는 이미지 폴더 존재", (SOURCE / "train" / "images").is_dir())

    print("\n[3] 스냅샷 등록 (copy=true) — train_data 와 같은 구조로 재생성")
    # 원본 클래스 순서 = 프로젝트 기본 클래스 순서여야 매핑이 통과한다.
    print(f"      프로젝트 클래스: {classes_svc.names()}")
    snap = dataset_svc.import_dir({"path": str(SOURCE), "name": "preprocessed_obb(스냅샷)",
                                   "copy": True})
    created.append(snap["dataset_id"])
    sdir = store.dataset_dir(snap["dataset_id"])
    print(f"      dataset_id={snap['dataset_id']} frames={snap['total_frames']} "
          f"objects={snap['total_objects']}")
    check("복사 표시", snap["copied"] is True)
    check("장수 일치", snap["counts"] == {"train": 4084, "valid": 1166, "test": 584},
          str(snap["counts"]))
    check("객체 수 일치", snap["total_objects"] == ref["total_objects"],
          f"{snap['total_objects']} vs {ref['total_objects']}")
    for split in ("train", "valid", "test"):
        n_img = len(list((sdir / split / "images").glob("*")))
        n_lbl = len(list((sdir / split / "labels").glob("*.txt")))
        check(f"{split} 이미지/라벨 짝 일치", n_img == n_lbl == snap["counts"][split],
              f"img={n_img} lbl={n_lbl}")
    sample = next(iter((sdir / "train" / "labels").glob("*.txt")))
    tokens = [len(ln.split()) for ln in sample.read_text(encoding="utf-8").splitlines() if ln.strip()]
    check("라벨이 OBB 8좌표(토큰 9개)", tokens and all(t == 9 for t in tokens), str(tokens[:3]))

    print("\n[4] 병합 빌드 (영상 없이 기존 데이터셋만)")
    job = dataset_svc.start({
        "name": "merged-obb",
        "video_ids": [],
        "task": "obb",
        "base_datasets": [ref["dataset_id"]],
    })
    deadline = time.time() + 900
    while job.status in ("queued", "running") and time.time() < deadline:
        print(f"      ... {job.status} {job.done}/{job.total} {job.message}")
        time.sleep(3)
    doc = job.to_dict()
    print(f"      job status={doc['status']} message={doc.get('message')}")
    check("병합 잡 성공", doc["status"] == "done", str(doc.get("error")))
    if doc["status"] == "done":
        merged = dataset_svc.get(job.target)
        created.append(merged["dataset_id"])
        print(f"      counts={merged['counts']} objects={merged['total_objects']}")
        print(f"      sources={[(s['name'], s['added']) for s in merged['sources']]}")
        check("기존 분할 그대로 유지",
              merged["counts"] == {"train": 4084, "valid": 1166, "test": 584},
              str(merged["counts"]))
        check("객체 수 보존", merged["total_objects"] == ref["total_objects"],
              f"{merged['total_objects']} vs {ref['total_objects']}")
        check("소스 이력 기록", len(merged["sources"]) == 1)
        cmap = merged["sources"][0]["class_map"]
        check("클래스가 이름으로 매핑됨(항등)",
              all(classes_svc.index_of(n) == i for n, i in cmap.items()), str(cmap))

    if "--train" in sys.argv:
        print("\n[5] 등록한 데이터셋으로 실제 OBB 학습 1 epoch")
        from services import trainer
        run = trainer.start({"dataset_id": ref["dataset_id"], "epochs": 1,
                             "batch": 4, "workers": 2, "note": "smoke_obb"})
        run_id = run["run_id"]
        print(f"      run_id={run_id} model={run['spec']['model_name']} "
              f"warnings={run['warnings']}")
        deadline = time.time() + 3600
        st = {}
        while time.time() < deadline:
            time.sleep(10)
            st = trainer.status(run_id, log_lines=0)
            print(f"      ... {st['status']} epoch={st['epoch']}/{st['epochs']} "
                  f"metrics={st['metrics']}")
            if st["status"] not in ("starting", "running"):
                break
        check("학습 완료", st.get("status") == "done", str(st.get("error")))
        check("best.pt 생성", "best" in (st.get("weights") or {}), str(st.get("weights")))
        if st.get("status") != "done":
            print(trainer.status(run_id, log_lines=60)["log_tail"])
            trainer.delete(run_id)
            return 1

        print(f"      metrics={st.get('metrics')}")

        # ── MLOps 루프의 마지막 고리 ──────────────────────────────
        # 사전학습 가중치(COCO/DOTA)는 탑다운 드론 영상의 10px 표적을 하나도 못 잡는다.
        # 그래서 "학습 → 승격 → 그 모델로 자동 라벨" 이 성립해야 파이프라인이 닫힌다.
        # 여기서 그 고리를 실제로 확인한다.
        print("\n[5-2] 승격 → 승격한 모델로 자동 라벨(추론)이 실제로 잡는지")
        import cv2
        from services import detector as detector_svc, registry
        entry = registry.promote(run_id, alias="smoke-obb-loop", which="best",
                                note="smoke_obb 루프 검증")
        check("승격 성공", Path(entry["abs_path"]).exists(), entry["path"])
        check("승격 이력에 출처 기록",
              entry["dataset_id"] == ref["dataset_id"] and entry["task"] == "obb",
              f"{entry['dataset_id']} / {entry['task']}")

        images = sorted((SOURCE / "valid" / "images").glob("*.jpg"))[:8]
        det = detector_svc.get_detector(entry["abs_path"])
        total_det = 0
        rotated = 0
        for p in images:
            frame = cv2.imread(str(p))
            dets = det.infer(frame, conf=0.25)
            total_det += len(dets)
            for d in dets:
                poly = d.get("poly") or []
                if len(poly) == 4:
                    ang = abs(math.degrees(math.atan2(poly[1][1] - poly[0][1],
                                                      poly[1][0] - poly[0][0]))) % 90
                    if 1.0 < ang < 89.0:
                        rotated += 1
        print(f"      {len(images)}장에서 탐지 {total_det}개 (회전된 박스 {rotated}개)")
        check("승격한 모델이 표적을 탐지함(자동 라벨 초안 가능)", total_det > 0, f"{total_det}개")
        check("탐지 결과가 회전박스(축정렬이 아님)", rotated > 0, f"{rotated}개")
        registry.unpromote("smoke-obb-loop")
        trainer.delete(run_id)

    print("\n[6] 정리")
    for did in created:
        res = dataset_svc.delete(did)
        print(f"      삭제 {did} (source_kept={res.get('source_kept')})")
    check("원본 폴더 보존", SOURCE.is_dir() and (SOURCE / "train" / "images").is_dir())
    check("워크스페이스 데이터셋 폴더 제거",
          all(not store.dataset_dir(d).exists() for d in created))

    print(f"\n{'=' * 60}\n결과: OK {_ok} / FAIL {_fail}\n{'=' * 60}")
    return 0 if _fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

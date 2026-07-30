"""
test/smoke_api.py
=================

파이프라인 전체를 실제 GPU 학습 없이 한 번 태워보는 스모크 테스트.

합성 영상(움직이는 사각형)을 만들어서:
  업로드 → 구간 지정 → 프레임 추출 + 자동 라벨 → 클래스 전파/승인
  → 데이터셋 빌드 → data.yaml 검증 → (학습은 1 epoch 옵션)

까지 돌린다. 라벨 좌표 변환·분할·파일 배치처럼 조용히 틀리기 쉬운 부분을
배포 전에 잡는 것이 목적이다.

    conda activate NIT
    cd NIT_train/test
    python smoke_api.py            # 학습 제외
    python smoke_api.py --train    # 1 epoch 학습까지 (GPU 필요)
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "src" / "v1"
sys.path.insert(0, str(APP_DIR))

import os  # noqa: E402

# Windows 콘솔 기본 코드페이지(cp949)에서 한글/기호가 깨져 죽지 않게 한다.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

# 실제 워크스페이스를 더럽히지 않도록 임시 워크스페이스를 쓴다.
_WS = Path(tempfile.mkdtemp(prefix="nittrain-smoke-"))
os.environ["NIT_TRAIN_WORKSPACE"] = str(_WS)
os.environ.setdefault("NIT_TRAIN_EXTRACT_FPS", "4")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

PASS, FAIL = "  [OK]", "  [FAIL]"
_failures: list = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print((PASS if cond else FAIL) + f" {label}" + (f" - {detail}" if detail else ""), flush=True)
    if not cond:
        _failures.append(label)


# 자동 라벨/트래킹 경로를 실제로 태우려면 사전학습 모델이 뭔가를 탐지하는 그림이어야
# 한다. 합성 도형은 아무것도 안 잡혀 파이프라인의 절반이 검증되지 않는다.
_REAL_FRAMES = Path(r"C:\project\tracker_py\train_data\preprocessed_obb\train\images")


def make_video(path: Path, seconds: int = 6, fps: int = 30) -> str:
    """스모크용 입력 영상. 실제 프레임이 있으면 그걸로, 없으면 합성 도형으로 만든다."""
    w, h = 640, 480
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    total = seconds * fps
    source = "synthetic"

    reals = sorted(_REAL_FRAMES.glob("01_frame_*.jpg"))[:60] if _REAL_FRAMES.exists() else []
    if reals:
        source = f"real({len(reals)} frames)"
        hold = max(1, total // len(reals))
        written = 0
        for p in reals:
            img = cv2.imread(str(p))
            if img is None:
                continue
            img = cv2.resize(img, (w, h))
            for _ in range(hold):
                writer.write(img)
                written += 1
        while written < total:
            writer.write(img)
            written += 1
    else:
        for i in range(total):
            frame = np.full((h, w, 3), 40, dtype=np.uint8)
            x = int(60 + (w - 200) * (i / max(1, total - 1)))
            cv2.rectangle(frame, (x, 200), (x + 90, 280), (200, 200, 200), -1)
            cv2.rectangle(frame, (x + 25, 150), (x + 65, 200), (170, 170, 170), -1)
            writer.write(frame)
    writer.release()
    return source


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="1 epoch 학습까지 수행(GPU 필요)")
    args = ap.parse_args()

    from app import app

    with TestClient(app) as client:
        print("\n[1] 서버 기본값")
        r = client.get("/api/meta")
        check("GET /api/meta", r.status_code == 200, str(r.status_code))
        meta = r.json()
        print(f"       기본 가중치: {meta['default_model']} (존재={meta['default_model_exists']})")
        print(f"       클래스: {meta['class_names']}")

        print("\n[2] 영상 업로드")
        vid_file = _WS / "sample.mp4"
        print(f"       입력 영상 소스: {make_video(vid_file)}")
        with vid_file.open("rb") as f:
            r = client.post("/api/videos", files={"file": ("sample.mp4", f, "video/mp4")})
        check("POST /api/videos", r.status_code == 200, r.text[:200])
        if r.status_code != 200:
            return 1
        video = r.json()
        video_id = video["video_id"]
        print(f"       video_id={video_id} fps={video['fps']} "
              f"{video['width']}x{video['height']} {video['duration_sec']}s")

        print("\n[3] 정상/비정상 구간")
        r = client.put(f"/api/videos/{video_id}/segments", json={"segments": [
            {"kind": "normal", "start_sec": 0.0, "end_sec": 6.0},
            {"kind": "abnormal", "start_sec": 2.0, "end_sec": 3.0, "note": "흔들림"},
        ]})
        check("PUT segments", r.status_code == 200, r.text[:200])
        sel = r.json()["selection_ranges"]
        check("비정상 구간이 제외됨", len(sel) == 2 and abs(sel[0][1] - 2.0) < 0.01, str(sel))
        print(f"       추출 대상 구간: {sel}")

        print("\n[4] 프레임 추출 + 자동 라벨")
        r = client.post(f"/api/videos/{video_id}/extract",
                        json={"fps": 4, "conf": 0.05, "track": True})
        check("POST extract", r.status_code == 200, r.text[:300])
        if r.status_code != 200:
            return 1
        job_id = r.json()["job_id"]
        job = _wait_job(client, job_id, timeout=600)
        check("추출 작업 완료", job.get("status") == "done",
              f"{job.get('status')} {job.get('error')}")
        print(f"       {job.get('message')}")
        n_frames = (job.get("result") or {}).get("frames_written", 0)
        check("프레임이 저장됨", n_frames > 0, f"{n_frames}장")

        print("\n[5] 라벨 확인")
        r = client.get(f"/api/videos/{video_id}/frames?limit=5")
        check("GET frames", r.status_code == 200)
        items = r.json()["items"]
        check("프레임 목록 조회", len(items) > 0, f"{r.json()['total']}장")
        first = items[0]["frame_id"]
        r = client.get(f"/api/videos/{video_id}/frames/{first}")
        doc = r.json()
        print(f"       {first}: 객체 {len(doc['objects'])}개 status={doc['status']}")
        r = client.get(f"/api/videos/{video_id}/frames/{first}/image?overlay=1")
        check("오버레이 이미지 렌더", r.status_code == 200 and len(r.content) > 1000,
              f"{len(r.content)}B")

        print("\n[6] 수동 라벨 + 승인")
        cls = meta["class_names"][0]
        r = client.put(f"/api/videos/{video_id}/frames/{first}", json={
            "objects": [{"class_name": cls,
                         "poly": [[100, 200], [190, 200], [190, 280], [100, 280]]}],
            "status": "approved",
        })
        check("PUT frame (수동 라벨+승인)", r.status_code == 200, r.text[:200])
        check("승인 상태 반영", r.json().get("status") == "approved")

        # 미확정 클래스가 있으면 승인이 거부돼야 한다(핵심 안전장치).
        target = next((it["frame_id"] for it in items
                       if it["frame_id"] != first and it["n_unresolved"] > 0), None)
        if target:
            r = client.post(f"/api/videos/{video_id}/frames/{target}/status",
                            json={"status": "approved"})
            check("미확정 클래스 승인 거부", r.status_code == 400,
                  f"{r.status_code} {r.text[:120]}")
        else:
            print("       (미확정 객체가 없어 거부 검증 생략)")

        print("\n[7] 객체 단위 클래스 전파")
        r = client.get(f"/api/videos/{video_id}/tracks")
        tracks = r.json()["items"]
        print(f"       track {len(tracks)}개: {[t['track_id'] for t in tracks][:8]}")
        propagated = 0
        for t in tracks:
            r = client.post(f"/api/videos/{video_id}/propagate",
                            json={"track_id": t["track_id"], "class_name": cls})
            propagated += r.json().get("objects", 0)
        check("track 기반 전파 동작", not tracks or propagated > 0, f"{propagated}개 객체")

        print("\n[8] 남은 미확정 객체 수동 지정 + 일괄 승인")
        r = client.get(f"/api/videos/{video_id}/frames?limit=1000")
        for it in r.json()["items"]:
            if it["n_unresolved"] == 0:
                continue
            doc = client.get(f"/api/videos/{video_id}/frames/{it['frame_id']}").json()
            objs = [{"class_name": o.get("class_name") or cls, "poly": o["poly"],
                     "track_id": o.get("track_id"), "score": o.get("score")}
                    for o in doc["objects"]]
            client.put(f"/api/videos/{video_id}/frames/{it['frame_id']}",
                       json={"objects": objs})
        # 자동 라벨이 못 잡은(객체 0개) 프레임도 배경 샘플로 승인된다.
        r = client.post(f"/api/videos/{video_id}/frames/status", json={"status": "approved"})
        check("일괄 승인", r.status_code == 200, r.text[:200])
        print(f"       {r.json()['updated']}장 승인, 실패 {len(r.json()['failed'])}장")
        check("전 프레임 승인", len(r.json()["failed"]) == 0, str(r.json()["failed"][:1]))

        r = client.get(f"/api/videos/{video_id}/progress")
        prog = r.json()
        print(f"       진행률: {prog['approved']}/{prog['frames']} 승인, "
              f"객체 {prog['objects']}개, 미확정 {prog['unresolved_objects']}개")
        check("승인된 프레임 존재", prog["approved"] > 0)

        print("\n[9] 데이터셋 빌드")
        r = client.post("/api/datasets", json={
            "name": "smoke", "video_ids": [video_id], "task": "detect",
            "splits": {"train": 0.6, "valid": 0.4, "test": 0.0},
        })
        check("POST /api/datasets", r.status_code == 200, r.text[:300])
        if r.status_code != 200:
            return 1
        ds_job = _wait_job(client, r.json()["job_id"], timeout=300)
        check("데이터셋 작업 완료", ds_job.get("status") == "done",
              f"{ds_job.get('status')} {ds_job.get('error')}")
        dataset_id = ds_job["target"]
        print(f"       {ds_job.get('message')}")

        r = client.get(f"/api/datasets/{dataset_id}")
        man = r.json()
        print(f"       counts={man['counts']} 클래스분포={man['class_histogram']}")
        check("train/valid 모두 채워짐",
              man["counts"]["train"] > 0 and man["counts"]["valid"] > 0, str(man["counts"]))

        ddir = _WS / "datasets" / dataset_id
        img_n = len(list((ddir / "train" / "images").glob("*.jpg")))
        lbl_n = len(list((ddir / "train" / "labels").glob("*.txt")))
        check("이미지/라벨 파일 수 일치", img_n == lbl_n == man["counts"]["train"],
              f"images={img_n} labels={lbl_n}")

        sample_lbl = next(iter((ddir / "train" / "labels").glob("*.txt")), None)
        if sample_lbl:
            body = sample_lbl.read_text(encoding="utf-8").strip()
            cols = len(body.split("\n")[0].split()) if body else 0
            check("detect 라벨 컬럼 수(5)", cols in (0, 5), f"{cols} cols: {body[:80]}")

        yaml_text = client.get(f"/api/datasets/{dataset_id}/data.yaml").text
        check("data.yaml 생성", "names:" in yaml_text and "train:" in yaml_text)
        print("       ---- data.yaml ----")
        for line in yaml_text.strip().splitlines():
            print(f"       {line}")

        print("\n[10] OBB 태스크 데이터셋(8좌표) 검증")
        r = client.post("/api/datasets", json={
            "name": "smoke-obb", "video_ids": [video_id], "task": "obb",
            "splits": {"train": 1.0, "valid": 0.0, "test": 0.0},
        })
        obb_job = _wait_job(client, r.json()["job_id"], timeout=300)
        check("OBB 데이터셋 빌드", obb_job.get("status") == "done", str(obb_job.get("error")))
        obb_dir = _WS / "datasets" / obb_job["target"] / "train" / "labels"
        sample = next((p for p in obb_dir.glob("*.txt") if p.read_text().strip()), None)
        if sample:
            cols = len(sample.read_text(encoding="utf-8").strip().split("\n")[0].split())
            check("obb 라벨 컬럼 수(9)", cols == 9, f"{cols} cols")

        print("\n[11] 모델 목록")
        r = client.get("/api/models")
        check("GET /api/models", r.status_code == 200)
        weights = r.json()["weights"]
        print(f"       가중치 {len(weights)}개: {[w['name'] for w in weights][:5]}")

        if args.train:
            print("\n[12] 학습 1 epoch")
            r = client.post("/api/train", json={
                "dataset_id": dataset_id, "epochs": 1, "batch": 2, "workers": 0,
            })
            check("POST /api/train", r.status_code == 200, r.text[:300])
            if r.status_code == 200:
                run_id = r.json()["run_id"]
                for w in r.json().get("warnings") or []:
                    print(f"       경고: {w}")
                st = _wait_run(client, run_id, timeout=1800)
                check("학습 완료", st.get("status") == "done",
                      f"{st.get('status')} {st.get('error')}")
                print(f"       weights={st.get('weights')}")
                if st.get("status") == "done":
                    r = client.post("/api/models/promote",
                                    json={"run_id": run_id, "alias": "smoke-model"})
                    check("모델 승격", r.status_code == 200, r.text[:200])

    print("\n" + "=" * 60)
    if _failures:
        print(f"실패 {len(_failures)}건: {_failures}")
    else:
        print("전체 통과")
    print(f"임시 워크스페이스: {_WS}")
    return 1 if _failures else 0


def _wait_job(client, job_id: str, timeout: int = 300) -> dict:
    t0 = time.time()
    last = ""
    while time.time() - t0 < timeout:
        doc = client.get(f"/api/jobs/{job_id}").json()
        msg = f"{doc['status']} {doc['done']}/{doc['total']} {doc.get('message', '')}"
        if msg != last:
            print(f"       … {msg}", flush=True)
            last = msg
        if doc["status"] in ("done", "error", "canceled"):
            return doc
        time.sleep(0.5)
    return {"status": "timeout"}


def _wait_run(client, run_id: str, timeout: int = 1800) -> dict:
    t0 = time.time()
    last = ""
    while time.time() - t0 < timeout:
        doc = client.get(f"/api/train/{run_id}?log_lines=0").json()
        msg = f"{doc['status']} epoch {doc['epoch']}/{doc['epochs']}"
        if msg != last:
            print(f"       … {msg}", flush=True)
            last = msg
        if doc["status"] in ("done", "error", "stopped"):
            return doc
        time.sleep(2)
    return {"status": "timeout"}


if __name__ == "__main__":
    code = main()
    if "--keep" not in sys.argv:
        shutil.rmtree(_WS, ignore_errors=True)
    sys.exit(code)

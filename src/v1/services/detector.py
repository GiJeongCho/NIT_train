"""
services/detector.py
====================

자동 라벨 초안을 만드는 YOLO 추론기.

tracker_py(`src/v1/services/detector.py`) 의 결과 파싱 로직을 그대로 이식했다.
운영 추론과 학습용 자동 라벨이 **같은 규칙으로 박스를 해석**해야, 라벨을 만든 기준과
배포 후 탐지 기준이 어긋나지 않는다. 이식하면서 두 가지를 더한다.

1. OBB 4꼭짓점(`poly`)을 버리지 않고 반환한다. tracker_py 는 트래커 입력용으로
   축정렬 bbox 만 넘기지만, 회전 박스 라벨을 만들려면 꼭짓점이 필요하다.
2. `model.track()` 으로 `track_id` 를 붙인다. 같은 객체가 프레임마다 같은 id 를
   받으므로, 프런트에서 클래스를 한 번 고치면 그 객체의 모든 프레임에 전파할 수 있다.
   (라벨링 공수가 프레임 수가 아니라 객체 수에 비례하게 된다.)

detect 모델과 obb 모델을 모두 지원한다. detect 모델의 축정렬 박스도 4점 폴리곤으로
바꿔 저장하므로, 라벨 자산의 형식은 모델 종류와 무관하게 하나다.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from core.config import PROJECT_ROOT, get_settings
from core import store
from utils.geometry import poly_to_xyxy, xyxy_to_poly

_WEIGHT_EXTS = (".pt", ".engine", ".onnx")


def resolve_model_path(raw: Optional[str]) -> Path:
    """모델 경로 문자열 → 실제 파일 경로.

    절대경로, `test_model/` 기준 상대경로, 워크스페이스 기준 상대경로, 프로젝트 루트
    기준 상대경로를 모두 받는다. 프런트가 어떤 형태로 보내도 동작하게 하려는 것.
    """
    s = get_settings()
    if not raw or not str(raw).strip():
        return s.base_model
    p = Path(str(raw).strip())
    if p.is_absolute():
        return p
    for base in (s.model_dir, store.workspace(), store.models_root(), PROJECT_ROOT):
        cand = base / p
        if cand.exists():
            return cand
    return s.model_dir / p


def infer_task(model_path: Path) -> Optional[str]:
    """파일명으로 태스크 추정. 엔진(.engine)은 메타가 없어 이름에 의존한다."""
    return "obb" if "obb" in model_path.as_posix().lower() else None


class Detector:
    """가중치 1개에 대응하는 추론기. 모델별로 하나씩 캐시된다."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = Path(model_path)
        self.model = None
        self.task = "detect"
        self.class_names: List[str] = []
        self._lock = threading.Lock()

    def load(self) -> "Detector":
        if self.model is not None:
            return self
        if not self.model_path.exists():
            raise FileNotFoundError(f"모델 파일 없음: {self.model_path}")

        from ultralytics import YOLO

        forced = infer_task(self.model_path)
        try:
            model = YOLO(str(self.model_path)) if forced is None else YOLO(str(self.model_path), task=forced)
        except Exception:
            # 태스크 자동 감지 실패(엔진 등) → detect 로 재시도.
            model = YOLO(str(self.model_path), task="detect")

        names_attr = getattr(model, "names", None)
        if isinstance(names_attr, dict):
            names = [str(names_attr[k]) for k in sorted(names_attr.keys())]
        elif isinstance(names_attr, list):
            names = [str(x) for x in names_attr]
        else:
            names = []

        self.model = model
        self.task = str(getattr(model, "task", None) or forced or "detect")
        self.class_names = names
        print(f"[detector] 로드 완료: {self.model_path.name} "
              f"(task={self.task}, classes={len(names)})", flush=True)
        return self

    def warmup(self, runs: int = 1) -> None:
        """더미 추론으로 첫 프레임 지연을 미리 소화한다."""
        if self.model is None:
            return
        s = get_settings()
        imgsz = s.autolabel_imgsz
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        try:
            for _ in range(max(1, runs)):
                self.model.predict(source=dummy, imgsz=imgsz, device=self._device(),
                                   verbose=False)
        except Exception as e:  # noqa: BLE001 - 워밍업 실패는 치명적이지 않다
            print(f"[detector] 워밍업 실패: {e}", flush=True)

    @staticmethod
    def _device():
        d = get_settings().device
        return int(d) if str(d).isdigit() else d

    def reset_tracker(self) -> None:
        """영상이 바뀔 때 트랙 상태를 비운다.

        비우지 않으면 이전 영상의 track_id 가 이어져, 서로 다른 영상의 객체가 같은
        id 를 공유해 클래스 전파가 엉뚱한 프레임까지 번진다.
        """
        with self._lock:
            if self.model is not None:
                # ultralytics 는 predictor 에 트래커 상태를 들고 있다. predictor 를
                # 버리면 다음 track() 호출에서 새로 만든다.
                self.model.predictor = None

    def infer(self, frame: np.ndarray, *, conf: Optional[float] = None,
              iou: Optional[float] = None, imgsz: Optional[int] = None,
              max_det: Optional[int] = None, track: bool = False) -> List[dict]:
        """프레임 1장 추론 → 라벨 초안 객체 리스트.

        반환 원소::

            {"class_id": int, "class_name": str, "score": float,
             "bbox": [x1, y1, x2, y2], "poly": [[x, y] * 4], "track_id": int | None}

        `poly` 는 픽셀 좌표다. obb 모델이면 실제 회전 꼭짓점, detect 모델이면
        축정렬 박스의 네 꼭짓점이 들어간다.
        """
        if self.model is None:
            self.load()
        s = get_settings()
        kwargs = dict(
            source=frame,
            imgsz=int(imgsz if imgsz is not None else s.autolabel_imgsz),
            conf=float(conf if conf is not None else s.autolabel_conf),
            iou=float(iou if iou is not None else s.autolabel_iou),
            max_det=int(max_det if max_det is not None else s.autolabel_max_det),
            device=self._device(),
            verbose=False,
        )

        # 모델 객체는 스레드 안전하지 않다. 잡이 여러 개 동시에 돌 수 있으므로 직렬화한다.
        with self._lock:
            try:
                if track:
                    results = self.model.track(persist=True, tracker=s.autolabel_tracker, **kwargs)
                else:
                    results = self.model.predict(**kwargs)
            except Exception as e:  # noqa: BLE001
                if track:
                    # 트래커 미지원 모델/버전에서도 라벨링 자체는 계속되게 폴백한다.
                    print(f"[detector] 트래킹 실패 → predict 폴백: {e}", flush=True)
                    results = self.model.predict(**kwargs)
                else:
                    raise

        if not results:
            return []
        return self._parse(results[0], frame.shape[1], frame.shape[0])

    def _parse(self, r, width: int, height: int) -> List[dict]:
        """ultralytics Results → 라벨 초안. (tracker_py detector._parse 동일 규칙)"""
        polys = None
        obb = getattr(r, "obb", None)
        if self.task == "obb" and obb is not None and len(obb) > 0:
            xyxy = obb.xyxy.detach().cpu().numpy()
            scores = obb.conf.detach().cpu().numpy()
            labels = obb.cls.detach().cpu().numpy().astype(int)
            polys = obb.xyxyxyxy.detach().cpu().numpy()  # (N, 4, 2)
            ids = getattr(obb, "id", None)
        else:
            boxes = getattr(r, "boxes", None)
            if boxes is None or boxes.shape[0] == 0:
                return []
            xyxy = boxes.xyxy.detach().cpu().numpy()
            scores = boxes.conf.detach().cpu().numpy()
            labels = boxes.cls.detach().cpu().numpy().astype(int)
            ids = getattr(boxes, "id", None)

        track_ids = (ids.detach().cpu().numpy().astype(int).tolist()
                     if ids is not None else None)

        out: List[dict] = []
        for i, (box, lbl_idx, score) in enumerate(zip(xyxy, labels, scores)):
            cid = int(lbl_idx)
            name = (self.class_names[cid] if 0 <= cid < len(self.class_names) else f"id{cid}")
            if polys is not None:
                poly = [[float(x), float(y)] for x, y in polys[i]]
                bbox = poly_to_xyxy(poly)
            else:
                x1, y1, x2, y2 = (float(v) for v in box)
                poly = xyxy_to_poly(x1, y1, x2, y2)
                bbox = [x1, y1, x2, y2]
            out.append({
                "class_id": cid,
                "class_name": name,
                "score": round(float(score), 4),
                "bbox": [round(v, 2) for v in bbox],
                "poly": [[round(x, 2), round(y, 2)] for x, y in poly],
                "track_id": (track_ids[i] if track_ids is not None and i < len(track_ids) else None),
            })
        return out

    def info(self) -> dict:
        return {
            "path": store.rel_to_workspace(self.model_path) or str(self.model_path),
            "name": self.model_path.name,
            "task": self.task,
            "class_names": self.class_names,
            "loaded": self.model is not None,
        }


# 모델 경로별 싱글톤. 같은 가중치로 여러 잡을 돌릴 때 재로드(수 초~수십 초)를 피한다.
_cache: Dict[str, Detector] = {}
_cache_lock = threading.Lock()


def get_detector(model: Optional[str] = None) -> Detector:
    path = resolve_model_path(model)
    key = str(path.resolve()) if path.exists() else str(path)
    with _cache_lock:
        det = _cache.get(key)
        if det is None:
            det = Detector(path)
            _cache[key] = det
    return det.load()


def list_weights() -> List[dict]:
    """자동 라벨/학습에 쓸 수 있는 가중치 목록.

    - `test_model/` : 사전학습 기본 가중치(기본값 yolo26l-obb.pt)
    - `workspace/models/` : 승격된 배포 후보
    - `workspace/runs/*/train/weights/` : 학습 산출물(best/last)
    """
    s = get_settings()
    items: List[dict] = []
    seen: set = set()

    def add(p: Path, origin: str) -> None:
        rp = p.resolve()
        if rp in seen or not p.exists():
            return
        seen.add(rp)
        items.append({
            "path": store.rel_to_workspace(p) or p.as_posix(),
            "abs_path": p.as_posix(),
            "name": p.name,
            "origin": origin,
            "task": infer_task(p) or "detect",
            "size_mb": round(p.stat().st_size / 1e6, 1),
            "is_default": rp == s.base_model.resolve() if s.base_model.exists() else False,
        })

    for ext in _WEIGHT_EXTS:
        for p in sorted(s.model_dir.glob(f"*{ext}")):
            add(p, "pretrained")
    for ext in _WEIGHT_EXTS:
        for p in sorted(store.models_root().glob(f"*{ext}")):
            add(p, "promoted")
    for run_id in store.list_run_ids():
        wdir = store.run_output_dir(run_id) / "weights"
        for name in ("best.pt", "last.pt"):
            p = wdir / name
            if p.exists():
                add(p, f"run:{run_id}")
    return items

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

# 추론 서비스(tracker_py)에서 이식한 커스텀 칼만 트래커. scipy 등 미설치 환경에서도
# 라벨링이 죽지 않도록 import 실패는 조용히 넘기고, 런타임에 ultralytics 트래커로 폴백한다.
try:
    from services.kalman_tracker import MultiObjectTracker
    _HAS_KALMAN = True
except Exception as _e:  # noqa: BLE001
    MultiObjectTracker = None  # type: ignore
    _HAS_KALMAN = False
    print(f"[detector] 커스텀 칼만 트래커 로드 실패 → ultralytics 트래커 사용: {_e}", flush=True)

_WEIGHT_EXTS = (".pt", ".engine", ".onnx")


def _iou_xyxy(a, b) -> float:
    """축정렬 박스 두 개의 IoU. 탐지 ↔ 트랙 매칭(track_id 부여)에 쓴다."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


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
        # 커스텀 칼만 트래커 상태(영상 단위). track=True 이고 커스텀 트래커를 쓸 때만 생성.
        self._mot = None

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
            # 커스텀 칼만 트래커도 초기화(트랙/ID 카운터 리셋). 안 비우면 이전 영상의
            # track_id 가 이어져 클래스 전파가 엉뚱한 영상까지 번진다.
            if self._mot is not None:
                self._mot.reset()

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

        # 커스텀 칼만 트래커를 쓸지 결정. track=True + 설정 ON + import 성공일 때만.
        use_custom = bool(track and _HAS_KALMAN and s.autolabel_custom_tracker)

        # 모델 객체는 스레드 안전하지 않다. 잡이 여러 개 동시에 돌 수 있으므로 직렬화한다.
        # 커스텀 트래커 상태(self._mot)도 같은 락으로 보호한다.
        with self._lock:
            try:
                if track and not use_custom:
                    # ultralytics 내장 트래커(폴백 경로).
                    results = self.model.track(persist=True, tracker=s.autolabel_tracker, **kwargs)
                else:
                    # 커스텀 트래커는 predict 결과를 직접 트래커에 넣어 track_id 를 붙인다.
                    results = self.model.predict(**kwargs)
            except Exception as e:  # noqa: BLE001
                if track and not use_custom:
                    # 트래커 미지원 모델/버전에서도 라벨링 자체는 계속되게 폴백한다.
                    print(f"[detector] 트래킹 실패 → predict 폴백: {e}", flush=True)
                    results = self.model.predict(**kwargs)
                else:
                    raise

            if not results:
                return []
            objects = self._parse(results[0], frame.shape[1], frame.shape[0])
            if use_custom:
                try:
                    objects = self._tracks_to_objects(objects)
                except Exception as e:  # noqa: BLE001
                    # 트래킹은 부가 기능이다. 실패해도 라벨(박스/클래스) 자체는 살린다.
                    print(f"[detector] 커스텀 트래킹 실패 → track_id 없이 진행: {e}", flush=True)
            return objects

    def _ensure_mot(self) -> None:
        """커스텀 트래커를 (저 fps 자동라벨에 맞춘 설정으로) 지연 생성한다."""
        if self._mot is None and MultiObjectTracker is not None:
            s = get_settings()
            # 트래커 기본 설정에는 ①적응형 감쇠 ②라벨 융합 비용 ③회귀 속도 되먹임 ④모션 Re-ID
            # ⑤라벨 고정 이 모두 ON 이다(kalman_tracker.TrackerConfig). 저 fps 샘플링에 맞춰
            # 표시 시작(min_hits)과 고스트 수명(max_age)만 낮춰 준다.
            try:
                from services.kalman_tracker import TrackerConfig  # noqa: WPS433
                cfg = TrackerConfig(
                    min_hits=max(1, int(s.autolabel_track_min_hits)),
                    max_age=max(1, int(s.autolabel_track_max_age)),
                )
                self._mot = MultiObjectTracker(config=cfg)
            except Exception:  # noqa: BLE001 — TrackerConfig import 실패 시 레거시 생성
                self._mot = MultiObjectTracker(min_hits=max(1, int(s.autolabel_track_min_hits)))

    def _tracks_to_objects(self, objects: List[dict]) -> List[dict]:
        """커스텀 칼만 트래커의 **출력 자체**로 라벨 객체 리스트를 다시 만든다.

        예전에는 실제 YOLO 탐지에만 track_id 만 얹고 나머지(고스트·라벨고정·스무딩)를 버렸다.
        여기서는 트래커가 내보낸 트랙을 1급 결과로 삼아, 트래커의 모든 기능을 그대로 반영한다.

        - 이번 프레임에 실탐지로 갱신된 트랙: 원래 탐지의 `poly`(회전박스 꼭짓점)를 살리되,
          클래스는 트래커의 **고정 라벨**(다수결/lock)을, id 는 track_id 를 쓴다.
        - **고스트(예측) 트랙**: 이번 프레임에 탐지가 없어도 칼만 예측 bbox 로 박스를 만들어
          내보낸다(`is_predicted=True`). → 깜빡임 없이 트래킹이 연속돼 보인다.
        - min_hits 미확정 신규 탐지처럼 트랙으로 안 잡힌 실탐지도 버리지 않고 track_id=None 으로
          함께 내보낸다(박스 자체는 보여야 하므로).
        """
        if MultiObjectTracker is None:
            return objects
        self._ensure_mot()
        s = get_settings()
        emit_ghost = bool(s.autolabel_emit_ghost)

        dets = [{
            "label": o.get("class_name") or o.get("model_class_name") or "?",
            "score": float(o.get("score") or 0.0),
            "bbox": [float(v) for v in o["bbox"]],
        } for o in objects]
        tracks = self._mot.update(dets)

        # 트랙 ↔ 이번 프레임 실탐지 매칭(회전 poly 를 재사용하기 위함).
        # 저 fps 에서는 칼만 보정 bbox 가 원본 탐지와 덜 겹칠 수 있으므로, IoU 우선 매칭 뒤
        # 남은 것은 '중심 거리' 로 한 번 더 이어 붙인다(트랙을 통째로 버리지 않도록).
        pairs = []
        for ti, t in enumerate(tracks):
            for oi, o in enumerate(objects):
                iou = _iou_xyxy(o["bbox"], t["bbox"])
                if iou > 0.0:
                    pairs.append((iou, ti, oi))
        pairs.sort(key=lambda p: p[0], reverse=True)
        trk_to_obj: dict = {}
        used_obj: set = set()
        for _iou, ti, oi in pairs:
            if ti in trk_to_obj or oi in used_obj:
                continue
            trk_to_obj[ti] = oi
            used_obj.add(oi)

        # 2차: IoU 로 못 붙은 실탐지·실트랙을 중심 거리로 매칭(게이트=트랙 박스 대각선).
        dpairs = []
        for ti, t in enumerate(tracks):
            if ti in trk_to_obj or t.get("is_predicted"):
                continue
            tb = t["bbox"]
            tcx = (tb[0] + tb[2]) / 2.0; tcy = (tb[1] + tb[3]) / 2.0
            diag = ((tb[2] - tb[0]) ** 2 + (tb[3] - tb[1]) ** 2) ** 0.5 or 1.0
            for oi, o in enumerate(objects):
                if oi in used_obj:
                    continue
                ob = o["bbox"]
                ocx = (ob[0] + ob[2]) / 2.0; ocy = (ob[1] + ob[3]) / 2.0
                dist = ((tcx - ocx) ** 2 + (tcy - ocy) ** 2) ** 0.5
                if dist <= 1.5 * diag:
                    dpairs.append((dist, ti, oi))
        dpairs.sort(key=lambda p: p[0])
        for _d, ti, oi in dpairs:
            if ti in trk_to_obj or oi in used_obj:
                continue
            trk_to_obj[ti] = oi
            used_obj.add(oi)

        out: List[dict] = []
        for ti, t in enumerate(tracks):
            is_pred = bool(t.get("is_predicted"))
            if is_pred and not emit_ghost:
                continue
            oi = trk_to_obj.get(ti)
            if oi is not None:
                # 실탐지에 매칭됨: 원래 회전 poly/모델 클래스를 살리고 트래커 라벨/ID 를 얹는다.
                base = dict(objects[oi])
                base["model_class_name"] = objects[oi].get("class_name")
                base["class_name"] = t.get("label") or objects[oi].get("class_name")
                base["track_id"] = int(t["track_id"])
                base["is_predicted"] = False
                base["track_status"] = t.get("status")
                out.append(base)
            else:
                # 매칭 실탐지가 없음 → 고스트(예측)이거나, 보정 박스가 원본과 안 겹친 실트랙.
                # 어느 쪽이든 트랙을 버리지 않고 트래커 bbox 로 박스를 만들어 내보낸다.
                bbox = [float(v) for v in t["bbox"]]
                out.append({
                    "class_name": t.get("label"),
                    "model_class_name": t.get("label"),
                    "score": float(t.get("score") or 0.0),
                    "bbox": bbox,
                    "poly": xyxy_to_poly(*bbox),
                    "track_id": int(t["track_id"]),
                    "is_predicted": is_pred,
                    "track_status": t.get("status") or ("ghost" if is_pred else None),
                })

        # 어떤 트랙에도 안 붙은 실탐지(예: min_hits 미확정 신규)는 박스만 보이게 유지.
        for oi, o in enumerate(objects):
            if oi in used_obj:
                continue
            base = dict(o)
            base.setdefault("track_id", None)
            base["is_predicted"] = False
            out.append(base)
        return out

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
# 미리보기 전용(트래킹 상태 격리). 추출 잡과 track_id 상태가 섞이지 않도록 캐시를 분리한다.
_preview_cache: Dict[str, Detector] = {}
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


def get_preview_detector(model: Optional[str] = None) -> Detector:
    """미리보기용 추론기. 추출 잡이 쓰는 싱글톤과 **별도 인스턴스**라, 미리보기의 트래킹
    reset/누적이 진행 중인 추출 잡의 track_id 를 오염시키지 않는다.
    """
    path = resolve_model_path(model)
    key = str(path.resolve()) if path.exists() else str(path)
    with _cache_lock:
        det = _preview_cache.get(key)
        if det is None:
            det = Detector(path)
            _preview_cache[key] = det
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

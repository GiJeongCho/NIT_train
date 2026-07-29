"""
services/segments.py
====================

영상의 **정상/비정상 구간** 관리.

드론 영상 한 편에는 학습에 쓸 만한 구간과 못 쓸 구간이 섞여 있다.
(이륙/착륙, 급기동으로 흔들린 구간, 렌즈 오염, 노출 폭주, 오탐이 쏟아지는 구간 …)
사용자는 프런트 타임라인에서 이 구간들을 표시하고, 학습 데이터는 **정상 구간에서
비정상 구간을 뺀 부분**에서만 뽑는다.

구간은 초(sec) 단위 실수로 받는다. 프레임 인덱스로 받으면 fps 가 다른 영상 간에
프런트 코드가 갈리고, 사용자가 보는 플레이어 시간과도 어긋난다.

핵심 규칙
- 정상 구간을 하나도 지정하지 않으면 **영상 전체를 정상**으로 본다(짧은 클립 편의).
- 정상 ∩ 비정상은 항상 비정상이 이긴다(사람이 "쓰지 말라"고 한 쪽을 존중).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from core import store
from services import video as video_svc

KINDS = ("normal", "abnormal")
Range = Tuple[float, float]

# 이보다 짧은 구간은 프레임이 1장도 안 나오거나 실수 클릭이므로 버린다.
_MIN_LEN_SEC = 0.05


def _merge(ranges: Sequence[Range]) -> List[Range]:
    """겹치거나 맞닿은 구간을 합친다."""
    items = sorted((float(a), float(b)) for a, b in ranges if float(b) - float(a) > 0)
    out: List[Range] = []
    for start, end in items:
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def _subtract(base: Sequence[Range], cut: Sequence[Range]) -> List[Range]:
    """base 구간에서 cut 구간을 뺀다."""
    result: List[Range] = []
    for start, end in _merge(base):
        pieces: List[Range] = [(start, end)]
        for cs, ce in _merge(cut):
            nxt: List[Range] = []
            for ps, pe in pieces:
                if ce <= ps or cs >= pe:
                    nxt.append((ps, pe))
                    continue
                if cs > ps:
                    nxt.append((ps, cs))
                if ce < pe:
                    nxt.append((ce, pe))
            pieces = nxt
        result.extend(p for p in pieces if p[1] - p[0] >= _MIN_LEN_SEC)
    return result


def normalize(raw_segments, duration_sec: float) -> List[dict]:
    """프런트 입력을 검증·정렬·병합한다.

    같은 종류끼리는 병합해 저장한다. 사용자가 드래그를 여러 번 해 생긴 중복 구간이
    그대로 남으면 프레임 추출에서 같은 프레임을 두 번 처리하게 된다.
    """
    buckets: dict = {k: [] for k in KINDS}
    notes: dict = {k: [] for k in KINDS}
    for seg in (raw_segments or []):
        if not isinstance(seg, dict):
            continue
        kind = str(seg.get("kind", "normal")).strip().lower()
        if kind not in KINDS:
            raise ValueError(f"kind 는 {list(KINDS)} 중 하나여야 합니다: {kind!r}")
        try:
            start = float(seg.get("start_sec", 0.0))
            end = float(seg.get("end_sec", 0.0))
        except (TypeError, ValueError):
            raise ValueError("start_sec / end_sec 는 숫자여야 합니다")
        if end < start:
            start, end = end, start
        limit = float(duration_sec) if duration_sec and duration_sec > 0 else end
        start = max(0.0, min(start, limit))
        end = max(0.0, min(end, limit))
        if end - start < _MIN_LEN_SEC:
            continue
        buckets[kind].append((start, end))
        note = str(seg.get("note", "") or "").strip()
        if note:
            notes[kind].append(note)

    out: List[dict] = []
    for kind in KINDS:
        for i, (start, end) in enumerate(_merge(buckets[kind])):
            out.append({
                "id": f"{kind[:3]}{i + 1}",
                "kind": kind,
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "duration_sec": round(end - start, 3),
                "note": "; ".join(notes[kind]) if notes[kind] and len(buckets[kind]) == 1 else "",
            })
    out.sort(key=lambda s: (s["start_sec"], s["kind"]))
    return out


def get(video_id: str) -> dict:
    doc = store.read_json(store.video_segments_path(video_id), None)
    if not isinstance(doc, dict):
        return {"video_id": video_id, "segments": [], "updated_at": None}
    return doc


def put(video_id: str, raw_segments) -> dict:
    meta = video_svc.get_meta(video_id)
    segs = normalize(raw_segments, float(meta.get("duration_sec") or 0.0))
    doc = {"video_id": video_id, "segments": segs, "updated_at": store.now_iso()}
    store.write_json(store.video_segments_path(video_id), doc)
    return doc


def summary(video_id: str) -> dict:
    segs = get(video_id).get("segments") or []
    out = {"total": len(segs)}
    for kind in KINDS:
        picked = [s for s in segs if s.get("kind") == kind]
        out[kind] = len(picked)
        out[f"{kind}_sec"] = round(sum(float(s.get("duration_sec") or 0.0) for s in picked), 2)
    return out


def selection_ranges(video_id: str, include: Optional[Sequence[str]] = None) -> List[Range]:
    """실제로 프레임을 뽑을 구간 목록.

    `include` 에 든 종류를 합친 뒤, 거기 없는 종류를 뺀다. 기본값(`("normal",)`)은
    "정상 구간에서 비정상 구간을 제외한 부분" 이 된다.
    """
    wanted = tuple(include) if include else ("normal",)
    for k in wanted:
        if k not in KINDS:
            raise ValueError(f"kind 는 {list(KINDS)} 중 하나여야 합니다: {k!r}")

    meta = video_svc.get_meta(video_id)
    duration = float(meta.get("duration_sec") or 0.0)
    segs = get(video_id).get("segments") or []

    base = [(float(s["start_sec"]), float(s["end_sec"])) for s in segs if s.get("kind") in wanted]
    if not base:
        if any(s.get("kind") in wanted for s in segs):
            return []
        # 해당 종류를 하나도 안 찍었으면 영상 전체를 대상으로 본다.
        if duration <= 0:
            raise ValueError("영상 길이를 알 수 없어 전체 구간을 쓸 수 없습니다. 구간을 직접 지정하세요.")
        base = [(0.0, duration)]

    cut = [(float(s["start_sec"]), float(s["end_sec"])) for s in segs if s.get("kind") not in wanted]
    return _subtract(base, cut)

"""
services/classes.py
===================

프로젝트 클래스 목록(라벨 공간) 관리.

자동 라벨을 만드는 모델의 클래스와 우리가 학습시키려는 클래스는 **다르다.**
기본 가중치 `yolo26l-obb.pt` 는 DOTA 15 클래스(plane/ship/large-vehicle…)로 회전박스를
찾아주지만, 실제 학습 목표는 표적 종류(Panther_II, VIDAR …)다. 그래서

- 모델이 준 클래스명은 `model_class_name` 으로 **참고용**만 보관하고,
- 라벨의 진짜 값은 이 목록에 있는 `class_name` 이다.

데이터셋을 만들 때 `class_name` → 목록의 인덱스로 변환해 라벨 txt 를 쓴다.
따라서 **목록의 순서가 곧 클래스 id** 이고, 순서를 바꾸면 기존에 만든 데이터셋과
클래스 id 가 어긋난다. 그래서 순서 변경/삭제는 막고 추가/이름변경만 허용한다.
"""

from __future__ import annotations

from typing import List, Optional

from core.config import get_settings
from core import store


def _path():
    return store.workspace() / "classes.json"


def get() -> dict:
    doc = store.read_json(_path(), None)
    if isinstance(doc, dict) and isinstance(doc.get("class_names"), list) and doc["class_names"]:
        return doc
    # 최초 호출: 설정 기본값(tracker_py OBB 데이터셋과 같은 순서)으로 초기화한다.
    doc = {
        "class_names": list(get_settings().class_names),
        "updated_at": store.now_iso(),
    }
    store.write_json(_path(), doc)
    return doc


def names() -> List[str]:
    return list(get()["class_names"])


def put(class_names) -> dict:
    """클래스 목록을 갱신한다.

    기존 항목의 **순서와 개수는 유지**해야 한다(이미 저장된 라벨/데이터셋의 클래스 id
    가 인덱스이기 때문). 뒤에 추가하거나 오타를 고치는 것만 허용한다.
    """
    new = [str(x).strip() for x in (class_names or []) if str(x).strip()]
    if not new:
        raise ValueError("클래스 목록이 비어 있습니다")
    if len(new) != len(set(new)):
        raise ValueError("중복된 클래스명이 있습니다")
    cur = names()
    if len(new) < len(cur):
        raise ValueError(
            f"클래스는 삭제할 수 없습니다(기존 {len(cur)}개 → 요청 {len(new)}개). "
            "이미 저장된 라벨의 클래스 id 가 인덱스이므로 순서가 어긋납니다."
        )
    doc = {"class_names": new, "updated_at": store.now_iso()}
    store.write_json(_path(), doc)
    return doc


def index_of(class_name: Optional[str]) -> int:
    """클래스명 → 인덱스. 목록에 없으면 -1(=미확정 라벨)."""
    if not class_name:
        return -1
    try:
        return names().index(str(class_name))
    except ValueError:
        return -1


def name_of(index: int) -> Optional[str]:
    ns = names()
    return ns[index] if 0 <= int(index) < len(ns) else None

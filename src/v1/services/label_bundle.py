"""Portable export/import for editable video labeling workspaces."""

from __future__ import annotations

import json
import os
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import Optional

from core import store
from services import annotations, classes as classes_svc, segments, video

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_BUNDLE_VERSION = 1


def export_zip(video_id: str, *, include_video: bool = False) -> Path:
    """Bundle source metadata, segments, extracted frames, editable labels and class order."""
    meta = video.get_meta(video_id)
    frame_ids = store.list_frame_ids(video_id)
    if not frame_ids:
        raise ValueError("내보낼 프레임/라벨이 없습니다.")

    meta_export = dict(meta)
    source = Path(str(meta.get("path") or ""))
    if include_video and (not source.is_file()):
        raise ValueError("원본 영상을 포함할 수 없습니다. 원본 파일 경로를 확인하세요.")
    # 이전 서버의 절대 경로를 다른 인스턴스에 노출하거나 잘못 재사용하지 않는다.
    meta_export["path"] = ""
    meta_export["managed"] = False
    safe_name = Path(str(meta.get("original_name") or video_id)).stem
    safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in safe_name).strip("._") or video_id
    out = store.video_dir(video_id) / f"{safe_name}_labeling.zip"
    manifest = {
        "format": "nit-train-labeling-bundle",
        "version": _BUNDLE_VERSION,
        "created_at": store.now_iso(),
        "frame_count": len(frame_ids),
        "includes_video": bool(include_video),
        "source_video_name": source.name if include_video else None,
        "class_names": classes_svc.names(),
    }

    with zipfile.ZipFile(out, "w", allowZip64=True) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        zf.writestr("meta.json", json.dumps(meta_export, ensure_ascii=False, indent=2))
        zf.writestr("segments.json", json.dumps(segments.get(video_id), ensure_ascii=False, indent=2))
        for frame_id in frame_ids:
            fid = store.safe_id(frame_id)
            image = store.frame_image_path(video_id, fid)
            label = store.frame_label_path(video_id, fid)
            if not image.is_file() or not label.is_file():
                raise ValueError(f"프레임 이미지 또는 라벨이 누락되었습니다: {fid}")
            zf.write(image, arcname=f"frames/{fid}.jpg")
            zf.write(label, arcname=f"labels/{fid}.json")
        if include_video:
            zf.write(source, arcname=f"source_video/{source.name}")
    return out


def _safe_extract(zip_path: Path, staging: Path) -> Path:
    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
        if len(infos) > 250_000:
            raise ValueError("압축 안 파일 수가 너무 많습니다.")
        total_size = sum(i.file_size for i in infos)
        if total_size > 1_000 * 1024**3:
            raise ValueError("압축 해제 데이터가 1TiB 제한을 초과합니다.")
        for info in infos:
            rel = PurePosixPath(info.filename)
            if rel.is_absolute() or ".." in rel.parts or not rel.parts:
                raise ValueError(f"안전하지 않은 zip 경로입니다: {info.filename}")
            mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(mode):
                raise ValueError("심볼릭 링크가 포함된 zip은 가져올 수 없습니다.")
            dest = staging.joinpath(*rel.parts)
            if os.path.commonpath([str(staging.resolve()), str(dest.resolve())]) != str(staging.resolve()):
                raise ValueError(f"압축 경로가 임시 폴더 밖을 가리킵니다: {info.filename}")
        zf.extractall(staging)

    root = staging
    if not (root / "manifest.json").is_file():
        candidates = [p.parent for p in staging.rglob("manifest.json")]
        if len(candidates) != 1:
            raise ValueError("라벨링 패키지 manifest.json을 찾을 수 없습니다.")
        root = candidates[0]
    return root


def import_zip(zip_path: Path) -> dict:
    """Import a portable bundle as a new editable video entry, remapping class indices."""
    staging = store.workspace() / ".label_import_staging" / store.new_id()
    staging.mkdir(parents=True, exist_ok=True)
    uploaded_video: Optional[Path] = None
    video_id: Optional[str] = None
    try:
        try:
            root = _safe_extract(Path(zip_path), staging)
        except zipfile.BadZipFile as e:
            raise ValueError("올바른 라벨링 zip 파일이 아닙니다.") from e
        try:
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            source_meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
            source_segments = json.loads((root / "segments.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise ValueError(f"라벨링 패키지의 JSON을 읽을 수 없습니다: {e}") from e
        if manifest.get("format") != "nit-train-labeling-bundle" or manifest.get("version") != _BUNDLE_VERSION:
            raise ValueError("지원하지 않는 라벨링 패키지 형식/버전입니다.")

        images = {p.stem: p for p in (root / "frames").rglob("*")
                  if p.is_file() and p.suffix.lower() in _IMAGE_EXTS}
        labels = {p.stem: p for p in (root / "labels").rglob("*.json") if p.is_file()}
        if not images or set(images) != set(labels):
            raise ValueError("프레임 이미지와 라벨 JSON이 한 쌍으로 있어야 합니다.")
        for fid in images:
            store.safe_id(fid)

        source_names = [str(n) for n in (manifest.get("class_names") or [])]
        current_names = classes_svc.names()
        merged_names = current_names + [n for n in source_names if n not in current_names]
        class_index = {name: i for i, name in enumerate(merged_names)}

        video_id = store.new_id()
        video_root = store.video_dir(video_id)
        store.frames_dir(video_id).mkdir(parents=True, exist_ok=True)
        store.labels_dir(video_id).mkdir(parents=True, exist_ok=True)

        video_files = [p for p in (root / "source_video").glob("*") if p.is_file()] if (root / "source_video").is_dir() else []
        if len(video_files) > 1:
            raise ValueError("source_video 폴더에는 원본 영상 하나만 포함할 수 있습니다.")
        if video_files:
            src = video_files[0]
            ext = video.check_ext(src.name)
            uploaded_video = store.uploads_root() / f"{video_id}{ext}"
            uploaded_video.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, uploaded_video)
            imported_meta = video._write_meta(
                video_id, uploaded_video,
                str(source_meta.get("original_name") or src.name), True,
            )
            imported_meta["preprocess"] = source_meta.get("preprocess") or {}
            imported_meta["source_video_included"] = True
        else:
            imported_meta = dict(source_meta)
            imported_meta.update({
                "video_id": video_id,
                "path": "",
                "managed": False,
                "source_video_included": False,
                "created_at": store.now_iso(),
            })
            imported_meta["size_mb"] = 0

        if merged_names != current_names:
            classes_svc.put(merged_names)

        for fid, image_path in images.items():
            shutil.copy2(image_path, store.frame_image_path(video_id, fid))
            doc = json.loads(labels[fid].read_text(encoding="utf-8"))
            doc["video_id"] = video_id
            doc["frame_id"] = fid
            for obj in doc.get("objects") or []:
                class_name = obj.get("class_name")
                if class_name in class_index:
                    obj["class_id"] = class_index[class_name]
                else:
                    old_id = int(obj.get("class_id", -1))
                    fallback_name = source_names[old_id] if 0 <= old_id < len(source_names) else None
                    obj["class_name"] = fallback_name
                    obj["class_id"] = class_index.get(fallback_name, -1)
            store.write_json(store.frame_label_path(video_id, fid), doc)

        source_segments["video_id"] = video_id
        store.write_json(store.video_segments_path(video_id), source_segments)
        imported_meta["imported_from_bundle"] = True
        store.write_json(store.video_meta_path(video_id), imported_meta)
        return {
            **imported_meta,
            "frame_count": len(images),
            "labeling": annotations.progress(video_id),
            "source_video_included": bool(video_files),
        }
    except Exception:
        if uploaded_video:
            uploaded_video.unlink(missing_ok=True)
        if video_id:
            shutil.rmtree(store.video_dir(video_id), ignore_errors=True)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)

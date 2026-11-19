"""Run isolation for reproducible video reconstruction.

No stage should write into a shared ``outputs`` directory during a new run.
Each :class:`RunContext` owns one immutable input identity and one fresh output
directory.  The manifest is updated in place only to record stage status and
does not infer provenance from filenames or existing artefacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
from typing import Any, Dict, Optional
from uuid import uuid4

import cv2


class RunManifestError(ValueError):
    """Raised when a requested run cannot be safely created or updated."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _video_metadata(path: Path) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RunManifestError(f"OpenCV cannot read input video: {path}")
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if frames <= 0 or fps <= 0 or width <= 0 or height <= 0:
        raise RunManifestError(f"Input video metadata is invalid: {path}")
    return {
        "frame_count": frames,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_seconds": frames / fps,
    }


@dataclass(frozen=True)
class RunContext:
    """A unique, single-video reconstruction run and its provenance manifest."""

    run_id: str
    root: Path
    video_path: Path
    manifest_path: Path

    @classmethod
    def create(
        cls,
        video_path: str | Path,
        *,
        output_root: str | Path = "outputs/runs",
        category: str = "auto",
        configuration: Optional[Dict[str, Any]] = None,
        camera_parameters: Optional[Dict[str, Any]] = None,
    ) -> "RunContext":
        source = Path(video_path).expanduser().resolve()
        if not source.is_file() or source.stat().st_size == 0:
            raise RunManifestError(f"Select one non-empty input video before starting: {source}")

        input_hash = _file_hash(source)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_stem = "".join(c if c.isalnum() else "_" for c in source.stem).strip("_") or "video"
        run_id = f"{timestamp}_{safe_stem}_{input_hash[:10]}_{uuid4().hex[:8]}"
        root = Path(output_root).resolve() / run_id
        # A UUID makes collision virtually impossible; fail rather than merge if it occurs.
        if root.exists():
            raise RunManifestError(f"Refusing to reuse an existing run directory: {root}")
        root.mkdir(parents=True)

        # Archive the exact submitted bytes inside the run.  A later upload with
        # the same filename must never change what an earlier run means.
        archived_input = root / "input" / source.name
        archived_input.parent.mkdir()
        shutil.copy2(source, archived_input)
        if _file_hash(archived_input) != input_hash:
            raise RunManifestError("Input copy hash mismatch; run was not created safely")

        manifest_path = root / "manifest.json"
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "created_at_utc": _utc_now(),
            "updated_at_utc": _utc_now(),
            "status": "created",
            "input": {
                "absolute_path": str(source),
                "archived_path": str(archived_input.relative_to(root)).replace("\\", "/"),
                "filename": source.name,
                "sha256": input_hash,
                "size_bytes": source.stat().st_size,
                "video_metadata": _video_metadata(source),
            },
            "category_request": category,
            "camera_parameters": camera_parameters or {"source": "auto_estimated"},
            "configuration": configuration or {},
            "software": {"python": sys.version, "platform": platform.platform()},
            "stages": [],
            "artifacts": {},
            "warnings": [
                "All geometry is non-metric unless a validated calibration and scale reference are recorded."
            ],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return cls(run_id=run_id, root=root, video_path=archived_input, manifest_path=manifest_path)

    def read_manifest(self) -> Dict[str, Any]:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def _write_manifest(self, data: Dict[str, Any]) -> None:
        data["updated_at_utc"] = _utc_now()
        self.manifest_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def path(self, *parts: str) -> Path:
        candidate = (self.root.joinpath(*parts)).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise RunManifestError("Artifact path escapes the current run directory")
        return candidate

    def stage_started(self, name: str) -> None:
        data = self.read_manifest()
        data["status"] = "running"
        data["stages"].append({"name": name, "status": "running", "started_at_utc": _utc_now()})
        self._write_manifest(data)

    def stage_finished(self, name: str, *, status: str, details: Optional[Dict[str, Any]] = None) -> None:
        if status not in {"passed", "failed", "partial", "skipped"}:
            raise RunManifestError(f"Unknown stage status: {status}")
        data = self.read_manifest()
        for stage in reversed(data["stages"]):
            if stage["name"] == name and stage["status"] == "running":
                stage.update({"status": status, "finished_at_utc": _utc_now(), "details": details or {}})
                break
        else:
            raise RunManifestError(f"Cannot finish stage that was not started: {name}")
        self._write_manifest(data)

    def register_artifact(self, name: str, artifact: str | Path) -> None:
        path = Path(artifact).resolve()
        if self.root not in path.parents:
            raise RunManifestError(f"Artifact is outside this run and cannot be registered: {path}")
        data = self.read_manifest()
        data["artifacts"][name] = str(path.relative_to(self.root)).replace("\\", "/")
        self._write_manifest(data)

    def finish(self, status: str, *, warning: Optional[str] = None) -> None:
        if status not in {"completed", "completed_with_warnings", "failed"}:
            raise RunManifestError(f"Unknown run status: {status}")
        data = self.read_manifest()
        data["status"] = status
        data["finished_at_utc"] = _utc_now()
        if warning:
            data["warnings"].append(warning)
        self._write_manifest(data)

"""Process downloaded Objectron clips into isolated project point-cloud runs.

This is intentionally separate from downloading: first inspect the subset
manifest, then process a bounded number of videos per class. Every clip is
passed as one input to run_isolated_reconstruction.py, which creates a fresh
run directory and labels it from Objectron's category metadata.

Example:
    python scripts/process_objectron_subset.py --per-class 1
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_CLASSES = {"bicycle", "book", "bottle", "camera", "cereal_box", "chair", "cup", "laptop", "shoe"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def existing_processed_hashes(runs_root: Path) -> set[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    for path in runs_root.glob("*/manifest.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            root = path.parent
            cloud = root / "reconstruction" / "object_points.ply"
            if data.get("status") in {"completed", "completed_with_warnings"} and cloud.is_file():
                seen.add((data.get("input", {}).get("sha256", ""), data.get("category_request", "")))
        except (OSError, json.JSONDecodeError):
            continue
    return seen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-manifest", type=Path,
                        default=PROJECT_ROOT / "data" / "objectron_subset" / "manifest.json")
    parser.add_argument("--runs-root", type=Path, default=PROJECT_ROOT / "outputs" / "runs")
    parser.add_argument("--per-class", type=int, default=1,
                        help="Maximum new videos per class in this invocation (default: 1).")
    parser.add_argument("--max-keyframes", type=int, default=20,
                        help="Keyframes per video for this pilot (default: 20; minimum: 8).")
    parser.add_argument("--classes", nargs="+", choices=sorted(ALLOWED_CLASSES), default=sorted(ALLOWED_CLASSES))
    parser.add_argument("--include-test", action="store_true",
                        help="Include Objectron test videos. Keep this off for training-cloud generation.")
    args = parser.parse_args()
    if args.per_class < 1 or args.max_keyframes < 8:
        parser.error("--per-class must be positive and --max-keyframes must be at least 8")
    if not args.subset_manifest.is_file():
        raise SystemExit(f"Subset manifest does not exist: {args.subset_manifest}. Run download_objectron_subset.py first.")

    manifest = json.loads(args.subset_manifest.read_text(encoding="utf-8"))
    split_allowed = {"train", "test"} if args.include_test else {"train"}
    runs_root = args.runs_root.resolve()
    runs_root.mkdir(parents=True, exist_ok=True)
    processed_hashes = existing_processed_hashes(runs_root)
    state_path = args.subset_manifest.parent / "processing_manifest.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {
        "schema_version": 1, "source_manifest": str(args.subset_manifest.resolve()), "runs": []
    }
    previous = {(r.get("sha256"), r.get("class")): r for r in state.get("runs", [])}
    selected_count: dict[str, int] = {}
    failures = []
    for sample in manifest.get("samples", []):
        label = sample.get("class")
        split = sample.get("split")
        if label not in args.classes or split not in split_allowed:
            continue
        annotation_validation = sample.get("annotation_validation", {})
        if annotation_validation.get("status") != "matched":
            # Supervised pipeline-cloud generation must not inherit a noisy
            # index label when its 3D annotation disagrees or was never audited.
            continue
        if selected_count.get(label, 0) >= args.per_class:
            continue
        video_info = sample.get("assets", {}).get("video", {})
        video = Path(video_info.get("path", ""))
        assets = sample.get("assets", {})
        if not video.is_file() or not all(name in assets for name in ("annotation", "geometry")):
            continue
        digest = video_info.get("sha256") or file_sha256(video)
        key = (digest, label)
        if key in processed_hashes:
            print(f"[SKIP] already processed {sample.get('source_key')} as {label}", flush=True)
            selected_count[label] = selected_count.get(label, 0) + 1
            continue
        selected_count[label] = selected_count.get(label, 0) + 1
        run_record = {
            "source_key": sample.get("source_key"), "class": label, "split": split,
            "sha256": digest, "video": str(video.resolve()), "started_at_utc": utc_now(),
        }
        print(f"[RUN] {sample.get('source_key')} -> {label}", flush=True)
        command = [
            sys.executable, str(PROJECT_ROOT / "scripts" / "run_isolated_reconstruction.py"),
            "--video", str(video.resolve()), "--category", label,
            "--output-root", str(runs_root), "--max-keyframes", str(args.max_keyframes),
        ]
        try:
            completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
            run_record["return_code"] = completed.returncode
            run_record["status"] = "completed" if completed.returncode == 0 else "completed_with_warnings"
            if completed.returncode != 0:
                failures.append({"source_key": sample.get("source_key"), "return_code": completed.returncode})
        except OSError as exc:
            run_record["status"] = "failed_to_start"
            run_record["error"] = f"{type(exc).__name__}: {exc}"
            failures.append({"source_key": sample.get("source_key"), "error": run_record["error"]})
        run_record["finished_at_utc"] = utc_now()
        previous[key] = run_record
        state["runs"] = list(previous.values())
        state["updated_at_utc"] = utc_now()
        state["failures"] = failures
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        if run_record["status"] == "completed":
            processed_hashes.add(key)

    state["runs"] = list(previous.values())
    state["updated_at_utc"] = utc_now()
    state["failures"] = failures
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(json.dumps({"videos_selected_per_class": selected_count, "failures": len(failures),
                      "processing_manifest": str(state_path)}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

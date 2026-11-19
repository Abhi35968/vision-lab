"""Resume annotation-box extraction for all completed Objectron subset runs."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-manifest", type=Path,
                        default=ROOT / "data" / "objectron_subset" / "manifest.json")
    parser.add_argument("--runs-root", type=Path, default=ROOT / "outputs" / "runs")
    parser.add_argument("--force", action="store_true", help="Re-run extraction for existing crop reports.")
    args = parser.parse_args()
    subset_path = args.subset_manifest.resolve()
    subset = json.loads(subset_path.read_text(encoding="utf-8"))
    state_path = subset_path.parent / "processing_manifest.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {"runs": []}

    source_by_hash = {}
    for sample in subset.get("samples", []):
        digest = sample.get("assets", {}).get("video", {}).get("sha256")
        if digest and sample.get("annotation_validation", {}).get("status") == "matched":
            source_by_hash[digest] = sample

    run_by_hash = {}
    for path in args.runs_root.resolve().glob("*/manifest.json"):
        try:
            m = json.loads(path.read_text(encoding="utf-8-sig"))
            digest = m.get("input", {}).get("sha256")
            if digest in source_by_hash and m.get("status") in {"completed", "completed_with_warnings"}:
                previous = run_by_hash.get(digest)
                if previous is None or path.stat().st_mtime > previous.stat().st_mtime:
                    run_by_hash[digest] = path
        except (OSError, json.JSONDecodeError):
            continue

    records, counts = [], Counter()
    for digest, sample in source_by_hash.items():
        run_manifest_path = run_by_hash.get(digest)
        if run_manifest_path is None:
            record = {"source_key": sample.get("source_key"), "class": sample.get("class"),
                      "sha256": digest, "split": sample.get("split"),
                      "status": "no_completed_pipeline_run"}
            records.append(record); counts[record["status"]] += 1
            continue
        run_dir = run_manifest_path.parent
        crop_report = run_dir / "reconstruction" / "objectron_box" / "report.json"
        if crop_report.is_file() and not args.force:
            try:
                report = json.loads(crop_report.read_text(encoding="utf-8"))
                record = {"source_key": sample.get("source_key"), "class": sample.get("class"),
                          "sha256": digest, "split": sample.get("split"),
                          "run_id": report.get("source_run_id", run_dir.name),
                          "status": report.get("status"), "crop_report": str(crop_report)}
                if report.get("point_filter"):
                    record["points_inside_box"] = report["point_filter"].get("points_inside_annotated_box")
                if report.get("pose_alignment"):
                    record["pose_pairs"] = report["pose_alignment"].get("paired_frames")
                    record["pose_rmse"] = report["pose_alignment"].get("rmse")
                records.append(record); counts[record["status"]] += 1
                continue
            except (OSError, json.JSONDecodeError):
                pass

        annotation = Path(sample.get("assets", {}).get("annotation", {}).get("path", ""))
        command = [sys.executable, str(ROOT / "scripts" / "extract_objectron_box_cloud.py"),
                   "--run-dir", str(run_dir), "--annotation", str(annotation)]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        log_path = run_dir / "reconstruction" / "objectron_box" / "extraction.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
        try:
            report = json.loads(crop_report.read_text(encoding="utf-8"))
            status = report.get("status", "missing_status")
            record = {"source_key": sample.get("source_key"), "class": sample.get("class"),
                      "sha256": digest, "split": sample.get("split"),
                      "run_id": run_dir.name, "status": status,
                      "extractor_exit_code": completed.returncode, "crop_report": str(crop_report)}
            if report.get("point_filter"):
                record["points_inside_box"] = report["point_filter"].get("points_inside_annotated_box")
            if report.get("pose_alignment"):
                record["pose_pairs"] = report["pose_alignment"].get("paired_frames")
                record["pose_rmse"] = report["pose_alignment"].get("rmse")
        except (OSError, json.JSONDecodeError):
            status = "extractor_failed"
            record = {"source_key": sample.get("source_key"), "class": sample.get("class"),
                      "sha256": digest, "split": sample.get("split"),
                      "run_id": run_dir.name, "status": status,
                      "extractor_exit_code": completed.returncode,
                      "error_log": str(log_path)}
        records.append(record); counts[status] += 1

    report = {"created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
              "source_manifest": str(subset_path), "processing_manifest": str(state_path),
              "source_split": subset.get("split", "train"),
              "matched_source_samples": len(source_by_hash), "run_records": records,
              "status_counts": dict(sorted(counts.items())),
              "train_ready": False,
              "note": "These are sparse SfM crops inside annotated boxes. Crop acceptance does not establish accurate dense or metric geometry."}
    output = subset_path.parent / "crop_processing_report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"matched_sources": len(source_by_hash), "status_counts": report["status_counts"],
                      "report": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

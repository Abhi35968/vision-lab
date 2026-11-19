"""Build a group-aware manifest from accepted Objectron box-cropped SfM clouds."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET_CLASSES = ["bicycle", "book", "bottle", "camera", "cereal_box",
                  "chair", "cup", "laptop", "shoe"]
MIN_SAMPLES_PER_CLASS = 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-report", type=Path,
                        default=ROOT / "data" / "objectron_subset" / "crop_processing_report.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data" / "objectron_box_dataset" / "manifest.json")
    args = parser.parse_args()
    report_path = args.crop_report.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    samples = []
    for item in report.get("run_records", []):
        if item.get("status") != "ok" or item.get("class") not in TARGET_CLASSES:
            continue
        crop_report_path = Path(item["crop_report"])
        crop = json.loads(crop_report_path.read_text(encoding="utf-8"))
        cloud = Path(crop["output_cloud"])
        if not cloud.is_file() or crop.get("status") != "ok":
            continue
        samples.append({
            "source_key": item["source_key"], "label": item["class"],
            "group_id": item["sha256"], "split": f"official_{item.get('split', 'train')}",
            "run_id": item["run_id"], "point_cloud": str(cloud.resolve()),
            "point_count": crop["point_filter"]["points_inside_annotated_box"],
            "annotation_category": crop["objectron_category"],
            "pose_pairs": crop["pose_alignment"]["paired_frames"],
            "pose_alignment_rmse_nominal_m": crop["pose_alignment"]["rmse"],
            "colmap_model": crop["colmap_model"],
        })
    counts = Counter(s["label"] for s in samples)
    total_target_classes_present = all(counts.get(label, 0) >= MIN_SAMPLES_PER_CLASS for label in TARGET_CLASSES)
    source_split = report.get("source_split", "train")
    test_samples_present = source_split == "test" and bool(samples)
    output = {
        "schema_version": 1,
        "dataset": "Objectron annotated-box crops of project COLMAP sparse clouds",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_crop_report": str(report_path),
        "official_source_split": source_split,
        "license": "Computational Use of Data Agreement 1.0 (C-UDA-1.0)",
        "license_url": "https://github.com/google-research-datasets/Objectron#license",
        "split_policy": f"This manifest contains only the official {source_split} index. Keep it separate from manifests built from other source splits; never split frames from the same source video across train and test.",
        "samples": samples,
        "samples_per_class": {label: counts.get(label, 0) for label in TARGET_CLASSES},
        "recording_sessions_per_class": {label: counts.get(label, 0) for label in TARGET_CLASSES},
        "training_ready": False,
        "readiness_checks": {
            "minimum_samples_per_class": MIN_SAMPLES_PER_CLASS,
            "all_classes_meet_minimum": total_target_classes_present,
            "official_test_samples_present_in_this_manifest": test_samples_present,
            "grouped_by_source_video_hash": True,
        },
        "limitations": [
            "Accepted samples are sparse SfM crops constrained by annotated boxes, not dense or watertight surfaces.",
            "Camera-center Sim(3) residual is not an independent metric geometry accuracy test.",
            "No independent official test split is present.",
        ],
    }
    out_path = args.output.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"accepted_samples": len(samples), "samples_per_class": output["samples_per_class"],
                      "training_ready": output["training_ready"], "output": str(out_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

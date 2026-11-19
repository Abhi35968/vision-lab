"""Create an honest manifest for project-specific PointNet training data.

It consumes only isolated runs with a measured ``object_points.ply`` and keeps
all samples from the same input hash in a single split.  It does not synthesize
labels or claim that a one-video-per-class collection generalizes.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

TARGET_CLASSES = ["bottle", "cup", "chair", "laptop", "camera", "shoe", "bicycle", "book", "cereal_box"]
MIN_TARGET_DETECTION_RATE = 0.70
MIN_EXACT_LABEL_RATE = 0.80
MIN_OBJECT_POINTS = 100
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def objectron_annotation_status() -> dict[str, dict]:
    """Index Objectron validation by source-video hash, including older pilots."""
    result = {}
    for manifest_path in (PROJECT_ROOT / "data" / "objectron_subset" / "manifest.json",
                          PROJECT_ROOT / "data" / "objectron_subset_pilot" / "manifest.json"):
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        for sample in manifest.get("samples", []):
            digest = sample.get("assets", {}).get("video", {}).get("sha256")
            if digest:
                validation = sample.get("annotation_validation", {})
                prior = result.get(digest)
                if prior is None or validation.get("status") != "matched":
                    result[digest] = {**validation, "source_key": sample.get("source_key")}
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="outputs/runs")
    parser.add_argument("--output", default="data/project_pointcloud_dataset/manifest.json")
    args = parser.parse_args()
    runs_root, output = Path(args.runs_root), Path(args.output)
    annotation_status = objectron_annotation_status()
    samples, groups, rejected = [], defaultdict(list), []
    for manifest_path in sorted(runs_root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        label = manifest.get("category_request", "auto")
        cloud = manifest_path.parent / "reconstruction" / "object_points.ply"
        if label not in TARGET_CLASSES or not cloud.exists():
            continue
        metrics_path = manifest_path.parent / "reconstruction" / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        reasons = []
        input_hash = manifest.get("input", {}).get("sha256", "")
        objectron_validation = annotation_status.get(input_hash)
        if objectron_validation and objectron_validation.get("status") != "matched":
            reasons.append("Objectron source annotation is not label-matched "
                           f"({objectron_validation.get('status', 'not audited')}: "
                           f"{objectron_validation.get('canonical_labels', objectron_validation.get('raw_labels', []))})")
        if metrics.get("status") != "ok":
            reasons.append(f"reconstruction status is {metrics.get('status', 'missing')!r}")
        if metrics.get("category_in_coco") is not True:
            reasons.append("target category has no supported exact YOLO mask mapping; use Objectron annotation geometry")
        detect_rate = metrics.get("detection_rate_pct")
        if not isinstance(detect_rate, (int, float)) or detect_rate < MIN_TARGET_DETECTION_RATE * 100:
            reasons.append(f"target detections cover less than {MIN_TARGET_DETECTION_RATE:.0%} of keyframes")
        exact_rate = metrics.get("exact_label_rate_pct")
        if not isinstance(exact_rate, (int, float)) or exact_rate < MIN_EXACT_LABEL_RATE * 100:
            reasons.append(f"exact target label agreement is below {MIN_EXACT_LABEL_RATE:.0%}")
        object_points = metrics.get("object_points", 0)
        if not isinstance(object_points, (int, float)) or object_points < MIN_OBJECT_POINTS:
            reasons.append(f"selected object cloud has fewer than {MIN_OBJECT_POINTS} points")
        if reasons:
            rejected.append({"run_id": manifest.get("run_id"), "label": label,
                             "metrics": {key: metrics.get(key) for key in
                                         ("detection_rate_pct", "exact_label_rate_pct", "object_points", "label_counts")},
                             "reasons": reasons})
            continue
        group = input_hash
        item = {"run_id": manifest["run_id"], "label": label, "group_id": group,
                "point_cloud": str(cloud.resolve()), "session": manifest["input"]["filename"],
                "object_points": int(object_points), "detection_rate_pct": detect_rate,
                "exact_label_rate_pct": exact_rate}
        groups[group].append(item)

    # Deterministic group-level split; no frame/cloud from one recording leaks.
    for group_id, items in sorted(groups.items()):
        bucket = int(group_id[:8], 16) % 10
        split = "test" if bucket == 0 else "validation" if bucket == 1 else "train"
        for item in items:
            item["split"] = split
            samples.append(item)
    counts = Counter(s["label"] for s in samples)
    sessions = Counter(s["label"] for s in {s["group_id"]: s for s in samples}.values())
    sufficient = all(sessions.get(label, 0) >= 3 for label in TARGET_CLASSES)
    report = {
        "schema_version": 1, "target_classes": TARGET_CLASSES, "samples": samples,
        "rejected_samples": rejected,
        "rejected_count": len(rejected),
        "samples_per_class": dict(counts), "recording_sessions_per_class": dict(sessions),
        "split_rule": "hash-grouped by original video; individual frames never cross splits",
        "acceptance_gates": {"minimum_target_detection_rate": MIN_TARGET_DETECTION_RATE,
                             "minimum_exact_label_rate": MIN_EXACT_LABEL_RATE,
                             "minimum_object_points": MIN_OBJECT_POINTS,
                             "reason": "Prevent wrong or unsupported masks from becoming training labels."},
        "training_ready": sufficient,
        "limitation": ("Insufficient independent recording sessions for one or more classes; do not train or claim "
                       "real-world generalization yet." if not sufficient else "Minimum session gate met; retain a held-out physical-object test set."),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("samples_per_class", "recording_sessions_per_class",
                                              "rejected_count", "training_ready", "limitation")}, indent=2))
    return 0 if sufficient else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Summarize isolated Objectron test predictions against audited labels."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-manifest", type=Path,
                        default=ROOT / "data" / "objectron_test_subset" / "manifest.json")
    parser.add_argument("--runs-root", type=Path, default=ROOT / "outputs" / "runs")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data" / "objectron_box_test_dataset" / "classification_summary.json")
    args = parser.parse_args()

    subset = read_json(args.subset_manifest.resolve())
    sources = {}
    for sample in subset.get("samples", []):
        digest = sample.get("assets", {}).get("video", {}).get("sha256")
        if digest and sample.get("annotation_validation", {}).get("status") == "matched":
            sources[digest] = sample

    latest: dict[str, tuple[float, Path, dict, dict]] = {}
    all_run_counts: Counter[str] = Counter()
    for manifest_path in args.runs_root.resolve().glob("*/manifest.json"):
        try:
            manifest = read_json(manifest_path)
            digest = manifest.get("input", {}).get("sha256")
            if digest not in sources or manifest.get("status") not in {"completed", "completed_with_warnings"}:
                continue
            all_run_counts[digest] += 1
            metrics_path = manifest_path.parent / "reconstruction" / "metrics.json"
            metrics = read_json(metrics_path) if metrics_path.is_file() else {}
            candidate = (manifest_path.stat().st_mtime, manifest_path, manifest, metrics)
            if digest not in latest or candidate[0] > latest[digest][0]:
                latest[digest] = candidate
        except (OSError, json.JSONDecodeError):
            continue

    records = []
    per_class: dict[str, list[dict]] = defaultdict(list)
    for digest, sample in sources.items():
        found = latest.get(digest)
        metrics = found[3] if found else {}
        manifest = found[2] if found else {}
        truth = sample.get("class")
        decision = metrics.get("classification_decision")
        pointnet_correct = metrics.get("pointnet_correct")
        record = {
            "source_key": sample.get("source_key"),
            "sha256": digest,
            "true_class": truth,
            "run_id": manifest.get("run_id"),
            "run_manifest": str(found[1].resolve()) if found else None,
            "classification_decision": decision,
            "classification_status": metrics.get("classification_status"),
            "pointnet_prediction": metrics.get("pointnet_prediction"),
            "pointnet_confidence_uncalibrated": metrics.get("pointnet_confidence"),
            "pointnet_correct": pointnet_correct,
            "category_in_modelnet": metrics.get("category_in_modelnet"),
            "source_run_count": all_run_counts.get(digest, 0),
            "status": "evaluated" if decision else "no_classification_decision",
        }
        records.append(record)
        per_class[truth].append(record)

    known = [r for r in records if r["classification_decision"] not in {None, "unknown"}]
    unknown = [r for r in records if r["classification_decision"] == "unknown"]
    no_decision = [r for r in records if r["classification_decision"] is None]
    correct = [r for r in known if r["classification_decision"] == r["true_class"]]
    pointnet_comparable = [r for r in records if r["pointnet_correct"] is not None]
    pointnet_correct = [r for r in pointnet_comparable if r["pointnet_correct"] is True]

    summary = {
        "schema_version": 1,
        "source_manifest": str(args.subset_manifest.resolve()),
        "created_from": "latest completed run manifest per audited source video hash",
        "source_video_count": len(sources),
        "source_video_hashes_with_completed_runs": len(latest),
        "run_manifests_for_test_hashes": sum(all_run_counts.values()),
        "duplicate_run_executions": sum(max(0, n - 1) for n in all_run_counts.values()),
        "final_decision_counts": {
            "known": len(known), "unknown": len(unknown), "no_decision": len(no_decision)
        },
        "known_decisions_correct": len(correct),
        "known_decisions_incorrect": len(known) - len(correct),
        "selective_accuracy": len(correct) / len(known) if known else None,
        "decision_coverage": len(known) / len(records) if records else None,
        "pointnet_comparable_count": len(pointnet_comparable),
        "pointnet_exact_count": len(pointnet_correct),
        "pointnet_exact_accuracy": len(pointnet_correct) / len(pointnet_comparable) if pointnet_comparable else None,
        "per_class": {},
        "records": records,
        "limitations": [
            "Predictions come from sparse reconstructed clouds and the existing ModelNet40 checkpoint.",
            "PointNet confidence scores are uncalibrated evidence, not probabilities.",
            "The official source split separates videos; physical-object identity across different videos was not independently verified.",
        ],
    }
    for label, rows in sorted(per_class.items()):
        known_rows = [r for r in rows if r["classification_decision"] not in {None, "unknown"}]
        correct_rows = [r for r in known_rows if r["classification_decision"] == label]
        comparable = [r for r in rows if r["pointnet_correct"] is not None]
        summary["per_class"][label] = {
            "videos": len(rows),
            "known_decisions": len(known_rows),
            "correct_known_decisions": len(correct_rows),
            "unknown_decisions": sum(r["classification_decision"] == "unknown" for r in rows),
            "no_decision": sum(r["classification_decision"] is None for r in rows),
            "pointnet_comparable": len(comparable),
            "pointnet_exact": sum(r["pointnet_correct"] is True for r in comparable),
        }

    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in (
        "source_video_count", "source_video_hashes_with_completed_runs",
        "run_manifests_for_test_hashes", "duplicate_run_executions",
        "final_decision_counts", "known_decisions_correct",
        "known_decisions_incorrect", "selective_accuracy", "decision_coverage",
        "pointnet_comparable_count", "pointnet_exact_count", "pointnet_exact_accuracy",
        "per_class")}, indent=2))
    print(f"output={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Check that each downloaded Objectron annotation agrees with its index class.

Writes `annotation_audit.json` and adds `annotation_validation` provenance to
the subset manifest. Only records with status `matched` should be used for
supervised object-cloud generation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
sys.path.insert(0, str(ROOT / "src"))
from objectron.schema.annotation_data_pb2 import Sequence  # noqa: E402

ALIASES = {
    "bike": "bicycle", "motobike": "bicycle",
    "cerealboxes": "cereal_box", "shoes": "shoe",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    path = args.manifest.resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    counts: Counter[str] = Counter()
    rows = []
    for sample in manifest.get("samples", []):
        category = sample.get("class", "")
        asset = sample.get("assets", {}).get("annotation", {}).get("path")
        if not asset or not Path(asset).is_file():
            record = {"status": "missing_annotation", "expected_class": category, "labels": []}
        else:
            try:
                sequence = Sequence()
                sequence.ParseFromString(Path(asset).read_bytes())
                raw = [obj.category for obj in sequence.objects]
                labels = [ALIASES.get(label, label) for label in raw]
                matched = bool(labels) and all(label == category for label in labels)
                record = {"status": "matched" if matched else "mismatch",
                          "expected_class": category, "raw_labels": raw,
                          "canonical_labels": labels,
                          "frame_annotations": len(sequence.frame_annotations)}
            except Exception as exc:
                record = {"status": "parse_error", "expected_class": category,
                          "error": f"{type(exc).__name__}: {exc}"}
        sample["annotation_validation"] = record
        counts[f"{category}:{record['status']}"] += 1
        rows.append({"source_key": sample.get("source_key"), **record})

    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    manifest["updated_at_utc"] = now
    manifest["samples"] = manifest.get("samples", [])
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = {"created_at_utc": now, "source_manifest": str(path),
              "canonical_label_aliases": ALIASES, "counts": dict(sorted(counts.items())),
              "matched_samples": sum(row["status"] == "matched" for row in rows),
              "total_samples": len(rows), "samples": rows}
    report_path = path.parent / "annotation_audit.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"matched_samples": report["matched_samples"],
                      "total_samples": report["total_samples"],
                      "counts": report["counts"], "report": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

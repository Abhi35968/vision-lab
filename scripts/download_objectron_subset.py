"""Download a small, resumable, class-balanced Objectron video subset.

The full Objectron archive is very large. This script uses the official
per-class train/test indexes and downloads only the requested samples and
their annotations/AR metadata. It writes provenance and hashes for every file.

Example:
    python scripts/download_objectron_subset.py --per-class 20 --split train
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_ROOT = "https://raw.githubusercontent.com/google-research-datasets/Objectron/master/index"
GCS_ROOT = "https://storage.googleapis.com/objectron"
CLASSES = {
    "bike": "bicycle",
    "book": "book",
    "bottle": "bottle",
    "camera": "camera",
    "cereal_box": "cereal_box",
    "chair": "chair",
    "cup": "cup",
    "laptop": "laptop",
    "shoe": "shoe",
}
ASSETS = ("video", "annotation", "geometry")
ANNOTATION_ALIASES = {"bike": "bicycle", "motobike": "bicycle",
                      "cerealboxes": "cereal_box", "shoes": "shoe"}
KEY_RE = re.compile(r"^(bike|book|bottle|camera|cereal_box|chair|cup|laptop|shoe)/batch-[0-9]+/[0-9]+$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def request_bytes(url: str, timeout: int = 30) -> bytes:
    request = Request(url, headers={"User-Agent": "3DCV-Project-Objectron-Subset/1.0"})
    with urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status} for {url}")
        return response.read()


def load_index(category: str, split: str, cache_root: Path) -> list[str]:
    index_path = cache_root / "indexes" / f"{category}_annotations_{split}"
    if not index_path.exists():
        data = request_bytes(f"{INDEX_ROOT}/{category}_annotations_{split}")
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_bytes(data)
    keys = [line.strip().strip("/") for line in index_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    keys = [key for key in keys if KEY_RE.fullmatch(key) and key.split("/")[0] == category]
    if not keys:
        raise RuntimeError(f"Official index is empty or malformed: {index_path}")
    return keys


def asset_url(key: str, asset: str) -> tuple[str, str]:
    if asset == "video":
        return f"{GCS_ROOT}/videos/{key}/video.MOV", "video.MOV"
    if asset == "geometry":
        return f"{GCS_ROOT}/videos/{key}/geometry.pbdata", "geometry.pbdata"
    return f"{GCS_ROOT}/annotations/{key}.pbdata", "annotation.pbdata"


def inspect_annotation(path: Path, expected_class: str) -> dict[str, Any]:
    # Import only for annotation-gated downloads; keep the basic downloader
    # usable without initializing protobuf for callers that do not request it.
    import os
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from objectron.schema.annotation_data_pb2 import Sequence
    sequence = Sequence()
    sequence.ParseFromString(path.read_bytes())
    raw = [obj.category for obj in sequence.objects]
    canonical = [ANNOTATION_ALIASES.get(label, label) for label in raw]
    matched = bool(canonical) and all(label == expected_class for label in canonical)
    return {"status": "matched" if matched else "mismatch", "expected_class": expected_class,
            "raw_labels": raw, "canonical_labels": canonical,
            "frame_annotations": len(sequence.frame_annotations)}


def download_asset(url: str, target: Path, max_bytes: int, retries: int = 3) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size > 0:
        return {"path": str(target.resolve()), "bytes": target.stat().st_size,
                "sha256": sha256(target), "downloaded": False}

    last_error: Exception | None = None
    for attempt in range(retries):
        part = target.with_suffix(target.suffix + ".part")
        try:
            request = Request(url, headers={"User-Agent": "3DCV-Project-Objectron-Subset/1.0"})
            digest = hashlib.sha256()
            total = 0
            with urlopen(request, timeout=60) as response, part.open("wb") as out:
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status} for {url}")
                declared_size = response.headers.get("Content-Length")
                if declared_size and int(declared_size) > max_bytes:
                    raise RuntimeError(f"File exceeds --max-file-mb limit: {declared_size} bytes")
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    total += len(block)
                    if total > max_bytes:
                        raise RuntimeError(f"File exceeded --max-file-mb limit ({max_bytes} bytes)")
                    out.write(block)
                    digest.update(block)
            if total == 0:
                raise RuntimeError(f"Downloaded empty file: {url}")
            part.replace(target)
            return {"path": str(target.resolve()), "bytes": total,
                    "sha256": digest.hexdigest(), "downloaded": True}
        except (HTTPError, URLError, TimeoutError, OSError, RuntimeError, ValueError) as exc:
            last_error = exc
            part.unlink(missing_ok=True)
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=5,
                        help="Maximum videos per category (default: 5; use 20+ to build a larger set).")
    parser.add_argument("--split", choices=("train", "test"), default="train",
                        help="Use the official Objectron split; do not mix test videos into training.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--classes", nargs="+", choices=tuple(CLASSES), default=list(CLASSES))
    parser.add_argument("--assets", nargs="+", choices=ASSETS, default=list(ASSETS))
    parser.add_argument("--max-file-mb", type=int, default=200,
                        help="Refuse any single asset larger than this limit.")
    parser.add_argument("--validate-annotations", action="store_true",
                        help="Count only clips whose annotation labels match the index class; continue through the official index to fill the requested valid count.")
    parser.add_argument("--max-candidates-per-class", type=int, default=100,
                        help="Bound annotation checks when searching for label-matched clips (default: 100).")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "objectron_subset")
    args = parser.parse_args()
    if args.per_class < 1 or args.max_file_mb < 1 or args.max_candidates_per_class < 1:
        parser.error("--per-class, --max-file-mb and --max-candidates-per-class must be positive")

    root = args.output.resolve()
    media_root = root / "media"
    manifest_path = root / "manifest.json"
    root.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "schema_version": 1,
            "dataset": "Google Objectron",
            "source_repository": "https://github.com/google-research-datasets/Objectron",
            "source_bucket": GCS_ROOT,
            "license": "Computational Use of Data Agreement 1.0 (C-UDA-1.0)",
            "license_url": "https://github.com/google-research-datasets/Objectron#license",
            "created_at_utc": utc_now(),
            "split": args.split,
            "samples": [],
        }
    if manifest.get("split") != args.split:
        raise SystemExit(f"Manifest split is {manifest.get('split')!r}; use a separate output directory for {args.split!r}.")

    by_key = {sample["source_key"]: sample for sample in manifest.get("samples", [])}
    max_bytes = args.max_file_mb * 1024 * 1024
    failures = []

    for category in args.classes:
        print(f"[INDEX] {category}", flush=True)
        try:
            keys = load_index(category, args.split, root)
        except Exception as exc:
            failures.append({"category": category, "error": str(exc)})
            print(f"[ERROR] index {category}: {exc}", file=sys.stderr, flush=True)
            continue
        # Class-local seeds keep a category's sample selection stable when
        # callers request a different set or ordering of categories.
        rng = random.Random(f"{args.seed}:{category}")
        rng.shuffle(keys)
        selected = (keys[:min(args.max_candidates_per_class, len(keys))] if args.validate_annotations
                    else keys[:min(args.per_class, len(keys))])
        valid_count = sum(1 for sample in by_key.values()
                          if sample.get("class") == CLASSES[category]
                          and sample.get("split") == args.split
                          and sample.get("annotation_validation", {}).get("status") == "matched"
                          and all(asset in sample.get("assets", {}) for asset in args.assets))
        for key in selected:
            if args.validate_annotations and valid_count >= args.per_class:
                break
            sample = by_key.setdefault(key, {
                "source_key": key,
                "class": CLASSES[category],
                "objectron_class": category,
                "split": args.split,
                "assets": {},
            })
            if args.validate_annotations and sample.get("annotation_validation", {}).get("status") == "matched" \
                    and all(asset in sample.get("assets", {}) for asset in args.assets):
                # Already counted in valid_count above.
                continue
            if args.validate_annotations:
                if "annotation" not in sample.get("assets", {}):
                    url, filename = asset_url(key, "annotation")
                    target = media_root / key / filename
                    try:
                        sample.setdefault("assets", {})["annotation"] = download_asset(url, target, max_bytes)
                    except Exception as exc:
                        failures.append({"source_key": key, "asset": "annotation", "error": str(exc)})
                        continue
                try:
                    validation = inspect_annotation(Path(sample["assets"]["annotation"]["path"]), CLASSES[category])
                except Exception as exc:
                    validation = {"status": "parse_error", "expected_class": CLASSES[category],
                                  "error": f"{type(exc).__name__}: {exc}"}
                sample["annotation_validation"] = validation
                if validation.get("status") != "matched":
                    print(f"[SKIP] {key} annotation {validation.get('status')}: {validation.get('raw_labels', [])}", flush=True)
                    manifest["samples"] = list(by_key.values())
                    manifest["updated_at_utc"] = utc_now()
                    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                    continue
                required_assets = list(dict.fromkeys([*args.assets, "annotation"]))
            else:
                required_assets = args.assets
            for asset in required_assets:
                if asset in sample.get("assets", {}):
                    continue
                url, filename = asset_url(key, asset)
                target = media_root / key / filename
                try:
                    sample["assets"][asset] = download_asset(url, target, max_bytes)
                    sample["updated_at_utc"] = utc_now()
                    print(f"[OK] {key} {asset} ({sample['assets'][asset]['bytes']} bytes)", flush=True)
                except Exception as exc:
                    failure = {"source_key": key, "asset": asset, "error": str(exc)}
                    failures.append(failure)
                    sample.setdefault("failures", []).append(failure)
                    print(f"[ERROR] {key} {asset}: {exc}", file=sys.stderr, flush=True)
                manifest["samples"] = list(by_key.values())
                manifest["updated_at_utc"] = utc_now()
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            if args.validate_annotations and all(asset in sample.get("assets", {}) for asset in required_assets):
                valid_count += 1

    manifest["samples"] = list(by_key.values())
    manifest["failures"] = failures
    manifest["updated_at_utc"] = utc_now()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    successful = sum(1 for sample in by_key.values() if all(a in sample.get("assets", {}) for a in args.assets))
    summary = {"requested_per_class": args.per_class, "split": args.split,
               "samples_in_manifest": len(by_key), "samples_with_all_requested_assets": successful,
               "annotation_validated": args.validate_annotations,
               "matched_samples_per_class": {category: sum(1 for sample in by_key.values()
                   if sample.get("class") == CLASSES[category]
                   and sample.get("annotation_validation", {}).get("status") == "matched"
                   and all(asset in sample.get("assets", {}) for asset in args.assets))
                   for category in args.classes},
               "failures": len(failures), "manifest": str(manifest_path)}
    print(json.dumps(summary, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())

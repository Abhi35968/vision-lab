# Objectron subset workflow

This folder is for a selected, locally downloaded subset of Google Objectron.
The project does not bundle or redistribute Objectron media. Downloaded video
and protobuf assets are ignored by Git; the generated manifest records source
keys, dataset split, file hashes, and the applicable license.

Objectron is released under the Computational Use of Data Agreement 1.0
(C-UDA-1.0). Read the terms before using or sharing any source media:
<https://github.com/google-research-datasets/Objectron#license>

The videos are labeled object-centric clips with camera/AR metadata, sparse
tracking points, and 3D box annotations. They are not precomputed clean object
clouds. To make samples relevant to this project, download a bounded subset,
then process each video into a fresh isolated project run. Keep the official
test split out of training and validation decisions.

The initial pilot found that the current YOLO masks can select the wrong
object for known categories (for example, a cereal box clip can yield a book
cloud). Do not train on every reconstructed cloud automatically. Run
`build_project_pointcloud_dataset.py` and inspect its accepted/rejected
records. `scripts/extract_objectron_box_cloud.py` now aligns the run's COLMAP
cameras to Objectron's annotated camera centers, checks the label, applies
track and reprojection filters, and crops the cloud to the annotated oriented
box. The pilot accepted the bicycle label/crop and rejected one cup-indexed
sample whose annotation says chair. This extractor outputs sparse SfM samples
and is not a dense surface reconstruction.

## Small pilot

```powershell
.\venv\Scripts\python.exe scripts\download_objectron_subset.py --per-class 5 --split train --validate-annotations
.\venv\Scripts\python.exe scripts\audit_objectron_annotations.py --manifest data\objectron_subset\manifest.json
.\venv\Scripts\python.exe scripts\process_objectron_subset.py --per-class 1
.\venv\Scripts\python.exe scripts\build_project_pointcloud_dataset.py
```

Increase `--per-class` for a larger download and processing batches after
reviewing the pilot cloud quality. The downloader is resumable and caps
annotation search at 100 candidates per class by default. Processing skips
unvalidated or label-mismatched clips and is limited to one clip per class per
invocation.

For a held-out evaluation, download from `--split test` into a separate output
directory (for example, `--output data/objectron_test_subset`), process with
`--include-test` and the matching `--subset-manifest`, and keep those clouds out of training.
The current training manifest is not sufficient to claim real-world
generalization until there are enough independent object instances and
recording sessions per category.

## Current downloaded snapshot

On 2026-11-18, the local train subset contains 50 complete clips whose
annotations match the indexed category (five for each of the nine classes,
plus five additional chair clips).
The manifest also contains 35 rejected annotation candidates used while
searching for replacements; only matched entries are eligible for processing.
The media tree is approximately 1.46 GB. The official test split was not
downloaded. See `annotation_audit.json` for per-clip labels and rejection
reasons.

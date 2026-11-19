# Input Isolation Audit

## Verdict: PASS for the new isolated-run workflow

New executions use `scripts/run_isolated_reconstruction.py`.  `RunContext` creates `outputs/runs/<run-id>/`, archives the submitted video under `input/`, records a SHA-256 hash and OpenCV metadata, and refuses to write outside the run root.  No stage reads a prior run's pose, cloud, mask, depth map, or mesh.

The Streamlit upload workflow in `demo_app.py` now uses the same context.  The older `app.py` upload control is blocked rather than silently applying its historical bottle cloud.

## Execution evidence

| Run ID | Source | SHA-256 prefix | Result directory | Status |
|---|---|---|---|---|
| `20261117T070142Z_03_cup_e2a4b0d4f8_df28a528` | `03_cup.MOV` | `e2a4b0d4f8` | `outputs/runs/...df28a528/` | completed with classification warning |
| `20261117T070339Z_04_chair_ee3b271ce9_adc54015` | `04_chair.MOV` | `ee3b271ce9` | `outputs/runs/...adc54015/` | completed with classification warning |
| `20261117T070738Z_03_cup_e2a4b0d4f8_70d9dc83` | repeated `03_cup.MOV` | `e2a4b0d4f8` | `outputs/runs/...70d9dc83/` | completed with classification warning |

The cup runs have the same source hash but distinct run IDs and directories. The chair has a distinct source hash. Each manifest lists only artifacts below its own directory. See `outputs/validation_comparison/comparison_manifest.json` and `outputs/validation_comparison/held_out_overlay_comparison.jpg`.

## Requirement status

| Requirement | Status | Evidence |
|---|---|---|
| Exactly one explicit input video per execution | PASS | Required `--video`; manifest has one archived input. |
| Unique run directory per execution | PASS | UUID suffix in all three run IDs. |
| No stale artifacts reused | PASS for new runner | Runner starts from archived input and uses a fresh run root. Historical `outputs/` remains comparison-only. |
| Origin/run associated with outputs | PASS | `manifest.json`, run-local `metrics.json`, and registered artifact paths. |
| Streamlit upload does not reuse stale cloud | PASS | `demo_app.py` uses `RunContext`; legacy `app.py` upload is explicitly blocked. |
| Two inputs and a repeat verified | PASS | Three executions above. |

## Remaining limitation

This does not migrate old shared `outputs/*` artifacts. They are intentionally preserved, but must not be cited as results from a new isolated run.

## Public held-out data execution (2026-11-18)

The Objectron official test index was downloaded separately to
`data/objectron_test_subset/`. The expanded set has 90 annotation-matched
clips (ten per class), each with a completed isolated run recorded in the
processing manifest. There are 99 test run manifests for 90 unique video
hashes because nine clips were processed a second time by overlapping
launches; each repeat still has a unique run ID and input hash. Crop results
are indexed separately at
`data/objectron_box_test_dataset/manifest.json`.
No test crop was added to the training manifest. The separate train and test
manifests record the official source split and video hash.

**Status: PASS for run/input separation.** Repeated runs created duplicate
work but no shared output paths. This validates isolation and split separation,
not reconstruction quality or classification accuracy.


## Expanded training and held-out split audit (2026-11-19)

The train acquisition contains 180 annotation-matched source videos (20 per
class). Each selected video was processed into an isolated run directory;
177 completed and three completed with warnings. The crop report associates
each video hash and class with its source run. The held-out test crop report
contains 90 completed official-test runs and 74 accepted clouds. The test
index has also been expanded to 20 annotation-matched sources per class, but
the additional 90 test videos were not reconstructed; they are not used for
the reported evaluation. Train and test manifests remain separate.

Evidence: data/objectron_subset/manifest.json,
data/objectron_subset/processing_manifest.json,
data/objectron_subset/crop_processing_report.json,
data/objectron_test_subset/manifest.json,
data/objectron_test_subset/processing_manifest.json, and
data/objectron_test_subset/crop_processing_report.json. Input hashes and
run-local manifests are retained under outputs/runs/.

**Status: PASS for the processed input/run separation.** The 143 train and
74 test accepted clouds have run-local provenance. This does not establish
physical-object independence across separate videos.

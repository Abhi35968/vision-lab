# Camera Geometry Audit

## Verdict: PARTIALLY VALIDATED

The new runner uses COLMAP first, validates camera-from-world convention (`Xc = R Xw + t`, `C = -Rᵀt`), writes a trajectory/reprojection diagnostic, and filters sparse tracks to at least three observations and at most 3 px point reprojection error before object association.

## Execution evidence

| Run | Registered | Median point reprojection | Median track length | Pose gate |
|---|---:|---:|---:|---|
| Cup, first | 40/40 | 0.500 px | 6 | PASS |
| Chair | 40/40 | 0.853 px | 5 | PASS |
| Cup, repeat | 40/40 | recorded in its run-local diagnostic | run-local | PASS |

The corresponding `camera_pose_validation.json` and `camera_pose_diagnostics.png` are located below each run's `reconstruction/diagnostics/` directory. Registration count is only one of five gates; a run also needs reprojection, track-length, rotation-orthonormality, and non-degenerate-trajectory checks.

## Calibration comparison

**FAIL — no valid calibrated-camera capture is available.** `config/camera_intrinsics.json` is explicitly `uncalibrated` with zero focal lengths, so it was not passed to COLMAP. The runner supports an OpenCV checkerboard calibration JSON and passes its scaled `fx, fy, cx, cy, k1, k2, p1, p2, k3` to COLMAP's OPENCV camera model, but an estimated-versus-calibrated accuracy comparison cannot honestly be produced until checkerboard images are captured.

## Objectron held-out camera evidence (2026-11-18)

All 90 official test clips (ten per class) completed isolated pipeline
invocations; the extractor accepted 74 annotated-box crops, rejected 15 for
insufficient object points, and rejected one for pose alignment. Accepted
crops passed the existing camera-pair alignment and track filters; this is a
dataset-generation gate, not a calibrated-camera accuracy study. Usable test
counts are bicycle 10, book 10, bottle 10, camera 8, cereal box 10, chair 7,
cup 5, laptop 6 and shoe 8. No calibrated phone capture was available, so the
calibrated-versus-auto comparison remains **FAIL / not measured**.

See `data/objectron_test_subset/crop_processing_report.json` and each run's
`reconstruction/diagnostics/camera_pose_validation.json` for per-run evidence.

## Required next capture

Capture at least 10–20 sharp checkerboard images with the same phone lens, resolution, focus mode, and zoom setting used for object videos. Run `CameraCalibrator`, then invoke the runner with `--calibration <result.json>`. Compare the two run-local diagnostics; do not compare against old shared outputs.


## Expanded Objectron pipeline camera evidence (2026-11-19)

The 20-video-per-class train crop report has 180 isolated run records. The
camera pose validator passed on 169/180 and failed on 11/180; median COLMAP
registration was 20/20 cameras. The median internal sampled reprojection
error was 0.385 px. The held-out ten-video-per-class test processing has 90
run records, with 87/90 pose-validator passes, median 20/20 registration,
and 0.395 px median sampled reprojection error. These are COLMAP
auto-estimated intrinsics and internal SfM consistency checks; they do not
establish calibrated phone accuracy or metric scale.

Execution evidence: data/objectron_subset/crop_processing_report.json,
data/objectron_test_subset/crop_processing_report.json, and the per-run
reconstruction/diagnostics/camera_pose_validation.json files. Calibrated
phone comparison remains **FAIL / not measured** because no calibration
capture is available. Dense/MVS geometry and scale are not validated by
these sparse-SfM camera diagnostics.

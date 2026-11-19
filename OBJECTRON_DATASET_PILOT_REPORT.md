# Objectron dataset pilot report

Date: 2026-11-18

## Purpose

Check whether public Objectron footage can quickly provide labeled, real-video
training examples for the project's nine target classes, and whether the
current reconstruction/segmentation pipeline extracts correctly labeled
object clouds from it.

## Download

- Source: Google Objectron official per-category training indexes and GCS
  assets; source keys and SHA-256 hashes are recorded in
  `data/objectron_subset_pilot/manifest.json`.
- Requested one clip per category, with video, 3D annotation and AR geometry
  metadata. Nine entries have all three requested assets. One additional cup
  video-only entry came from the initial connectivity check.
- The dataset is governed by C-UDA 1.0; see the license link recorded in the
  manifest and the local `data/objectron_subset/README.md`.
- No test-split samples were downloaded or processed.

## Reconstruction pilot

The first attempt used 40 keyframes and remained in COLMAP exhaustive feature
matching. It was stopped and retained as a failed run. The pilot was then
rerun with 20 keyframes. Five categories completed before processing was
stopped: the current YOLO mask selection was visibly assigning off-target
classes for several known Objectron labels. All five completed COLMAP runs
registered 20/20 cameras and passed the project's pose checks. This verifies
camera tracking for these clips, not object-cloud label quality.

| Objectron label | YOLO keyframe detection | Exact label rate | Selected object points | PointNet output | Dataset decision |
|---|---:|---:|---:|---|---|
| bicycle | 100% | 100% | 392 | person, 0.331 | Accepted by current mask gate; one sample only |
| book | 15% | 0% | 156 | lamp, 0.684 | Rejected; masks labeled oven |
| bottle | 95% | 68.4% | 180 | lamp, 0.538 | Rejected; inconsistent masks and too few points |
| camera | 50% | No supported mapping | 862 | person, 0.904 | Rejected; masks labeled truck/suitcase |
| cereal box | 100% | No supported mapping | 7,204 | person, 0.267 | Rejected; masks labeled book/refrigerator |

The selected clouds for book, camera and cereal box must not be treated as
ground-truth object clouds. Their large point counts do not repair incorrect
mask selection. The pipeline stopped before chair, cup, laptop and shoe were
processed.

## Current dataset status

`scripts/build_project_pointcloud_dataset.py` now records and rejects runs
with missing category support, weak target coverage, poor exact-label
agreement, or fewer than 100 selected object points. The generated
`data/project_pointcloud_dataset/manifest.json` currently accepts two chair
sessions, one bicycle session and one laptop session. It rejects the old
cup-indexed run because its Objectron annotation says chair. It records ten
rejected runs and correctly reports `training_ready: false`. No new PointNet
model was trained.

## Next step

The expanded processing batch and crop gates are now recorded below. Continue
by collecting more independent chair examples and a held-out official test
subset before considering model training or evaluation.

## Annotation-based crop pilot (2026-11-18)

Added `scripts/extract_objectron_box_cloud.py` to align each completed COLMAP
run to Objectron's per-frame camera centers, reject high-residual pose fits,
filter short/high-error SfM tracks, and retain points inside the ground-truth
oriented 3D box. It writes a PLY and a run-local `report.json`; it does not
reuse any other run's cloud. A requested-class/annotation-class check blocks
label-mismatched crops. The extractor still produces sparse SfM points, not a
dense reconstruction or watertight object surface.

| Sample | Annotation label check | Matched poses | Camera-center alignment RMSE | Filtered points in 3D box | Outcome |
|---|---|---:|---:|---:|---|
| bike/batch-4/47 | bike → bicycle | 20/20 | 0.0102 m | 780 | Crop accepted for pilot inspection only |
| book/batch-2/11 | book → book | 20/20 | 0.0020 m | 659 | Crop accepted for pilot inspection only |
| bottle/batch-28/31 | bottle → bottle | 20/20 | 0.0050 m | 187 | Crop accepted for pilot inspection only |
| camera/batch-6/33 | camera → camera | 20/20 | 0.0016 m | 1,418 | Crop accepted for pilot inspection only |
| cereal_box/batch-23/39 | cerealboxes → cereal_box | 20/20 | 0.0034 m | 7,006 | Crop accepted for pilot inspection only |
| cup/batch-20/34 | chair ≠ cup | 20/20 | 0.0035 m | Not accepted | Rejected; mismatch written to run-local report |

For the bicycle run, the pipeline's own 392-point mask cloud had 87.8% of its
points inside the annotated box. The side-by-side 3D plot is
`outputs/runs/20261118T074500Z_video_dd9dc59e77_83954ec9/reconstruction/objectron_box/mask_vs_annotated_box.png`.
The two sparse clouds broadly occupy the same object, but neither plot nor box
membership proves surface accuracy; the box itself contains empty volume.

The cup example shows that strong pose alignment does not validate the semantic
label. The downloaded pilot annotations were parsed for all nine indexed
samples: eight agree after canonicalizing `bike` to `bicycle` and
`cerealboxes` to `cereal_box`; `cup/batch-20/34` is indexed as cup but its
annotation says chair. This sample must not enter cup training/evaluation.

The bicycle cloud contains 780 points inside the annotated box after enforcing
track length ≥3 and COLMAP reprojection error ≤3 px, out of 3,326 points passing
those filters. Camera alignment maps COLMAP units to Objectron's annotated
world scale; treat this as nominal metric scale from the dataset, not an
independent accuracy validation. Low residual checks camera-center consistency
only; it does not prove the object surface is accurate. A COLMAP output-selection
bug was found during the bottle crop: COLMAP wrote multiple sparse models.
`sparse/0` contained only two registered images, while `sparse/1` contained the
20-image best model that the pipeline's metrics described. The extractor now
selects the saved model with the most registered images (breaking ties by
point count) and records all candidates. Bottle then aligned 20/20 frames with
0.0050 m RMSE and yielded 187 in-box points. This shows why the first saved
sparse model cannot be assumed to match a run's metrics.

## Training decision

No model was trained. Current new-data evidence is seven accepted annotation-box
pilot crops (bicycle, book, bottle, camera, cereal box, chair, cup) and multiple
confirmed index/annotation mismatches. The laptop crop is below the point-count gate and the shoe
reconstruction has too few registered views. These remain pilot
artifacts rather than a balanced training dataset. The project-specific dataset manifest remains
`training_ready: false`; this is not enough independent objects or recording
sessions to make a credible real-world performance claim. Continue by
downloading a balanced, moderate Objectron train subset, validating each sample
against its annotation, generating run-local box crops, and holding Objectron's
official test split out of training. Evaluate against physical-object/session
separated data before fine-tuning PointNet. The legal terms remain C-UDA 1.0;
check and comply with the agreement before redistributing clips or derived
assets.

## Expanded annotation-verified training subset

The downloader now supports `--validate-annotations` and a bounded candidate
scan. It downloads annotation protobufs first, normalizes the known aliases
(`bike`/`motobike`, `cerealboxes`, `shoes`), and downloads video/AR geometry
only for label-matched entries. The first five-index sample per class contained
multiple mismatches, especially entries annotated as chair. After replacing
those using the official train index, the manifest has five annotation-matched
complete clips in each of the nine target classes, then added five more valid
chair clips (50 total). It also records 35 rejected index/annotation
candidates; no test-split assets were requested. The media tree is 1.46 GB.
`scripts/process_objectron_subset.py` now skips
samples whose annotation validation is absent or not `matched`.

Additional 20-keyframe reconstructions and annotation-box crops:

| Class | COLMAP registered views | Annotation-box result | PointNet-only top result | Run-local decision |
|---|---:|---|---|---|
| chair | 20/20 | 1,619 in-box points | bookshelf, 0.141 | unknown |
| cup | 20/20 | 250 in-box points | person, 0.451 | YOLO-only cup |
| laptop | 20/20 | 35 points; below 100-point pilot gate | lamp, 0.970 | YOLO-only laptop |
| shoe | 4/20 | Rejected; insufficient registered views | lamp, 0.470 | unknown |

This confirms the dataset can supply class labels and useful 3D box supervision,
but point-cloud yield varies substantially by object and camera tracking. The
laptop's confident lamp ranking and the shoe tracking loss show that downloading
more videos alone does not solve PointNet domain shift or reconstruction
quality. The five-per-class subset is a bounded data-generation pilot, not an
adequate basis for claiming generalization or training a deployable classifier.

## Initial five-per-class train subset outcome (historical)

All 45 annotation-matched clips were submitted to the isolated pipeline with
20 keyframes each. The run manifest records 44 `completed` and one
`completed_with_warnings` result (cup/batch-41/31). The crop pass then reported
36 `ok`, 6 `insufficient_object_points`, 2 `rejected_insufficient_pose_pairs`,
and 1 `rejected_pose_alignment` among the 45 matched sources. Per-class usable
counts were:

| Class | Accepted crops |
|---|---:|
| Bicycle | 5 |
| Book | 5 |
| Bottle | 4 |
| Camera | 5 |
| Cereal box | 4 |
| Chair | 2 |
| Cup | 3 |
| Laptop | 4 |
| Shoe | 4 |

The initial 36 accepted crops were collected in
`data/objectron_box_dataset/manifest.json`. That manifest remains
`training_ready: false`: chair has only two usable clouds, the data contains no
official test split, and the output remains sparse SfM rather than dense object
surfaces. The full batch log is `outputs/objectron_processing_batch.log`; per
sample run, crop, and label provenance is preserved under `outputs/runs/` and
`data/objectron_subset/`.

### Chair class extension

Five more verified chair clips were downloaded and processed, bringing the
pipeline batch to 50 completed/warning runs. The refreshed crop report accepts
40 total clouds: bicycle 5, book 5, bottle 4, camera 5, cereal box 4, chair 6,
cup 3, laptop 4 and shoe 4. The ten rejected crops comprise seven low-point
clouds, two insufficient-pose cases and one pose-alignment failure. The updated
dataset manifest now meets the three-sample minimum in every class, but remains
`training_ready: false` because the official test split is absent and these
small counts do not support reliable generalization.

## Official test split follow-up (2026-11-18)

The preceding train-set section records the state before test data was
downloaded. A separate Objectron official test subset now contains 45
annotation-matched videos (five per target class). The processing manifest
records 45 completed source videos; 51 run directories exist because six
videos were processed twice during an overlapping launch. All runs remain
isolated by run ID and input hash.

The final crop pass accepted 36 held-out clouds and rejected eight for too few
in-box points and one for pose alignment. Accepted per-class counts are
bicycle 5, book 5, bottle 5, camera 3, cereal box 5, chair 4, cup 2, laptop 2
and shoe 5. This official split is not included in the train manifest.

Files: `data/objectron_test_subset/manifest.json`,
`data/objectron_test_subset/processing_manifest.json`,
`data/objectron_test_subset/crop_processing_report.json`, and
`data/objectron_box_test_dataset/manifest.json`.

## Ten-per-class expansion (2026-11-18)

Both official indexes were extended to ten annotation-matched videos per
class. The train crop pass accepted 73/90 videos (bicycle 10, book 10, bottle
8, camera 10, cereal box 9, chair 6, cup 6, laptop 6, shoe 8). The separate
test pass accepted 74/90 (bicycle 10, book 10, bottle 10, camera 8, cereal
box 10, chair 7, cup 5, laptop 6, shoe 8). Test crop rejections were 15
insufficient-point cases and one pose-alignment failure. The larger set is
still marked `training_ready: false`; physical-object identity across videos
is not independently established, and the number of accepted clouds per class
is too small for a reliable generalization claim.

Existing project PointNet evaluation on the expanded test set: 0/39 exact
PointNet predictions for classes represented in ModelNet40. The final
classifier emitted 25 known labels (13 correct), 57 `unknown`, and eight
without a decision. This yields 52.0% selective accuracy at 27.8% coverage,
not a general accuracy estimate. See
`data/objectron_box_test_dataset/classification_summary.json`.

## Project-specific PointNet training pilot

Using the expanded 73-cloud train set, I fine-tuned the PointNet feature
extractor with video-hash-separated train/validation groups (58/15) and kept
the official test set out of checkpoint selection. Best validation checkpoint
was epoch 1; validation accuracy was 1/15 (6.7%). On held-out crops, the pilot
scored 9/74 (12.2%) top-1, close to nine-class chance. The checkpoint is saved
as `outputs/objectron_pointnet_pilot/pointnet_objectron_pilot.pth` with full
evidence in `outputs/objectron_pointnet_pilot/training_report.json`. It was not
made the app default because it did not improve performance; confidence is
uncalibrated and unknown rejection is not implemented in this experimental
checkpoint.

## Expanded train acquisition and retraining (2026-11-19)

The official Objectron train index was extended to 20 annotation-matched
videos per class. All 180 selected videos received isolated pipeline runs;
177 completed and three returned warnings. The annotation-box crop gate
accepted 143 clouds and rejected 37 (28 insufficient in-box object points,
five with insufficient pose pairs, and four pose-alignment failures).
Accepted counts: bicycle 20, book 17, bottle 16, camera 19, cereal box 18,
chair 14, cup 13, laptop 10 and shoe 16. The separate test data remains
untouched at its original 74 accepted clouds (10 annotation-matched source
videos per category, with fewer accepted camera/chair/cup/laptop/shoe crops).

Retraining on the accepted train clouds used 114 train videos and 29
video-hash-disjoint validation videos. The best validation-loss checkpoint
was epoch 15: 8/29 validation correct (27.6%) and 14/74 official test correct
(18.9%). This exceeds the previous pilot's 9/74 (12.2%) by five examples on
the same test set, but remains only modestly above the 10/74 majority-class
baseline. It does not support a real-world accuracy claim. Per-class official
test accuracy: bicycle 3/10, book 1/10, bottle 0/10, camera 1/8, cereal box
0/10, chair 5/7, cup 2/5, laptop 2/6, shoe 0/8.

New execution evidence: `outputs/objectron_expand20_train_download.log`,
`outputs/objectron_expand20_train_processing.log`,
`data/objectron_subset/annotation_audit.json`,
`data/objectron_subset/crop_processing_report.json`,
`data/objectron_box_dataset/manifest.json`,
`outputs/objectron_pointnet_training20.log`, and
`outputs/objectron_pointnet_pilot20/training_report.json`. Training data
remain sparse SfM geometry and are video-disjoint only; physical-object
identity across separate Objectron recordings is unknown. This model remains
experimental and is not the app default.

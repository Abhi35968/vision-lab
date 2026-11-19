# Classification Validation Report

## Verdict: PASS for uncertainty handling; FAIL for real-world PointNet accuracy

The final classifier now exposes independent `yolo_only`, `pointnet_only`, and `fused_diagnostic` evidence. The fused score is explicitly uncalibrated and cannot increase reported confidence. When modalities disagree, the final label is `unknown`.

## Execution evidence

| Run | YOLO-only | PointNet-only | Final result |
|---|---|---|---|
| Cup, first | cup, evidence 0.59 | stairs, evidence 0.50 | `unknown`, `UNCERTAIN_REJECTED` |
| Chair | chair, evidence 0.56 | plant, evidence 0.92 | `unknown`, `UNCERTAIN_REJECTED` |
| Cup, repeat | cup, evidence 0.59 | lamp, evidence 0.46 | `unknown`, `UNCERTAIN_REJECTED` |

These are deliberate rejections, not successful real-world classifications. The run-local `classification.json` files preserve the evidence.

### Objectron follow-up runs

| Class | PointNet-only | Final result | Reconstruction note |
|---|---|---|---|
| Chair | bookshelf, 0.141 | `unknown` | 20/20 cameras; annotated-box crop has 1,619 points |
| Cup | person, 0.451 | YOLO-only `cup` | 20/20 cameras; annotated-box crop has 250 points |
| Laptop | lamp, 0.970 | YOLO-only `laptop` | 20/20 cameras; only 35 points inside the annotation box |
| Shoe | lamp, 0.470 | `unknown` | only 4/20 cameras; insufficient pose alignment |

These were train-split pilot examples. The independent Objectron test split is
now processed separately; see the test results below. The high lamp score on
the laptop clip demonstrates why this ModelNet40 softmax cannot be read as a
calibrated real-world probability.

## Model suitability

The existing model remains a ModelNet40 CAD classifier. It cannot represent camera, shoe, bicycle, book, or cereal box as project labels, and its 84.6% ModelNet40 result does not generalize to sparse real reconstructions. It is therefore unsuitable as a final project classifier today.

`scripts/build_project_pointcloud_dataset.py` creates a group-safe manifest from
YOLO-selected clouds and rejects Objectron videos whose annotation disagrees
with the requested class. Its current mask-based counts are book 3, chair 5,
cup 2, bottle 2, bicycle 2 and laptop 4; the other target classes have no
accepted mask-based clouds. `data/project_pointcloud_dataset/manifest.json`
marks `training_ready: false`; no generalization claim is made.

The expanded annotation-box dataset contains 73 usable train crops: bicycle 10,
book 10, bottle 8, camera 10, cereal box 9, chair 6, cup 6, laptop 6 and shoe 8.
These counts remain too small for a credible generalization claim, and
augmentation cannot replace independent objects and sessions.

`augment_partial_real_scan` implements full SO(3) rotation, dropout, partial-view occlusion, outlier injection, scale, and translation augmentation for a future real-scan trainer. A project-specific train/validation/test split must remain grouped by original video hash and physical recording session.

## Official held-out test results (2026-11-18)

The expanded test set contains ten annotation-verified videos per class. The
pipeline produced 25 known final labels, 57 `unknown` decisions, and eight
runs without a classification decision because object evidence was absent or
too sparse. Of the 25 known decisions, 13 matched the annotated class and 12
were wrong. Selective accuracy is 13/25 (52.0%) at 25/90 (27.8%) coverage;
this small benchmark is not a generalization claim.

PointNet's top prediction was exact for 0/39 cases where the ground-truth
category exists in ModelNet40 (bottle, chair, cup and laptop).
The other five project categories are outside ModelNet40's output labels, so
the checkpoint cannot classify them correctly by construction. For most
videos, the system rejected the model evidence as unknown rather than forcing
these incorrect predictions. Scores remain uncalibrated evidence.

The annotation-box crop set has 74 usable held-out clouds: bicycle 10, book
10, bottle 10, camera 8, cereal box 10, chair 7, cup 5, laptop 6 and shoe 8.
Cup remains the weakest group with five held-out clouds. Source files:
`data/objectron_test_subset/crop_processing_report.json`,
`data/objectron_box_test_dataset/manifest.json`, and run-local `reconstruction/metrics.json`.
The summary is reproducible with `scripts/evaluate_objectron_test_classification.py`
and is saved at `data/objectron_box_test_dataset/classification_summary.json`.
The official split is disjoint by source-video index/hash; physical object
identity across different recordings was not independently verified, so that
stronger grouping condition remains **PARTIALLY VALIDATED**.

## Project-specific PointNet pilot (2026-11-18)

I fine-tuned the existing PointNet feature extractor on the real partial/noisy
Objectron SfM crops. The training-only split used 58 videos for fitting and 15
video-hash-disjoint videos for validation. The official test set remained
untouched during checkpoint selection. Training stopped after seven epochs;
best validation loss was at epoch 1, with 1/15 validation predictions correct
(6.7%). The resulting experimental checkpoint scored 9/74 (12.2%) top-1 on
the official test crops, close to the 11.1% nine-class chance level. It is not
an improvement worth deploying.

Per-class test results: bicycle 0/10, book 0/10, bottle 0/10, camera 1/8,
cereal box 1/10, chair 1/7, cup 0/5, laptop 2/6 and shoe 4/8. These are small
supports and physical-object identity across recordings is not verified. The
checkpoint is intentionally separate from the app's default ModelNet40 model;
its softmax remains uncalibrated and it has no unknown-class rejection.

Execution evidence: `scripts/train_objectron_pointnet_pilot.py`,
`outputs/objectron_pointnet_pilot/training_report.json`, and
`outputs/objectron_pointnet_pilot/pointnet_objectron_pilot.pth`. I loaded this
checkpoint through `PointNetInferenceEngine` with the matching nine-class file
and ran it on an accepted test cloud; loading succeeded. The weak held-out
accuracy means training alone did not solve the domain gap.

## Expanded 20-video-per-class training pilot (2026-11-19)

The official train index was expanded to 20 annotation-matched videos per
class. The reconstruction/crop pipeline accepted 143 of 180 source videos;
the accepted counts are bicycle 20, book 17, bottle 16, camera 19, cereal
box 18, chair 14, cup 13, laptop 10 and shoe 16. Of those 143 clouds, 114
were used for fitting and 29 for video-hash-disjoint validation. The official
test split was kept separate at 74 accepted clouds.

The experimental checkpoint selected epoch 15 and reached 8/29 validation
accuracy (27.6%) and 14/74 official test accuracy (18.9%). The preceding
10-video-per-class pilot scored 1/15 validation (6.7%) and 9/74 test
(12.2%) on the same test clouds. This is a measured improvement of five test
examples, but absolute test accuracy remains low: 14/74 is only modestly
above both nine-class uniform chance (11.1%) and the test majority-class
baseline (10/74, 13.5%). Per-class test accuracy remains uneven: bicycle
3/10, book 1/10, bottle 0/10, camera 1/8, cereal box 0/10, chair 5/7, cup
2/5, laptop 2/6 and shoe 0/8.

The checkpoint is preserved separately at
`outputs/objectron_pointnet_pilot20/pointnet_objectron_pilot.pth`; full
metrics are in `outputs/objectron_pointnet_pilot20/training_report.json` and
the execution log is `outputs/objectron_pointnet_training20.log`. It remains
experimental: inputs are sparse SfM crops, confidence is uncalibrated, the
checkpoint has no unknown-class rejection, and Objectron does not establish
physical-object identity across videos. It was not made the app default.
The checkpoint and nine-class label file load through
PointNetInferenceEngine; on a held-out bicycle crop (1,756 points) the model
returned cup at 0.168 uncalibrated confidence. This is an execution/loading
check, not an additional accuracy measurement.

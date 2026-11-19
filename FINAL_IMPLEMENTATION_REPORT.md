# Final Implementation Report

## Overall status: PARTIALLY VALIDATED — not ready to claim reliable metric reconstruction or general real-world classification

## Implemented

1. Run-scoped provenance: `src/runtime/run_manager.py`, `scripts/run_isolated_reconstruction.py`, and the demo upload flow.
2. Conservative COLMAP pose validation and reliable-track gate: `src/sfm/pose_validation.py`.
3. Rejection of synthetic geometry/classification fallbacks and unvalidated own-SfM dense fallback.
4. Explicit non-metric warning for relative monocular depth.
5. Classification uncertainty/rejection, independent modality display, and no confidence inflation from fusion.
6. Group-safe project-dataset manifest and real-scan augmentation utility.
7. Side-by-side run-local comparison figures.

## Validation status

| Requirement | Status |
|---|---|
| Input isolation, two videos, repeated input | PASS |
| Run manifest/provenance | PASS |
| COLMAP pose convention and trajectory diagnostics | PASS for three auto-calibrated runs |
| Calibrated vs estimated camera comparison | FAIL — no calibration capture available |
| Segmentation quality | PARTIALLY VALIDATED — mask rate and held-out projection recorded, no manual ground truth |
| Dense reconstruction / mesh topology | PARTIALLY VALIDATED — unsafe fallbacks removed; no new calibrated MVS/RGB-D experiment |
| Unknown classification | PASS |
| Real-world PointNet accuracy | FAIL — observed disagreement is rejected, not solved |
| Project-specific training | FAIL — insufficient independent labelled sessions |

## Test evidence

- `pytest tests/test_run_manager.py tests/test_pointnet.py -q`: **13 passed**.
- `pytest tests/test_pointnet.py tests/test_dense_depth.py tests/test_real_sfm.py -q`: **23 passed** before the final run-manager tests were added.
- Three real isolated COLMAP executions are recorded in `outputs/runs/`; comparison figures are in `outputs/validation_comparison/`.

## Next decision gate

Do not train the current PointNet again on ModelNet40. First collect at least three independent recording sessions per class (preferably substantially more), include labelled partial/noisy output clouds, calibrate the phone, and maintain a held-out physical-object test set. Then compare calibrated COLMAP MVS, visual hull, and RGB-D against a known-size reference before making accuracy claims.

## Objectron dataset follow-up (2026-11-18)

- Added a resumable official-index downloader and isolated per-video pipeline runner.
- Added `scripts/extract_objectron_box_cloud.py` to align COLMAP camera centers to Objectron camera poses and crop sparse tracks to the annotated oriented 3D box.
- Execution evidence: bicycle, book, bottle, camera and cereal-box pilot samples passed label checks and yielded 780, 659, 187, 1,418 and 7,006 filtered points inside their boxes, respectively. Each alignment used 20 camera pairs; RMSE ranged from 0.0016 to 0.0102 m. These are sparse SfM crops, not verified surfaces.
- Execution evidence: a cup-indexed sample had a chair annotation; its crop was rejected and a run-local mismatch report written.
- Found and fixed COLMAP model selection in the extractor: the bottle run sparse/0 had two cameras, while sparse/1 had the 20-camera model used by run metrics. The extractor now selects the model with the most registered images and records all model candidates.
- A bicycle side-by-side figure compares the project's mask cloud with the annotated-box crop under that run's `reconstruction/objectron_box/` directory.
- Pilot annotation check: eight of the initial nine clips agreed after class-name canonicalization; one cup index pointed to a chair annotation.
- PointNet was inspected: its 40-output checkpoint is ModelNet40, not a model trained on this project's partial SfM clouds. Its softmax scores are not real-world calibrated probabilities.
- Expanded the train-only subset to five annotation-matched complete clips in each class, plus five extra chair clips (50 usable clips; 1.46 GB media tree). The manifest also records 35 rejected annotation candidates. At that stage the official test split had not yet been downloaded; see the follow-up below.
- Processed 50 annotation-matched train clips: 49 runs completed and one completed with warnings. The crop pass accepted 40 sparse clouds (bicycle 5, book 5, bottle 4, camera 5, cereal box 4, chair 6, cup 3, laptop 4, shoe 4); ten failed point-count or pose-quality gates.
- Added `scripts/build_objectron_box_dataset.py`; its manifest records the accepted train crops and remains `training_ready: false` because class counts are too small for robust generalization. No project-specific model had yet been trained at that stage.

## Official held-out data follow-up (2026-11-18)

The official train and test indexes have each been expanded to ten
annotation-matched videos per class. Train crop extraction accepted 73 of 90
videos; test accepted 74 of 90. The test run tree contains 99 manifests for
90 unique source hashes because nine clips were processed a second time during
overlapping launches; every run still has its own output directory and input
hash. The test manifest is `data/objectron_box_test_dataset/manifest.json`,
separate from the train manifest. Per-class counts are recorded in both.
The official split separates source-video hashes, but physical-object identity
across separate recordings is not independently verified.

### Held-out classification evidence

- Final decisions: 57 `unknown`, 25 known labels, and eight videos with no
  classification result because object evidence was absent or too sparse.
- Of 25 known labels, 13 matched Objectron's annotation and 12 were wrong.
  Selective accuracy is 52.0% at 27.8% coverage over 90 test videos; this is
  not a broad accuracy claim.
- The existing PointNet was exact on 0 of 39 cases for its four overlapping
  ModelNet40 target classes (bottle, chair, cup, laptop). The five other
  project categories are not output classes in this checkpoint.
- The uncertain/unknown rejection behavior works as intended, but the model's
  real-world classification performance fails. Its softmax scores remain
  uncalibrated.

### Experimental project-specific training

I added and ran `scripts/train_objectron_pointnet_pilot.py` on the 73 accepted
training clouds. It used 58 video groups for fitting and 15 for validation,
with the official test split untouched for checkpoint selection. Early
stopping selected epoch 1; validation accuracy was 1/15 (6.7%). The pilot
checkpoint reached 9/74 (12.2%) top-1 test accuracy, near nine-class chance
(11.1%). Training alone did not fix the domain gap. The checkpoint and report
are under `outputs/objectron_pointnet_pilot/`; the default app checkpoint was
left unchanged because the pilot is not an improvement and lacks unknown
rejection/calibration.

### Requirement status after the test run

| Area | Status | Evidence / gap |
|---|---|---|
| Input isolation and train/test separation | PASS | 90 test videos; 99 test run manifests including nine repeats; separate test manifest and per-run source hashes. |
| Camera geometry | PARTIALLY VALIDATED | COLMAP ran on all clips; one crop failed pose alignment. Calibrated phone comparison is still unavailable. |
| Segmentation and object crop | PARTIALLY VALIDATED | 74 annotation-box sparse crops accepted; masks are not manually ground-truthed. |
| Dense reconstruction and mesh topology | NOT VALIDATED | No calibrated PatchMatch, RGB-D, metric reference, or new mesh-topology evaluation. |
| Unknown classification | PASS | 57 low-evidence cases rejected rather than forced. |
| Real-world PointNet accuracy | FAIL | 0/39 exact PointNet predictions for overlapping classes; 13/25 selective final labels were correct. |
| Project-specific training | FAIL as an accuracy improvement; pilot executed | 58 fit / 15 validation groups; 9/74 held-out accuracy, near chance. 73 train / 74 test crops are small/uneven and physical-object overlap across recordings is unverified. |
| Side-by-side comparisons | PARTIALLY VALIDATED | Per-run figures and `outputs/validation_comparison/objectron_test_class_montage.png`; no dense/metric ground-truth comparison. |

This is a useful benchmark and acquisition pilot, not a completed real-world
reconstruction system. The experimental PointNet run did not improve results.
The next useful work is to collect more independent partial clouds or find a
pretrained model closer to real scanned objects, while separately obtaining
phone-camera checkerboard calibration. Physical-object identity must be
verified across train and test before claiming generalization.


## Latest expanded-data training run (2026-11-19)

The official Objectron train index was expanded to 20 annotation-matched
videos per class. The isolated pipeline processed 180 unique train videos
(177 completed, three with warnings); annotation-box extraction accepted
143 sparse clouds. The separate held-out test set remains the original 74
accepted clouds from 90 processed videos. Train/validation was grouped by
source-video hash (114/29 clouds); no physical-object identity grouping is
available.

The new PointNet pilot selected epoch 15 and scored 8/29 validation (27.6%)
and 14/74 official test (18.9%). The earlier ten-per-class pilot scored
9/74 (12.2%) on those same test clouds, so the expanded training gained five
correct cases. Its test performance remains only modestly above the majority
class baseline (10/74, 13.5%) and has substantial class variation. Exact
per-class results and the confusion matrix are in
outputs/objectron_pointnet_pilot20/training_report.json. The checkpoint is
kept separate at outputs/objectron_pointnet_pilot20/; it is not the app
default because its evidence is too weak for reliable real-world use and it
has no unknown rejection or confidence calibration.

### Updated requirement status

| Requirement | Status | Execution evidence / remaining limit |
|---|---|---|
| Input isolation and video/run provenance | PASS | 180 train runs and 90 held-out test runs have unique run directories and source hashes; train/test datasets and manifests are separate. |
| Camera trajectory and pose checks | PARTIALLY VALIDATED | Pose validator passed 169/180 train and 87/90 test runs; auto-estimated intrinsics only, no calibrated phone comparison. |
| Segmentation/object isolation | PARTIALLY VALIDATED | 143 train and 74 test annotation-box crops accepted; no manual mask ground truth. |
| Dense geometry and metric scale | FAIL / NOT VALIDATED | Outputs used for training are sparse SfM crops; no calibrated MVS, RGB-D, or independent scale reference. |
| Mesh topology | FAIL / NOT VALIDATED | No new mesh-topology validation was run. |
| Classification uncertainty | PARTIALLY VALIDATED | The app's existing reject path is preserved, but this experimental checkpoint has no unknown detection or calibration and is not integrated. |
| Project-specific PointNet training | PARTIALLY VALIDATED | 143 accepted source clouds; video-hash-disjoint validation and untouched official test; 14/74 top-1. Physical-object identity across recordings remains unknown. |
| Real-world classification quality | FAIL for deployment | Test accuracy is 18.9%, uneven by class, and only modestly above the majority baseline. |
| Side-by-side comparisons | PASS for classifier pilot; PARTIALLY VALIDATED for geometry | outputs/validation_comparison/pointnet_pilot10_vs_20_test.png compares both checkpoints on the same held-out clouds; existing run-local mask-versus-box figures remain. No dense/metric ground-truth comparison exists. |

The next useful step is still collecting more independently identified
physical objects and sessions, adding phone calibration, and evaluating
against metric RGB-D or known-scale references. This run shows that training
on the pipeline's own sparse point clouds helps somewhat, but it does not
make the current reconstruction/classification pipeline reliable yet.

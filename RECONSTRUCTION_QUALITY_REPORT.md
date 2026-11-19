# Reconstruction Quality Report

## Verdict: PARTIALLY VALIDATED

The isolated sparse-object path is validated for provenance and held-out mask consistency. It is **not** validated as metric geometry.

## New-run evidence

| Run | Object points after track gate | YOLO mask rate | Held-out mask consistency | Interpretation |
|---|---:|---:|---:|---|
| Cup, first | 71 | 100% | 92.2% | Sparse object association only; inadequate for reliable 3D classification. |
| Chair | 342 | 100% | 83.7% | Object association is plausible, but worst held-out view is 3.3%, so geometry is not uniformly consistent. |
| Cup, repeat | 70 | 100% | run-local | Separate repeat, not merged with the first cup run. |

Side-by-side result figures are in `outputs/validation_comparison/run_metrics_comparison.png` and `outputs/validation_comparison/held_out_overlay_comparison.jpg`.

## Corrective changes

- Background/tabletop entry is reduced by mask association plus the new track-quality gate.
- Dense fusion now refuses synthetic poses, synthetic sparse points, and unvalidated own-SfM fallback.
- The Depth Anything scale heuristic is documented as arbitrary SfM units; it no longer calls an assumed depth span metres.
- Poisson output is described by topology diagnostics, not as automatically watertight.

## Dense geometry / smooth objects

**PARTIALLY VALIDATED.** Smooth bottles and cups remain hard because they contain too few persistent features. Accurate segmentation helps association but does not add physical depth observations. A visual hull is appropriate as a conservative silhouette envelope; it cannot recover hidden concavities, internal geometry, or textureless backside detail. It must be labelled as a silhouette-based approximation.

**NOT YET EXECUTED:** calibrated COLMAP PatchMatch Stereo and RGB-D reconstruction. PatchMatch needs calibrated cameras, sufficient texture, and substantially more CPU/GPU/RAM; a smooth/glossy bottle can still fail. RGB-D gives the most direct metric depth but requires a depth sensor and sensor calibration. Metric scale requires an ArUco/checkerboard/known-size reference or calibrated RGB-D; monocular relative depth alone is not a metric source.

Temporary removable visual markers or a textured label are recommended for controlled evaluation. They should be removed from the final texture/model and reported as an acquisition aid.

## Objectron official test subset (2026-11-18)

The separate official test set now contains 90 annotation-matched videos (ten
per class), processed to isolated run directories. Annotation-box extraction
accepted 74 sparse SfM crops (82.2%): bicycle 10, book 10, bottle 10, camera
8, cereal box 10, chair 7, cup 5, laptop 6 and shoe 8. Fifteen clips failed
the minimum in-box point count and one failed camera-pose alignment. At least
one cup clip had no viable initial COLMAP pair; process exit status alone
therefore overstates usable reconstruction success.

These are sparse points inside Objectron's annotated oriented boxes, not dense
surface reconstructions or independently verified metric geometry. They do
not establish mesh topology, calibrated MVS performance, or real-world object
surface accuracy. See `data/objectron_test_subset/crop_processing_report.json`
and `data/objectron_box_test_dataset/manifest.json`. The nine-class
side-by-side comparison montage is
`outputs/validation_comparison/objectron_test_class_montage.png`.

## Expanded Objectron train subset (2026-11-18)

Five annotation-matched train clips per target class, plus five additional
chair clips, were processed in isolated 20-keyframe runs. Of 50 matched
sources, 49 pipeline runs completed and one finished with warnings. The
Objectron camera-pose alignment and oriented-box crop pass accepted 40 sparse
clouds; seven failed the 100-point gate, two had too few matched poses, and one
failed pose alignment. Usable counts were bicycle 5, book 5, bottle 4, camera
5, cereal box 4, chair 6, cup 3, laptop 4 and shoe 4. Per-run side-by-side
mask-versus-box figures are saved under each run's
`reconstruction/objectron_box/` directory where a pipeline mask cloud exists.

This validates label-conditioned sparse crop generation on several categories,
not dense surface quality or metric reconstruction accuracy. No calibrated
PatchMatch, RGB-D, or independent scale-reference comparison was performed.
The separate held-out test subset is now present, but its results are listed
above; the train crop manifest still does not support robust model training
because per-class counts are small and the clouds are sparse.


## Expanded Objectron crop quality (2026-11-19)

The annotation-verified train index now has 20 selected videos per class.
All 180 received isolated SfM runs: 177 completed and three completed with
warnings. Crop extraction accepted 143/180 (79.4%): bicycle 20, book 17,
bottle 16, camera 19, cereal box 18, chair 14, cup 13, laptop 10 and shoe
16. It rejected 28 for insufficient in-box object points, five for too few
pose pairs, and four for pose alignment. The median object crop contains
675 sparse SfM points. On the separate existing 90-video test processing,
74/90 crops were accepted, with 15 low-point and one alignment rejection;
median object-point metric across runs is 514.5 points.

Detection coverage was high (median 95.0% on train runs and 92.5% on test
runs), but this is detector coverage rather than a manual segmentation
accuracy score. Internal camera diagnostics passed in 169/180 train and
87/90 test runs. None of these checks makes sparse SfM a dense or metric
surface: no calibrated PatchMatch MVS, RGB-D fusion, independent scale
reference, or mesh-topology validation was performed. Hidden concavities
and internal geometry are therefore not reconstructed or verified.

Execution evidence:
data/objectron_subset/crop_processing_report.json,
data/objectron_test_subset/crop_processing_report.json,
data/objectron_box_dataset/manifest.json, and
outputs/validation_comparison/pointnet_pilot10_vs_20_test.png. Previous
per-run mask-versus-annotated-box point-cloud comparisons are preserved
under each run's reconstruction/objectron_box/ directory.

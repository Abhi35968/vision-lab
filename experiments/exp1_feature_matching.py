"""
Experiment 1 - Feature matching comparison: SIFT vs ORB.

Part A (pair level): for every consecutive keyframe pair of the real video, extract up to
2000 features with each detector, match with Lowe's ratio test, and verify the matches with
an essential-matrix RANSAC. Records counts, inlier ratio and timings.

Part B (pipeline level): runs the project's full incremental SfM with SIFT and with ORB over
several RANSAC seeds and compares registered cameras, 3D points, reprojection error and runtime.

    python -m experiments.exp1_feature_matching
"""

import shutil
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.common import (
    K, keyframe_paths, out_dir, run_sfm, save_json, style_axes, C_SIFT, C_ORB,
)
from src.sfm.feature_extractor import FeatureExtractor
from src.sfm.feature_matcher import FeatureMatcher

DETECTORS = ["SIFT", "ORB"]
COLORS = {"SIFT": C_SIFT, "ORB": C_ORB}
N_FEATURES = 2000
PIPELINE_SEEDS = [0, 1, 2, 3, 4]


def pair_level(paths, out):
    rows = []
    vis_pair = len(paths) // 2  # pair used for the qualitative match figure
    for det in DETECTORS:
        extractor = FeatureExtractor(detector=det, n_features=N_FEATURES)
        matcher = FeatureMatcher(matcher_type="FLANN" if det == "SIFT" else "BF", lowe_ratio=0.75)
        feats = []
        for p in paths:
            feats.append(extractor.extract(p))
        for i in range(len(paths) - 1):
            f1, f2 = feats[i], feats[i + 1]
            m = matcher.match(f1, f2)
            inliers, t_ransac = 0, 0.0
            mask = None
            if m.final_matches_count >= 8:
                t0 = time.perf_counter()
                _, mask = cv2.findEssentialMat(m.matched_pts1, m.matched_pts2, K, method=cv2.RANSAC,
                                               prob=0.999, threshold=1.0)
                t_ransac = (time.perf_counter() - t0) * 1000.0
                inliers = int(mask.sum()) if mask is not None else 0
            rows.append({
                "detector": det, "pair": f"{i}-{i + 1}", "pair_index": i,
                "keypoints_1": f1.num_keypoints, "keypoints_2": f2.num_keypoints,
                "ratio_test_matches": m.final_matches_count,
                "ransac_inliers": inliers,
                "inlier_ratio": inliers / m.final_matches_count if m.final_matches_count else 0.0,
                "extract_ms_per_image": (f1.extraction_time_ms + f2.extraction_time_ms) / 2.0,
                "match_ms": m.matching_time_ms,
                "ransac_ms": t_ransac,
            })
            if i == vis_pair and mask is not None:
                good = [mm for mm, keep in zip(m.final_matches, mask.ravel()) if keep]
                img1, img2 = cv2.imread(str(paths[i])), cv2.imread(str(paths[i + 1]))
                vis = FeatureMatcher.visualize_matches(img1, f1, img2, f2, good, max_draw=150,
                                                       title=f"{det} RANSAC inliers")
                cv2.imwrite(str(out / f"matches_{det.lower()}_pair_{i}_{i + 1}.jpg"), vis,
                            [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"[INFO] {det}: pair-level statistics done for {len(paths) - 1} pairs")
    df = pd.DataFrame(rows)
    df.to_csv(out / "pair_level_metrics.csv", index=False)
    return df


def pipeline_level(paths, out):
    """Full SfM with each detector, repeated over several RANSAC seeds (the outcome is seed-dependent)."""
    rows = []
    work = out / "_runs"
    for det in DETECTORS:
        for seed in PIPELINE_SEEDS:
            row = run_sfm(paths, work / f"{det.lower()}_s{seed}", detector=det, seed=seed)
            rows.append(row)
            print(f"[INFO] {det} full SfM seed={seed}: registered {row['registered_cameras']}, "
                  f"{row['points_3d']} pts, median err {row['reproj_median_px']} px, {row['runtime_s']} s", flush=True)
    shutil.rmtree(work, ignore_errors=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(out / "pipeline_level_runs.csv", index=False)
    agg = runs.groupby("detector").agg(
        runs=("seed", "count"),
        registered_cameras=("registered_cameras", "median"),
        registered_min=("registered_cameras", "min"), registered_max=("registered_cameras", "max"),
        points_3d=("points_3d", "median"), points_min=("points_3d", "min"), points_max=("points_3d", "max"),
        mean_track_length=("mean_track_length", "median"),
        reproj_median_px=("reproj_median_px", "median"), reproj_mean_px=("reproj_mean_px", "median"),
        runtime_s=("runtime_s", "median"),
    ).reindex(DETECTORS).reset_index()
    agg.to_csv(out / "pipeline_level_metrics.csv", index=False)
    pipe = agg.to_dict(orient="records")
    save_json(pipe, out / "pipeline_level_metrics.json")
    return pipe


def plots(df, pipe, out):
    g = df.groupby("detector")
    summary = g.agg(
        keypoints=("keypoints_1", "mean"), ratio_test_matches=("ratio_test_matches", "mean"),
        ransac_inliers=("ransac_inliers", "mean"), inlier_ratio=("inlier_ratio", "mean"),
        extract_ms=("extract_ms_per_image", "mean"), match_ms=("match_ms", "mean"),
        ransac_ms=("ransac_ms", "mean"),
    ).reindex(DETECTORS)
    summary.to_csv(out / "pair_level_summary.csv")

    # Figure 1: matching quality
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    x = np.arange(3); w = 0.36
    for k, det in enumerate(DETECTORS):
        vals = summary.loc[det, ["keypoints", "ratio_test_matches", "ransac_inliers"]].values
        bars = axs[0].bar(x + (k - 0.5) * w, vals, w, color=COLORS[det], label=det)
        for b in bars:
            axs[0].text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.0f}",
                        ha="center", va="bottom", fontsize=9)
    axs[0].set_xticks(x); axs[0].set_xticklabels(["Keypoints / image", "Ratio-test matches", "RANSAC inliers"])
    axs[0].set_ylabel("Mean count per consecutive pair"); axs[0].set_title("Feature and match counts")
    axs[0].legend(frameon=False); style_axes(axs[0])

    for det in DETECTORS:
        d = df[df.detector == det]
        axs[1].plot(d.pair_index, d.ransac_inliers, color=COLORS[det], lw=2, label=det)
    axs[1].set_xlabel("Consecutive keyframe pair index"); axs[1].set_ylabel("RANSAC inlier matches")
    axs[1].set_title("Verified matches along the video"); axs[1].legend(frameon=False); style_axes(axs[1])

    data = [df[df.detector == det].inlier_ratio.values * 100 for det in DETECTORS]
    bp = axs[2].boxplot(data, tick_labels=DETECTORS, patch_artist=True, widths=0.5,
                        medianprops=dict(color="black"))
    for patch, det in zip(bp["boxes"], DETECTORS):
        patch.set_facecolor(COLORS[det]); patch.set_alpha(0.85)
    axs[2].set_ylabel("Inlier ratio (%)"); axs[2].set_title("Share of matches that survive RANSAC")
    style_axes(axs[2])
    plt.tight_layout(); plt.savefig(out / "matching_quality.png", dpi=150); plt.close()

    # Figure 2: processing time
    fig, ax = plt.subplots(figsize=(7, 4.2))
    stages = ["extract_ms", "match_ms", "ransac_ms"]
    labels = ["Extraction (per image)", "Matching (per pair)", "RANSAC (per pair)"]
    x = np.arange(len(stages))
    for k, det in enumerate(DETECTORS):
        vals = summary.loc[det, stages].values
        bars = ax.bar(x + (k - 0.5) * w, vals, w, color=COLORS[det], label=det)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.0f}",
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("Mean time (ms)")
    ax.set_title("Processing time, 1080x1920 keyframes (CPU)"); ax.legend(frameon=False); style_axes(ax)
    plt.tight_layout(); plt.savefig(out / "processing_time.png", dpi=150); plt.close()

    # Figure 3: full pipeline (median over seeds; whiskers show min-max where available)
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.9))
    items = [("registered_cameras", "Registered cameras", ("registered_min", "registered_max")),
             ("points_3d", "Sparse 3D points", ("points_min", "points_max")),
             ("reproj_median_px", "Median reprojection error (px)", None), ("runtime_s", "SfM runtime (s)", None)]
    for ax, (key, title, rng) in zip(axs, items):
        vals = np.array([r[key] for r in pipe], dtype=float)
        yerr = None
        if rng:
            lo = np.array([r[rng[0]] for r in pipe], dtype=float); hi = np.array([r[rng[1]] for r in pipe], dtype=float)
            yerr = [vals - lo, hi - vals]
        bars = ax.bar(DETECTORS, vals, color=[COLORS[d] for d in DETECTORS], width=0.55, yerr=yerr, capsize=5,
                      ecolor="#444444")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, 0.02 * max(vals.max(), 1e-9), f"{v:g}", ha="center",
                    va="bottom", fontsize=10, color="white", fontweight="bold")
        ax.set_title(title, fontsize=11); style_axes(ax)
    plt.suptitle(f"Full incremental SfM with each detector (47 keyframes, median of {len(PIPELINE_SEEDS)} seeds, "
                 "whiskers = min-max)", fontsize=12)
    plt.tight_layout(); plt.savefig(out / "pipeline_comparison.png", dpi=150); plt.close()
    return summary


def main():
    out = out_dir("exp1_feature_matching")
    paths = keyframe_paths()
    print(f"[INFO] Experiment 1 on {len(paths)} keyframes")
    df = pair_level(paths, out)
    pipe = pipeline_level(paths, out)
    summary = plots(df, pipe, out)
    save_json({"pair_level_mean": summary.reset_index().to_dict(orient="records"),
               "pipeline_level": pipe, "n_features": N_FEATURES, "num_pairs": len(paths) - 1},
              out / "summary.json")
    print(summary.round(2).to_string())
    print(f"[INFO] Results written to {out}")


if __name__ == "__main__":
    main()

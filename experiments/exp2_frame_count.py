"""
Experiment 2 - Reconstruction quality versus number of video frames, and run-to-run stability.

Part A: incremental SfM on N keyframes sampled uniformly from the 47 available, so every run
covers the same camera sweep at a different frame density. Each N is repeated with several
RANSAC seeds because the pipeline's outcome depends on the random draws.

Part B: the full 47-keyframe reconstruction repeated with more seeds, to show how much the
number of registered cameras varies with nothing changed but the random seed.

    python -m experiments.exp2_frame_count
"""

import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.common import keyframe_paths, out_dir, run_sfm, save_json, style_axes, C_A, C_B, C_GREY

FRAME_COUNTS = [8, 12, 16, 24, 32, 40, 47]
SEEDS_PER_COUNT = [0, 1, 2]
STABILITY_SEEDS = list(range(10))
STORED_RUN = {"registered_cameras": 32, "points_3d": 2077}  # outputs/real_video_sfm/focal_1300


def main():
    out = out_dir("exp2_frame_count")
    work = out / "_runs"
    paths = keyframe_paths()

    rows = []
    for n in FRAME_COUNTS:
        idx = np.unique(np.round(np.linspace(0, len(paths) - 1, n)).astype(int))
        subset = [paths[i] for i in idx]
        seeds = STABILITY_SEEDS if n == len(paths) else SEEDS_PER_COUNT
        for seed in seeds:
            row = run_sfm(subset, work / f"n{n:02d}_s{seed}", seed=seed)
            rows.append(row)
            print(f"[INFO] N={n:2d} seed={seed}: registered {row['registered_cameras']:2d} "
                  f"(frames {row['first_registered']}..{row['last_registered']} of subset), "
                  f"{row['points_3d']} pts, median err {row['reproj_median_px']} px, {row['runtime_s']} s",
                  flush=True)
    shutil.rmtree(work, ignore_errors=True)  # per-run dumps are large; the table keeps what matters

    df = pd.DataFrame(rows)
    df.to_csv(out / "all_runs.csv", index=False)
    agg = df.groupby("frames_used").agg(
        runs=("seed", "count"),
        registered_median=("registered_cameras", "median"), registered_min=("registered_cameras", "min"),
        registered_max=("registered_cameras", "max"),
        rate_median_pct=("registration_rate_pct", "median"),
        points_median=("points_3d", "median"), points_min=("points_3d", "min"), points_max=("points_3d", "max"),
        track_len_median=("mean_track_length", "median"),
        reproj_median_px=("reproj_median_px", "median"), reproj_mean_px=("reproj_mean_px", "median"),
        runtime_median_s=("runtime_s", "median"),
    ).reset_index()
    agg.to_csv(out / "frame_count_summary.csv", index=False)

    full = df[df.frames_used == len(paths)]
    stability = {
        "frames": len(paths), "seeds": STABILITY_SEEDS,
        "registered_cameras_per_seed": full.registered_cameras.tolist(),
        "registered_range_per_seed": [f"{a}..{b}" for a, b in zip(full.first_registered, full.last_registered)],
        "points_per_seed": full.points_3d.tolist(),
        "registered_min": int(full.registered_cameras.min()), "registered_max": int(full.registered_cameras.max()),
        "registered_median": float(full.registered_cameras.median()),
        "stored_main_reconstruction": STORED_RUN,
    }
    save_json({"per_frame_count": agg.to_dict(orient="records"), "stability_47_frames": stability},
              out / "summary.json")

    # --- Figure A: effect of frame count (median over seeds, range as error bar)
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    x = agg.frames_used.values

    def panel(ax, med, lo, hi, title, ylabel=None):
        ax.errorbar(x, med, yerr=[med - lo, hi - med], fmt="-o", color=C_A, lw=2, ms=7, capsize=4, ecolor=C_GREY)
        for xi, yi in zip(x, med):
            ax.annotate(f"{yi:g}", (xi, yi), textcoords="offset points", xytext=(8, 6), fontsize=9)
        ax.set_title(title); ax.set_xlabel("Frames given to SfM"); ax.set_xticks(list(x))
        if ylabel:
            ax.set_ylabel(ylabel)
        style_axes(ax)

    panel(axs[0, 0], agg.registered_median.values, agg.registered_min.values, agg.registered_max.values,
          "Registered cameras (median, min-max over seeds)")
    axs[0, 0].plot(x, x, "--", color=C_GREY, lw=1.2, label="All frames registered"); axs[0, 0].legend(frameon=False)
    panel(axs[0, 1], agg.points_median.values, agg.points_min.values, agg.points_max.values, "Sparse 3D points")
    for ax, key, title in ((axs[0, 2], "track_len_median", "Mean track length (views)"),
                           (axs[1, 0], "rate_median_pct", "Registration rate (%)"),
                           (axs[1, 2], "runtime_median_s", "SfM runtime (s)")):
        ax.plot(x, agg[key], "-o", color=C_A, lw=2, ms=7)
        for xi, yi in zip(x, agg[key]):
            ax.annotate(f"{yi:g}", (xi, yi), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9)
        ax.set_title(title); ax.set_xlabel("Frames given to SfM"); ax.set_xticks(list(x)); style_axes(ax)
    axs[1, 0].set_ylim(0, 110)
    ax = axs[1, 1]
    ax.plot(x, agg.reproj_median_px, "-o", color=C_A, lw=2, ms=7, label="Median")
    ax.plot(x, agg.reproj_mean_px, "-s", color=C_B, lw=2, ms=6, label="Mean")
    ax.set_title("Reprojection error (px)"); ax.set_xlabel("Frames given to SfM"); ax.set_xticks(list(x))
    ax.legend(frameon=False); style_axes(ax)
    plt.suptitle("Reconstruction versus number of frames (uniformly sampled from the 47 keyframes)", fontsize=13)
    plt.tight_layout(); plt.savefig(out / "frame_count_comparison.png", dpi=150); plt.close()

    # --- Figure B: same input, same settings, different random seed
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.3))
    ax = axs[0]
    bars = ax.bar(full.seed.astype(str), full.registered_cameras, color=C_A, width=0.6)
    for b, a, c in zip(bars, full.first_registered, full.last_registered):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{int(b.get_height())}\n({a}-{c})",
                ha="center", va="bottom", fontsize=8)
    ax.axhline(len(paths), color=C_GREY, ls="--", lw=1.2, label="47 keyframes available")
    ax.axhline(STORED_RUN["registered_cameras"], color=C_B, ls="-", lw=1.5, label="Stored main reconstruction (32)")
    ax.set_ylim(0, 56); ax.set_xlabel("RANSAC seed"); ax.set_ylabel("Registered cameras")
    ax.set_title("Registered cameras, with the registered frame range"); ax.legend(frameon=False, fontsize=8)
    style_axes(ax)
    ax = axs[1]
    ax.bar(full.seed.astype(str), full.points_3d, color=C_A, width=0.6)
    ax.axhline(STORED_RUN["points_3d"], color=C_B, ls="-", lw=1.5, label="Stored main reconstruction (2,077)")
    ax.set_xlabel("RANSAC seed"); ax.set_ylabel("Sparse 3D points"); ax.set_title("Sparse 3D points")
    ax.legend(frameon=False, fontsize=8); style_axes(ax)
    plt.suptitle("Run-to-run stability: 47 keyframes, identical settings, only the random seed changes", fontsize=12)
    plt.tight_layout(); plt.savefig(out / "seed_stability.png", dpi=150); plt.close()

    print(agg.to_string(index=False))
    print("stability:", stability["registered_cameras_per_seed"], stability["registered_range_per_seed"])
    print(f"[INFO] Results written to {out}")


if __name__ == "__main__":
    main()

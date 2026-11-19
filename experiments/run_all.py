"""
Runs every report experiment in order and regenerates all tables and figures.

    python -m experiments.run_all                 # everything except PointNet training
    python -m experiments.run_all --train         # also retrain PointNet first (GPU recommended)

Total time without training is roughly 40 minutes on a laptop (the ten-video batch takes about 25).
"""

import argparse
import subprocess
import sys
import time

STEPS = [
    ("Object-level 3D outputs and scene figures", "experiments.build_object_scene"),
    ("Experiment 1: SIFT vs ORB", "experiments.exp1_feature_matching"),
    ("Experiment 2: frame count and seed stability (own SfM)", "experiments.exp2_frame_count"),
    ("Experiment 5: COLMAP baseline", "experiments.exp5_colmap_baseline"),
    ("Experiment 2b: frame count with COLMAP", "experiments.exp2b_colmap_frame_count"),
    ("Experiment 3: object localisation consistency (own SfM)", "experiments.exp3_object_localization"),
    ("Experiment 6: object localisation on COLMAP", "experiments.exp6_colmap_object"),
    ("Experiment 4: PointNet accuracy and robustness", "experiments.exp4_pointnet_eval"),
    ("Multi-object evaluation on the ten object videos", "experiments.objectron_batch"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="retrain PointNet on ModelNet40 before evaluating")
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()

    steps = list(STEPS)
    if args.train:
        steps.insert(0, ("PointNet training", "experiments.train_pointnet"))
    failed = []
    for title, module in steps:
        print(f"\n{'=' * 78}\n {title}\n{'=' * 78}", flush=True)
        cmd = [sys.executable, "-m", module]
        if module.endswith("train_pointnet"):
            cmd += ["--epochs", str(args.epochs)]
        t0 = time.time()
        code = subprocess.call(cmd)
        print(f"[run_all] {module}: {'ok' if code == 0 else f'FAILED (exit {code})'} in {time.time() - t0:.0f} s",
              flush=True)
        if code != 0:
            failed.append(module)
    print("\n[run_all] finished" + (f"; failed: {failed}" if failed else "; all steps succeeded"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

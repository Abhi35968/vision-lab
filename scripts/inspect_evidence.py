"""
Inspect Visual Evidence for Selected 3D Points
Phase 5.3.2.1 Audit Step 6
"""

import json
from pathlib import Path
import re
import cv2
import numpy as np
import pandas as pd

def inspect_visual_evidence():
    sfm_dir = Path("outputs/real_video_sfm/focal_1300")
    seg_dir = Path("outputs/object_segmentation")
    assoc_dir = Path("outputs/object_association")

    cams = json.load(open(sfm_dir / "cameras.json", "r", encoding="utf-8"))
    cam_orig = {}
    for cid_str, cinfo in cams.items():
        img_name = cinfo.get("image_name", "")
        m = re.match(r"keyframe_(\d+)_orig_(\d+)\.png", img_name)
        if m:
            cam_orig[int(cid_str)] = (int(m.group(1)), int(m.group(2)), img_name)

    df = pd.read_csv(assoc_dir / "point_mask_associations.csv")
    tracks = json.load(open(sfm_dir / "feature_tracks.json", "r", encoding="utf-8"))["tracks"]
    tracks_by_id = {t["track_id"]: t for t in tracks}

    # Load masks
    masks_dir = seg_dir / "masks"
    masks = {}
    for p in masks_dir.glob("*.png"):
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is not None:
            if m.ndim == 3:
                m = m[:, :, 0]
            masks[p.name] = m

    strong_sample_ids = [187, 584, 737, 766, 816, 818, 820, 848, 850, 954]
    weak_ids = df[df["classification"] == "WEAK_OBJECT_CANDIDATE"]["point_id"].tolist()
    bg_sample_ids = [0, 10, 100, 500, 1000]

    print("=" * 80)
    print("STEP 6: VISUAL EVIDENCE AUDIT")
    print("=" * 80)

    print("\n" + "=" * 40)
    print("1. STRONG CANDIDATES (10 Sample Points)")
    print("=" * 40)
    for pid in strong_sample_ids:
        row = df[df["point_id"] == pid].iloc[0]
        tid = int(row["track_id"])
        tr = tracks_by_id[tid]
        xw, yw, zw = row["x_world"], row["y_world"], row["z_world"]
        print(f"\nPoint ID {pid} (Track {tid}): 3D = [{xw:.3f}, {yw:.3f}, {zw:.3f}]")
        print(f"  Track Length: {row['total_observations']}, Masked Views: {row['masked_observations']}, Inside Mask: {row['inside_mask_observations']}, Ratio: {row['mask_support_ratio']:.3f}, Mean Reproj Error: {row['mean_reprojection_error_px']:.3f} px")
        
        # Per observation evidence
        print("  Observations:")
        for obs in tr["observations"]:
            cid = obs.get("image_id", obs.get("camera_id"))
            u, v = obs.get("coords", (0.0, 0.0))
            _, orig_idx, img_name = cam_orig.get(cid, (cid, None, f"frame_{cid}.png"))
            mask_name = Path(img_name).stem + "_mask.png"
            mask = masks.get(mask_name)
            is_in = False
            if mask is not None:
                col, r = int(round(u)), int(round(v))
                if 0 <= col < mask.shape[1] and 0 <= r < mask.shape[0]:
                    is_in = (mask[r, col] > 0)
            print(f"    Cam {cid:2d} (Orig Frame {orig_idx}): Obs Coords=({u:6.1f}, {v:6.1f}), Inside Mask={is_in}")

    print("\n" + "=" * 40)
    print("2. ALL WEAK CANDIDATES (20 Points)")
    print("=" * 40)
    for pid in weak_ids:
        row = df[df["point_id"] == pid].iloc[0]
        tid = int(row["track_id"])
        tr = tracks_by_id[tid]
        xw, yw, zw = row["x_world"], row["y_world"], row["z_world"]
        sup_cams = str(row["supporting_cameras"])
        print(f"\nPoint ID {pid} (Track {tid}): 3D = [{xw:.3f}, {yw:.3f}, {zw:.3f}]")
        print(f"  Track Length: {row['total_observations']}, Masked: {row['masked_observations']}, Inside: {row['inside_mask_observations']}, Ratio: {row['mask_support_ratio']:.3f}, Mean Reproj Error: {row['mean_reprojection_error_px']:.3f} px")
        print(f"  Supporting Cameras: {sup_cams}")
        for obs in tr["observations"]:
            cid = obs.get("image_id", obs.get("camera_id"))
            u, v = obs.get("coords", (0.0, 0.0))
            _, orig_idx, img_name = cam_orig.get(cid, (cid, None, f"frame_{cid}.png"))
            mask_name = Path(img_name).stem + "_mask.png"
            mask = masks.get(mask_name)
            is_in = False
            if mask is not None:
                col, r = int(round(u)), int(round(v))
                if 0 <= col < mask.shape[1] and 0 <= r < mask.shape[0]:
                    is_in = (mask[r, col] > 0)
            print(f"    Cam {cid:2d} (Orig Frame {orig_idx}): Obs=({u:6.1f}, {v:6.1f}), Inside={is_in}")

    print("\n" + "=" * 40)
    print("3. SAMPLE BACKGROUND CANDIDATES (5 Points)")
    print("=" * 40)
    for pid in bg_sample_ids:
        row = df[df["point_id"] == pid].iloc[0]
        tid = int(row["track_id"])
        tr = tracks_by_id[tid]
        xw, yw, zw = row["x_world"], row["y_world"], row["z_world"]
        print(f"\nPoint ID {pid} (Track {tid}): 3D = [{xw:.3f}, {yw:.3f}, {zw:.3f}]")
        print(f"  Track Length: {row['total_observations']}, Masked: {row['masked_observations']}, Inside: {row['inside_mask_observations']}, Ratio: {row['mask_support_ratio']:.3f}, Mean Reproj Error: {row['mean_reprojection_error_px']:.3f} px")
        for obs in tr["observations"][:5]:  # show first 5
            cid = obs.get("image_id", obs.get("camera_id"))
            u, v = obs.get("coords", (0.0, 0.0))
            _, orig_idx, img_name = cam_orig.get(cid, (cid, None, f"frame_{cid}.png"))
            mask_name = Path(img_name).stem + "_mask.png"
            mask = masks.get(mask_name)
            is_in = False
            if mask is not None:
                col, r = int(round(u)), int(round(v))
                if 0 <= col < mask.shape[1] and 0 <= r < mask.shape[0]:
                    is_in = (mask[r, col] > 0)
            print(f"    Cam {cid:2d} (Orig Frame {orig_idx}): Obs=({u:6.1f}, {v:6.1f}), Inside={is_in}")

if __name__ == "__main__":
    inspect_visual_evidence()


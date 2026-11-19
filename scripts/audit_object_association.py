"""
Independent Audit Script for Multi-View 2D-to-3D Object Association (Phase 5.3.2.1)
Reads saved SfM reconstruction, masks, and feature tracks directly,
recomputes all metrics and classifications from scratch using raw equations,
and compares against Phase 5.3.2 artifacts.
"""

import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

def run_independent_audit():
    print("=" * 80)
    print("INDEPENDENT RECOMPUTATION AUDIT FOR PHASE 5.3.2")
    print("=" * 80)

    sfm_dir = Path("outputs/real_video_sfm/focal_1300")
    seg_dir = Path("outputs/object_segmentation")
    assoc_dir = Path("outputs/object_association")

    # 1. Load inputs
    with open(sfm_dir / "cameras.json", "r", encoding="utf-8") as f:
        cams_raw = json.load(f)
    reg_cams = {int(k): v for k, v in cams_raw.items() if v.get("is_registered", False)}

    with open(sfm_dir / "feature_tracks.json", "r", encoding="utf-8") as f:
        tracks_raw = json.load(f)
    tracks = tracks_raw.get("tracks", [])

    with open(seg_dir / "segmentation_metadata.json", "r", encoding="utf-8") as f:
        seg_meta = json.load(f)
    frame_statuses = {f["image_name"]: f.get("target_status", "UNKNOWN") for f in seg_meta.get("frames", [])}

    # Load masks
    masks_dir = seg_dir / "masks"
    masks = {}
    for p in masks_dir.glob("*.png"):
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is not None:
            if m.ndim == 3:
                m = m[:, :, 0]
            masks[p.name] = m

    # Intrinsics
    fx, fy = 1300.0, 1300.0
    cx, cy = 540.0, 960.0
    w_img, h_img = 1080, 1920

    # 2. Recompute per point
    recomputed_points = []
    
    tot_obs_eval = 0
    tot_valid_proj = 0
    tot_invalid_proj = 0

    for idx, tr in enumerate(tracks):
        tid = tr.get("track_id", idx)
        p3d = np.array(tr.get("point_3d", [0, 0, 0]), dtype=np.float64)
        obs_list = tr.get("observations", [])
        
        valid_projs = 0
        masked_obs = 0
        inside_obs = 0
        outside_obs = 0
        reproj_errs = []
        depths = []
        supporting_cams = []

        for obs in obs_list:
            tot_obs_eval += 1
            cid = obs.get("image_id", obs.get("camera_id"))
            img_name = obs.get("image_name", f"keyframe_{cid:04d}.png")
            u_obs, v_obs = obs.get("coords", (0.0, 0.0))

            if cid not in reg_cams:
                tot_invalid_proj += 1
                continue

            cinfo = reg_cams[cid]
            R = np.array(cinfo["rotation_matrix"], dtype=np.float64)
            t = np.array(cinfo["translation_vector"], dtype=np.float64).reshape(3, 1)

            # Transform: X_c = R * X_w + t
            Xc = R @ p3d.reshape(3, 1) + t
            depth = float(Xc[2, 0])

            if depth < 1e-4:
                tot_invalid_proj += 1
                continue

            u_proj = fx * float(Xc[0, 0]) / depth + cx
            v_proj = fy * float(Xc[1, 0]) / depth + cy

            if not (0.0 <= u_proj < float(w_img) and 0.0 <= v_proj < float(h_img)):
                tot_invalid_proj += 1
                continue

            valid_projs += 1
            tot_valid_proj += 1
            depths.append(depth)
            
            err = np.sqrt((u_proj - u_obs) ** 2 + (v_proj - v_obs) ** 2)
            reproj_errs.append(err)

            # Mask sampling
            mask_name = Path(img_name).stem + "_mask.png"
            mask = masks.get(mask_name, masks.get(img_name))
            status = frame_statuses.get(img_name, "VALID" if (mask is not None and np.any(mask > 0)) else "NO_TARGET")

            if mask is None or status != "VALID":
                continue

            masked_obs += 1
            col = int(round(u_obs))
            row = int(round(v_obs))

            if 0 <= col < w_img and 0 <= row < h_img:
                if mask[row, col] > 0:
                    inside_obs += 1
                    supporting_cams.append(cid)
                else:
                    outside_obs += 1
            else:
                outside_obs += 1

        support_ratio = (inside_obs / float(masked_obs)) if masked_obs > 0 else 0.0
        mean_err = float(np.mean(reproj_errs)) if reproj_errs else 0.0
        med_err = float(np.median(reproj_errs)) if reproj_errs else 0.0

        # Classification rule evaluation
        if masked_obs < 2:
            cls_name = "INSUFFICIENT_EVIDENCE"
        elif masked_obs >= 3 and inside_obs >= 3 and support_ratio >= 0.60:
            cls_name = "STRONG_OBJECT_CANDIDATE"
        elif (inside_obs >= 2 and support_ratio >= 0.35) or (inside_obs >= 1 and masked_obs == 2 and support_ratio >= 0.50):
            cls_name = "WEAK_OBJECT_CANDIDATE"
        elif inside_obs > 0 and (0.25 <= support_ratio < 0.60):
            cls_name = "AMBIGUOUS"
        elif inside_obs == 0 or support_ratio < 0.25:
            cls_name = "BACKGROUND_CANDIDATE"
        else:
            cls_name = "AMBIGUOUS"

        recomputed_points.append({
            "point_id": idx,
            "track_id": tid,
            "total_observations": len(obs_list),
            "valid_projections": valid_projs,
            "masked_observations": masked_obs,
            "inside_mask_observations": inside_obs,
            "outside_mask_observations": outside_obs,
            "mask_support_ratio": round(support_ratio, 4),
            "mean_reprojection_error_px": round(mean_err, 3),
            "median_reprojection_error_px": round(med_err, 3),
            "classification": cls_name,
            "supporting_cameras": supporting_cams,
        })

    # 3. Compare with saved CSV
    csv_path = assoc_dir / "point_mask_associations.csv"
    saved_df = pd.read_csv(csv_path)

    mismatches = []
    for i, r_recomp in enumerate(recomputed_points):
        r_saved = saved_df.iloc[i]
        for field in ["total_observations", "valid_projections", "masked_observations",
                      "inside_mask_observations", "outside_mask_observations", "classification"]:
            val_recomp = r_recomp[field]
            val_saved = r_saved[field]
            if val_recomp != val_saved:
                mismatches.append(f"Pt {i} {field}: saved={val_saved}, recomp={val_recomp}")

    print(f"\nTotal points evaluated: {len(recomputed_points)}")
    print(f"Total field-level mismatches vs point_mask_associations.csv: {len(mismatches)}")
    if mismatches:
        for m in mismatches[:10]:
            print("  ", m)

    # 4. Compare master summary counts
    classes_recomp = pd.Series([p["classification"] for p in recomputed_points]).value_counts()
    print("\nRecomputed Class Breakdown:")
    for k, v in classes_recomp.items():
        print(f"  {k:25s}: {v}")

    with open(assoc_dir / "association_metadata.json", "r", encoding="utf-8") as f:
        saved_meta = json.load(f)["summary"]

    saved_breakdown = saved_meta["classification_breakdown"]
    print("\nSaved Metadata Breakdown:")
    print(f"  STRONG_OBJECT_CANDIDATE  : {saved_breakdown['strong_object_candidates']['count']}")
    print(f"  WEAK_OBJECT_CANDIDATE    : {saved_breakdown['weak_object_candidates']['count']}")
    print(f"  AMBIGUOUS                : {saved_breakdown['ambiguous_points']['count']}")
    print(f"  BACKGROUND_CANDIDATE     : {saved_breakdown['background_candidates']['count']}")
    print(f"  INSUFFICIENT_EVIDENCE    : {saved_breakdown['insufficient_evidence']['count']}")

    print("\nRecomputed Projections & Observations:")
    print(f"  Total observations evaluated: {tot_obs_eval} (saved: {saved_meta['projections_and_observations']['total_observations_evaluated']})")
    print(f"  Total valid projections     : {tot_valid_proj} (saved: {saved_meta['projections_and_observations']['total_valid_projections']})")
    print(f"  Total invalid projections   : {tot_invalid_proj} (saved: {saved_meta['projections_and_observations']['total_invalid_projections']})")

    # Reprojection errors
    all_errs = [p["mean_reprojection_error_px"] for p in recomputed_points if p["valid_projections"] > 0]
    strong_errs = [p["mean_reprojection_error_px"] for p in recomputed_points if p["classification"] == "STRONG_OBJECT_CANDIDATE" and p["valid_projections"] > 0]
    bg_errs = [p["mean_reprojection_error_px"] for p in recomputed_points if p["classification"] == "BACKGROUND_CANDIDATE" and p["valid_projections"] > 0]

    print("\nReprojection Errors:")
    print(f"  Mean all points : recomp={np.mean(all_errs):.3f}, saved={saved_meta['reprojection_errors_px']['mean_all_points']}")
    print(f"  Mean strong     : recomp={np.mean(strong_errs):.3f}, saved={saved_meta['reprojection_errors_px']['mean_strong_candidates']}")
    print(f"  Median strong   : recomp={np.median(strong_errs):.3f}, saved={saved_meta['reprojection_errors_px']['median_strong_candidates']}")
    print(f"  Mean background : recomp={np.mean(bg_errs):.3f}, saved={saved_meta['reprojection_errors_px']['mean_background']}")

if __name__ == "__main__":
    run_independent_audit()


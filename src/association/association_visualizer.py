"""
Association Visualizer Module
Renders 2D multi-view association overlays (bounding boxes + classified SfM observations)
and 3D sparse scene visualizations highlighting candidate object clusters in arbitrary_units.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from src.detection.detection_models import Detection
from src.detection.detection_visualizer import DetectionVisualizer
from src.sfm.camera_registration import CameraPose
from src.association.association_models import (
    AssociationStatus,
    ObservationAssociation,
    TrackAssociation,
    ObjectPointGroup,
)


class AssociationVisualizer:
    """
    Renders 2D keyframe association overlays and 3D object-level spatial point distributions.
    Maintains deterministic per-class color fidelity across 2D images and 3D Matplotlib scenes.
    """

    def __init__(self, output_dir: Union[str, Path] = "outputs/association/association_visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def visualize_keyframe_association(
        self,
        image_or_path: Union[np.ndarray, str, Path],
        detections: List[Detection],
        observation_associations: List[ObservationAssociation],
        output_filename: str,
    ) -> Path:
        """
        Draws 2D bounding boxes and classifies 2D SfM feature observations into:
        - Matched (colored by class)
        - Ambiguous (orange marker)
        - Unmatched (faint gray dot)
        """
        if isinstance(image_or_path, (str, Path)):
            img = cv2.imread(str(image_or_path))
            if img is None:
                raise FileNotFoundError(f"Failed to read image at: {image_or_path}")
        else:
            img = image_or_path.copy()

        h, w = img.shape[:2]

        # 1. Draw detection bounding boxes
        for det in detections:
            color = DetectionVisualizer.get_class_color(det.class_id)
            x1 = max(0, min(w - 1, int(round(det.x1))))
            y1 = max(0, min(h - 1, int(round(det.y1))))
            x2 = max(0, min(w - 1, int(round(det.x2))))
            y2 = max(0, min(h - 1, int(round(det.y2))))

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            tag_y1 = max(0, y1 - th - baseline - 4)
            tag_y2 = y1 if y1 - th - baseline - 4 >= 0 else y1 + th + baseline + 4
            tag_x2 = min(w - 1, x1 + tw + 6)
            cv2.rectangle(img, (x1, tag_y1), (tag_x2, tag_y2), color, -1)
            text_pos_y = tag_y2 - baseline - 2 if tag_y2 == y1 else tag_y2 - 2
            cv2.putText(
                img, label, (x1 + 3, text_pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
            )

        # 2. Draw SfM 2D observations
        for obs in observation_associations:
            ox = max(0, min(w - 1, int(round(obs.coords[0]))))
            oy = max(0, min(h - 1, int(round(obs.coords[1]))))

            if not obs.is_matched:
                # Unmatched observation: small subtle gray dot
                cv2.circle(img, (ox, oy), 2, (180, 180, 180), -1)
            elif obs.is_ambiguous:
                # Ambiguous observation: highlighted orange diamond
                color_ambig = (0, 140, 255)  # Orange BGR
                pts_diamond = np.array([[ox, oy - 4], [ox + 4, oy], [ox, oy + 4], [ox - 4, oy]], np.int32)
                cv2.fillConvexPoly(img, pts_diamond, color_ambig)
                cv2.polylines(img, [pts_diamond], True, (0, 0, 0), 1)
            else:
                # Clean matched observation: colored by assigned class
                cid = obs.class_id if obs.class_id is not None else 0
                c_color = DetectionVisualizer.get_class_color(cid)
                cv2.circle(img, (ox, oy), 4, c_color, -1)
                cv2.circle(img, (ox, oy), 4, (0, 0, 0), 1)

        # 3. Add visual legend overlay in top-right corner
        leg_x, leg_y = w - 210, 15
        cv2.rectangle(img, (leg_x - 5, leg_y - 5), (w - 10, leg_y + 65), (20, 20, 20), -1)
        cv2.rectangle(img, (leg_x - 5, leg_y - 5), (w - 10, leg_y + 65), (200, 200, 200), 1)

        cv2.circle(img, (leg_x + 10, leg_y + 10), 4, (0, 255, 0), -1)
        cv2.putText(img, "Matched Obs", (leg_x + 25, leg_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        pts_d = np.array([[leg_x + 10, leg_y + 26], [leg_x + 14, leg_y + 30], [leg_x + 10, leg_y + 34], [leg_x + 6, leg_y + 30]], np.int32)
        cv2.fillConvexPoly(img, pts_d, (0, 140, 255))
        cv2.putText(img, "Ambiguous Obs", (leg_x + 25, leg_y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.circle(img, (leg_x + 10, leg_y + 50), 2, (180, 180, 180), -1)
        cv2.putText(img, "Unmatched Obs", (leg_x + 25, leg_y + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        out_path = self.output_dir / output_filename
        cv2.imwrite(str(out_path), img)
        return out_path

    def visualize_3d_object_point_groups(
        self,
        track_associations: List[TrackAssociation],
        object_point_groups: Dict[str, ObjectPointGroup],
        registered_cameras: Optional[Dict[int, CameraPose]] = None,
        filename: str = "3d_scene_object_associations.png",
    ) -> Path:
        """
        Plots 3D point cloud in arbitrary_units, distinguishing unassociated points
        from candidate object point clusters with consistent color coding.
        """
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111, projection="3d")

        # 1. Plot all unassociated triangulated points (background context)
        unassoc_pts = [
            ta.point_3d for ta in track_associations
            if ta.is_triangulated and ta.point_3d is not None
            and ta.association_status == AssociationStatus.UNASSOCIATED.value
        ]
        if unassoc_pts:
            arr_u = np.array(unassoc_pts)
            ax.scatter(
                arr_u[:, 0], arr_u[:, 1], arr_u[:, 2],
                c="#b0b0b0", s=6, alpha=0.35, label=f"Unassociated Points ({len(arr_u)})"
            )

        # 2. Plot candidate object points grouped by class
        cmap = plt.get_cmap("tab10")
        for idx, (c_name, grp) in enumerate(object_point_groups.items()):
            pts = [
                ta.point_3d for ta in track_associations
                if ta.track_id in grp.track_ids and ta.point_3d is not None
            ]
            if pts:
                arr_c = np.array(pts)
                color = cmap(idx % 10)
                ax.scatter(
                    arr_c[:, 0], arr_c[:, 1], arr_c[:, 2],
                    color=color, s=24, edgecolors="black", linewidth=0.5, alpha=0.85,
                    label=f"Class: {c_name} ({len(arr_c)} pts)"
                )

        # 3. Plot camera trajectory
        if registered_cameras:
            cams = [cam for cam in sorted(registered_cameras.values(), key=lambda c: c.image_id) if cam.is_registered]
            if cams:
                centers = np.array([c.camera_center for c in cams])
                ax.plot(
                    centers[:, 0], centers[:, 1], centers[:, 2],
                    color="crimson", linestyle="--", linewidth=1.2, alpha=0.8
                )
                ax.scatter(
                    centers[:, 0], centers[:, 1], centers[:, 2],
                    c="navy", marker="^", s=60, edgecolors="white", label=f"Cameras ({len(cams)})"
                )
                for c in cams:
                    C = c.camera_center
                    ax.text(C[0], C[1], C[2], f"C{c.image_id}", fontsize=7.5, color="darkred")

        ax.set_title(
            "3D Scene & Candidate Object Point Associations (arbitrary units)",
            fontsize=12, fontweight="bold", pad=12
        )
        ax.set_xlabel("X (arbitrary_units)", labelpad=8)
        ax.set_ylabel("Y (arbitrary_units)", labelpad=8)
        ax.set_zlabel("Z (arbitrary_units)", labelpad=8)
        ax.legend(loc="upper right", framealpha=0.9, fontsize=8.5)
        ax.grid(True, linestyle=":", alpha=0.6)

        out_path = self.output_dir / filename
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path


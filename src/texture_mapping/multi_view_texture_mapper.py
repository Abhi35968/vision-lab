"""
Multi-View Texture Mapping Module (Phase 4)
Projects and blends multi-view RGB keyframe imagery onto 3D mesh surfaces:
1. Reuses existing calibrated SfM camera poses [R | t] and intrinsics K.
2. Visibility and occlusion gating: ray-normal angle check (cos theta > 0.15) and frustum boundary test.
3. Multi-view color blending: weights pixel contributions by viewpoint obliqueness and image sharpness.
4. Generates both a textured mesh with UV coordinates (texture atlas PNG + OBJ/MTL) and per-vertex colors.
5. Exports Stanford PLY (with RGB), Wavefront OBJ (with MTL texture map), and binary GLB formats.
"""

from dataclasses import dataclass, asdict, field
import json
import logging
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image
import trimesh

from src.object_reconstruction.bottle_models import Mesh3D

logger = logging.getLogger("MultiViewTextureMapper")


@dataclass
class TexturingReport:
    """Summary of multi-view texture projection and blending."""
    num_vertices_total: int
    num_vertices_colored: int
    unobserved_vertices_count: int
    num_cameras_used: int
    average_views_per_vertex: float
    texturing_method: str               # "UV_ATLAS_AND_VERTEX_COLORS" or "VERTEX_COLOR_FALLBACK"
    texture_image_path: Optional[str]
    obj_mesh_path: Optional[str]
    ply_mesh_path: Optional[str]
    glb_mesh_path: Optional[str]
    execution_time_ms: float
    limitations: List[str]

    def to_dict(self) -> dict:
        return asdict(self)


class MultiViewTextureMapper:
    """
    Back-projects RGB pixels from calibrated keyframes onto triangular 3D meshes.
    """

    def __init__(
        self,
        min_cos_angle: float = 0.15,       # Reject grazing angle viewings (< ~81 deg from normal)
        blur_filter_threshold: float = 8.0, # Exclude severely blurred frames
        texture_resolution: int = 1024,    # Dimension of square texture atlas (1024x1024)
        default_color: Tuple[int, int, int] = (180, 180, 180),  # Neutral gray fallback for unobserved vertices
    ):
        self.min_cos_angle = min_cos_angle
        self.blur_filter_threshold = blur_filter_threshold
        self.texture_resolution = texture_resolution
        self.default_color = default_color

    def _compute_frame_sharpness(self, img_bgr: np.ndarray) -> float:
        """Computes variance of Laplacian sharpness weight."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def bake_texture(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        keyframes: Dict[str, np.ndarray],          # {frame_name: bgr_image}
        camera_poses: Dict[str, dict],             # {frame_name: {"R": (3,3), "t": (3,1) or "camera_center": ...}}
        camera_matrix: np.ndarray,                 # (3, 3) K
        normals: Optional[np.ndarray] = None,
        masks: Optional[Dict[str, np.ndarray]] = None,
        output_dir: Union[str, Path] = "outputs/textured_mesh",
        mesh_tag: str = "reconstructed_object",
    ) -> Tuple[trimesh.Trimesh, TexturingReport]:
        """
        Projects RGB colors from keyframes onto vertices and generates a texture atlas.
        """
        t0 = time.time()
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)

        v = np.asarray(vertices, dtype=np.float64)
        f = np.asarray(faces, dtype=np.int32)
        num_v = len(v)
        limitations: List[str] = []

        # 1. Compute vertex normals if not provided
        if normals is not None and len(normals) == num_v:
            vn = np.asarray(normals, dtype=np.float64)
        else:
            mesh_temp = Mesh3D(v, f)
            vn = mesh_temp.compute_vertex_normals()

        # 2. Filter Keyframes by Sharpness
        valid_cameras: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, float]] = []
        K = np.asarray(camera_matrix, dtype=np.float64)

        # Build comprehensive pose lookup dictionary
        pose_lookup: Dict[str, dict] = {}
        for k, c in camera_poses.items():
            if isinstance(c, dict):
                if "image_name" in c:
                    pose_lookup[c["image_name"]] = c
                pose_lookup[k] = c

        for fname, img in keyframes.items():
            if fname not in pose_lookup:
                continue
            c_data = pose_lookup[fname]
            if not c_data.get("is_registered", True):
                continue
            if "translation_vector" in c_data and len(c_data["translation_vector"]) != 3:
                continue

            sharpness = self._compute_frame_sharpness(img)
            if sharpness < self.blur_filter_threshold:
                continue

            R = None
            if "rotation_matrix" in c_data:
                R = np.array(c_data["rotation_matrix"], dtype=np.float64)
            elif "R" in c_data:
                R = np.array(c_data["R"], dtype=np.float64)

            t = None
            if "translation_vector" in c_data:
                t = np.array(c_data["translation_vector"], dtype=np.float64).reshape(3, 1)
            elif "t" in c_data:
                t = np.array(c_data["t"], dtype=np.float64).reshape(3, 1)

            if R is None and "camera_center" in c_data:
                # If only camera_center is provided, derive look-at pose facing object centroid
                c_center = np.array(c_data["camera_center"], dtype=np.float64)
                z_cam = np.mean(v, axis=0) - c_center
                z_cam /= max(1e-6, np.linalg.norm(z_cam))
                up = np.array([0, -1, 0], dtype=np.float64)
                x_cam = np.cross(up, z_cam)
                x_cam /= max(1e-6, np.linalg.norm(x_cam))
                y_cam = np.cross(z_cam, x_cam)
                R = np.vstack([x_cam, y_cam, z_cam])
                t = -R @ c_center.reshape(3, 1)

            if R is not None and t is not None:
                valid_cameras.append((fname, img, R, t, sharpness))

        if not valid_cameras:
            limitations.append("No registered camera views passed the sharpness and pose validity check.")
            # Neutral gray fallback
            vertex_colors = np.full((num_v, 4), [*self.default_color, 255], dtype=np.uint8)
            tm_mesh = trimesh.Trimesh(vertices=v, faces=f, vertex_colors=vertex_colors, process=False)
            rep = TexturingReport(
                num_vertices_total=num_v,
                num_vertices_colored=0,
                unobserved_vertices_count=num_v,
                num_cameras_used=0,
                average_views_per_vertex=0.0,
                texturing_method="VERTEX_COLOR_FALLBACK",
                texture_image_path=None,
                obj_mesh_path=None,
                ply_mesh_path=None,
                glb_mesh_path=None,
                execution_time_ms=0.0,
                limitations=limitations,
            )
            return tm_mesh, rep

        # 3. Multi-View Projection & Color Blending per Vertex
        accum_rgb = np.zeros((num_v, 3), dtype=np.float64)
        accum_weights = np.zeros(num_v, dtype=np.float64)
        view_counts = np.zeros(num_v, dtype=np.int32)

        for fname, img_bgr, R, t, sharpness in valid_cameras:
            h, w = img_bgr.shape[:2]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float64)

            # Camera center in world space
            c_center = (-R.T @ t).flatten()

            # Transform vertices to camera coordinate frame: X_c = R * X + t
            v_cam = (R @ v.T + t).T  # (N, 3)
            depths = v_cam[:, 2]

            # Ray vectors from camera center to vertices
            ray_dirs = v - c_center
            ray_norms = np.linalg.norm(ray_dirs, axis=1, keepdims=True)
            ray_norms[ray_norms == 0] = 1.0
            ray_dirs_unit = ray_dirs / ray_norms

            # Angle of incidence: cos(theta) = -ray_unit . normal
            cos_thetas = -np.sum(ray_dirs_unit * vn, axis=1)

            # Perspective projection: [u, v, 1]^T ~ K * X_c
            proj = (K @ v_cam.T).T
            valid_z = depths > 0.05
            u = np.zeros(num_v)
            v_px = np.zeros(num_v)
            u[valid_z] = proj[valid_z, 0] / depths[valid_z]
            v_px[valid_z] = proj[valid_z, 1] / depths[valid_z]

            # Check inside image boundaries
            inside_bounds = (
                valid_z
                & (cos_thetas > self.min_cos_angle)
                & (u >= 2.0) & (u < (w - 2.0))
                & (v_px >= 2.0) & (v_px < (h - 2.0))
            )

            valid_indices = np.where(inside_bounds)[0]
            if len(valid_indices) == 0:
                continue

            # Mask verification if segmentation masks are supplied
            if masks is not None and fname in masks:
                mask = masks[fname]
                mh, mw = mask.shape[:2]
                mask_y = np.clip(np.round(v_px[valid_indices]).astype(int), 0, mh - 1)
                mask_x = np.clip(np.round(u[valid_indices]).astype(int), 0, mw - 1)
                valid_mask = mask[mask_y, mask_x] > 0
                valid_indices = valid_indices[valid_mask]
                if len(valid_indices) == 0:
                    continue

            # Bilinear interpolation of RGB colors
            u_valid = u[valid_indices]
            v_valid = v_px[valid_indices]
            u0 = np.floor(u_valid).astype(int)
            v0 = np.floor(v_valid).astype(int)
            u1 = u0 + 1
            v1 = v0 + 1

            du = (u_valid - u0)[:, np.newaxis]
            dv = (v_valid - v0)[:, np.newaxis]

            c00 = img_rgb[v0, u0]
            c10 = img_rgb[v0, u1]
            c01 = img_rgb[v1, u0]
            c11 = img_rgb[v1, u1]

            sampled_colors = (
                c00 * (1 - du) * (1 - dv)
                + c10 * du * (1 - dv)
                + c01 * (1 - du) * dv
                + c11 * du * dv
            )

            # Weight by cos_theta and normalized sharpness
            weights = cos_thetas[valid_indices] * (sharpness / 100.0)
            accum_rgb[valid_indices] += sampled_colors * weights[:, np.newaxis]
            accum_weights[valid_indices] += weights
            view_counts[valid_indices] += 1

        # 4. Finalize Vertex Colors
        observed_mask = accum_weights > 0
        final_vertex_colors = np.full((num_v, 4), [*self.default_color, 255], dtype=np.uint8)

        if observed_mask.sum() > 0:
            blended = accum_rgb[observed_mask] / accum_weights[observed_mask, np.newaxis]
            blended = np.clip(blended, 0, 255).astype(np.uint8)
            final_vertex_colors[observed_mask, :3] = blended

        unobserved_count = int((~observed_mask).sum())
        if unobserved_count > 0:
            limitations.append(f"{unobserved_count} vertices were not directly visible in camera views; filled with neutral material tone.")

        # 5. Generate Texture Atlas with Proper GLTF/OpenGL UV Parameterization
        v_centered = v - np.mean(v, axis=0)
        theta = np.arctan2(v_centered[:, 0], v_centered[:, 2])
        uv_u = (theta + np.pi) / (2.0 * np.pi)

        # Height normalized: In GLTF, v=0 is at bottom (y_max), v=1 is at top (y_min)
        y_min = np.min(v[:, 1])
        y_max = np.max(v[:, 1])
        y_range = max(1e-4, y_max - y_min)
        uv_v = (y_max - v[:, 1]) / y_range
        uv_coords = np.column_stack([uv_u, uv_v])

        tex_res = self.texture_resolution

        # Check if mesh is suitable for continuous cylindrical/parametric surface texturing
        if num_v >= 10 and y_range > 0.10:
            # Dense UV surface grid (256x256) projected across cameras
            gh, gw = 256, 256
            grid_us = np.linspace(0.0, 1.0, gw, endpoint=False)
            grid_vs = np.linspace(0.0, 1.0, gh, endpoint=False)

            # Row 0 is top (y_min / v=1), row gh-1 is bottom (y_max / v=0)
            grid_y = y_min + grid_vs * y_range

            # Determine profile radii
            p_json = Path("outputs/object_reconstruction/reconstruction_parameters.json")
            param_obj = None
            if p_json.exists():
                try:
                    from src.object_reconstruction.bottle_models import BottleParameters
                    from src.object_reconstruction.parametric_bottle import ParametricBottle
                    with open(p_json, "r") as pf:
                        pdict = json.load(pf)
                    b_params = BottleParameters(**pdict)
                    param_bottle = ParametricBottle(b_params)
                    b_profile = param_bottle.generate_profile()
                    zs_prof = np.array([p.z_axial for p in b_profile])
                    rs_prof = np.array([p.radius for p in b_profile])
                    param_obj = (b_params, zs_prof, rs_prof)
                except Exception:
                    param_obj = None

            centroid_xz = np.mean(v[:, [0, 2]], axis=0)
            if param_obj is not None:
                b_params, zs_prof, rs_prof = param_obj
                c_cap = np.array(b_params.cap_center_world, dtype=np.float64)
                c_base = c_cap + np.array([0, b_params.total_height, 0], dtype=np.float64)
                z_axials = (1.0 - grid_vs) * b_params.total_height
                grid_radii = np.interp(z_axials, zs_prof, rs_prof)
                center_x = c_base[0]
                center_z = c_base[2]
            else:
                radii_from_center = np.linalg.norm(v[:, [0, 2]] - centroid_xz, axis=1)
                sort_idx = np.argsort(v[:, 1])
                grid_radii = np.interp(grid_y, v[sort_idx, 1], radii_from_center[sort_idx])
                center_x = centroid_xz[0]
                center_z = centroid_xz[2]

            grid_thetas = grid_us * 2.0 * np.pi - np.pi
            sin_th = np.sin(grid_thetas)
            cos_th = np.cos(grid_thetas)

            X_grid = np.zeros((gh, gw, 3), dtype=np.float64)
            X_grid[:, :, 0] = center_x + grid_radii[:, None] * sin_th[None, :]
            X_grid[:, :, 1] = grid_y[:, None]
            X_grid[:, :, 2] = center_z + grid_radii[:, None] * cos_th[None, :]

            N_grid = np.zeros((gh, gw, 3), dtype=np.float64)
            N_grid[:, :, 0] = sin_th[None, :]
            N_grid[:, :, 2] = cos_th[None, :]

            pts_flat = X_grid.reshape(-1, 3)
            norms_flat = N_grid.reshape(-1, 3)
            n_grid_pts = len(pts_flat)

            grid_accum_bgr = np.zeros((n_grid_pts, 3), dtype=np.float64)
            grid_accum_w = np.zeros(n_grid_pts, dtype=np.float64)

            for fname, img_bgr, R, t, sharpness in valid_cameras:
                c_center = (-R.T @ t).flatten()
                ray = pts_flat - c_center
                ray_dist = np.linalg.norm(ray, axis=1, keepdims=True)
                ray_dist[ray_dist == 0] = 1.0
                ray_dir = ray / ray_dist
                cos_inc = -np.sum(ray_dir * norms_flat, axis=1)

                Xc = (R @ pts_flat.T + t).T
                depths_g = Xc[:, 2]
                valid_zg = depths_g > 0.05

                proj_g = (K @ Xc.T).T
                u_g = np.zeros(n_grid_pts)
                v_g = np.zeros(n_grid_pts)
                u_g[valid_zg] = proj_g[valid_zg, 0] / depths_g[valid_zg]
                v_g[valid_zg] = proj_g[valid_zg, 1] / depths_g[valid_zg]

                ih, iw = img_bgr.shape[:2]
                inside_g = (
                    valid_zg
                    & (cos_inc > self.min_cos_angle)
                    & (u_g >= 2.0) & (u_g < (iw - 2.0))
                    & (v_g >= 2.0) & (v_g < (ih - 2.0))
                )

                if masks is not None and fname in masks:
                    mask_g = masks[fname]
                    mh, mw = mask_g.shape[:2]
                    g_idx = np.where(inside_g)[0]
                    if len(g_idx) > 0:
                        px_v_int = np.clip(np.round(v_g[g_idx]).astype(int), 0, mh - 1)
                        px_u_int = np.clip(np.round(u_g[g_idx]).astype(int), 0, mw - 1)
                        mask_pass = mask_g[px_v_int, px_u_int] > 0
                        inside_g[g_idx[~mask_pass]] = False

                idx_g = np.where(inside_g)[0]
                if len(idx_g) == 0:
                    continue

                u_vals = u_g[idx_g].astype(int)
                v_vals = v_g[idx_g].astype(int)
                sampled_bgr = img_bgr[v_vals, u_vals].astype(np.float64)
                w_vals = cos_inc[idx_g] * (sharpness / 100.0)

                grid_accum_bgr[idx_g] += sampled_bgr * w_vals[:, None]
                grid_accum_w[idx_g] += w_vals

            grid_obs = grid_accum_w > 0
            raw_atlas_bgr = np.zeros((n_grid_pts, 3), dtype=np.uint8)
            if grid_obs.sum() > 0:
                raw_atlas_bgr[grid_obs] = np.clip(
                    grid_accum_bgr[grid_obs] / grid_accum_w[grid_obs, None], 0, 255
                ).astype(np.uint8)
            else:
                raw_atlas_bgr[:] = [self.default_color[2], self.default_color[1], self.default_color[0]]

            raw_atlas_bgr = raw_atlas_bgr.reshape(gh, gw, 3)

            # Inpaint unobserved rear surface regions seamlessly
            inpaint_mask = (~grid_obs.reshape(gh, gw)).astype(np.uint8) * 255
            if grid_obs.sum() > 0 and inpaint_mask.sum() > 0:
                filled_atlas_bgr = cv2.inpaint(
                    raw_atlas_bgr, inpaint_mask, inpaintRadius=15, flags=cv2.INPAINT_TELEA
                )
            else:
                filled_atlas_bgr = raw_atlas_bgr

            atlas_final_bgr = cv2.resize(filled_atlas_bgr, (tex_res, tex_res), interpolation=cv2.INTER_CUBIC)
            atlas_final_bgr = cv2.GaussianBlur(atlas_final_bgr, (3, 3), 0.5)
            atlas_rgb = cv2.cvtColor(atlas_final_bgr, cv2.COLOR_BGR2RGB)
        else:
            # Fallback for planar / sparse synthetic meshes
            atlas_rgb = np.full((tex_res, tex_res, 3), self.default_color, dtype=np.uint8)
            if observed_mask.sum() > 0:
                mean_col = np.mean(final_vertex_colors[observed_mask, :3], axis=0).astype(np.uint8)
                atlas_rgb[:] = mean_col
                grid_u = np.clip((uv_u * (tex_res - 1)).astype(int), 0, tex_res - 1)
                grid_v = np.clip(((1.0 - uv_v) * (tex_res - 1)).astype(int), 0, tex_res - 1)
                for idx in range(num_v):
                    if observed_mask[idx]:
                        cv2.circle(
                            atlas_rgb,
                            (grid_u[idx], grid_v[idx]),
                            radius=int(tex_res * 0.1),
                            color=[int(c) for c in final_vertex_colors[idx, :3]],
                            thickness=-1,
                        )

        texture_pil = Image.fromarray(atlas_rgb)
        tex_path = out_p / f"{mesh_tag}_texture_atlas.png"
        texture_pil.save(tex_path)

        # 6. Build Trimesh with Materials and UVs
        material = trimesh.visual.material.PBRMaterial(
            name=f"{mesh_tag}_pbr",
            baseColorTexture=texture_pil,
            metallicFactor=0.1,
            roughnessFactor=0.6,
        )
        visual = trimesh.visual.TextureVisuals(
            uv=uv_coords,
            image=texture_pil,
            material=material,
        )
        tm_mesh = trimesh.Trimesh(
            vertices=v,
            faces=f,
            vertex_colors=final_vertex_colors,
            visual=visual,
            process=False,
        )

        # 7. Export Files (PLY with vertex colors, OBJ with MTL, GLB binary)
        ply_path = out_p / f"{mesh_tag}_textured.ply"
        obj_path = out_p / f"{mesh_tag}_textured.obj"
        glb_path = out_p / f"{mesh_tag}_textured.glb"

        tm_mesh.export(str(ply_path), file_type="ply")
        tm_mesh.export(str(obj_path), file_type="obj")
        tm_mesh.export(str(glb_path), file_type="glb")

        t_ms = (time.time() - t0) * 1000.0
        avg_views = float(np.mean(view_counts[observed_mask])) if observed_mask.sum() > 0 else 0.0

        report = TexturingReport(
            num_vertices_total=num_v,
            num_vertices_colored=int(observed_mask.sum()),
            unobserved_vertices_count=unobserved_count,
            num_cameras_used=len(valid_cameras),
            average_views_per_vertex=round(avg_views, 2),
            texturing_method="UV_ATLAS_AND_VERTEX_COLORS",
            texture_image_path=str(tex_path),
            obj_mesh_path=str(obj_path),
            ply_mesh_path=str(ply_path),
            glb_mesh_path=str(glb_path),
            execution_time_ms=round(t_ms, 2),
            limitations=limitations,
        )

        return tm_mesh, report

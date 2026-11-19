"""
Multi-View Stereo (MVS) Comparator Module
Compares dense 3D reconstruction paradigms against project constraints:
Windows 11 OS, CPU-only execution, 16 GB RAM, transparent refractive water bottle,
and existing SfM camera poses.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List


@dataclass
class ReconstructionMethodEvaluation:
    """Detailed capability and compatibility scorecard for a 3D reconstruction method."""
    method_name: str
    reconstruction_paradigm: str
    windows_compatibility: str
    hardware_requirement: str
    max_ram_footprint: str
    surface_reflectance_assumption: str
    handles_transparency_and_refraction: bool
    sfm_pose_compatibility: str
    installation_complexity: str
    dataset_feasibility_verdict: str  # "FEASIBLE", "DEGRADED", "INFEASIBLE"
    technical_justification: str

    def to_dict(self) -> dict:
        return asdict(self)


class MVSComparator:
    """
    Evaluates and ranks multi-view dense reconstruction and neural representation
    techniques specifically for the tabletop water bottle project environment.
    """

    @staticmethod
    def get_method_comparisons() -> List[ReconstructionMethodEvaluation]:
        return [
            ReconstructionMethodEvaluation(
                method_name="Classical PatchMatch MVS",
                reconstruction_paradigm="Photometric Patch Correlation (PMVS2 / PatchMatch Stereo)",
                windows_compatibility="Supported (Native or precompiled C++ binaries)",
                hardware_requirement="CPU-compatible (Multi-threaded x86-64)",
                max_ram_footprint="4 - 8 GB RAM (within 16 GB limit with downsampled frames)",
                surface_reflectance_assumption="Strictly Lambertian (diffuse color constancy across views)",
                handles_transparency_and_refraction=False,
                sfm_pose_compatibility="Directly compatible with calibrated K, R, t from SfM",
                installation_complexity="Moderate (requires precompiled binaries or lightweight Python bindings)",
                dataset_feasibility_verdict="INFEASIBLE",
                technical_justification=(
                    "Fails on the water bottle because the surface is transparent clear plastic. "
                    "Photometric consistency (NCC) assumes color constancy, which is violated by moving "
                    "specular highlights and refracted background rays. Produces hollow voids or points on the background."
                ),
            ),
            ReconstructionMethodEvaluation(
                method_name="COLMAP Dense Reconstruction",
                reconstruction_paradigm="Dense Depth/Normal Estimation + Multi-view Fusion + Poisson Meshing",
                windows_compatibility="Supported (Official Windows binaries available)",
                hardware_requirement="NVIDIA GPU with CUDA REQUIRED (CUDA 11/12 with >= 6 GB VRAM)",
                max_ram_footprint="8 - 14 GB RAM (near ceiling for full resolution)",
                surface_reflectance_assumption="Lambertian with geometric multi-view consistency filtering",
                handles_transparency_and_refraction=False,
                sfm_pose_compatibility="Native (COLMAP binary or project export)",
                installation_complexity="Low (official Windows executable zip package)",
                dataset_feasibility_verdict="INFEASIBLE",
                technical_justification=(
                    "Blocked on two fatal grounds: (1) Hardware: Dense PatchMatch in COLMAP strictly "
                    "requires an NVIDIA CUDA GPU; on CPU-only machines, dense reconstruction cannot execute. "
                    "(2) Physics: COLMAP's geometric consistency filter explicitly purges non-Lambertian "
                    "refractive transmission, carving away the transparent bottle body."
                ),
            ),
            ReconstructionMethodEvaluation(
                method_name="OpenMVS",
                reconstruction_paradigm="DensifyPointCloud + ReconstructMesh + RefineMesh + TextureMesh",
                windows_compatibility="Supported via pre-built binaries (requires VC++ runtime)",
                hardware_requirement="CPU-only supported (CUDA optional for acceleration)",
                max_ram_footprint="10 - 16 GB RAM (high risk of Out-Of-Memory on 16 GB RAM at 1080p)",
                surface_reflectance_assumption="Lambertian photometric consistency across stereo pairs",
                handles_transparency_and_refraction=False,
                sfm_pose_compatibility="Compatible via OpenMVS project export format (cameras, points)",
                installation_complexity="High (requires external dependencies: Eigen, Boost, Ceres, CGAL)",
                dataset_feasibility_verdict="INFEASIBLE",
                technical_justification=(
                    "CPU densification is computationally intensive and memory-heavy (can exceed 16 GB RAM). "
                    "More critically, OpenMVS assumes opaque diffuse materials. It cannot correlate "
                    "features on smooth transparent surfaces and will either reconstruct the tablecloth beneath "
                    "or produce severe surface noise."
                ),
            ),
            ReconstructionMethodEvaluation(
                method_name="Shape-from-Silhouette / Visual Hull (Voxel Carving)",
                reconstruction_paradigm="Volumetric intersection of back-projected 2D segmentation masks",
                windows_compatibility="100% Native Python / NumPy / SciPy (cross-platform)",
                hardware_requirement="CPU-only (low compute, runs in seconds to minutes)",
                max_ram_footprint="1 - 3 GB RAM (highly efficient octree / voxel grid)",
                surface_reflectance_assumption="Invariant to reflectance (purely geometric silhouette boundary)",
                handles_transparency_and_refraction=True,
                sfm_pose_compatibility="Directly uses camera projection matrices P = K[R|t] and binary masks",
                installation_complexity="Zero (already implemented in project's existing environment)",
                dataset_feasibility_verdict="DEGRADED (Orbit Constrained)",
                technical_justification=(
                    "Theoretically the best representation for transparent objects because it completely bypasses "
                    "the Lambertian assumption and uses YOLO masks. However, on this dataset, camera orbit spans "
                    "only 57.3 degrees (15.9% of circle). Visual hull back-projection with a 57-degree wedge creates "
                    "an unconstrained elongated frustum stretching to infinity along the viewing direction. "
                    "Requires >= 180-360 degree orbit."
                ),
            ),
            ReconstructionMethodEvaluation(
                method_name="Neural Radiance Fields (NeRF / 3D Gaussian Splatting)",
                reconstruction_paradigm="Differentiable Volumetric Radiance / 3D Gaussian Primitive Optimization",
                windows_compatibility="Supported (PyTorch / CUDA build chains)",
                hardware_requirement="NVIDIA GPU REQUIRED (>= 8-12 GB dedicated VRAM)",
                max_ram_footprint="12 - 16 GB RAM + 8 GB VRAM",
                surface_reflectance_assumption="View-dependent specular (standard); Refraction requires specialized models (Ref-NeRF)",
                handles_transparency_and_refraction=True,
                sfm_pose_compatibility="Compatible with standard transforms.json / COLMAP poses",
                installation_complexity="Very High (CUDA toolkit, Visual Studio C++ build tools, large wheels)",
                dataset_feasibility_verdict="INFEASIBLE",
                technical_justification=(
                    "Completely incompatible with CPU-only 16 GB RAM environment: training NeRF or 3DGS on CPU "
                    "takes tens of hours and lacks optimized rasterization CUDA kernels. Furthermore, standard NeRF "
                    "models straight ray paths and fails on refraction without Ref-NeRF, and 57-degree orbit causes "
                    "massive floaters outside the viewing cone."
                ),
            ),
            ReconstructionMethodEvaluation(
                method_name="Parametric Surface Prior (Cylindrical / Spline Fitting)",
                reconstruction_paradigm="Geometric primitive fitting constrained by silhouette rays and 3D inliers",
                windows_compatibility="100% Native Python / SciPy / NumPy",
                hardware_requirement="CPU-only (lightweight numerical optimization, < 5 seconds)",
                max_ram_footprint="< 1 GB RAM",
                surface_reflectance_assumption="Invariant to surface texture (relies on geometric symmetry prior)",
                handles_transparency_and_refraction=True,
                sfm_pose_compatibility="Directly aligns with existing camera poses, 3D centroids, and mask bounding rays",
                installation_complexity="Zero (pure Python scientific stack)",
                dataset_feasibility_verdict="FEASIBLE (with symmetry prior)",
                technical_justification=(
                    "Tabletop water bottles are rotational solids (figures of revolution). By fitting a parametric "
                    "axis of revolution to the 14 high-confidence 3D points and bounding the radial profile "
                    "using multi-view silhouette rays, a complete, watertight 3D bottle model can be reconstructed "
                    "even with only 57.3 degrees of camera orbit, overcoming texturelessness and transparency."
                ),
            ),
        ]


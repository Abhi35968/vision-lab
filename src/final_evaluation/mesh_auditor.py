"""
Mesh Auditor (Phase 5.6).
Performs topological, geometric, and format equivalence auditing for reconstructed 3D meshes.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import trimesh


@dataclass
class MeshAuditResult:
    """Quantitative results of mesh geometric and topological audit."""
    num_vertices: int
    num_faces: int
    num_edges_unique: int
    num_directed_edges: int
    num_connected_components: int
    is_watertight: bool
    is_edge_manifold: bool
    num_boundary_edges: int
    num_non_manifold_edges: int
    num_degenerate_faces: int
    min_face_area: float
    max_face_area: float
    num_duplicate_vertices: int
    num_duplicate_faces: int
    consistent_winding_edges_count: int
    inconsistent_winding_edges_count: int
    winding_consistency_ratio: float
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float
    dim_x: float
    dim_y: float
    dim_z: float
    enclosed_volume: float
    euler_characteristic: int
    genus: int
    ply_obj_max_vertex_diff: float
    ply_obj_equivalent: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MeshAuditor:
    """Audits 3D mesh files for geometric validity, topological consistency, and format fidelity."""

    def __init__(self, tolerance: float = 1e-5):
        self.tolerance = tolerance

    def audit(self, ply_path: Path, obj_path: Optional[Path] = None) -> MeshAuditResult:
        """Audits PLY mesh and optionally compares with OBJ counterpart."""
        ply_path = Path(ply_path)
        if not ply_path.exists():
            raise FileNotFoundError(f"PLY mesh not found at {ply_path}")

        mesh_ply = trimesh.load(str(ply_path), process=False)

        # Basic topology
        v = np.asarray(mesh_ply.vertices, dtype=np.float64)
        f = np.asarray(mesh_ply.faces, dtype=np.int64)

        num_vertices = len(v)
        num_faces = len(f)

        # Unique undirected edges & edge manifold status
        edges = mesh_ply.edges_sorted
        unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
        num_edges_unique = len(unique_edges)
        num_directed_edges = len(mesh_ply.edges)

        boundary_edges = int(np.sum(counts == 1))
        non_manifold_edges = int(np.sum(counts > 2))
        is_edge_manifold = bool(boundary_edges == 0 and non_manifold_edges == 0 and np.all(counts == 2))

        # Connected components
        bodies = trimesh.graph.connected_components(mesh_ply.face_adjacency)
        num_components = len(bodies)

        # Watertightness
        is_watertight = bool(mesh_ply.is_watertight)

        # Face degeneracy (zero area or degenerate vertex indices)
        areas = mesh_ply.area_faces
        degenerate_area = int(np.sum(areas <= 1e-10))
        degenerate_indices = int(np.sum(
            (f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 2] == f[:, 0])
        ))
        num_degenerate_faces = max(degenerate_area, degenerate_indices)
        min_face_area = float(np.min(areas)) if len(areas) > 0 else 0.0
        max_face_area = float(np.max(areas)) if len(areas) > 0 else 0.0

        # Duplicate elements
        rounded_v = np.round(v, decimals=6)
        num_duplicate_vertices = num_vertices - len(np.unique(rounded_v, axis=0))

        sorted_f = np.sort(f, axis=1)
        num_duplicate_faces = num_faces - len(np.unique(sorted_f, axis=0))

        # Winding consistency analysis
        # Check opposite edge traversals across faces
        edge_dir_counts: Dict[Tuple[int, int], int] = {}
        for face in f:
            for i in range(3):
                e = (int(face[i]), int(face[(i + 1) % 3]))
                edge_dir_counts[e] = edge_dir_counts.get(e, 0) + 1

        consistent_edges = 0
        inconsistent_edges = 0
        visited_undirected = set()

        for (u, w) in edge_dir_counts:
            edge_undirected = tuple(sorted((u, w)))
            if edge_undirected in visited_undirected:
                continue
            visited_undirected.add(edge_undirected)

            count_forward = edge_dir_counts.get((u, w), 0)
            count_reverse = edge_dir_counts.get((w, u), 0)

            if count_forward == 1 and count_reverse == 1:
                consistent_edges += 1
            else:
                inconsistent_edges += 1

        total_tested_edges = consistent_edges + inconsistent_edges
        consistency_ratio = (
            float(consistent_edges) / float(total_tested_edges)
            if total_tested_edges > 0 else 1.0
        )

        # Bounds and dimensions
        min_coords = np.min(v, axis=0)
        max_coords = np.max(v, axis=0)
        dims = max_coords - min_coords

        # Enclosed volume (using divergence theorem)
        try:
            enclosed_volume = float(mesh_ply.volume)
        except Exception:
            enclosed_volume = 0.0

        # Euler characteristic and genus
        # For closed 2-manifold surface: chi = V - E + F = 2 - 2g
        euler_char = int(num_vertices - num_edges_unique + num_faces)
        genus = int(max(0, (2 - euler_char) // 2))

        # PLY vs OBJ equivalence check
        ply_obj_max_diff = 0.0
        ply_obj_equivalent = True

        if obj_path is not None and Path(obj_path).exists():
            mesh_obj = trimesh.load(str(obj_path), process=False)
            v_obj = np.asarray(mesh_obj.vertices, dtype=np.float64)
            f_obj = np.asarray(mesh_obj.faces, dtype=np.int64)

            if len(v_obj) != num_vertices or len(f_obj) != num_faces:
                ply_obj_equivalent = False
                ply_obj_max_diff = 999.0
            else:
                ply_obj_max_diff = float(np.max(np.abs(v - v_obj)))
                ply_obj_equivalent = bool(ply_obj_max_diff <= self.tolerance)

        return MeshAuditResult(
            num_vertices=num_vertices,
            num_faces=num_faces,
            num_edges_unique=num_edges_unique,
            num_directed_edges=num_directed_edges,
            num_connected_components=num_components,
            is_watertight=is_watertight,
            is_edge_manifold=is_edge_manifold,
            num_boundary_edges=boundary_edges,
            num_non_manifold_edges=non_manifold_edges,
            num_degenerate_faces=num_degenerate_faces,
            min_face_area=min_face_area,
            max_face_area=max_face_area,
            num_duplicate_vertices=num_duplicate_vertices,
            num_duplicate_faces=num_duplicate_faces,
            consistent_winding_edges_count=consistent_edges,
            inconsistent_winding_edges_count=inconsistent_edges,
            winding_consistency_ratio=consistency_ratio,
            min_x=float(min_coords[0]),
            max_x=float(max_coords[0]),
            min_y=float(min_coords[1]),
            max_y=float(max_coords[1]),
            min_z=float(min_coords[2]),
            max_z=float(max_coords[2]),
            dim_x=float(dims[0]),
            dim_y=float(dims[1]),
            dim_z=float(dims[2]),
            enclosed_volume=enclosed_volume,
            euler_characteristic=euler_char,
            genus=genus,
            ply_obj_max_vertex_diff=ply_obj_max_diff,
            ply_obj_equivalent=ply_obj_equivalent,
        )

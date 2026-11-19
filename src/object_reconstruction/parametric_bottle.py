"""
Parametric Bottle Geometry Generator (Phase 5.5).
Generates a smooth surface of revolution with anatomical sections (base, body, shoulder, neck, cap)
and transforms local geometry into the calibrated 3D Structure from Motion world coordinate frame.
"""

from typing import List, Tuple
import numpy as np

from src.object_reconstruction.bottle_models import (
    BottleParameters,
    BottleProfileSample,
    Mesh3D,
)


class ParametricBottle:
    """
    Constructs a watertight, surface-of-revolution 3D triangular mesh of a bottle.
    """

    def __init__(self, params: BottleParameters):
        self.params = params

    def generate_profile(self) -> List[BottleProfileSample]:
        """
        Samples the 1D axial radial profile r(z) for z in [0, H] where
        z = 0 is the base of the bottle and z = H is the top of the cap.
        """
        H = self.params.total_height
        R_body = self.params.body_radius
        R_neck = R_body * self.params.neck_radius_ratio
        R_cap = R_body * self.params.cap_radius_ratio
        R_base = R_body * self.params.base_radius_ratio

        z_base = H * self.params.base_chamfer_z_ratio
        z_shoulder = H * self.params.shoulder_z_ratio
        z_neck = H * self.params.neck_z_ratio
        z_cap = H * self.params.cap_z_ratio

        samples: List[BottleProfileSample] = []
        zs = np.linspace(0.0, H, self.params.axial_resolution)

        for z in zs:
            if z <= z_base:
                # Rounded base chamfer: smooth quarter-circle transition from R_base to R_body
                t = z / max(1e-6, z_base)
                r = R_base + (R_body - R_base) * np.sin(0.5 * np.pi * t)
                region = "base"
            elif z <= z_shoulder:
                # Main cylindrical body
                r = R_body
                region = "body"
            elif z <= z_neck:
                # Smooth S-curve shoulder taper (cosine transition from R_body to R_neck)
                t = (z - z_shoulder) / max(1e-6, z_neck - z_shoulder)
                # Cosine blend: 1 at t=0, 0 at t=1
                blend = 0.5 * (1.0 + np.cos(np.pi * t))
                r = R_neck + (R_body - R_neck) * blend
                region = "shoulder"
            elif z <= z_cap:
                # Cylindrical neck
                r = R_neck
                region = "neck"
            else:
                # Cap section: slight stepped ridge
                r = R_cap
                region = "cap"

            samples.append(BottleProfileSample(z_axial=round(float(z), 5), radius=round(float(r), 5), region=region))

        return samples

    def build_mesh(self) -> Mesh3D:
        """
        Revolves the 1D profile around the longitudinal axis and transforms into SfM world space.
        Produces a completely closed, watertight 2-manifold triangular mesh.
        """
        profile = self.generate_profile()
        n_axial = len(profile)
        n_angular = self.params.angular_resolution
        thetas = np.linspace(0, 2 * np.pi, n_angular, endpoint=False)

        # 1. Compute World Coordinate Frame for Bottle
        # Cap center is given in world coordinates
        c_cap = np.asarray(self.params.cap_center_world, dtype=np.float64)

        # Base nominal direction pointing downwards (+Y in OpenCV camera/world convention)
        ax_deg = np.radians(self.params.axis_tilt_x_deg)
        az_deg = np.radians(self.params.axis_tilt_z_deg)

        # Rotation around X then Z
        Rx = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(ax_deg), -np.sin(ax_deg)],
            [0.0, np.sin(ax_deg), np.cos(ax_deg)]
        ])
        Rz = np.array([
            [np.cos(az_deg), -np.sin(az_deg), 0.0],
            [np.sin(az_deg), np.cos(az_deg), 0.0],
            [0.0, 0.0, 1.0]
        ])
        R_tilt = Rz @ Rx

        # Nominal downward vector from cap to base
        nominal_down = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        u_down = R_tilt @ nominal_down
        u_down /= np.linalg.norm(u_down)

        # Bottle axis points from base to cap (upwards):
        u_up = -u_down

        # Construct orthonormal basis for radial plane (u_r1, u_r2, u_up)
        # Choose a reference vector not collinear with u_up
        ref = np.array([1.0, 0.0, 0.0]) if abs(u_up[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
        u_r1 = np.cross(u_up, ref)
        u_r1 /= np.linalg.norm(u_r1)
        u_r2 = np.cross(u_up, u_r1)
        u_r2 /= np.linalg.norm(u_r2)

        # Base center in world space:
        c_base = c_cap + self.params.total_height * u_down

        # 2. Generate Vertices
        vertices: List[List[float]] = []

        # Ring vertices: index = ring_idx * n_angular + ang_idx
        for sample in profile:
            z_val = sample.z_axial
            r_val = sample.radius
            ring_center = c_base + z_val * u_up
            for th in thetas:
                pt = ring_center + r_val * (np.cos(th) * u_r1 + np.sin(th) * u_r2)
                vertices.append(pt.tolist())

        # Top cap center vertex (index = n_axial * n_angular)
        top_center_idx = len(vertices)
        vertices.append((c_base + self.params.total_height * u_up).tolist())

        # Bottom base center vertex (index = n_axial * n_angular + 1)
        bottom_center_idx = len(vertices)
        vertices.append(c_base.tolist())

        # 3. Generate Triangular Faces with Consistent Outward Winding
        faces: List[List[int]] = []

        # Side quad strips between adjacent rings
        for i in range(n_axial - 1):
            ring_curr = i * n_angular
            ring_next = (i + 1) * n_angular
            for j in range(n_angular):
                j_next = (j + 1) % n_angular
                v0 = ring_curr + j
                v1 = ring_curr + j_next
                v2 = ring_next + j_next
                v3 = ring_next + j
                # Triangle 1 (v0, v1, v2)
                faces.append([v0, v1, v2])
                # Triangle 2 (v0, v2, v3)
                faces.append([v0, v2, v3])

        # Top cap fan (connected to last ring: i = n_axial - 1)
        last_ring = (n_axial - 1) * n_angular
        for j in range(n_angular):
            j_next = (j + 1) % n_angular
            v0 = last_ring + j
            v1 = last_ring + j_next
            # CCW viewed from outside top:
            faces.append([v0, top_center_idx, v1])

        # Bottom base fan (connected to first ring: i = 0)
        first_ring = 0
        for j in range(n_angular):
            j_next = (j + 1) % n_angular
            v0 = first_ring + j
            v1 = first_ring + j_next
            # CCW viewed from outside bottom:
            faces.append([v0, v1, bottom_center_idx])

        return Mesh3D(
            vertices=np.array(vertices, dtype=np.float64),
            faces=np.array(faces, dtype=np.int32),
        )


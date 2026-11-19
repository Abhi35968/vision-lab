"""
GLB Export and Model Sharing Module (Phase 5)
Handles:
1. Exporting clean, self-contained Binary glTF (.glb) files with embedded PBR materials and vertex colors.
2. Generating scannable QR codes pointing strictly to reachable network HTTP/HTTPS endpoints.
3. Guards against invalid local filesystem URI schemes for QR sharing.
4. Provides format inspection, file size telemetry, and metadata extraction.
"""

from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import socket
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import qrcode
import trimesh

logger = logging.getLogger("GLBExporter")


@dataclass
class ExportResult:
    """Telemetry and file properties of the exported GLB model."""
    glb_path: str
    file_size_kb: float
    num_vertices: int
    num_faces: int
    has_embedded_texture: bool
    has_vertex_colors: bool
    is_watertight: bool
    qr_code_image_path: Optional[str]
    reachable_share_url: Optional[str]

    def to_dict(self) -> dict:
        return asdict(self)


class GLBExporter:
    """
    Exports 3D meshes to production-standard GLB format and generates mobile sharing QR codes.
    """

    def __init__(self, default_output_dir: Union[str, Path] = "outputs/exported_models"):
        self.output_dir = Path(default_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_host_network_ip() -> str:
        """Discovers host LAN IP address for local network mobile device sharing."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    def export(
        self,
        mesh_or_trimesh: Union[trimesh.Trimesh, object],
        filename: str = "model_3d.glb",
        texture_image: Optional[Image.Image] = None,
        vertex_colors: Optional[np.ndarray] = None,
        share_url: Optional[str] = None,
    ) -> ExportResult:
        """
        Exports a mesh to a self-contained GLB binary file and optional sharing QR code.
        """
        out_path = self.output_dir / filename
        if not out_path.suffix.lower() == ".glb":
            out_path = out_path.with_suffix(".glb")

        # Convert to Trimesh if custom Mesh3D or Scene was passed
        if isinstance(mesh_or_trimesh, trimesh.Scene):
            if hasattr(mesh_or_trimesh, "to_geometry"):
                tm = mesh_or_trimesh.to_geometry()
            else:
                dumped = mesh_or_trimesh.dump(concatenate=True)
                tm = dumped[0] if isinstance(dumped, list) else dumped
        elif not isinstance(mesh_or_trimesh, trimesh.Trimesh):
            v = np.asarray(mesh_or_trimesh.vertices, dtype=np.float64)
            f = np.asarray(mesh_or_trimesh.faces, dtype=np.int32)
            tm = trimesh.Trimesh(vertices=v, faces=f, process=False)
        else:
            tm = mesh_or_trimesh.copy()

        has_v_colors = False
        if vertex_colors is not None and len(vertex_colors) == len(tm.vertices):
            tm.visual.vertex_colors = np.asarray(vertex_colors, dtype=np.uint8)
            has_v_colors = True
        elif hasattr(tm.visual, "vertex_colors") and tm.visual.vertex_colors is not None:
            has_v_colors = True

        has_tex = False
        if texture_image is not None:
            mat = trimesh.visual.material.PBRMaterial(
                name="model_material",
                baseColorTexture=texture_image,
                metallicFactor=0.1,
                roughnessFactor=0.6,
            )
            tm.visual.material = mat
            has_tex = True
        elif hasattr(tm.visual, "material") and tm.visual.material is not None:
            has_tex = True

        # Export binary GLB
        glb_bytes = tm.export(file_type="glb")
        with open(out_path, "wb") as f:
            f.write(glb_bytes)

        file_size_kb = round(out_path.stat().st_size / 1024.0, 2)
        is_watertight = bool(tm.is_watertight)

        # Generate Share QR Code if reachable URL provided
        qr_path_str: Optional[str] = None
        if share_url:
            qr_img, qr_path = self.generate_qr_code(share_url, qr_filename=f"{out_path.stem}_qr.png")
            qr_path_str = str(qr_path)

        return ExportResult(
            glb_path=str(out_path),
            file_size_kb=file_size_kb,
            num_vertices=len(tm.vertices),
            num_faces=len(tm.faces),
            has_embedded_texture=has_tex,
            has_vertex_colors=has_v_colors,
            is_watertight=is_watertight,
            qr_code_image_path=qr_path_str,
            reachable_share_url=share_url,
        )

    def generate_qr_code(
        self,
        url: str,
        qr_filename: str = "model_qr.png",
    ) -> Tuple[Image.Image, Path]:
        """
        Generates a mobile-scannable QR code.
        Strictly enforces reachable network URLs (http:// or https://) and rejects local filesystem paths.
        """
        clean_url = str(url).strip()

        # Strict validation: prevent inaccessible local paths
        if clean_url.startswith("file://") or ":\\" in clean_url or clean_url.startswith("/") or clean_url.startswith("./"):
            raise ValueError(
                f"Invalid QR code URL: '{clean_url}'. QR codes must point to reachable network HTTP/HTTPS addresses, "
                f"not local computer filesystem paths."
            )

        if not (clean_url.startswith("http://") or clean_url.startswith("https://")):
            # Auto-prefix http://
            clean_url = f"http://{clean_url}"

        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=10,
            border=4,
        )
        qr.add_data(clean_url)
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color="#1D3557", back_color="#FFFFFF")
        pil_img = getattr(qr_img, "_img", qr_img)

        out_qr_path = self.output_dir / qr_filename
        qr_img.save(out_qr_path)

        return pil_img, out_qr_path

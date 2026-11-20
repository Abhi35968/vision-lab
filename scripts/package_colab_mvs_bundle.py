"""
Bundles keyframes, COLMAP sparse reconstruction, and MVS runner into a lightweight zip
for one-click execution on Google Colab (T4 GPU).
"""

from pathlib import Path
import shutil
import zipfile

VALIDATION_DIR = Path("outputs/deployment_live_validation")
KEYFRAMES_DIR = VALIDATION_DIR / "keyframes"
COLMAP_SPARSE_DIR = VALIDATION_DIR / "colmap/sparse/0"
SCRIPT_PATH = Path("scripts/colab_patchmatch_mvs.py")
OUT_ZIP = Path("outputs/colab_mvs_input.zip")

def package():
    OUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
    if OUT_ZIP.exists():
        OUT_ZIP.unlink()

    print(f"Packaging Colab bundle into {OUT_ZIP}...")
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        # 1. Add MVS runner script
        if SCRIPT_PATH.exists():
            z.write(SCRIPT_PATH, arcname="colab_patchmatch_mvs.py")
            print(f"Added {SCRIPT_PATH.name}")

        # 2. Add Sparse COLMAP files
        for f in COLMAP_SPARSE_DIR.glob("*.bin"):
            z.write(f, arcname=f"sparse/{f.name}")
            print(f"Added sparse/{f.name}")

        # 3. Add Keyframes
        kf_count = 0
        for f in KEYFRAMES_DIR.glob("*.jpg"):
            z.write(f, arcname=f"keyframes/{f.name}")
            kf_count += 1
        print(f"Added {kf_count} keyframes")

    size_mb = OUT_ZIP.stat().st_size / (1024 * 1024)
    print(f"Bundle created successfully! Size: {size_mb:.2f} MB at {OUT_ZIP}")

if __name__ == "__main__":
    package()

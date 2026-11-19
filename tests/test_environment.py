"""
Environment and Dependency Verification Script
3D Scene and Object Reconstruction Using SfM and Deep Learning
"""

import sys
import shutil
import subprocess

def run_component_check(name, test_fn):
    """Executes a component test function and prints PASS, FAIL, or NOT AVAILABLE."""
    try:
        status, details = test_fn()
        symbol = "[ PASS ]" if status == "PASS" else ("[ NOT AVAILABLE ]" if status == "NOT AVAILABLE" else "[ FAIL ]")
        print(f"{symbol:<18} | {name:<26} | {details}")
        return status
    except Exception as e:
        print(f"{'[ FAIL ]':<18} | {name:<26} | Exception: {str(e)}")
        return "FAIL"

def check_python_version():
    ver = sys.version_info
    ver_str = f"{ver.major}.{ver.minor}.{ver.micro}"
    if ver.major == 3 and ver.minor in (10, 11):
        return "PASS", f"Python {ver_str} (Recommended 3.10/3.11)"
    elif ver.major == 3 and ver.minor >= 8:
        return "PASS", f"Python {ver_str} (Compatible)"
    else:
        return "FAIL", f"Python {ver_str} (Unsupported version)"

def check_numpy():
    import numpy as np
    return "PASS", f"v{np.__version__}"

def check_opencv():
    import cv2
    return "PASS", f"v{cv2.__version__}"

def check_pytorch():
    import torch
    return "PASS", f"v{torch.__version__}"

def check_pytorch_device():
    import torch
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        return "PASS", f"CUDA Available: {device_name} ({torch.cuda.device_count()} GPU(s))"
    else:
        return "PASS", "CPU-only mode (No NVIDIA CUDA detected, CPU inference enabled)"

def check_ultralytics():
    import ultralytics
    return "PASS", f"v{ultralytics.__version__}"

def check_open3d():
    import open3d as o3d
    return "PASS", f"v{o3d.__version__}"

def check_scipy():
    import scipy
    return "PASS", f"v{scipy.__version__}"

def check_sklearn():
    import sklearn
    return "PASS", f"v{sklearn.__version__}"

def check_matplotlib():
    import matplotlib
    return "PASS", f"v{matplotlib.__version__}"

def check_colmap():
    colmap_path = shutil.which("colmap")
    if not colmap_path:
        return "NOT AVAILABLE", "COLMAP not found in system PATH (Manual Windows setup required)"
    
    try:
        res = subprocess.run(["colmap", "-h"], capture_output=True, text=True, timeout=5)
        # Check if output contains COLMAP banner
        output = res.stdout or res.stderr
        first_line = output.splitlines()[0] if output.splitlines() else "COLMAP present"
        return "PASS", f"{first_line} ({colmap_path})"
    except Exception as e:
        return "FAIL", f"COLMAP error: {e}"

def main():
    print("=" * 78)
    print(" 3DCV PROJECT: ENVIRONMENT & DEPENDENCY VERIFICATION")
    print("=" * 78)
    print(f"{'STATUS':<18} | {'COMPONENT':<26} | {'DETAILS'}")
    print("-" * 78)

    tests = [
        ("Python Version", check_python_version),
        ("NumPy", check_numpy),
        ("OpenCV (cv2)", check_opencv),
        ("PyTorch", check_pytorch),
        ("PyTorch Device / CUDA", check_pytorch_device),
        ("Ultralytics YOLO", check_ultralytics),
        ("Open3D", check_open3d),
        ("SciPy", check_scipy),
        ("Scikit-learn", check_sklearn),
        ("Matplotlib", check_matplotlib),
        ("COLMAP CLI", check_colmap),
    ]

    results = {}
    for name, fn in tests:
        results[name] = run_component_check(name, fn)

    print("-" * 78)
    passed_count = sum(1 for s in results.values() if s == "PASS")
    na_count = sum(1 for s in results.values() if s == "NOT AVAILABLE")
    failed_count = sum(1 for s in results.values() if s == "FAIL")

    print(f"Summary: {passed_count} PASS, {na_count} NOT AVAILABLE, {failed_count} FAIL")
    print("=" * 78)

    if failed_count > 0:
        sys.exit(1)
    sys.exit(0)

def test_environment_dependencies():
    """Verify that essential core Python modules pass verification."""
    assert check_python_version()[0] == "PASS"
    assert check_numpy()[0] == "PASS"
    assert check_opencv()[0] == "PASS"
    assert check_pytorch()[0] == "PASS"
    assert check_scipy()[0] == "PASS"

if __name__ == "__main__":
    main()


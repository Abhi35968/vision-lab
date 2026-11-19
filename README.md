# 3D Scene and Object Reconstruction Using Structure from Motion and Deep Learning

A course project that takes a handheld phone video, reconstructs the scene and camera path with Structure from Motion (SfM), finds an object in the frames with YOLO, lifts it into 3D, and classifies 3D shapes with PointNet.

**Results:** see [docs/FINAL_RESULTS.md](docs/FINAL_RESULTS.md) for every table and figure.

## Pipeline

1. **Preprocessing** (`src/preprocessing/`): extracts frames, rejects blurred and near-duplicate ones, selects keyframes.
2. **Structure from Motion** (`src/sfm/`): own OpenCV pipeline with SIFT or ORB features, essential-matrix RANSAC, triangulation, PnP registration and bundle adjustment. COLMAP (through `pycolmap`) is used as the reference engine.
3. **2D perception** (`src/detection/`, `src/segmentation/`): pretrained YOLO detection and instance segmentation.
4. **2D-to-3D association** (`src/association/`, `src/refinement/`): assigns sparse 3D points to the object using the masks, filters them, and computes 3D bounding boxes.
5. **Object model** (`src/object_reconstruction/`, `src/final_evaluation/`): fits a parametric bottle mesh to the silhouettes and evaluates it.
6. **3D classification** (`src/classification_3d/`): PointNet trained on ModelNet40.
7. **Visualisation** (`app.py`, `outputs/objects_3d/`): Streamlit dashboard, static figures and an interactive 3D scene.

## Headline results

| Item | Result |
|---|---|
| Development video | 9.6 s video of a bottle, 47 keyframes at 1080×1920 |
| Own SfM (stored run) | 32 of 47 cameras, 2,077 points, median reprojection error 1.03 px |
| COLMAP | 47 of 47 cameras, 4,853 points, mean reprojection error 0.79 px |
| Object segmentation | Bottle found in 40 of 47 keyframes |
| Object localisation (COLMAP) | 84% of object points stay inside the mask in held-out views |
| PointNet on ModelNet40 | 84.6% test accuracy (40 epochs) |
| Ten object videos | All 10 reconstructed by COLMAP (40 of 40 cameras each); 3D object cloud and box for 9; 92% of object points stay inside the mask in held-out views |
| Tests | 212 passing |

## Setup (Windows, Python 3.12)

```powershell
cd D:\Class_material\3Dcomputer\3DCV_proj
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1

# PyTorch first: GPU build (NVIDIA) ...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# ... or CPU build
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
python main.py --phase test_env
```

`test_env` reports the COLMAP command-line tool as not available. That is expected: COLMAP is used through the `pycolmap` package, which needs no separate install.

If PowerShell blocks the activation script, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` first.

## Running

```powershell
python -m pytest tests -q                 # unit tests
python -m experiments.run_all             # all report experiments and figures (about 40 minutes)
python -m experiments.objectron_batch     # only the ten-video evaluation
streamlit run demo_app.py                 # project demo (or double-click run_demo.bat)
streamlit run app.py                      # upload-and-process dashboard
python main.py --phase classify_3d        # PointNet on the object points
python main.py --help                     # every pipeline phase
```

### PointNet training

The trained weights are in `models/weights/pointnet_modelnet40.pth`. To retrain:

1. Download `modelnet40_ply_hdf5_2048.zip` (435 MB) and extract it to `data/modelnet40/` so that `data/modelnet40/modelnet40_ply_hdf5_2048/ply_data_train0.h5` exists.
2. Run `python -m experiments.train_pointnet --epochs 40`.

A GPU is recommended. `pointnet_colab.zip` contains the files needed to run the same script on Google Colab.

## Directory structure

```
3DCV_proj/
├── app.py                     Streamlit dashboard
├── main.py                    Command-line runner for every pipeline phase
├── config/                    Pipeline settings, camera intrinsics, PointNet class list
├── data/
│   ├── raw_videos/            Input video
│   └── modelnet40/            ModelNet40 point clouds (for PointNet training)
├── docs/
│   ├── FINAL_RESULTS.md       Results, figures and limitations
│   ├── phase2_sfm.md          Two-view SfM notes
│   └── phase3_incremental_sfm.md
├── experiments/               Report experiments (run_all.py runs them all)
├── Objectron_10_Videos/       Ten object videos for the multi-object evaluation
├── models/weights/            Trained PointNet weights
├── outputs/
│   ├── real_video_validation/ Keyframes and preprocessing reports
│   ├── real_video_sfm/        Own SfM reconstructions (focal_1300 is the main one)
│   ├── colmap/                COLMAP reconstruction and object localisation
│   ├── object_segmentation/   YOLO masks
│   ├── object_association/    2D-to-3D association
│   ├── object_refinement/     Filtered object points
│   ├── objects_3d/            Object point cloud, 3D boxes, scene figures
│   ├── object_reconstruction/ Fitted bottle mesh
│   ├── final_evaluation/      Mesh evaluation
│   ├── classification_3d/     PointNet results on the real object
│   ├── objectron/             Results for the ten object videos
│   └── experiments/           Tables and figures for the report experiments
├── src/                       Pipeline source code
└── tests/                     Unit tests
```

## Known limitations

- Every video shows one object, so several objects are never separated within a single scene.
- The camera is not calibrated and there is no metric scale; 3D dimensions are in arbitrary units.
- The project's own SfM gives different results from run to run; COLMAP is the reliable reconstruction.
- The object point cloud of the development bottle is sparse (29 points) because its body has no texture; textured objects in the ten-video run give hundreds to thousands of points.
- PointNet does not classify the reconstructed real objects correctly.

Details are in [docs/FINAL_RESULTS.md](docs/FINAL_RESULTS.md), section 9.

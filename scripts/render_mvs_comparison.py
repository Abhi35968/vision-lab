"""
Render Multi-Angle Visual Comparisons of all 3 reconstruction methods:
1. Baseline: Depth Anything + Gradient Filter
2. Open3D TSDF Volume Integration
3. COLMAP CUDA PatchMatch MVS
"""
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def render_model_views(mesh_or_pcd, title, is_mesh=False):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=800, height=800)
    vis.add_geometry(mesh_or_pcd)
    
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.95, 0.95, 0.97])
    opt.point_size = 2.0
    if is_mesh:
        opt.mesh_show_wireframe = False
        opt.mesh_show_back_face = True
        
    ctr = vis.get_view_control()
    
    # 4 distinct viewing angles: Front, Side, Perspective, Top
    angles = [
        {"zoom": 0.75, "front": [0, 0, -1], "up": [0, -1, 0]},
        {"zoom": 0.75, "front": [-1, 0, 0], "up": [0, -1, 0]},
        {"zoom": 0.75, "front": [0.6, 0.4, -0.7], "up": [0, -1, 0]},
        {"zoom": 0.75, "front": [0, -1, 0.05], "up": [0, 0, -1]}
    ]
    
    snapshots = []
    for ang in angles:
        ctr.set_zoom(ang["zoom"])
        ctr.set_front(ang["front"])
        ctr.set_up(ang["up"])
        vis.poll_events()
        vis.update_renderer()
        img = vis.capture_screen_float_buffer(do_render=True)
        snapshots.append(np.asarray(img))
        
    vis.destroy_window()
    return snapshots

def main():
    root = Path(".")
    out_dir = root / "outputs" / "colab_mvs_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load 3 point clouds
    pcd1 = o3d.io.read_point_cloud("outputs/deployment_live_validation/final_object_only_cloud.ply")
    pcd2 = o3d.io.read_point_cloud("outputs/deployment_live_validation/tsdf_fusion/tsdf_volume_point_cloud.ply")
    pcd3 = o3d.io.read_point_cloud("outputs/colab_mvs_results/fused.ply")
    
    views1 = render_model_views(pcd1, "Baseline")
    views2 = render_model_views(pcd2, "TSDF")
    views3 = render_model_views(pcd3, "COLMAP MVS")
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    row_titles = [
        "1. Baseline: Depth Anything + Gradient Filter\n(112,767 pts, Radial Std Err: 50.4%)",
        "2. Open3D TSDF Volume Integration\n(53,027 pts, Radial Std Err: 84.0%)",
        "3. COLMAP CUDA PatchMatch MVS\n(504,386 pts, Radial Std Err: 24.3%)"
    ]
    col_titles = ["Front View", "Side View", "Perspective View", "Top-Down (Circularity)"]
    
    all_views = [views1, views2, views3]
    for row_idx in range(3):
        for col_idx in range(4):
            ax = axes[row_idx, col_idx]
            ax.imshow(all_views[row_idx][col_idx])
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=12, fontweight="bold", pad=8)
        axes[row_idx, 0].text(-0.1, 0.5, row_titles[row_idx], transform=axes[row_idx, 0].transAxes,
                              rotation=90, va='center', ha='right', fontsize=11, fontweight="bold")
                              
    plt.tight_layout()
    comparison_img = out_dir / "mvs_3way_visual_comparison.png"
    plt.savefig(comparison_img, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved visual comparison to {comparison_img}")

if __name__ == "__main__":
    main()

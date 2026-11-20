"""
Render Multi-Angle Visual Comparisons of all 3 reconstruction meshes:
1. Baseline Poisson Mesh
2. Open3D TSDF Marching Cubes Mesh
3. COLMAP PatchMatch Poisson Mesh
"""
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def render_mesh_views(mesh):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=800, height=800)
    
    # Paint default neutral tone if uncolored
    if not mesh.has_vertex_colors():
        mesh.paint_uniform_color([0.7, 0.75, 0.82])
    mesh.compute_vertex_normals()
    vis.add_geometry(mesh)
    
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.96, 0.96, 0.98])
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
    
    mesh1 = o3d.io.read_triangle_mesh("outputs/deployment_live_validation/final_reconstructed_mesh.ply")
    mesh2 = o3d.io.read_triangle_mesh("outputs/deployment_live_validation/tsdf_fusion/tsdf_volume_mesh.ply")
    mesh3 = o3d.io.read_triangle_mesh("outputs/colab_mvs_results/mvs_poisson_mesh.ply")
    
    views1 = render_mesh_views(mesh1)
    views2 = render_mesh_views(mesh2)
    views3 = render_mesh_views(mesh3)
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    row_titles = [
        "1. Baseline Poisson Mesh\n(148,817 verts, 296,527 faces)",
        "2. TSDF Marching Cubes Mesh\n(31,724 verts, 61,004 faces)",
        "3. COLMAP PatchMatch Poisson Mesh\n(242,096 verts, 481,373 faces)"
    ]
    col_titles = ["Front View", "Side View", "Perspective View", "Top-Down View"]
    
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
    comparison_img = out_dir / "mvs_mesh_3way_visual_comparison.png"
    plt.savefig(comparison_img, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved mesh visual comparison to {comparison_img}")

if __name__ == "__main__":
    main()

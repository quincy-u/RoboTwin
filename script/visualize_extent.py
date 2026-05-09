"""Visualize model_data.json extents+center overlaid on GLB mesh geometry.

Renders top/front/side orthographic projections of the mesh vertices with
the OBB box edges drawn on top. Saves a PNG so no display is needed.

Usage:
  python script/visualize_extent.py --name 001_bottle --id 2 [--out extent_vis.png]
"""
import argparse
import json
from pathlib import Path
import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_mesh_vertices(glb_path: str) -> np.ndarray:
    """Load GLB vertices with scene node transforms applied (Z-up world frame)."""
    scene = trimesh.load(glb_path, force="scene")
    try:
        mesh = trimesh.util.concatenate(scene.dump())
        return np.array(mesh.vertices)
    except Exception:
        return np.concatenate([np.array(g.vertices) for g in scene.geometry.values()])


def obb_corners(center, extents):
    half = np.array(extents) / 2.0
    signs = np.array([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],
                      [ 1,-1,-1],[ 1,-1,1],[ 1,1,-1],[ 1,1,1]], dtype=float)
    return np.array(center) + signs * half


EDGES = [
    (0,1),(1,3),(3,2),(2,0),  # bottom face
    (4,5),(5,7),(7,6),(6,4),  # top face
    (0,4),(1,5),(2,6),(3,7),  # verticals
]

# (axis_h, axis_v, title)
VIEWS = [
    (0, 1, "Top (X-Y)"),
    (0, 2, "Front (X-Z)"),
    (1, 2, "Side (Y-Z)"),
]


def draw_view(ax, verts, corners, aabb_corners, ah, av, title):
    ax.scatter(verts[:, ah], verts[:, av], s=0.3, alpha=0.3, color="steelblue")
    for a, b in EDGES:
        ax.plot([aabb_corners[a, ah], aabb_corners[b, ah]],
                [aabb_corners[a, av], aabb_corners[b, av]],
                color="gray", linewidth=1.0, linestyle="--", alpha=0.6)
        ax.plot([corners[a, ah], corners[b, ah]],
                [corners[a, av], corners[b, av]], "r-", linewidth=1.2)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlabel(["X", "Y", "Z"][ah])
    ax.set_ylabel(["X", "Y", "Z"][av])


def load_model_data(base: Path, mid: int) -> dict:
    """Load model_data for a given index, handling flat and URDF subdirectory layouts."""
    flat = base / f"model_data{mid}.json"
    if flat.exists():
        md = json.loads(flat.read_text())
        if not isinstance(md.get("scale", 1), list):
            md["scale"] = [md["scale"]] * 3
        return md

    subdirs = sorted(
        [d for d in base.iterdir() if d.is_dir() and d.name != "visual"],
        key=lambda d: int(d.name) if d.name.isdigit() else 0,
    )
    subdir = subdirs[mid] if mid < len(subdirs) else subdirs[0]
    md = json.loads((subdir / "model_data.json").read_text())
    if not isinstance(md.get("scale", 1), list):
        md["scale"] = [md["scale"]] * 3
    bb_file = subdir / "bounding_box.json"
    if bb_file.exists() and not md.get("extents"):
        bb = json.loads(bb_file.read_text())
        bb_min, bb_max = np.array(bb["min"]), np.array(bb["max"])
        md["extents"] = (bb_max - bb_min).tolist()
        md["center"] = ((bb_min + bb_max) / 2).tolist()
    return md


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Object dir name, e.g. 001_bottle")
    parser.add_argument("--id", required=True, type=int, help="Model variant index, e.g. 2")
    parser.add_argument("--out", default="extent_vis.png")
    args = parser.parse_args()

    base = Path("assets/objects") / args.name
    glb = base / "visual" / f"base{args.id}.glb"

    md = load_model_data(base, args.id)

    scale = np.array(md["scale"])
    center = np.array(md.get("center", [0, 0, 0])) * scale
    extents = np.array(md["extents"]) * scale

    verts_raw = load_mesh_vertices(str(glb))

    corners = obb_corners(center, extents)

    # Detect whether stored extents are in GLB Y-up or trimesh Z-up frame.
    # Use UNSCALED quantities so non-uniform scale doesn't corrupt the axis-swap test.
    # Rx(-90°) swaps Y and Z: stored [X,Y,Z]_GLB -> trimesh [X,Z,Y].
    # When frames differ, rotate VERTICES into stored (GLB) frame FIRST, then apply
    # scale — so scale[i] is applied to GLB axis i, matching the simulation convention.
    trimesh_ext_us = verts_raw.max(axis=0) - verts_raw.min(axis=0)
    ext_us = np.array(md["extents"])
    err_no_rot = np.linalg.norm(ext_us - trimesh_ext_us)
    err_rotated = np.linalg.norm(ext_us[[0, 2, 1]] - trimesh_ext_us)
    if err_rotated < err_no_rot:
        Rx_pos90 = np.array([[1,  0, 0],
                             [0,  0, -1],
                             [0,  1, 0]], dtype=float)
        verts = (Rx_pos90 @ verts_raw.T).T * scale  # rotate to GLB frame, then scale
    else:
        verts = verts_raw * scale

    aabb_min = verts.min(axis=0)
    aabb_max = verts.max(axis=0)
    aabb_corners = obb_corners((aabb_min + aabb_max) / 2, aabb_max - aabb_min)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (ah, av, title) in zip(axes, VIEWS):
        draw_view(ax, verts, corners, aabb_corners, ah, av, title)

    plt.suptitle(f"{args.name} id={args.id}", fontsize=8)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()

"""Visualize 3D OBB (center+extents from model_data.json) on the mesh for all assets.

For each asset variant, renders three orthographic projections (top/front/side)
of the visual mesh vertices with the OBB box edges overlaid. Saves a PNG per
variant under obb_visualizations/<asset>/<variant>.png and a contact-sheet
grid at obb_visualizations/index.html for browsing.

Usage:
    python script/visualize_obb_all.py                  # all assets
    python script/visualize_obb_all.py 001_bottle       # one asset
    python script/visualize_obb_all.py --out /tmp/vis   # custom out dir
    python script/visualize_obb_all.py --workers 8      # parallel
"""
import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ASSETS_ROOT = Path(__file__).resolve().parents[1] / "assets" / "objects"

EDGES = [
    (0, 1), (1, 3), (3, 2), (2, 0),
    (4, 5), (5, 7), (7, 6), (6, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]
VIEWS = [(0, 1, "Top (X-Y)"), (0, 2, "Front (X-Z)"), (1, 2, "Side (Y-Z)")]


def obb_corners(center, extents):
    half = np.asarray(extents) / 2.0
    signs = np.array([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
    ], dtype=float)
    return np.asarray(center) + signs * half


def load_mesh_vertices(glb_path: Path):
    scene = trimesh.load(str(glb_path), force="scene")
    try:
        mesh = trimesh.util.concatenate(scene.dump())
        return np.asarray(mesh.vertices)
    except Exception:
        return np.concatenate([np.asarray(g.vertices) for g in scene.geometry.values()])


def load_urdf_vertices(subdir: Path) -> np.ndarray:
    """Combine per-link visual meshes from a URDF in the URDF root frame.

    Walks parent joints to accumulate joint origin offsets; ignores joint
    rotations (uses the canonical/initial pose) since we just need vertex
    positions for OBB visualization.
    """
    urdf_path = subdir / "mobility.urdf"
    root = ET.parse(str(urdf_path)).getroot()
    joints_by_child = {
        j.find("child").get("link"): j for j in root.findall("joint")
    }

    def chain_offset(link_name: str) -> np.ndarray:
        off = np.zeros(3)
        while link_name in joints_by_child:
            j = joints_by_child[link_name]
            orig = j.find("origin")
            if orig is not None:
                xyz = list(map(float, orig.get("xyz", "0 0 0").split()))
                off = np.array(xyz) + off
            link_name = j.find("parent").get("link")
        return off

    parts = []
    for link in root.findall("link"):
        loff = chain_offset(link.get("name"))
        for visual in link.findall("visual"):
            origin = visual.find("origin")
            voff = np.zeros(3)
            if origin is not None:
                voff = np.array(list(map(float, origin.get("xyz", "0 0 0").split())))
            mesh_node = visual.find("geometry/mesh")
            if mesh_node is None:
                continue
            mesh_file = subdir / mesh_node.get("filename")
            if not mesh_file.exists():
                continue
            m = trimesh.load(str(mesh_file), force="mesh")
            if hasattr(m, "vertices"):
                parts.append(np.asarray(m.vertices) + loff + voff)
    if not parts:
        return np.zeros((0, 3))
    return np.concatenate(parts, axis=0)


def find_visual_mesh(asset_dir: Path, idx) -> Path | None:
    """Return the visual mesh file for the variant; falls back to collision."""
    suffix = "" if idx is None else str(idx)
    for folder in [asset_dir / "visual", asset_dir / "collision", asset_dir]:
        if not folder.exists():
            continue
        for fn in [f"base{suffix}.glb", f"textured{suffix}.obj"]:
            p = folder / fn
            if p.exists():
                return p
    return None


def is_articulated(asset_dir: Path) -> bool:
    return any(
        (sub / "mobility.urdf").exists()
        for sub in asset_dir.iterdir()
        if sub.is_dir() and sub.name != "visual"
    )


def variants_for_asset(asset_dir: Path):
    """Yield (variant_label, model_data_path, mesh_path)."""
    if is_articulated(asset_dir):
        subdirs = sorted(
            [d for d in asset_dir.iterdir() if d.is_dir() and d.name != "visual"],
            key=lambda d: int(d.name) if d.name.isdigit() else 0,
        )
        for i, sd in enumerate(subdirs):
            jf = sd / "model_data.json"
            if not jf.exists():
                continue
            mesh = find_visual_mesh(asset_dir, i)
            # mesh path None => signal URDF-based loader by passing subdir/mobility.urdf
            if mesh is None:
                urdf = sd / "mobility.urdf"
                mesh = urdf if urdf.exists() else None
            yield sd.name, jf, mesh
    else:
        for jf in sorted(asset_dir.glob("model_data*.json")):
            stem = jf.stem.replace("model_data", "")
            idx = int(stem) if stem.isdigit() else None
            mesh = find_visual_mesh(asset_dir, idx)
            label = str(idx) if idx is not None else "0"
            yield label, jf, mesh


def render_variant(asset_name: str, variant: str, md_path: Path, mesh_path: Path, out_path: Path):
    if mesh_path is None or not mesh_path.exists():
        return f"skip:no-mesh {asset_name}/{variant}"
    try:
        md = json.loads(md_path.read_text())
    except Exception as e:
        return f"err:{asset_name}/{variant}:{e}"
    center = md.get("center")
    extents = md.get("extents")
    if center is None or extents is None:
        return f"skip:no-obb {asset_name}/{variant}"

    try:
        if mesh_path.suffix == ".urdf":
            verts = load_urdf_vertices(mesh_path.parent)
        else:
            verts = load_mesh_vertices(mesh_path)
    except Exception as e:
        return f"err:{asset_name}/{variant}:mesh:{e}"
    if verts.size == 0:
        return f"skip:empty-mesh {asset_name}/{variant}"

    box_corners = obb_corners(np.array(center), np.array(extents))
    aabb_lo, aabb_hi = verts.min(0), verts.max(0)
    aabb_corners = obb_corners((aabb_lo + aabb_hi) / 2, aabb_hi - aabb_lo)

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, (ah, av, title) in zip(axes, VIEWS):
        ax.scatter(verts[:, ah], verts[:, av], s=0.3, alpha=0.3, color="steelblue")
        for a, b in EDGES:
            ax.plot([aabb_corners[a, ah], aabb_corners[b, ah]],
                    [aabb_corners[a, av], aabb_corners[b, av]],
                    color="gray", linewidth=1.0, linestyle="--", alpha=0.5)
            ax.plot([box_corners[a, ah], box_corners[b, ah]],
                    [box_corners[a, av], box_corners[b, av]],
                    color="red", linewidth=1.3)
        ax.set_title(title, fontsize=8)
        ax.set_aspect("equal")
        ax.set_xlabel(["X", "Y", "Z"][ah], fontsize=8)
        ax.set_ylabel(["X", "Y", "Z"][av], fontsize=8)
        ax.tick_params(labelsize=6)

    ext = np.array(extents)
    scale = md.get("scale", [1, 1, 1])
    if not isinstance(scale, (list, tuple)):
        scale = [scale] * 3
    real = ext * np.array(scale)
    plt.suptitle(
        f"{asset_name}  variant={variant}\n"
        f"extents (mesh) = [{ext[0]:.3f}, {ext[1]:.3f}, {ext[2]:.3f}]   "
        f"scale = [{scale[0]:.3f}, {scale[1]:.3f}, {scale[2]:.3f}]   "
        f"real (m) = [{real[0]:.3f}, {real[1]:.3f}, {real[2]:.3f}]",
        fontsize=8,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110)
    plt.close(fig)
    return f"ok:{asset_name}/{variant}"


def render_worker(args):
    return render_variant(*args)


def collect_jobs(asset_dirs, out_dir: Path):
    jobs = []
    for asset_dir in asset_dirs:
        asset_name = asset_dir.name
        for variant, md_path, mesh_path in variants_for_asset(asset_dir):
            out = out_dir / asset_name / f"{variant}.png"
            jobs.append((asset_name, variant, md_path, mesh_path, out))
    return jobs


def write_index_html(out_dir: Path):
    """Write a per-asset HTML page and an index linking them all."""
    assets = sorted([d for d in out_dir.iterdir() if d.is_dir()])
    lines = ["<html><head><title>OBB Visualizations</title>",
             "<style>body{font-family:sans-serif;margin:20px}"
             "a{display:inline-block;margin:2px;padding:4px 8px;background:#eef;text-decoration:none;color:#225}"
             "</style></head><body>",
             f"<h1>OBB Visualizations ({len(assets)} assets)</h1>"]
    for a in assets:
        n = len(list(a.glob("*.png")))
        lines.append(f'<a href="{a.name}/index.html">{a.name} ({n})</a>')

        pages = sorted(a.glob("*.png"))
        sub = [f"<html><head><title>{a.name}</title>",
               "<style>body{font-family:sans-serif;margin:20px}"
               "img{max-width:1000px;display:block;margin:12px 0;border:1px solid #ccc}"
               ".v{font-weight:bold;margin-top:18px}</style></head><body>",
               f'<a href="../index.html">&larr; back</a><h1>{a.name}</h1>']
        for p in pages:
            sub.append(f'<div class="v">{p.stem}</div><img src="{p.name}">')
        sub.append("</body></html>")
        (a / "index.html").write_text("\n".join(sub))
    lines.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("assets", nargs="*")
    parser.add_argument("--out", default="obb_visualizations")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    parser.add_argument("--no-html", action="store_true")
    args = parser.parse_args()

    if args.assets:
        asset_dirs = [ASSETS_ROOT / a for a in args.assets]
    else:
        asset_dirs = sorted([
            d for d in ASSETS_ROOT.iterdir()
            if d.is_dir() and not d.name.startswith("_")
        ])
    asset_dirs = [d for d in asset_dirs if d.exists()]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = collect_jobs(asset_dirs, out_dir)
    print(f"Rendering {len(jobs)} variants across {len(asset_dirs)} assets → {out_dir}")

    ok, skipped, errored = 0, 0, 0
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(render_worker, j) for j in jobs]
            for i, fu in enumerate(as_completed(futures), 1):
                r = fu.result()
                if r.startswith("ok:"):
                    ok += 1
                elif r.startswith("skip:"):
                    skipped += 1
                    print(f"  {r}")
                else:
                    errored += 1
                    print(f"  {r}")
                if i % 50 == 0:
                    print(f"  ... {i}/{len(jobs)}")
    else:
        for i, j in enumerate(jobs, 1):
            r = render_worker(j)
            if r.startswith("ok:"):
                ok += 1
            elif r.startswith("skip:"):
                skipped += 1
                print(f"  {r}")
            else:
                errored += 1
                print(f"  {r}")
            if i % 50 == 0:
                print(f"  ... {i}/{len(jobs)}")

    print(f"\nDone: ok={ok} skipped={skipped} errored={errored}")

    if not args.no_html:
        write_index_html(out_dir)
        print(f"Index: {out_dir}/index.html")


if __name__ == "__main__":
    main()

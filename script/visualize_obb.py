"""Visualize oriented bounding box projected onto camera images from a demo episode."""
import argparse
import io
import json
import re
import h5py
import numpy as np
import cv2
from PIL import Image
from pathlib import Path

CAMERAS = ["head_camera", "front_camera", "left_camera", "right_camera"]
BOX_COLOR = (0, 255, 0)   # green
BOX_THICKNESS = 2

# 12 edges of a box defined by corner indices (see _box_corners)
BOX_EDGES = [
    (0,1),(1,3),(3,2),(2,0),  # bottom face
    (4,5),(5,7),(7,6),(6,4),  # top face
    (0,4),(1,5),(2,6),(3,7),  # verticals
]


def decode_rgb(raw_bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(bytes(raw_bytes))))


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """[qw, qx, qy, qz] -> 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def compute_obb(pose_7: np.ndarray, model_data: dict):
    """Return (center_world, R, half_extents) of the OBB in world space."""
    pos = pose_7[:3]
    quat = pose_7[3:]  # [qw, qx, qy, qz]
    scale = np.array(model_data["scale"])
    if scale.ndim == 0:
        scale = np.array([scale, scale, scale])
    trans_mat = np.array(model_data.get("transform_matrix", np.eye(4)))
    pos = pos - trans_mat[:3, 3]
    if "body_q" in model_data:
        R = R @ quat_to_mat(np.array(model_data["body_q"]))
    center_local = np.array(model_data.get("center", [0, 0, 0])) * scale
    half_extents = np.array(model_data["extents"]) * scale / 2.0

    R = quat_to_mat(quat)
    center_world = pos + R @ center_local
    return center_world, R, half_extents


def box_corners(center: np.ndarray, R: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Return (8, 3) world-space corners of the OBB."""
    signs = np.array([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],
                      [ 1,-1,-1],[ 1,-1,1],[ 1,1,-1],[ 1,1,1]], dtype=float)
    return center + (signs * half) @ R.T


def project_points(pts_world: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    """Project (N,3) world points to (N,2) pixel coords."""
    ones = np.ones((pts_world.shape[0], 1))
    pts_h = np.concatenate([pts_world, ones], axis=1)   # (N, 4)
    pts_cam = (extrinsic @ pts_h.T).T                   # (N, 3)
    pts_img = (intrinsic @ pts_cam.T).T                 # (N, 3)
    px = pts_img[:, :2] / pts_img[:, 2:3]
    return px.astype(int)



def make_grid(frames):
    n = len(frames)
    if n == 1: return frames[0]
    if n == 2: return np.concatenate(frames, axis=1)
    row1 = np.concatenate(frames[:2], axis=1)
    pad = [np.zeros_like(frames[0])] * (4 - n)
    row2 = np.concatenate(frames[2:4] + pad, axis=1)
    return np.concatenate([row1, row2], axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf5", help="Path to episode HDF5 file")
    parser.add_argument("--model_json", default=None, help="Path to model_dataX.json (only needed when proj_3d_obb not in HDF5)")
    parser.add_argument("--out", default="obb_vis.mp4")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--cameras", nargs="+", default=CAMERAS)
    parser.add_argument("--object", default=None, help="Object name in object_pose (auto-detected if omitted)")
    args = parser.parse_args()

    model_data = None
    if args.model_json:
        with open(args.model_json) as f:
            model_data = json.load(f)
        model_dir = Path(args.model_json).parent
        if "extents" not in model_data:
            bb_file = model_dir / "bounding_box.json"
            if bb_file.exists():
                bb = json.loads(bb_file.read_text())
                bb_min, bb_max = np.array(bb["min"]), np.array(bb["max"])
                model_data["extents"] = (bb_max - bb_min).tolist()
                model_data["center"] = ((bb_min + bb_max) / 2).tolist()

    with h5py.File(args.hdf5, "r") as f:
        cams = [c for c in args.cameras if f"observation/{c}/rgb" in f]
        n_frames = f[f"observation/{cams[0]}/rgb"].shape[0]
        obj_name = args.object or list(f["object_pose"].keys())[0]
        poses = f[f"object_pose/{obj_name}"][:]
        rgb_data   = {c: f[f"observation/{c}/rgb"][:] for c in cams}
        intrinsics = {c: f[f"observation/{c}/intrinsic_cv"][:] for c in cams}
        extrinsics = {c: f[f"observation/{c}/extrinsic_cv"][:] for c in cams}
        saved_obb = {}
        for c in cams:
            k = f"observation/{c}/proj_3d_obb/{obj_name}"
            if k in f:
                saved_obb[c] = f[k][:]

    missing = [c for c in cams if c not in saved_obb]
    if missing and model_data is None:
        raise ValueError(f"proj_3d_obb missing for {missing} and no --model_json provided for fallback")

    sample = decode_rgb(rgb_data[cams[0]][0])
    h, w = sample.shape[:2]
    n_cams = len(cams)
    grid_w = w * min(n_cams, 2)
    grid_h = h * ((n_cams + 1) // 2)

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (grid_w, grid_h))

    for i in range(n_frames):
        corners_world = None
        if missing:
            center, R, half = compute_obb(poses[i], model_data)
            corners_world = box_corners(center, R, half)

        frames = []
        for c in cams:
            rgb = decode_rgb(rgb_data[c][i])
            vis = rgb.copy()
            if c in saved_obb:
                pts = saved_obb[c][i].reshape(8, 2).astype(int)
            else:
                pts = project_points(corners_world, intrinsics[c][i], extrinsics[c][i])
            for a, b in BOX_EDGES:
                cv2.line(vis, tuple(pts[a]), tuple(pts[b]), BOX_COLOR, BOX_THICKNESS)
            cv2.putText(vis, c, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
            frames.append(vis)

        grid = make_grid(frames)
        writer.write(cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"Saved {n_frames} frames → {args.out}")


if __name__ == "__main__":
    main()

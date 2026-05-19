"""Compute and save proj_3d_obb into each episode HDF5.

For rigid objects, saves:
  observation/{camera}/proj_3d_obb/bottle  shape (T, 16)  float32
  columns: [x0,y0, x1,y1, ..., x7,y7]

For articulated objects (with object_qpos in HDF5), saves:
  observation/{camera}/proj_3d_obb/laptop  shape (T, 32)  float32
  columns: [link0_x0,link0_y0,...,link0_x7,link0_y7, link1_x0,...,link1_x7,link1_y7]
"""
import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as _SciRotation


CAMERAS = ["head_camera", "front_camera", "left_camera", "right_camera"]


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def box_corners_world(pose_7: np.ndarray, model_data: dict) -> np.ndarray:
    pos = pose_7[:3]
    R = quat_to_mat(pose_7[3:])
    scale = np.array(model_data["scale"])
    # NOTE: do NOT subtract trans_mat[:3, 3] from pos — get_pose() returns the
    # actor's actual world pose, which already accounts for the trans_mat
    # offset that create_sapien_urdf_obj added at set_pose time. Subtracting
    # it again would shift the OBB by trans_mat.translation (e.g. ~7 cm for
    # the microwave, putting the box below the table).
    if "body_q" in model_data:
        R = R @ quat_to_mat(np.array(model_data["body_q"]))
    center_local = np.array(model_data.get("center", [0, 0, 0])) * scale
    half = np.array(model_data["extents"]) * scale / 2.0
    center_world = pos + R @ center_local
    signs = np.array([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],
                      [ 1,-1,-1],[ 1,-1,1],[ 1,1,-1],[ 1,1,1]], dtype=float)
    return center_world + (signs * half) @ R.T   # (8, 3)


def project(pts_world: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    ones = np.ones((pts_world.shape[0], 1))
    pts_cam = (extrinsic @ np.concatenate([pts_world, ones], axis=1).T).T
    pts_img = (intrinsic @ pts_cam.T).T
    return (pts_img[:, :2] / pts_img[:, 2:3]).astype(np.float32)


def proj_3d_obb(corners_2d: np.ndarray) -> np.ndarray:
    """Flatten (N,2) projected corners to 2N values [x0,y0,...,xN-1,yN-1]."""
    return corners_2d.flatten().astype(np.float32)


def kinematics_from_urdf(urdf_path: Path):
    """Parse the kinematic chain from a URDF for articulated objects.

    Finds the fixed joint (base → anchor_link) and the revolute joint
    (anchor_link → moving_link), handling both link-naming conventions.

    Returns a dict with keys:
      body_q            [qw,qx,qy,qz] or None
      anchor_link       str  — link attached to 'base' via fixed joint
      moving_link       str  — child of the revolute joint
      revolute_origin   [x,y,z] in anchor_link frame (unscaled model units)
      revolute_axis     [x,y,z]
    Returns None if no revolute joint is present (rigid object).
    """
    import xml.etree.ElementTree as ET
    import transforms3d as t3d

    def rpy_to_quat(rpy):
        R = t3d.euler.euler2mat(*rpy, axes="sxyz")
        return t3d.quaternions.mat2quat(R).tolist()

    try:
        root = ET.parse(str(urdf_path)).getroot()
        anchor_link = None
        body_q = None
        revolute = None

        for joint in root.findall("joint"):
            jtype = joint.get("type")
            parent_el = joint.find("parent")
            child_el = joint.find("child")
            origin_el = joint.find("origin")

            if jtype == "fixed" and parent_el is not None and parent_el.get("link") == "base":
                anchor_link = child_el.get("link") if child_el is not None else None
                rpy_str = origin_el.get("rpy", "0 0 0") if origin_el is not None else "0 0 0"
                rpy = [float(v) for v in rpy_str.split()]
                if any(abs(v) > 1e-6 for v in rpy):
                    body_q = rpy_to_quat(rpy)

            elif jtype == "revolute":
                axis_el = joint.find("axis")
                xyz_str = origin_el.get("xyz", "0 0 0") if origin_el is not None else "0 0 0"
                axis_str = axis_el.get("xyz", "0 0 1") if axis_el is not None else "0 0 1"
                revolute = {
                    "parent": parent_el.get("link") if parent_el is not None else None,
                    "child": child_el.get("link") if child_el is not None else None,
                    "origin": [float(v) for v in xyz_str.split()],
                    "axis": [float(v) for v in axis_str.split()],
                }

        if revolute is None:
            return None

        return {
            "body_q": body_q,
            "anchor_link": anchor_link,           # e.g. "link_0" or "link_1"
            "moving_link": revolute["child"],      # the link that rotates
            "revolute_origin": revolute["origin"], # in anchor_link frame
            "revolute_axis": revolute["axis"],
        }
    except Exception:
        return None


def box_corners_world_articulated(pose_7: np.ndarray, qpos: float, model_data: dict,
                                  obj_name: str = "") -> np.ndarray:
    """Compute a single unified OBB (8 corners) for an articulated object.

    Computes the actual world-space corners of both the anchor and moving links,
    then returns 8 corners of the tightest OBB in anchor-link orientation that
    contains all actual corners.  No phantom corners, single box output.

    For laptop, only the moving (lid) link's corners are used so the OBB
    tracks the lid rather than the whole body+lid envelope.

    Returns (8, 3).
    """
    pos = pose_7[:3].copy()
    R_body = quat_to_mat(pose_7[3:])
    scale_raw = model_data["scale"]
    scale = np.array(scale_raw) if isinstance(scale_raw, list) else np.full(3, float(scale_raw))

    # See box_corners_world above — get_pose() already returns the world pose
    # including trans_mat offset; do NOT subtract trans_mat translation here.
    R_fixed = quat_to_mat(np.array(model_data["body_q"])) if "body_q" in model_data else np.eye(3)
    R_link0 = R_body @ R_fixed
    pos_link0 = pos  # fixed joint has xyz=[0,0,0]

    signs = np.array([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],
                      [1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]], dtype=float)

    link_bboxes = model_data["link_bboxes"]
    anchor_link = model_data["anchor_link"]
    moving_link = model_data["moving_link"]
    # Track the moving link only (door / lid) rather than the union envelope
    # for articulated objects whose meaningful manipulation target IS the
    # opening part. Extend as needed.
    only_moving = obj_name in ("laptop", "microwave")

    def corners_for_link(origin_world, R_world, bbox):
        bb_min = np.array(bbox["min"]) * scale
        bb_max = np.array(bbox["max"]) * scale
        center = (bb_min + bb_max) / 2
        half = (bb_max - bb_min) / 2
        pts_local = center + signs * half
        return origin_world + (R_world @ pts_local.T).T

    # Revolute joint: anchor_link → moving_link
    joint_origin_local = np.array(model_data["revolute_origin"], dtype=float) * scale
    joint_axis = np.array(model_data["revolute_axis"], dtype=float)
    p_joint_world = pos_link0 + R_link0 @ joint_origin_local
    R_revolute = _SciRotation.from_rotvec(joint_axis * float(qpos)).as_matrix()
    R_moving = R_link0 @ R_revolute
    corners_moving = corners_for_link(p_joint_world, R_moving, link_bboxes[moving_link])

    if only_moving:
        # Lid corners are already a tight OBB aligned with the lid's own frame.
        return corners_moving

    corners_anchor = corners_for_link(pos_link0, R_link0, link_bboxes[anchor_link])

    # Fit tight OBB in anchor-link orientation over all actual world corners.
    all_corners = np.vstack([corners_anchor, corners_moving])
    local = (all_corners - pos_link0) @ R_link0
    bb_min, bb_max = local.min(axis=0), local.max(axis=0)
    center_l = (bb_min + bb_max) / 2
    half_l   = (bb_max - bb_min) / 2
    obb_local = center_l + signs * half_l                      # (8, 3)
    return pos_link0 + obb_local @ R_link0.T                   # (8, 3) world


def model_id_from_scene_info(scene_info: dict, episode_idx: int) -> int:
    key = f"episode_{episode_idx}"
    label = scene_info[key]["info"]["{A}"]   # e.g. "001_bottle/base16"
    return int(re.search(r"base(\d+)", label).group(1))


def process_episode(hdf5_path: Path, model_data: dict, cameras: list[str], obj_name: str = "bottle"):
    with h5py.File(hdf5_path, "r+") as f:
        cams = [c for c in cameras if f"observation/{c}/rgb" in f]
        pose_key = f"object_pose/{obj_name}"
        if pose_key not in f:
            print(f"  {hdf5_path.name}: no {pose_key}, skipping")
            return
        poses = f[pose_key][:]
        T = poses.shape[0]

        qpos_key = f"object_qpos/{obj_name}"
        is_articulated = (
            qpos_key in f
            and "link_bboxes" in model_data
            and "anchor_link" in model_data
        )
        if is_articulated:
            qpos_all = f[qpos_key][:]
            if qpos_all.ndim > 1:
                qpos_all = qpos_all[:, 0]
        D = 16  # 8 projected corners × 2 coords

        for c in cams:
            intrinsics = f[f"observation/{c}/intrinsic_cv"][:]
            extrinsics = f[f"observation/{c}/extrinsic_cv"][:]
            obb_data = np.zeros((T, D), dtype=np.float32)
            for t in range(T):
                if is_articulated:
                    corners_w = box_corners_world_articulated(poses[t], qpos_all[t], model_data, obj_name=obj_name)
                else:
                    corners_w = box_corners_world(poses[t], model_data)
                corners_2d = project(corners_w, intrinsics[t], extrinsics[t])
                obb_data[t] = proj_3d_obb(corners_2d)

            key = f"observation/{c}/proj_3d_obb/{obj_name}"
            if key in f:
                del f[key]
            f.create_dataset(key, data=obb_data)

        suffix = " [articulated]" if is_articulated else ""
        print(f"  {hdf5_path.name}: wrote proj_3d_obb for {len(cams)} cameras, {T} steps{suffix}")


def load_model_data(model_dir: Path, mid: int) -> dict:
    """Load model_data for a given model index, handling both flat and URDF layouts."""
    import sys, xml.etree.ElementTree as ET
    import transforms3d as t3d

    def rpy_to_quat(rpy):
        roll, pitch, yaw = rpy
        R = t3d.euler.euler2mat(roll, pitch, yaw, axes='sxyz')
        return t3d.quaternions.mat2quat(R).tolist()

    def body_q_from_urdf(urdf_path):
        try:
            tree = ET.parse(str(urdf_path))
            for joint in tree.getroot().findall('joint'):
                if joint.get('type') == 'fixed':
                    parent_el = joint.find('parent')
                    origin_el = joint.find('origin')
                    if parent_el is not None and parent_el.get('link') == 'base':
                        rpy_str = origin_el.get('rpy', '0 0 0') if origin_el is not None else '0 0 0'
                        rpy = [float(v) for v in rpy_str.split()]
                        if any(abs(v) > 1e-6 for v in rpy):
                            return rpy_to_quat(rpy)
        except Exception:
            pass
        return None

    # Flat layout: model_dir/model_data{mid}.json
    flat = model_dir / f"model_data{mid}.json"
    if flat.exists():
        data = json.loads(flat.read_text())
        if not isinstance(data["scale"], list):
            data["scale"] = [data["scale"]] * 3
        return data

    # URDF layout: model_dir/{subdir}/model_data.json (sorted by numeric name)
    subdirs = sorted(
        [d for d in model_dir.iterdir() if d.is_dir() and d.name != "visual"],
        key=lambda d: int(d.name) if d.name.isdigit() else 0,
    )
    if mid < len(subdirs):
        subdir = subdirs[mid]
    else:
        subdir = next((d for d in subdirs if d.name == str(mid)), subdirs[0])

    data = json.loads((subdir / "model_data.json").read_text())
    if not isinstance(data["scale"], list):
        data["scale"] = [data["scale"]] * 3
    bb_file = subdir / "bounding_box.json"
    if bb_file.exists() and "extents" not in data:
        bb = json.loads(bb_file.read_text())
        bb_min, bb_max = np.array(bb["min"]), np.array(bb["max"])
        data["extents"] = (bb_max - bb_min).tolist()
        data["center"] = ((bb_min + bb_max) / 2).tolist()
    urdf = subdir / "mobility.urdf"
    lbc_file = subdir / "link_bboxes_cache.json"
    if urdf.exists():
        kin = kinematics_from_urdf(urdf)
        if kin is not None:
            # Articulated: store full kinematic chain info
            if kin["body_q"] is not None and "body_q" not in data:
                data["body_q"] = kin["body_q"]
            data.setdefault("anchor_link", kin["anchor_link"])
            data.setdefault("moving_link", kin["moving_link"])
            data.setdefault("revolute_origin", kin["revolute_origin"])
            data.setdefault("revolute_axis", kin["revolute_axis"])
            if lbc_file.exists():
                data.setdefault("link_bboxes", json.loads(lbc_file.read_text()))
        else:
            # Rigid with URDF: just get body_q from fixed joint
            if "body_q" not in data:
                q = body_q_from_urdf(urdf)
                if q is not None:
                    data["body_q"] = q
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Path to demo data dir (contains episodeX.hdf5)")
    parser.add_argument("model_dir", help="Path to object model dir")
    parser.add_argument("scene_info", help="Path to scene_info.json")
    parser.add_argument("--cameras", nargs="+", default=CAMERAS)
    parser.add_argument("--object", default="bottle", help="Object name key in object_pose")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)

    with open(args.scene_info) as f:
        scene_info = json.load(f)

    episodes = sorted(data_dir.glob("episode*.hdf5"),
                      key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))

    for ep_path in episodes:
        ep_idx = int(re.search(r"(\d+)", ep_path.stem).group(1))
        mid = model_id_from_scene_info(scene_info, ep_idx)
        model_data = load_model_data(model_dir, mid)
        process_episode(ep_path, model_data, args.cameras, obj_name=args.object)

    print("Done.")


if __name__ == "__main__":
    main()

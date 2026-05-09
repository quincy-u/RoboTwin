"""
Get 3D center of mass of the bottle in adjust_bottle task,
project it onto the head camera 2D image plane,
and save pixel coordinates to ~/dev/vjepa2/center_of_mass.npy
"""
import sys
import os

sys.path.append("./")
sys.path.append("./policy")
sys.path.append("./description/utils")

from envs import CONFIGS_PATH
import numpy as np
import yaml
import importlib

SEED = 0


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def build_args():
    with open("./task_config/demo_clean.yml", "r") as f:
        args = yaml.safe_load(f)

    args["task_name"] = "adjust_bottle"
    args["task_config"] = "demo_clean"
    args["ckpt_setting"] = "test"
    args["eval_mode"] = False
    args["save_data"] = False
    args["now_ep_num"] = 0
    args["seed"] = SEED
    args["need_plan"] = False  # no motion planning needed for scene setup
    args["render_freq"] = 0

    embodiment_type = args.get("embodiment")  # e.g. ['aloha-agilex']
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    robot_file = _embodiment_types[embodiment_type[0]]["file_path"]
    args["left_robot_file"] = robot_file
    args["right_robot_file"] = robot_file
    args["dual_arm_embodied"] = True

    args["left_embodiment_config"] = get_embodiment_config(robot_file)
    args["right_embodiment_config"] = get_embodiment_config(robot_file)

    with open(CONFIGS_PATH + "_camera_config.yml", "r") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    return args


def project_world_to_pixel(point_world, intrinsic, extrinsic):
    """
    Project a 3D world-frame point onto the image plane.

    Args:
        point_world: (3,) array, world coordinates
        intrinsic:   (3, 3) OpenCV intrinsic matrix  K
        extrinsic:   (4, 4) world-to-camera matrix   [R|t] (OpenCV convention)

    Returns:
        (u, v) pixel coordinates as float array
    """
    p_h = np.array([*point_world, 1.0])       # homogeneous world point
    p_cam = extrinsic @ p_h                    # camera-frame coords
    p_2d_h = intrinsic @ p_cam[:3]            # perspective projection
    u = p_2d_h[0] / p_2d_h[2]
    v = p_2d_h[1] / p_2d_h[2]
    return np.array([u, v])


def main():
    args = build_args()

    envs_module = importlib.import_module("envs.adjust_bottle")
    env_class = getattr(envs_module, "adjust_bottle")
    env = env_class()

    env.setup_demo(is_test=False, **args)

    # Sync renderer so camera matrices are up-to-date
    env.scene.update_render()

    # ── Active hand end-effector 3D position ─────────────────────────────────
    # adjust_bottle uses left arm when qpose_tag==0, right when qpose_tag==1
    arm_tag = "right" if env.qpose_tag == 1 else "left"
    if arm_tag == "left":
        ee_pos_world = np.array(env.robot.get_left_ee_pose()[:3])
    else:
        ee_pos_world = np.array(env.robot.get_right_ee_pose()[:3])
    print(f"Active arm: {arm_tag}")
    print(f"End-effector 3D world position: {ee_pos_world}")

    # ── Head camera intrinsic & extrinsic ────────────────────────────────────
    head_cam_idx = env.cameras.head_camera_id
    head_camera = env.cameras.static_camera_list[head_cam_idx]

    K = head_camera.get_intrinsic_matrix()   # 3×3, OpenCV format
    E = head_camera.get_extrinsic_matrix()   # 4×4, world-to-cam, OpenCV

    print(f"Intrinsic matrix K:\n{K}")
    print(f"Extrinsic matrix E (world→cam):\n{E}")

    # ── Project to 2D — shape (2,): [u, v] ───────────────────────────────────
    pixel = project_world_to_pixel(ee_pos_world, K, E)
    print(f"Projected pixel (u, v): {pixel}")

    img_w = args["head_camera_w"]
    img_h = args["head_camera_h"]
    print(f"Image size: {img_w} x {img_h}")
    in_frame = 0 <= pixel[0] < img_w and 0 <= pixel[1] < img_h
    print(f"Pixel in image frame: {in_frame}")

    # ── Save as (2,) ─────────────────────────────────────────────────────────
    out_path = os.path.expanduser("~/dev/vjepa2/center_of_mass.npy")
    np.save(out_path, pixel)
    print(f"Saved pixel coordinates to {out_path}")

    env.close_env()


if __name__ == "__main__":
    main()

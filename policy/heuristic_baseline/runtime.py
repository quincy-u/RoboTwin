"""Adapters connecting simple-grasp orchestration to RoboTwin."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import transforms3d as t3d

from simple_grasp.m2t2_adapter import M2T2GraspGenerator
from simple_grasp.mink_ik import (
    MinkArmConfig,
    MinkCollisionConfig,
    MinkIKConfig,
    MinkIKSolver,
)
from simple_grasp.policy import PolicyConfig, SimpleGraspPolicy
from simple_grasp.types import GraspCandidate, ObjectState, SceneObservation

from .errors import NoFeasiblePlanFailure, TargetSelectionFailure
from .m2t2_backend import RoboTwinM2T2Backend
from .task_plan import Handoff, Pick, Place


M2T2_TO_ROBOTWIN = np.array(
    [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        # M2T2's pose origin is already shifted 0.1034 m from the predicted
        # contact by build_6d_grasp().  This is only the remaining calibrated
        # offset from that wrist frame to RoboTwin's gripper command frame.
        [1.0, 0.0, 0.0, -0.0166],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


CANONICAL_COMMAND_QUATERNIONS = {
    "right": np.array([-0.353523, 0.61239, -0.353524, -0.61239]),
    "left": np.array([-0.61239, 0.353523, -0.61239, -0.353524]),
}
SELECTED_GRASP_COLOR = "#ff2d95"
SELECTED_GRASP_RGB = (255, 45, 149)
M2T2_GRIPPER_POLYLINE = np.array(
    [
        [0.05268743, -0.00005996, 0.10527314],
        [0.05268743, -0.00005996, 0.05900000],
        [0.0, 0.0, 0.05900000],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.05900000],
        [-0.05268743, 0.00005996, 0.05900000],
        [-0.05268743, 0.00005996, 0.10527314],
    ],
    dtype=np.float64,
)
# The M2T2 gripper's finger-base plane is local +Z=0.059 m. Keep a
# 4 mm tolerance for segmented-point noise while preventing target geometry
# from extending into the gripper palm/body after an orientation transform.
M2T2_MIN_TARGET_PALM_DEPTH_M = 0.055


def _grasp_wireframes(world_grasp_poses: np.ndarray) -> np.ndarray:
    """Transform M2T2's canonical seven-point gripper into world space."""
    poses = np.asarray(world_grasp_poses, dtype=np.float64)
    if poses.size == 0:
        return np.empty((0, len(M2T2_GRIPPER_POLYLINE), 3), dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("world_grasp_poses must have shape (N, 4, 4)")
    local_h = np.concatenate(
        (
            M2T2_GRIPPER_POLYLINE,
            np.ones((len(M2T2_GRIPPER_POLYLINE), 1), dtype=np.float64),
        ),
        axis=1,
    )
    world_h = np.einsum("nij,pj->npi", poses, local_h)
    return world_h[..., :3]


def _project_world_points_cv(
    points_world: np.ndarray,
    intrinsic_cv: np.ndarray,
    extrinsic_cv: np.ndarray,
    *,
    near_m: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points with RoboTwin's OpenCV world-to-camera matrices."""
    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim < 2 or points.shape[-1] != 3:
        raise ValueError("points_world must end in xyz coordinates")
    intrinsic = np.asarray(intrinsic_cv, dtype=np.float64)
    extrinsic = np.asarray(extrinsic_cv, dtype=np.float64)
    if intrinsic.shape != (3, 3):
        raise ValueError("intrinsic_cv must have shape (3, 3)")
    if extrinsic.shape == (4, 4):
        extrinsic = extrinsic[:3]
    if extrinsic.shape != (3, 4):
        raise ValueError("extrinsic_cv must have shape (3, 4) or (4, 4)")
    if near_m <= 0.0:
        raise ValueError("near_m must be positive")

    flat = points.reshape(-1, 3)
    homogeneous = np.concatenate(
        (flat, np.ones((len(flat), 1), dtype=np.float64)), axis=1
    )
    camera = homogeneous @ extrinsic.T
    projected_h = camera @ intrinsic.T
    pixels = np.full((len(flat), 2), np.nan, dtype=np.float64)
    valid = np.all(np.isfinite(projected_h), axis=1)
    valid &= camera[:, 2] >= near_m
    pixels[valid] = projected_h[valid, :2] / projected_h[valid, 2:3]
    return pixels.reshape(*points.shape[:-1], 2), camera.reshape(
        *points.shape[:-1], 3
    )


def save_grasp_visualization(
    output_path: str | Path,
    scene: SceneObservation,
    target: ObjectState,
    candidates: list[GraspCandidate],
    selected: GraspCandidate | None,
    *,
    arm: str,
    grasp_to_robotwin: np.ndarray,
    camera_rgb: np.ndarray,
    camera_intrinsic: np.ndarray,
    camera_extrinsic: np.ndarray,
    rejected_candidates: list[GraspCandidate] | None = None,
    policy_ranked_candidates: list[GraspCandidate] | None = None,
    executed_command_pose: np.ndarray | None = None,
    raw_trace: dict[str, np.ndarray] | None = None,
    max_grasps: int | None = None,
    max_points: int = 30_000,
    image_scale: int = 4,
) -> Path:
    """Draw every M2T2 gripper directly over the synchronized head RGB frame."""
    if max_grasps is not None and max_grasps <= 0:
        raise ValueError("max_grasps must be positive or None for all grasps")
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    if image_scale <= 0:
        raise ValueError("image_scale must be positive")
    grasp_to_robotwin = np.asarray(grasp_to_robotwin, dtype=np.float64)
    if grasp_to_robotwin.shape != (4, 4):
        raise ValueError("grasp_to_robotwin must have shape (4, 4)")
    if executed_command_pose is not None:
        executed_command_pose = np.asarray(executed_command_pose, dtype=np.float64)
        if executed_command_pose.shape != (4, 4):
            raise ValueError("executed_command_pose must have shape (4, 4)")
    rejected_candidates = list(rejected_candidates or [])
    policy_ranked_candidates = list(
        candidates
        if policy_ranked_candidates is None
        else policy_ranked_candidates
    )
    raw_trace = raw_trace or {}

    import colorsys
    import cv2
    from PIL import Image

    image = np.asarray(camera_rgb)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("camera_rgb must have shape (height, width, 3)")
    image = image[..., :3]
    if image.dtype != np.uint8:
        image = np.asarray(image, dtype=np.float64)
        if image.size and float(np.nanmax(image)) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    else:
        image = image.copy()
    native_height, native_width = image.shape[:2]
    canvas = cv2.resize(
        image,
        (native_width * image_scale, native_height * image_scale),
        interpolation=cv2.INTER_LINEAR,
    )
    height, width = canvas.shape[:2]

    def candidate_arrays(
        items: list[GraspCandidate],
    ) -> tuple[np.ndarray, np.ndarray]:
        poses = np.asarray(
            [item.world_grasp_pose for item in items], dtype=np.float64
        ).reshape(-1, 4, 4)
        scores = np.asarray(
            [item.confidence for item in items], dtype=np.float64
        )
        return poses, scores

    displayed = list(
        candidates if max_grasps is None else candidates[:max_grasps]
    )
    rejected_by_score = sorted(
        rejected_candidates,
        key=lambda candidate: candidate.confidence,
        reverse=True,
    )
    displayed_rejected = list(
        rejected_by_score
        if max_grasps is None
        else rejected_by_score[:max_grasps]
    )
    m2t2_poses, m2t2_scores = candidate_arrays(candidates)
    ranked_poses, ranked_scores = candidate_arrays(policy_ranked_candidates)
    displayed_poses, _ = candidate_arrays(displayed)
    rejected_poses, rejected_scores = candidate_arrays(rejected_candidates)
    displayed_rejected_poses, _ = candidate_arrays(displayed_rejected)
    selected_pose = (
        np.empty((0, 4, 4), dtype=np.float64)
        if selected is None
        else np.asarray(selected.world_grasp_pose, dtype=np.float64)[None, :, :]
    )

    ranked_wireframes = _grasp_wireframes(displayed_poses)
    rejected_wireframes = _grasp_wireframes(displayed_rejected_poses)
    selected_wireframe = _grasp_wireframes(selected_pose)
    ranked_pixels, ranked_camera = _project_world_points_cv(
        ranked_wireframes, camera_intrinsic, camera_extrinsic
    )
    rejected_pixels, rejected_camera = _project_world_points_cv(
        rejected_wireframes, camera_intrinsic, camera_extrinsic
    )
    selected_pixels, selected_camera = _project_world_points_cv(
        selected_wireframe, camera_intrinsic, camera_extrinsic
    )

    near_m = 0.1
    image_rect = (0, 0, width, height)

    def draw_camera_polyline(
        destination: np.ndarray,
        camera_points: np.ndarray,
        *,
        color_rgb: tuple[int, int, int],
        thickness: int,
    ) -> int:
        edges_drawn = 0
        intrinsic = np.asarray(camera_intrinsic, dtype=np.float64)
        for first, second in zip(camera_points[:-1], camera_points[1:]):
            first = np.asarray(first, dtype=np.float64).copy()
            second = np.asarray(second, dtype=np.float64).copy()
            if not (
                np.all(np.isfinite(first)) and np.all(np.isfinite(second))
            ):
                continue
            if first[2] < near_m and second[2] < near_m:
                continue
            if first[2] < near_m:
                fraction = (near_m - first[2]) / (second[2] - first[2])
                first += fraction * (second - first)
            if second[2] < near_m:
                fraction = (near_m - second[2]) / (first[2] - second[2])
                second += fraction * (first - second)
            projected = np.stack((intrinsic @ first, intrinsic @ second))
            pixels = projected[:, :2] / projected[:, 2:3]
            pixels *= image_scale
            if not np.all(np.isfinite(pixels)):
                continue
            pixels = np.clip(np.rint(pixels), -2_000_000, 2_000_000)
            point_a = tuple(int(value) for value in pixels[0])
            point_b = tuple(int(value) for value in pixels[1])
            visible, point_a, point_b = cv2.clipLine(
                image_rect, point_a, point_b
            )
            if not visible:
                continue
            cv2.line(
                destination,
                point_a,
                point_b,
                color_rgb,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )
            edges_drawn += 1
        return edges_drawn

    base_thickness = max(1, int(round(image_scale * 0.25)))
    options_layer = canvas.copy()
    rejected_edges_drawn = np.zeros(len(rejected_camera), dtype=np.int64)
    for index in reversed(range(len(rejected_camera))):
        rejected_edges_drawn[index] = draw_camera_polyline(
            options_layer,
            rejected_camera[index],
            color_rgb=(255, 64, 64),
            thickness=base_thickness,
        )

    ranked_edges_drawn = np.zeros(len(ranked_camera), dtype=np.int64)
    rank_denominator = max(len(ranked_camera) - 1, 1)
    for index in reversed(range(len(ranked_camera))):
        hue = 0.72 * index / rank_denominator
        color = colorsys.hsv_to_rgb(hue, 0.92, 1.0)
        color_rgb = tuple(int(round(channel * 255.0)) for channel in color)
        ranked_edges_drawn[index] = draw_camera_polyline(
            options_layer,
            ranked_camera[index],
            color_rgb=color_rgb,
            thickness=base_thickness,
        )
    canvas = cv2.addWeighted(options_layer, 0.76, canvas, 0.24, 0.0)

    selected_edges_drawn = 0
    if len(selected_camera):
        selected_edges_drawn = draw_camera_polyline(
            canvas,
            selected_camera[0],
            color_rgb=(15, 23, 42),
            thickness=base_thickness + 6,
        )
        draw_camera_polyline(
            canvas,
            selected_camera[0],
            color_rgb=SELECTED_GRASP_RGB,
            thickness=base_thickness + 4,
        )

    status = (
        f"M2T2 grasps {len(displayed)}/{len(candidates)}"
        if selected is not None
        else f"M2T2 grasps {len(displayed)}/{len(candidates)} | no feasible IK"
    )
    font_scale = max(0.55, min(height, width) / 1200.0)
    font_thickness = max(1, int(round(font_scale * 2.0)))
    text_size, baseline = cv2.getTextSize(
        status, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
    )
    box_right = min(width - 8, 24 + text_size[0] + (80 if selected else 0))
    box_bottom = 22 + text_size[1] + baseline
    label_layer = canvas.copy()
    cv2.rectangle(
        label_layer,
        (8, 8),
        (box_right, box_bottom),
        (0, 0, 0),
        thickness=-1,
    )
    canvas = cv2.addWeighted(label_layer, 0.62, canvas, 0.38, 0.0)
    cv2.putText(
        canvas,
        status,
        (18, 17 + text_size[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        font_thickness,
        cv2.LINE_AA,
    )
    if selected is not None:
        sample_x = min(width - 66, 28 + text_size[0])
        sample_y = 17 + text_size[1] // 2
        cv2.line(
            canvas,
            (sample_x, sample_y),
            (sample_x + 42, sample_y),
            (15, 23, 42),
            thickness=base_thickness + 6,
            lineType=cv2.LINE_AA,
        )
        cv2.line(
            canvas,
            (sample_x, sample_y),
            (sample_x + 42, sample_y),
            SELECTED_GRASP_RGB,
            thickness=base_thickness + 4,
            lineType=cv2.LINE_AA,
        )

    selected_source_command = (
        np.empty((0, 4, 4), dtype=np.float64)
        if selected is None
        else selected_pose @ grasp_to_robotwin
    )
    accepted_command = (
        np.empty((0, 4, 4), dtype=np.float64)
        if executed_command_pose is None
        else executed_command_pose[None, :, :]
    )
    raw_poses = np.asarray(
        raw_trace.get("poses", np.empty((0, 4, 4))), dtype=np.float64
    ).reshape(-1, 4, 4)
    raw_scores = np.asarray(
        raw_trace.get("scores", np.empty(0)), dtype=np.float64
    )
    raw_contacts = np.asarray(
        raw_trace.get("contacts", np.empty((0, 3))), dtype=np.float64
    ).reshape(-1, 3)
    raw_target_contacts = np.asarray(
        raw_trace.get("target_contacts", np.empty(0)), dtype=bool
    )
    raw_query_ids = np.asarray(
        raw_trace.get("query_ids", np.empty((0, 2))), dtype=np.int64
    ).reshape(-1, 2)

    scene_points = np.asarray(scene.xyz, dtype=np.float64)
    scene_colors = np.asarray(scene.rgb, dtype=np.float64)
    finite_scene = np.all(np.isfinite(scene_points), axis=1)
    finite_scene &= np.all(np.isfinite(scene_colors), axis=1)
    finite_indices = np.flatnonzero(finite_scene)
    if len(finite_indices) > max_points:
        finite_indices = finite_indices[
            np.linspace(0, len(finite_indices) - 1, max_points, dtype=int)
        ]
    selected_index = -1
    if selected is not None and len(ranked_poses):
        errors = np.linalg.norm(
            ranked_poses - selected_pose[0], axis=(1, 2)
        )
        selected_index = int(np.argmin(errors))

    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(output_path)
    np.savez_compressed(
        output_path.with_suffix(".npz"),
        target_name=np.asarray(target.name),
        arm=np.asarray(arm),
        target_world_pose=target.world_pose,
        target_points=np.asarray(
            scene.xyz[scene.instance_labels == target.instance_id],
            dtype=np.float32,
        ),
        target_rgb=np.asarray(
            scene.rgb[scene.instance_labels == target.instance_id],
            dtype=np.float32,
        ),
        scene_world_points=np.asarray(
            scene_points[finite_indices], dtype=np.float32
        ),
        scene_rgb=np.asarray(scene_colors[finite_indices], dtype=np.float32),
        scene_point_count=np.asarray(int(finite_scene.sum()), dtype=np.int64),
        camera_rgb=np.asarray(image, dtype=np.uint8),
        camera_intrinsic_cv=np.asarray(camera_intrinsic, dtype=np.float64),
        camera_extrinsic_cv=np.asarray(camera_extrinsic, dtype=np.float64),
        visualization_scale=np.asarray(image_scale, dtype=np.int64),
        raw_world_grasp_poses=raw_poses,
        raw_confidences=raw_scores,
        raw_contacts=raw_contacts,
        raw_target_contacts=raw_target_contacts,
        raw_query_ids=raw_query_ids,
        m2t2_world_grasp_poses=m2t2_poses,
        m2t2_confidences=m2t2_scores,
        ranked_world_grasp_poses=ranked_poses,
        ranking_scores=ranked_scores,
        rendered_world_grasp_poses=displayed_poses,
        rendered_grasp_wireframes_world=np.asarray(
            ranked_wireframes, dtype=np.float32
        ),
        projected_grasp_polylines=np.asarray(ranked_pixels, dtype=np.float32),
        grasp_camera_points=np.asarray(ranked_camera, dtype=np.float32),
        grasp_edges_drawn=ranked_edges_drawn,
        rejected_world_grasp_poses=rejected_poses,
        rejected_confidences=rejected_scores,
        projected_rejected_polylines=np.asarray(
            rejected_pixels, dtype=np.float32
        ),
        rejected_edges_drawn=rejected_edges_drawn,
        selected_world_grasp_pose=selected_pose,
        selected_source_command_pose=selected_source_command,
        selected_grasp_wireframe_world=np.asarray(
            selected_wireframe, dtype=np.float32
        ),
        projected_selected_polyline=np.asarray(
            selected_pixels, dtype=np.float32
        ),
        selected_grasp_color=np.asarray(SELECTED_GRASP_COLOR),
        selected_grasp_index=np.asarray(selected_index, dtype=np.int64),
        selected_edges_drawn=np.asarray(
            selected_edges_drawn, dtype=np.int64
        ),
        mink_accepted_command_pose=accepted_command,
        grasp_to_robotwin=grasp_to_robotwin,
    )
    return output_path


class SceneSnapshot:
    def __init__(self) -> None:
        self.scene: SceneObservation | None = None

    def update(self, scene: SceneObservation) -> None:
        self.scene = scene

    def observe(self) -> SceneObservation:
        if self.scene is None:
            raise RuntimeError("scene snapshot has not been initialized")
        return self.scene

    def object_state(self, name: str) -> ObjectState:
        try:
            return self.observe().objects[name]
        except KeyError as exc:
            available = ", ".join(sorted(self.observe().objects))
            raise TargetSelectionFailure(
                f"unknown target {name!r}; available: {available}"
            ) from exc


def _arm_joint_state(task_env: Any, arm: str) -> np.ndarray:
    """Read physical arm joints when RoboTwin exposes them, else drive targets."""
    robot = task_env.robot
    getter = getattr(robot, f"get_{arm}_arm_real_jointState", None)
    if getter is None:
        getter = getattr(robot, f"get_{arm}_arm_jointState")
    state = np.asarray(getter(), dtype=np.float64)
    if state.ndim != 1 or len(state) < 2 or not np.all(np.isfinite(state)):
        raise ValueError(f"invalid {arm} arm joint state {state}")
    return state


def _aloha_self_collision_config(model: mujoco.MjModel) -> MinkCollisionConfig:
    """Constrain active arms against themselves, each other, and the robot."""
    for prefix in ("fl_joint", "fr_joint"):
        for index in range(1, 7):
            joint_name = f"{prefix}{index}"
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if joint_id < 0:
                raise ValueError(f"Mink model lacks arm joint {joint_name}")
            # RoboTwin's URDF uses the placeholder range [-10, 10], which
            # permits multi-turn IK branches the physical arm cannot execute.
            model.jnt_limited[joint_id] = 1
            model.jnt_range[joint_id] = (-np.pi, np.pi)

    def body_name(geom_id: int) -> str:
        return (
            mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                int(model.geom_bodyid[geom_id]),
            )
            or ""
        )

    arm_groups: dict[str, tuple[int, ...]] = {}
    movable_groups: dict[str, tuple[int, ...]] = {}
    for prefix in ("fl_link", "fr_link"):
        arm_groups[prefix] = tuple(
            geom_id
            for geom_id in range(model.ngeom)
            if body_name(geom_id).startswith(prefix)
        )
        # link1 intentionally overlaps its fixed shoulder mount at qpos0.
        movable_groups[prefix] = tuple(
            geom_id
            for geom_id in arm_groups[prefix]
            if body_name(geom_id) != f"{prefix}1"
        )
        if len(arm_groups[prefix]) < 2 or not movable_groups[prefix]:
            raise ValueError(f"Mink model has no collision group for {prefix}")
    fixed_group = tuple(
        geom_id
        for geom_id in range(model.ngeom)
        if not body_name(geom_id).startswith(("fl_link", "fr_link"))
    )
    if not fixed_group:
        raise ValueError("Mink model has no fixed robot collision group")

    left = arm_groups["fl_link"]
    right = arm_groups["fr_link"]
    geom_pairs = (
        (left, left),
        (right, right),
        (left, right),
        (movable_groups["fl_link"], fixed_group),
        (movable_groups["fr_link"], fixed_group),
    )
    return MinkCollisionConfig(
        geom_pairs=geom_pairs,
        minimum_distance_m=0.002,
        detection_distance_m=0.03,
        bound_relaxation=0.001,
    )


class RoboTwinMinkIK:
    """Mink pose IK with joint-space interpolation for RoboTwin qpos control."""

    def __init__(
        self,
        task_env: Any,
        grasp_to_robotwin: np.ndarray,
        *,
        model_path: str | Path,
        max_joint_step_rad: float = 0.12,
        max_waypoints_per_segment: int = 64,
        relax_orientation_on_failure: bool = True,
        canonical_seed_on_failure: bool = False,
        ik_config: MinkIKConfig | None = None,
    ) -> None:
        self.task_env = task_env
        transform = np.asarray(grasp_to_robotwin, dtype=np.float64)
        if transform.shape != (4, 4):
            raise ValueError("grasp_to_robotwin must have shape (4, 4)")
        if max_joint_step_rad <= 0.0 or max_waypoints_per_segment < 2:
            raise ValueError("invalid joint interpolation configuration")
        self.grasp_to_robotwin = transform
        self.max_joint_step_rad = float(max_joint_step_rad)
        self.max_waypoints_per_segment = int(max_waypoints_per_segment)
        self.relax_orientation_on_failure = bool(relax_orientation_on_failure)
        self.canonical_seed_on_failure = bool(canonical_seed_on_failure)
        robot = task_env.robot
        arms = {
            "left": MinkArmConfig(
                "fl_link6", tuple(robot.left_arm_joints_name), frame_type="body"
            ),
            "right": MinkArmConfig(
                "fr_link6", tuple(robot.right_arm_joints_name), frame_type="body"
            ),
        }
        self._candidate_seed: dict[str, np.ndarray] = {}
        self._joint_start_overrides: dict[str, np.ndarray] = {}
        self._candidate_paths: list[np.ndarray] = []
        self._completed_paths: list[np.ndarray] = []
        self.solver = MinkIKSolver.from_xml_path(
            model_path,
            arms,
            self._joint_positions,
            collision_provider=_aloha_self_collision_config,
            world_from_model=self._world_from_model,
            config=ik_config or MinkIKConfig(),
        )
        self._support_planes: dict[str, tuple[float, float, float]] = {}
        self._neutral_contact_depths: dict[tuple[int, int], float] = {}
        model = getattr(self.solver, "model", None)
        if model is not None:
            neutral = mujoco.MjData(model)
            neutral.qpos[:] = model.qpos0
            mujoco.mj_forward(model, neutral)
            for contact in neutral.contact[: neutral.ncon]:
                if contact.dist >= 0.0:
                    continue
                key = tuple(sorted((int(contact.geom1), int(contact.geom2))))
                self._neutral_contact_depths[key] = min(
                    float(contact.dist),
                    self._neutral_contact_depths.get(key, 0.0),
                )
        self.reset_stats()

    def _joint_positions(self, arm: str) -> np.ndarray:
        if arm in self._candidate_seed:
            return self._candidate_seed[arm].copy()
        if arm in self._joint_start_overrides:
            return self._joint_start_overrides[arm].copy()
        return _arm_joint_state(self.task_env, arm)[:-1].copy()

    def set_joint_start_override(
        self, arm: str, joints: np.ndarray | None
    ) -> None:
        """Override the measured seed for independently planned future motion."""
        if arm not in {"left", "right"}:
            raise ValueError(f"unknown arm {arm!r}")
        if joints is None:
            self._joint_start_overrides.pop(arm, None)
            return
        values = np.asarray(joints, dtype=np.float64)
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            raise ValueError("joint start override must be a finite 6-vector")
        self._joint_start_overrides[arm] = values.copy()

    def plan_joint_transition(
        self, arm: str, start: np.ndarray, target: np.ndarray
    ) -> np.ndarray | None:
        """Interpolate and collision-check a joint-space transition."""
        start_values = np.asarray(start, dtype=np.float64)
        target_values = np.asarray(target, dtype=np.float64)
        if (
            arm not in {"left", "right"}
            or start_values.shape != (6,)
            or target_values.shape != (6,)
            or not np.all(np.isfinite(start_values))
            or not np.all(np.isfinite(target_values))
        ):
            raise ValueError("joint transition requires an arm and finite 6-vectors")
        path = self._path(start_values, target_values)
        if len(path) > self.max_waypoints_per_segment:
            self.failures["WaypointLimit"] = (
                self.failures.get("WaypointLimit", 0) + 1
            )
            return None
        if not self._path_is_safe(arm, path, start_values):
            return None
        return path.copy()

    def _world_from_model(self, arm: str) -> np.ndarray:
        pose = getattr(self.task_env.robot, f"{arm}_entity_origion_pose")
        return np.asarray(pose.to_transformation_matrix(), dtype=np.float64)

    def set_support_plane(
        self,
        arm: str,
        support_z: float | None,
        *,
        clearance_m: float = 0.003,
        max_joint_step_rad: float = 0.03,
    ) -> None:
        """Set or clear one arm's horizontal world support plane."""
        if arm not in {"left", "right"}:
            raise ValueError(f"unknown arm {arm!r}")
        clearance = float(clearance_m)
        step = float(max_joint_step_rad)
        if (
            not np.isfinite(clearance)
            or clearance < 0.0
            or not np.isfinite(step)
            or step <= 0.0
        ):
            raise ValueError("invalid support collision configuration")
        if support_z is None:
            self._support_planes.pop(arm, None)
            return
        plane = float(support_z)
        if not np.isfinite(plane):
            raise ValueError("support_z must be finite or None")
        self._support_planes[arm] = (plane, clearance, step)

    def clear_support_planes(self) -> None:
        self._support_planes.clear()

    def support_plane_z(self, arm: str) -> float | None:
        setting = self._support_planes.get(arm)
        return None if setting is None else float(setting[0])

    def _arm_qpos_indices(self, arm: str) -> tuple[int, ...]:
        if arm not in {"left", "right"}:
            raise ValueError(f"unknown arm {arm!r}")
        model = self.solver.model
        indices: list[int] = []
        for joint_name in getattr(
            self.task_env.robot, f"{arm}_arm_joints_name"
        ):
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if joint_id < 0:
                raise ValueError(f"Mink model lacks arm joint {joint_name}")
            indices.append(int(model.jnt_qposadr[joint_id]))
        return tuple(indices)

    def active_arm_body_ids(self, arm: str) -> frozenset[int]:
        """Return kinematic descendants of the controlled arm root body."""
        model = self.solver.model
        joint_names = tuple(
            getattr(self.task_env.robot, f"{arm}_arm_joints_name")
        )
        if not joint_names:
            raise ValueError(f"{arm} arm has no controlled joints")
        root_joint = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_names[0]
        )
        if root_joint < 0:
            raise ValueError(f"Mink model lacks arm joint {joint_names[0]}")
        root_body = int(model.jnt_bodyid[root_joint])
        descendants: set[int] = set()
        for body_id in range(model.nbody):
            current = body_id
            visited: set[int] = set()
            while current not in visited:
                if current == root_body:
                    descendants.add(body_id)
                    break
                if current == 0:
                    break
                visited.add(current)
                current = int(model.body_parentid[current])
        if root_body not in descendants:
            descendants.add(root_body)
        return frozenset(descendants)

    def active_arm_geom_ids(self, arm: str) -> tuple[int, ...]:
        bodies = self.active_arm_body_ids(arm)
        return tuple(
            geom_id
            for geom_id in range(self.solver.model.ngeom)
            if int(self.solver.model.geom_bodyid[geom_id]) in bodies
        )

    @staticmethod
    def _geom_minimum_world_z(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        geom_id: int,
        world_from_model: np.ndarray,
    ) -> float:
        """Exact primitive/mesh minimum along world Z."""
        world_model = _pose_matrix(
            world_from_model, name="world_from_model"
        )
        model_rotation = np.asarray(
            data.geom_xmat[geom_id], dtype=np.float64
        ).reshape(3, 3)
        world_rotation = world_model[:3, :3] @ model_rotation
        world_position = (
            world_model[:3, :3]
            @ np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
            + world_model[:3, 3]
        )
        z_row = world_rotation[2]
        size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
        geom_type = int(model.geom_type[geom_id])
        sphere = int(mujoco.mjtGeom.mjGEOM_SPHERE)
        capsule = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
        ellipsoid = int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)
        cylinder = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
        box = int(mujoco.mjtGeom.mjGEOM_BOX)
        mesh = int(mujoco.mjtGeom.mjGEOM_MESH)
        if geom_type == sphere:
            extent = float(size[0])
        elif geom_type == capsule:
            extent = float(size[0] + abs(z_row[2]) * size[1])
        elif geom_type == ellipsoid:
            extent = float(np.linalg.norm(z_row * size))
        elif geom_type == cylinder:
            radial = float(np.linalg.norm(z_row[:2])) * size[0]
            extent = float(radial + abs(z_row[2]) * size[1])
        elif geom_type == box:
            extent = float(np.dot(np.abs(z_row), size))
        elif geom_type == mesh:
            mesh_id = int(model.geom_dataid[geom_id])
            start = int(model.mesh_vertadr[mesh_id])
            stop = start + int(model.mesh_vertnum[mesh_id])
            vertices = np.asarray(
                model.mesh_vert[start:stop], dtype=np.float64
            )
            if len(vertices) == 0:
                return float(world_position[2])
            return float(np.min(
                vertices @ world_rotation[2] + world_position[2]
            ))
        else:
            # Active robot geoms should be standard primitives or meshes.
            # A bounding sphere remains conservative for an unusual type.
            extent = float(np.linalg.norm(size))
        return float(world_position[2] - extent)

    def path_has_support_collision(
        self,
        arm: str,
        path: np.ndarray,
        support_z: float,
        *,
        clearance_m: float = 0.003,
        max_joint_step_rad: float = 0.03,
        start: np.ndarray | None = None,
    ) -> bool:
        """Densely reject active-arm URDF geometry near a support plane."""
        plane = float(support_z)
        clearance = float(clearance_m)
        step = float(max_joint_step_rad)
        if (
            not np.isfinite(plane)
            or not np.isfinite(clearance)
            or clearance < 0.0
            or not np.isfinite(step)
            or step <= 0.0
        ):
            raise ValueError("invalid support collision configuration")
        rows = np.asarray(path, dtype=np.float64)
        joint_count = len(self._arm_qpos_indices(arm))
        if rows.size == 0:
            return False
        if (
            rows.ndim != 2
            or rows.shape[1] != joint_count
            or not np.all(np.isfinite(rows))
        ):
            raise ValueError(
                f"path must be a finite Nx{joint_count} joint array"
            )
        current = (
            _arm_joint_state(self.task_env, arm)[:-1]
            if start is None
            else np.asarray(start, dtype=np.float64)
        )
        if current.shape != (joint_count,) or not np.all(np.isfinite(current)):
            raise ValueError("support path start has invalid joint shape")
        model = self.solver.model
        qpos_indices = self._arm_qpos_indices(arm)
        active_geoms = self.active_arm_geom_ids(arm)
        if not active_geoms:
            return False
        world_model = self._world_from_model(arm)
        data = mujoco.MjData(model)
        self.support_collision_checks += 1
        for target in rows:
            steps = max(
                1,
                int(np.ceil(
                    float(np.max(np.abs(target - current))) / step
                )),
            )
            for alpha in np.linspace(0.0, 1.0, steps + 1):
                joints = current + float(alpha) * (target - current)
                data.qpos[:] = model.qpos0
                data.qpos[list(qpos_indices)] = joints
                mujoco.mj_forward(model, data)
                if any(
                    self._geom_minimum_world_z(
                        model, data, geom_id, world_model
                    )
                    < plane + clearance
                    for geom_id in active_geoms
                ):
                    return True
            current = target
        return False

    def _path_is_safe(
        self, arm: str, path: np.ndarray, start: np.ndarray
    ) -> bool:
        support = self._support_planes.get(arm)
        if support is not None and self.path_has_support_collision(
            arm,
            path,
            support[0],
            clearance_m=support[1],
            max_joint_step_rad=support[2],
            start=start,
        ):
            self.failures["SupportClearance"] = (
                self.failures.get("SupportClearance", 0) + 1
            )
            return False
        if self._path_has_self_collision(arm, path):
            self.failures["SelfCollision"] = (
                self.failures.get("SelfCollision", 0) + 1
            )
            return False
        return True

    def reset_stats(self) -> None:
        self.calls = 0
        self.planner_attempts = 0
        self.successes = 0
        self.relaxed_successes = 0
        self.canonical_seed_successes = 0
        self.over_limit_successes = 0
        self.support_collision_checks = 0
        self.failures: dict[str, int] = {}
        self.first_target: np.ndarray | None = None
        self._stage = 0
        self._candidate_seed = {}
        self._candidate_paths = []
        self._completed_paths = []
        self._candidate_command_rotation: np.ndarray | None = None
        self._candidate_command_targets: list[np.ndarray] = []
        self._completed_command_targets: list[np.ndarray] = []

    @property
    def selected_grasp_command_pose(self) -> np.ndarray | None:
        """World command pose Mink accepted for the completed grasp stage."""
        if len(self._completed_command_targets) != 3:
            return None
        return self._completed_command_targets[1].copy()

    @property
    def completed_command_targets(self) -> tuple[np.ndarray, ...]:
        """Accepted world command poses for pregrasp, grasp, and retreat."""
        return tuple(item.copy() for item in self._completed_command_targets)

    @property
    def completed_paths(self) -> tuple[np.ndarray, ...]:
        """Joint paths for the last complete three-stage grasp plan."""
        return tuple(item.copy() for item in self._completed_paths)

    @staticmethod
    def _with_command_rotation(
        target: np.ndarray, rotation: np.ndarray
    ) -> np.ndarray:
        """Change wrist rotation while preserving the 12 cm logical TCP."""
        result = np.asarray(target, dtype=np.float64).copy()
        tcp = result[:3, 3] + 0.12 * result[:3, 0]
        result[:3, :3] = np.asarray(rotation, dtype=np.float64)
        result[:3, 3] = tcp - 0.12 * result[:3, 0]
        return result

    @staticmethod
    def _nearest_safe_revolute_solution(
        joints: np.ndarray, reference: np.ndarray
    ) -> np.ndarray | None:
        """Use an equivalent [-pi, pi] branch without a >pi stage jump."""
        joints = np.asarray(joints, dtype=np.float64)
        reference = np.asarray(reference, dtype=np.float64)
        if joints.shape != reference.shape or not np.all(np.isfinite(joints)):
            return None
        wrapped = (joints + np.pi) % (2.0 * np.pi) - np.pi
        if np.any(np.abs(reference) > np.pi + 1e-6):
            return None
        if np.any(np.abs(wrapped - reference) > np.pi + 1e-6):
            return None
        return wrapped

    def _path(self, start: np.ndarray, target: np.ndarray) -> np.ndarray:
        delta = float(np.max(np.abs(target - start)))
        steps = max(2, int(np.ceil(delta / self.max_joint_step_rad)))
        return np.linspace(start, target, steps + 1, dtype=np.float64)[1:]

    def _has_disallowed_arm_contact(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        monitored_body_ids: set[int],
        allowed_body_pairs: frozenset[tuple[int, int]] = frozenset(),
    ) -> bool:
        """Reject new arm penetrations while preserving neutral mount overlap."""
        for contact in data.contact[: data.ncon]:
            if contact.dist >= -1e-4:
                continue
            first_body = int(model.geom_bodyid[contact.geom1])
            second_body = int(model.geom_bodyid[contact.geom2])
            body_pair = tuple(sorted((first_body, second_body)))
            if body_pair in allowed_body_pairs:
                continue
            if (
                first_body not in monitored_body_ids
                and second_body not in monitored_body_ids
            ):
                continue
            key = tuple(sorted((int(contact.geom1), int(contact.geom2))))
            neutral_depth = self._neutral_contact_depths.get(key)
            if (
                neutral_depth is not None
                and contact.dist >= neutral_depth - 1e-3
            ):
                continue
            self._last_disallowed_contact = (
                first_body, second_body, float(contact.dist)
            )
            return True
        return False

    def _path_has_self_collision(self, arm: str, path: np.ndarray) -> bool:
        """Return whether the active arm intersects robot or fixed geometry."""
        model = getattr(self.solver, "model", None)
        if model is None:
            return False
        arm_body_ids = set(self.active_arm_body_ids(arm))
        joint_names = tuple(
            getattr(self.task_env.robot, f"{arm}_arm_joints_name")
        )
        qpos_indices = []
        for joint_name in joint_names:
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if joint_id < 0:
                raise ValueError(f"Mink model lacks arm joint {joint_name}")
            qpos_indices.append(int(model.jnt_qposadr[joint_id]))

        data = mujoco.MjData(model)
        for joints in np.asarray(path, dtype=np.float64):
            data.qpos[:] = model.qpos0
            for other_arm, override in self._joint_start_overrides.items():
                data.qpos[list(self._arm_qpos_indices(other_arm))] = override
            data.qpos[qpos_indices] = joints
            mujoco.mj_forward(model, data)
            if self._has_disallowed_arm_contact(model, data, arm_body_ids):
                return True
        return False

    def _solve_safe_target(
        self, arm: str, target: np.ndarray, start: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        self.planner_attempts += 1
        joints = self.solver.solve(arm, target)
        if joints is None:
            return None, None
        joints = self._nearest_safe_revolute_solution(joints, start)
        if joints is None:
            self.failures["UnsafeJointBranch"] = (
                self.failures.get("UnsafeJointBranch", 0) + 1
            )
            return None, None
        path = self._path(start, joints)
        if len(path) > self.max_waypoints_per_segment:
            self.failures["WaypointLimit"] = (
                self.failures.get("WaypointLimit", 0) + 1
            )
            return None, None
        if not self._path_is_safe(arm, path, start):
            return None, None
        return joints, path

    def solve_command_target(
        self,
        arm: str,
        world_command_pose: np.ndarray,
        start: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Solve one exact command-frame follow-up without changing stage state."""
        target = np.asarray(world_command_pose, dtype=np.float64)
        start = np.asarray(start, dtype=np.float64)
        if target.shape != (4, 4):
            raise ValueError("world_command_pose must have shape (4, 4)")
        self.calls += 1
        self._candidate_seed = {arm: start.copy()}
        try:
            joints, path = self._solve_safe_target(arm, target, start)
        finally:
            self._candidate_seed = {}
        if (
            (joints is None or path is None)
            and self.canonical_seed_on_failure
        ):
            canonical_rotation = t3d.quaternions.quat2mat(
                CANONICAL_COMMAND_QUATERNIONS[arm]
            )
            canonical_target = self._with_command_rotation(
                target, canonical_rotation
            )
            canonical_joints, canonical_path = self._solve_safe_target(
                arm, canonical_target, start
            )
            if canonical_joints is not None and canonical_path is not None:
                self._candidate_seed = {arm: canonical_joints.copy()}
                exact_joints, exact_path = self._solve_safe_target(
                    arm, target, canonical_joints
                )
                if exact_joints is not None and exact_path is not None:
                    exact_joints = self._nearest_safe_revolute_solution(
                        exact_joints, start
                    )
                    if exact_joints is not None:
                        direct_path = self._path(start, exact_joints)
                        if (
                            len(direct_path) <= self.max_waypoints_per_segment
                            and self._path_is_safe(
                                arm, direct_path, start
                            )
                        ):
                            joints, path = exact_joints, direct_path
                            self.canonical_seed_successes += 1
        if joints is None or path is None:
            self.failures["NoSolution"] = self.failures.get("NoSolution", 0) + 1
            return None
        self.successes += 1
        return joints.copy(), path.copy(), target.copy()

    def full_robot_path_has_self_collision(
        self,
        actions: list[np.ndarray],
        *,
        max_joint_step_rad: float = 0.03,
        allowed_body_pairs: frozenset[tuple[int, int]] = frozenset(),
    ) -> bool:
        """Check dense paired-arm motion against all robot/fixed geometry."""
        if max_joint_step_rad <= 0.0:
            raise ValueError("max_joint_step_rad must be positive")
        model = getattr(self.solver, "model", None)
        if model is None:
            return False
        rows = [np.asarray(action, dtype=np.float64) for action in actions]
        if any(
            row.shape != (14,) or not np.all(np.isfinite(row))
            for row in rows
        ):
            raise ValueError("bimanual actions must be finite 14D qpos rows")
        if not rows:
            return False
        self.last_full_robot_collision = None

        arm_indices: dict[str, list[int]] = {}
        arm_body_ids: set[int] = set()
        for arm in ("left", "right"):
            indices: list[int] = []
            for joint_name in getattr(
                self.task_env.robot, f"{arm}_arm_joints_name"
            ):
                joint_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                )
                if joint_id < 0:
                    raise ValueError(f"Mink model lacks arm joint {joint_name}")
                indices.append(int(model.jnt_qposadr[joint_id]))
            arm_indices[arm] = indices
            arm_body_ids.update(self.active_arm_body_ids(arm))

        current = np.concatenate(
            (
                _arm_joint_state(self.task_env, "left")[:-1],
                _arm_joint_state(self.task_env, "right")[:-1],
            )
        )
        data = mujoco.MjData(model)
        for row_index, row in enumerate(rows):
            target = np.concatenate((row[:6], row[7:13]))
            steps = max(
                1,
                int(
                    np.ceil(
                        float(np.max(np.abs(target - current)))
                        / max_joint_step_rad
                    )
                ),
            )
            for alpha in np.linspace(0.0, 1.0, steps + 1)[1:]:
                joints = current + float(alpha) * (target - current)
                data.qpos[:] = model.qpos0
                data.qpos[arm_indices["left"]] = joints[:6]
                data.qpos[arm_indices["right"]] = joints[6:]
                mujoco.mj_forward(model, data)
                if self._has_disallowed_arm_contact(
                    model, data, arm_body_ids, allowed_body_pairs
                ):
                    first, second, depth = self._last_disallowed_contact
                    self.last_full_robot_collision = {
                        "row_index": row_index,
                        "body_pair": tuple(
                            mujoco.mj_id2name(
                                model, mujoco.mjtObj.mjOBJ_BODY, body
                            ) or "world"
                            for body in (first, second)
                        ),
                        "depth_m": depth,
                    }
                    return True
            current = target
        return False

    def handoff_path_has_self_collision(
        self,
        actions: list[np.ndarray],
        *,
        max_joint_step_rad: float = 0.03,
    ) -> bool:
        """Check a handoff while allowing terminal-link rendezvous contact."""
        model = getattr(self.solver, "model", None)
        if model is None:
            return False
        terminal_roots = []
        for name in ("fl_link6", "fr_link6"):
            body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, name
            )
            if body_id < 0:
                raise ValueError(f"Mink model lacks handoff body {name}")
            terminal_roots.append(int(body_id))

        def descendants(root: int) -> frozenset[int]:
            bodies = set()
            for body_id in range(model.nbody):
                current = body_id
                visited = set()
                while current not in visited:
                    if current == root:
                        bodies.add(body_id)
                        break
                    if current == 0:
                        break
                    visited.add(current)
                    current = int(model.body_parentid[current])
            return frozenset(bodies)

        left_terminal = descendants(terminal_roots[0])
        right_terminal = descendants(terminal_roots[1])
        allowed = frozenset(
            tuple(sorted((left, right)))
            for left in left_terminal
            for right in right_terminal
        )
        return self.full_robot_path_has_self_collision(
            actions,
            max_joint_step_rad=max_joint_step_rad,
            allowed_body_pairs=allowed,
        )

    def consume_path(self, arm: str, target: np.ndarray) -> np.ndarray | None:
        if not self._completed_paths:
            return None
        path = self._completed_paths.pop(0)
        if not np.allclose(path[-1], target):
            raise ValueError("Mink path endpoint does not match the policy target")
        return path

    def solve(self, arm: str, world_grasp_pose: np.ndarray) -> np.ndarray | None:
        self.calls += 1
        stage = self._stage
        self._stage = (self._stage + 1) % 3
        if stage == 0:
            self._candidate_seed = {arm: self._joint_positions(arm)}
            self._candidate_paths = []
            self._completed_paths = []
            self._candidate_command_rotation = None
            self._candidate_command_targets = []
            self._completed_command_targets = []
        if arm not in self._candidate_seed:
            return None

        target = np.asarray(world_grasp_pose, dtype=np.float64) @ self.grasp_to_robotwin
        if self.first_target is None:
            self.first_target = target.copy()

        start = self._candidate_seed[arm]
        accepted_command_target = target.copy()
        if self._candidate_command_rotation is not None:
            accepted_command_target = self._with_command_rotation(
                target, self._candidate_command_rotation
            )
            joints, path = self._solve_safe_target(
                arm, accepted_command_target, start
            )
        else:
            joints, path = self._solve_safe_target(
                arm, accepted_command_target, start
            )
            if (
                joints is None
                and stage == 0
                and (
                    self.canonical_seed_on_failure
                    or self.relax_orientation_on_failure
                )
            ):
                canonical_rotation = t3d.quaternions.quat2mat(
                    CANONICAL_COMMAND_QUATERNIONS[arm]
                )
                canonical_target = self._with_command_rotation(
                    target, canonical_rotation
                )
                canonical_joints, canonical_path = self._solve_safe_target(
                    arm, canonical_target, start
                )
                if (
                    canonical_joints is not None
                    and canonical_path is not None
                    and self.canonical_seed_on_failure
                ):
                    # Use the canonical pose only as a numerical seed. The
                    # simulator must move directly from the real start state
                    # to the exact M2T2 solution, never through the seed pose.
                    self._candidate_seed[arm] = canonical_joints.copy()
                    exact_joints, exact_seed_path = self._solve_safe_target(
                        arm, target, canonical_joints
                    )
                    if exact_joints is not None and exact_seed_path is not None:
                        exact_joints = self._nearest_safe_revolute_solution(
                            exact_joints, start
                        )
                        if exact_joints is None:
                            self.failures["UnsafeJointBranch"] = (
                                self.failures.get("UnsafeJointBranch", 0) + 1
                            )
                        else:
                            direct_path = self._path(start, exact_joints)
                            if len(direct_path) > self.max_waypoints_per_segment:
                                self.failures["WaypointLimit"] = (
                                    self.failures.get("WaypointLimit", 0) + 1
                                )
                            elif not self._path_is_safe(
                                arm, direct_path, start
                            ):
                                pass
                            else:
                                joints, path = exact_joints, direct_path
                                accepted_command_target = target.copy()
                                self.canonical_seed_successes += 1
                if (
                    joints is None
                    and canonical_joints is not None
                    and canonical_path is not None
                    and self.relax_orientation_on_failure
                ):
                    joints, path = canonical_joints, canonical_path
                    accepted_command_target = canonical_target
                    self.relaxed_successes += 1
            if joints is not None:
                self._candidate_command_rotation = (
                    accepted_command_target[:3, :3].copy()
                )

        if joints is None or path is None:
            self.failures["NoSolution"] = self.failures.get("NoSolution", 0) + 1
            self._candidate_seed = {}
            self._candidate_paths = []
            self._candidate_command_rotation = None
            self._candidate_command_targets = []
            return None
        self._candidate_paths.append(path)
        self._candidate_command_targets.append(accepted_command_target.copy())
        self._candidate_seed[arm] = joints.copy()
        self.successes += 1
        if stage == 2:
            self._completed_paths = [item.copy() for item in self._candidate_paths]
            self._completed_command_targets = [
                item.copy() for item in self._candidate_command_targets
            ]
            self._candidate_paths = []
            self._candidate_command_rotation = None
            self._candidate_command_targets = []
            self._candidate_seed = {}
        return joints.copy()


class QposActionBuffer:
    """Record controller calls as bounded full RoboTwin qpos waypoints."""

    def __init__(
        self,
        task_env: Any,
        planner_ik: RoboTwinMinkIK,
        *,
        max_waypoints_per_segment: int = 64,
        gripper_settle_actions: int = 5,
    ) -> None:
        if max_waypoints_per_segment < 2:
            raise ValueError("max_waypoints_per_segment must be at least 2")
        if gripper_settle_actions < 1:
            raise ValueError("gripper_settle_actions must be at least 1")
        self.task_env = task_env
        self.planner_ik = planner_ik
        self.max_waypoints_per_segment = int(max_waypoints_per_segment)
        self.gripper_settle_actions = int(gripper_settle_actions)
        self.actions: list[np.ndarray] = []
        self.left = np.empty(0)
        self.right = np.empty(0)
        self.left_gripper = 1.0
        self.right_gripper = 1.0
        self.metadata: list[dict[str, Any]] = []
        self._motion_segment_index = 0

    def reset(self) -> None:
        left = _arm_joint_state(self.task_env, "left")
        right = _arm_joint_state(self.task_env, "right")
        self.left, self.left_gripper = left[:-1].copy(), float(left[-1])
        self.right, self.right_gripper = right[:-1].copy(), float(right[-1])
        self.actions = []
        self.metadata = []
        self._motion_segment_index = 0

    def _append(
        self,
        *,
        phase: str,
        arm: str,
        endpoint: bool,
        waypoint_index: int = 1,
        waypoint_count: int = 1,
        command_pose: np.ndarray | None = None,
    ) -> None:
        self.actions.append(
            np.concatenate(
                (self.left, [self.left_gripper], self.right, [self.right_gripper])
            )
        )
        target_qpos = self.left if arm == "left" else self.right
        target_gripper = (
            self.left_gripper if arm == "left" else self.right_gripper
        )
        self.metadata.append(
            {
                "phase": phase,
                "arm": arm,
                "endpoint": bool(endpoint),
                "waypoint_index": int(waypoint_index),
                "waypoint_count": int(waypoint_count),
                "target_qpos": target_qpos.copy(),
                "target_gripper": float(target_gripper),
                "command_pose": (
                    None
                    if command_pose is None
                    else np.asarray(command_pose, dtype=np.float64).copy()
                ),
            }
        )

    def _subsample(self, path: np.ndarray) -> np.ndarray:
        if len(path) > self.max_waypoints_per_segment:
            raise RuntimeError(
                "Mink path exceeds max_waypoints_per_segment; refusing to "
                "violate max_joint_step_rad"
            )
        return path

    def move_joints(self, arm: str, joints: np.ndarray) -> None:
        if arm not in {"left", "right"}:
            raise ValueError(f"unknown arm {arm!r}")
        target = np.asarray(joints, dtype=np.float64)
        current = self.left if arm == "left" else self.right
        if target.shape != current.shape:
            raise ValueError(
                f"{arm} arm target has shape {target.shape}; expected {current.shape}"
            )
        path = self.planner_ik.consume_path(arm, target)
        if path is None:
            raise RuntimeError("planner path cache is missing for joint target")

        phases = ("pregrasp", "grasp", "retreat")
        if self._motion_segment_index >= len(phases):
            raise RuntimeError("received more than three grasp motion segments")
        phase = phases[self._motion_segment_index]
        command_targets = self.planner_ik.completed_command_targets
        command_pose = (
            command_targets[self._motion_segment_index]
            if len(command_targets) == len(phases)
            else None
        )
        waypoints = self._subsample(path)
        waypoint_count = len(waypoints)
        for waypoint_index, waypoint in enumerate(waypoints, start=1):
            if arm == "left":
                self.left = waypoint[: len(self.left)].copy()
            else:
                self.right = waypoint[: len(self.right)].copy()
            endpoint = waypoint_index == waypoint_count
            self._append(
                phase=phase,
                arm=arm,
                endpoint=endpoint,
                waypoint_index=waypoint_index,
                waypoint_count=waypoint_count,
                command_pose=command_pose if endpoint else None,
            )
        self._motion_segment_index += 1

    def open_gripper(self, arm: str) -> None:
        self._set_gripper(arm, 1.0)

    def close_gripper(self, arm: str) -> None:
        self._set_gripper(arm, 0.0)

    def _set_gripper(self, arm: str, value: float) -> None:
        if arm == "left":
            self.left_gripper = value
        elif arm == "right":
            self.right_gripper = value
        else:
            raise ValueError(f"unknown arm {arm!r}")
        phase = "open" if value > 0.5 else "close"
        repeat_count = 1 if phase == "open" else self.gripper_settle_actions
        for waypoint_index in range(1, repeat_count + 1):
            self._append(
                phase=phase,
                arm=arm,
                endpoint=waypoint_index == repeat_count,
                waypoint_index=waypoint_index,
                waypoint_count=repeat_count,
            )


class StagedQposActionBuffer:
    """Compose named one- or two-arm stages into full 14D qpos rows."""

    def __init__(self, task_env: Any, *, max_waypoints_per_segment: int = 64) -> None:
        if max_waypoints_per_segment < 2:
            raise ValueError("max_waypoints_per_segment must be at least 2")
        self.task_env = task_env
        self.max_waypoints_per_segment = int(max_waypoints_per_segment)
        self.actions: list[np.ndarray] = []
        self.metadata: list[dict[str, Any]] = []
        self.left = np.empty(0)
        self.right = np.empty(0)
        self.left_gripper = 1.0
        self.right_gripper = 1.0

    def reset(self) -> None:
        left = _arm_joint_state(self.task_env, "left")
        right = _arm_joint_state(self.task_env, "right")
        self.left, self.left_gripper = left[:-1].copy(), float(left[-1])
        self.right, self.right_gripper = right[:-1].copy(), float(right[-1])
        self.actions = []
        self.metadata = []

    @staticmethod
    def _resample_path(start: np.ndarray, path: np.ndarray, count: int) -> np.ndarray:
        start = np.asarray(start, dtype=np.float64)
        path = np.asarray(path, dtype=np.float64)
        if path.ndim != 2 or path.shape[1:] != start.shape or len(path) < 1:
            raise ValueError("each staged path must contain joint waypoints")
        source = np.vstack((start, path))
        source_progress = np.linspace(0.0, 1.0, len(source))
        target_progress = np.linspace(0.0, 1.0, count + 1)[1:]
        return np.column_stack([
            np.interp(target_progress, source_progress, source[:, joint])
            for joint in range(source.shape[1])
        ])

    @staticmethod
    def _validate_arms(arms: set[str]) -> tuple[str, ...]:
        if not arms or not arms <= {"left", "right"}:
            raise ValueError("stage arms must be a nonempty subset of left/right")
        return tuple(arm for arm in ("left", "right") if arm in arms)

    def _append(
        self,
        *,
        phase: str,
        arms: tuple[str, ...],
        endpoint: bool,
        waypoint_index: int,
        waypoint_count: int,
        command_poses: dict[str, np.ndarray | None] | None,
        target_names: dict[str, str] | None,
        arm_sources: dict[str, str] | None,
        gripper_arms: tuple[str, ...] | None,
    ) -> None:
        command_poses = command_poses or {}
        target_names = target_names or {}
        arm_sources = arm_sources or {}
        self.actions.append(np.concatenate((
            self.left, [self.left_gripper], self.right, [self.right_gripper]
        )))

        def arm_target(arm: str) -> dict[str, Any]:
            qpos = self.left if arm == "left" else self.right
            gripper = self.left_gripper if arm == "left" else self.right_gripper
            command_pose = command_poses.get(arm)
            return {
                "target_qpos": qpos.copy(),
                "target_gripper": float(gripper),
                "command_pose": (
                    None if command_pose is None
                    else np.asarray(command_pose, dtype=np.float64).copy()
                ),
                "target_name": target_names.get(arm),
                "arm_source": arm_sources.get(arm),
            }

        record: dict[str, Any] = {
            "phase": str(phase),
            "arm": arms[0] if len(arms) == 1 else "both",
            "endpoint": bool(endpoint),
            "waypoint_index": int(waypoint_index),
            "waypoint_count": int(waypoint_count),
        }
        if gripper_arms is not None:
            record["gripper_arms"] = list(gripper_arms)
        if len(arms) == 1:
            record.update(arm_target(arms[0]))
        else:
            record["arm_targets"] = {arm: arm_target(arm) for arm in arms}
            sources = {
                arm_sources[arm] for arm in arms if arm_sources.get(arm) is not None
            }
            record["arm_source"] = sources.pop() if len(sources) == 1 else "mixed"
        self.metadata.append(record)

    def move_phase(
        self,
        phase: str,
        paths: dict[str, np.ndarray],
        command_poses: dict[str, np.ndarray] | None = None,
        target_names: dict[str, str] | None = None,
        arm_sources: dict[str, str] | None = None,
    ) -> None:
        arms = self._validate_arms(set(paths))
        arrays = {arm: np.asarray(paths[arm], dtype=np.float64) for arm in arms}
        if any(
            path.ndim != 2
            or path.shape[1:] != (6,)
            or len(path) < 1
            or len(path) > self.max_waypoints_per_segment
            or not np.all(np.isfinite(path))
            for path in arrays.values()
        ):
            raise ValueError("staged paths must be finite bounded Nx6 arrays")
        count = max(len(path) for path in arrays.values())
        starts = {"left": self.left.copy(), "right": self.right.copy()}
        waypoints = {
            arm: self._resample_path(starts[arm], arrays[arm], count)
            for arm in arms
        }
        for index in range(count):
            for arm in arms:
                if arm == "left":
                    self.left = waypoints[arm][index].copy()
                else:
                    self.right = waypoints[arm][index].copy()
            endpoint = index == count - 1
            self._append(
                phase=phase,
                arms=arms,
                endpoint=endpoint,
                waypoint_index=index + 1,
                waypoint_count=count,
                command_poses=command_poses if endpoint else None,
                target_names=target_names,
                arm_sources=arm_sources,
                gripper_arms=None,
            )

    def gripper_phase(
        self,
        phase: str,
        targets: dict[str, float],
        repeats: int,
        target_names: dict[str, str] | None = None,
        arm_sources: dict[str, str] | None = None,
    ) -> None:
        arms = self._validate_arms(set(targets))
        if repeats < 1:
            raise ValueError("gripper phase repeats must be at least one")
        for arm in arms:
            value = float(targets[arm])
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("gripper targets must be finite values in [0, 1]")
            if arm == "left":
                self.left_gripper = value
            else:
                self.right_gripper = value
        for index in range(1, repeats + 1):
            self._append(
                phase=phase,
                arms=arms,
                endpoint=index == repeats,
                waypoint_index=index,
                waypoint_count=repeats,
                command_poses=None,
                target_names=target_names,
                arm_sources=arm_sources,
                gripper_arms=arms,
            )


@dataclass(frozen=True)
class _SingleArmPlacePlan:
    """A prevalidated full placement or safe grasp-and-lift attempt."""

    arm: str
    target_name: str
    arm_source: str
    candidate: GraspCandidate
    paths: tuple[np.ndarray, ...]
    command_targets: tuple[np.ndarray, ...]
    desired_object_pose: np.ndarray
    orientation_source: str = "m2t2"
    completion_level: str = "place"


@dataclass(frozen=True)
class _HandoffArmPlan:
    """One arm's exact, prevalidated stages in an atomic handoff plan."""

    arm: str
    role: str
    target_name: str
    arm_source: str
    candidate: GraspCandidate
    paths: tuple[np.ndarray, ...]
    command_targets: tuple[np.ndarray, ...]
    contact_local_point: tuple[float, float, float]
    gripper_target: float = 0.0
    orientation_source: str = "m2t2"


@dataclass(frozen=True)
class _SimultaneousPickArmPlan:
    """One arm's prevalidated role in a same-object simultaneous pick."""

    arm: str
    target_name: str
    arm_source: str
    candidate: GraspCandidate
    paths: tuple[np.ndarray, np.ndarray, np.ndarray]
    command_targets: tuple[np.ndarray, np.ndarray, np.ndarray]
    contact_local_point: tuple[float, float, float]
    gripper_target: float = 0.0
    orientation_source: str = "m2t2"


@dataclass(frozen=True)
class _BimanualArmPlan:
    """One arm's complete grasp, lift, and held-transport plan."""

    arm: str
    target_name: str
    arm_source: str
    candidate: GraspCandidate
    paths: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    command_targets: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    orientation_source: str = "m2t2"


def _repair_pose_rotation(rotation: np.ndarray, *, name: str) -> np.ndarray:
    """Return a nearby proper rotation, repairing small metadata drift."""
    value = np.asarray(rotation, dtype=np.float64)
    if value.shape != (3, 3) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} rotation must be a finite 3x3 matrix")
    gram = value.T @ value
    determinant = float(np.linalg.det(value))
    if (
        np.allclose(gram, np.eye(3), atol=1e-6, rtol=0.0)
        and np.isclose(determinant, 1.0, atol=1e-6, rtol=0.0)
    ):
        return value.copy()

    left, singular_values, right_transpose = np.linalg.svd(value)
    if (
        singular_values[-1] <= 1e-8
        or np.max(np.abs(singular_values - 1.0)) > 0.05
    ):
        raise ValueError(f"{name} rotation is too malformed to repair")
    repaired = left @ right_transpose
    if np.linalg.det(repaired) < 0.0:
        left[:, -1] *= -1.0
        repaired = left @ right_transpose
    return repaired


def _pose_matrix(pose: Any, *, name: str) -> np.ndarray:
    if hasattr(pose, "to_transformation_matrix"):
        matrix = np.array(
            pose.to_transformation_matrix(), dtype=np.float64, copy=True
        )
    else:
        values = np.asarray(pose, dtype=np.float64)
        if values.shape == (4, 4):
            matrix = values.copy()
        elif values.shape == (7,):
            matrix = np.eye(4, dtype=np.float64)
            matrix[:3, 3] = values[:3]
            matrix[:3, :3] = t3d.quaternions.quat2mat(values[3:])
        else:
            raise ValueError(f"{name} must be a pose7 or a 4x4 matrix")
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 4x4 rigid transform")
    matrix[:3, :3] = _repair_pose_rotation(matrix[:3, :3], name=name)
    return matrix


def _pose7_from_matrix(matrix: np.ndarray) -> np.ndarray:
    pose = _pose_matrix(matrix, name="pose")
    return np.concatenate(
        (pose[:3, 3], t3d.quaternions.mat2quat(pose[:3, :3]))
    )


def _desired_object_pose(
    world_object_pose: np.ndarray,
    world_source_reference_pose: np.ndarray,
    desired_world_source_reference_pose: np.ndarray,
) -> np.ndarray:
    """Move the full object so its source reference equals the destination."""
    world_object = _pose_matrix(world_object_pose, name="world_object_pose")
    world_source = _pose_matrix(
        world_source_reference_pose, name="world_source_reference_pose"
    )
    desired_source = _pose_matrix(
        desired_world_source_reference_pose,
        name="desired_world_source_reference_pose",
    )
    object_from_source = np.linalg.inv(world_object) @ world_source
    return desired_source @ np.linalg.inv(object_from_source)


def _rigid_place_command_pose(
    world_object_pose: np.ndarray,
    world_source_reference_pose: np.ndarray,
    world_grasp_command_pose: np.ndarray,
    desired_world_source_reference_pose: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return desired object and gripper poses while preserving attachment."""
    world_object = _pose_matrix(world_object_pose, name="world_object_pose")
    world_grasp = _pose_matrix(
        world_grasp_command_pose, name="world_grasp_command_pose"
    )
    desired_object = _desired_object_pose(
        world_object,
        world_source_reference_pose,
        desired_world_source_reference_pose,
    )
    object_from_gripper = np.linalg.inv(world_object) @ world_grasp
    return desired_object, desired_object @ object_from_gripper


def _aligned_place_reference_pose(
    world_source_reference_pose: np.ndarray,
    world_destination_pose: np.ndarray,
    world_grasp_command_pose: np.ndarray,
    *,
    arm: str,
    constrain: str,
    z_transform: bool,
) -> np.ndarray:
    """Apply the same reference-frame orientation rules as place_actor()."""
    if arm not in {"left", "right"}:
        raise ValueError(f"unknown placement arm {arm!r}")
    source = _pose_matrix(
        world_source_reference_pose, name="world_source_reference_pose"
    )
    destination = _pose_matrix(
        world_destination_pose, name="world_destination_pose"
    )
    grasp = _pose_matrix(
        world_grasp_command_pose, name="world_grasp_command_pose"
    )
    if constrain not in {"auto", "align", "free"}:
        raise ValueError(f"unknown RoboTwin placement constraint {constrain!r}")

    from envs.utils.transforms import get_place_pose

    kwargs: dict[str, Any] = {"z_transform": bool(z_transform)}
    resolved_constrain = constrain
    if constrain == "auto":
        actor_axis = source[:3, 3] - grasp[:3, 3]
        if np.linalg.norm(actor_axis) < 1e-8:
            actor_axis = grasp[:3, 0].copy()
        resolved_constrain = "align"
        kwargs.update(
            actor_axis=actor_axis,
            actor_axis_type="world",
        )
        if abs(float(np.dot(actor_axis, [0.0, 0.0, 1.0]))) <= 0.1:
            kwargs["align_axis"] = (
                [1.0, 1.0, 0.0]
                if arm == "left"
                else [-1.0, 1.0, 0.0]
            )
        else:
            kwargs["actor_axis"] = grasp[:3, 2].copy()
            kwargs["align_axis"] = [0.0, 1.0, 0.0]

    try:
        with np.errstate(invalid="raise", divide="raise"):
            result = get_place_pose(
                _pose7_from_matrix(source),
                _pose7_from_matrix(destination),
                constrain=resolved_constrain,
                **kwargs,
            )
    except (FloatingPointError, np.linalg.LinAlgError):
        # The shared helper computes arccos(dot) without clipping and can turn
        # numerically valid, nearly parallel vectors into NaNs. Reproduce its
        # orientation rules with clipped dot products as a local fallback.
        actor2world = np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
        ).T
        source_axis = source[:3, :3] @ (
            actor2world[:3, 2]
            if z_transform
            else np.array([0.0, 0.0, 1.0])
        )
        destination_axis = destination[:3, 2]
        source_axis /= np.linalg.norm(source_axis)
        destination_axis /= np.linalg.norm(destination_axis)
        cross = np.cross(source_axis, destination_axis)
        sine = float(np.linalg.norm(cross))
        cosine = float(
            np.clip(np.dot(source_axis, destination_axis), -1.0, 1.0)
        )
        if sine < 1e-8:
            if cosine >= 0.0:
                alignment = np.eye(3)
            else:
                basis = np.eye(3)[np.argmin(np.abs(source_axis))]
                axis = np.cross(source_axis, basis)
                axis /= np.linalg.norm(axis)
                alignment = t3d.axangles.axangle2mat(axis, np.pi)
        else:
            alignment = t3d.axangles.axangle2mat(
                cross / sine, np.arctan2(sine, cosine)
            )
        recovered = source.copy()
        recovered[:3, :3] = alignment @ source[:3, :3]
        recovered[:3, 3] = destination[:3, 3]
        if resolved_constrain == "align":
            align_axes = kwargs.get("align_axis")
            if align_axes is None:
                align_axes = destination[:3, 0].reshape(3, 1)
            else:
                align_axes = np.asarray(
                    align_axes, dtype=np.float64
                ).reshape(-1, 3).T
            align_axes /= np.linalg.norm(align_axes, axis=0)

            actor_axis = np.asarray(
                kwargs.get("actor_axis", [1.0, 0.0, 0.0]),
                dtype=np.float64,
            ).reshape(3)
            if kwargs.get("actor_axis_type", "actor") == "actor":
                actor_axis = recovered[:3, :3] @ actor_axis
            selected_axis = align_axes[:, int(
                np.argmax(actor_axis @ align_axes)
            )]
            target_x = destination[:3, 0]
            target_y = destination[:3, 1]
            actor_projected = (
                np.dot(target_x, actor_axis) * target_x
                + np.dot(target_y, actor_axis) * target_y
            )
            selected_projected = (
                np.dot(target_x, selected_axis) * target_x
                + np.dot(target_y, selected_axis) * target_y
            )
            for projected, label in (
                (actor_projected, "actor"),
                (selected_projected, "alignment"),
            ):
                if np.linalg.norm(projected) < 1e-8:
                    raise ValueError(
                        f"{label} placement axis has no target-plane projection"
                    )
            actor_projected /= np.linalg.norm(actor_projected)
            selected_projected /= np.linalg.norm(selected_projected)
            cross = np.cross(actor_projected, selected_projected)
            sine = float(np.linalg.norm(cross))
            cosine = float(np.clip(
                np.dot(actor_projected, selected_projected), -1.0, 1.0
            ))
            if sine < 1e-8:
                planar_alignment = (
                    np.eye(3) if cosine >= 0.0
                    else t3d.axangles.axangle2mat(destination[:3, 2], np.pi)
                )
            else:
                planar_alignment = t3d.axangles.axangle2mat(
                    cross / sine, np.arctan2(sine, cosine)
                )
            recovered[:3, :3] = (
                planar_alignment @ recovered[:3, :3]
            )
        result = _pose7_from_matrix(recovered)
    return _pose_matrix(np.asarray(result), name="aligned_place_reference_pose")


def _place_offset_axis(
    preplace_axis: str | np.ndarray,
    desired_world_object_pose: np.ndarray,
    desired_world_grasp_command_pose: np.ndarray,
    world_destination_pose: np.ndarray,
) -> np.ndarray:
    """Resolve RoboTwin's pre-placement displacement direction in world."""
    if isinstance(preplace_axis, str):
        if preplace_axis == "grasp":
            axis = (
                np.asarray(desired_world_object_pose)[:3, 3]
                - np.asarray(desired_world_grasp_command_pose)[:3, 3]
            )
        elif preplace_axis == "fp":
            axis = np.asarray(world_destination_pose)[:3, 2]
        else:
            raise ValueError(f"unknown preplace_axis {preplace_axis!r}")
    else:
        axis = np.asarray(preplace_axis, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if axis.shape != (3,) or not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("placement offset axis must be a finite nonzero vector")
    return axis / norm


def _recorded_contact_region(
    value: Any, *, name: str
) -> np.ndarray | None:
    """Normalize an optional procedural contact region in object coordinates."""
    if value is None:
        return None
    points = np.asarray(value, dtype=np.float64)
    if points.size == 0:
        return None
    if points.shape == (3,):
        points = points[None, :]
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not np.all(np.isfinite(points))
    ):
        raise ValueError(f"{name} must contain finite object-local 3D points")
    return points.copy()


def _handoff_contact_regions(
    local_target_points: np.ndarray,
    giver_recorded: Any,
    receiver_recorded: Any,
    *,
    giver_reference_local: np.ndarray | None,
    receiver_reference_local: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    """Resolve role regions from trace data, else PCA endpoints and arm geometry."""
    giver = _recorded_contact_region(
        giver_recorded, name="giver contact region"
    )
    receiver = _recorded_contact_region(
        receiver_recorded, name="receiver contact region"
    )
    local_target = np.asarray(local_target_points, dtype=np.float64)
    if local_target.ndim != 2 or local_target.shape[1:] != (3,):
        raise ValueError("local target points must have shape (N, 3)")
    local_target = local_target[np.all(np.isfinite(local_target), axis=1)]
    source = "recorded_regions"

    if giver is None or receiver is None:
        if len(local_target) < 2:
            raise TargetSelectionFailure(
                "handoff lacks recorded contact regions and segmented geometry"
            )
        centered = local_target - np.mean(local_target, axis=0)
        _, singular_values, axes = np.linalg.svd(
            centered, full_matrices=False
        )
        if not len(singular_values) or singular_values[0] < 1e-6:
            raise TargetSelectionFailure(
                "segmented handoff target has no usable principal axis"
            )
        projection = centered @ axes[0]
        order = np.argsort(projection)
        tail_count = max(1, int(np.ceil(0.15 * len(local_target))))
        low = local_target[order[:tail_count]]
        high = local_target[order[-tail_count:]]
        source = "segmented_pca_fallback"

        def centroid_distance(
            region: np.ndarray, reference: np.ndarray
        ) -> float:
            return float(np.linalg.norm(np.mean(region, axis=0) - reference))

        if giver is not None:
            receiver = max(
                (low, high),
                key=lambda region: float(
                    np.linalg.norm(
                        np.mean(region, axis=0) - np.mean(giver, axis=0)
                    )
                ),
            )
        elif receiver is not None:
            giver = max(
                (low, high),
                key=lambda region: float(
                    np.linalg.norm(
                        np.mean(region, axis=0) - np.mean(receiver, axis=0)
                    )
                ),
            )
        else:
            direct = 0.0
            reverse = 0.0
            if giver_reference_local is not None:
                direct += centroid_distance(low, giver_reference_local)
                reverse += centroid_distance(high, giver_reference_local)
            if receiver_reference_local is not None:
                direct += centroid_distance(high, receiver_reference_local)
                reverse += centroid_distance(low, receiver_reference_local)
            giver, receiver = (
                (low, high) if direct <= reverse else (high, low)
            )

    assert giver is not None and receiver is not None
    centroid_separation = float(
        np.linalg.norm(np.mean(giver, axis=0) - np.mean(receiver, axis=0))
    )
    object_span = (
        float(np.linalg.norm(np.ptp(local_target, axis=0)))
        if len(local_target) >= 2
        else centroid_separation
    )
    scale = centroid_separation if centroid_separation > 1e-6 else object_span
    if object_span > 1e-6:
        scale = min(scale, object_span)
    minimum_separation = max(0.005, 0.25 * scale)
    return giver, receiver, minimum_separation, source


def _grasp_command_tcp(
    world_grasp_pose: np.ndarray,
    grasp_to_robotwin: np.ndarray,
) -> np.ndarray:
    """Return M2T2's predicted contact as RoboTwin's logical TCP."""
    grasp = _pose_matrix(world_grasp_pose, name="world_grasp_pose")
    transform = _pose_matrix(grasp_to_robotwin, name="grasp_to_robotwin")
    command = grasp @ transform
    return command[:3, 3] + 0.12 * command[:3, 0]


def _align_grasp_pose_to_local_contact(
    world_grasp_pose: np.ndarray,
    world_object_pose: np.ndarray,
    contact_local_point: np.ndarray,
    grasp_to_robotwin: np.ndarray,
) -> np.ndarray:
    """Translate a source grasp so its logical TCP reaches a local contact."""
    grasp = _pose_matrix(
        world_grasp_pose, name="world_grasp_pose"
    ).copy()
    world_object = _pose_matrix(
        world_object_pose, name="world_object_pose"
    )
    local_contact = np.asarray(contact_local_point, dtype=np.float64)
    if local_contact.shape != (3,) or not np.all(np.isfinite(local_contact)):
        raise ValueError("contact_local_point must be a finite 3-vector")
    desired_world_tcp = (
        world_object[:3, :3] @ local_contact
        + world_object[:3, 3]
    )
    grasp[:3, 3] += desired_world_tcp - _grasp_command_tcp(
        grasp, grasp_to_robotwin
    )
    return grasp


def _approach_offset_command_pose(
    world_command_pose: np.ndarray, distance_m: float
) -> np.ndarray:
    """Withdraw a RoboTwin command along its calibrated approach axis."""
    command = _pose_matrix(
        world_command_pose, name="world_command_pose"
    ).copy()
    distance = float(distance_m)
    if not np.isfinite(distance) or distance < 0.0:
        raise ValueError("approach offset must be finite and nonnegative")
    command[:3, 3] -= distance * command[:3, 0]
    return command


def _rigid_transport_command_pose(
    world_object_pose: np.ndarray,
    world_functional_pose: np.ndarray,
    world_grasp_command_pose: np.ndarray,
    desired_world_functional_pose: np.ndarray,
) -> np.ndarray:
    """Translate a rigid grasp so its functional point reaches target XYZ."""
    _pose_matrix(world_object_pose, name="world_object_pose")
    world_functional = _pose_matrix(
        world_functional_pose, name="world_functional_pose"
    )
    world_grasp = _pose_matrix(
        world_grasp_command_pose, name="world_grasp_command_pose"
    )
    desired_functional = _pose_matrix(
        desired_world_functional_pose,
        name="desired_world_functional_pose",
    )
    translation = np.eye(4, dtype=np.float64)
    translation[:3, 3] = (
        desired_functional[:3, 3] - world_functional[:3, 3]
    )
    return translation @ world_grasp


def _robot_facing_grasp_pose(
    world_grasp_pose: np.ndarray,
    current_ee_position: np.ndarray,
    grasp_to_robotwin: np.ndarray,
) -> np.ndarray:
    """Keep M2T2's contact while aiming its approach from robot to object."""
    transform = _pose_matrix(
        grasp_to_robotwin, name="grasp_to_robotwin"
    )
    command = _pose_matrix(
        world_grasp_pose, name="world_grasp_pose"
    ) @ transform
    current = np.asarray(current_ee_position, dtype=np.float64)
    if current.shape != (3,) or not np.all(np.isfinite(current)):
        raise ValueError("current_ee_position must be a finite 3-vector")

    tcp = command[:3, 3] + 0.12 * command[:3, 0]
    approach = tcp - current
    approach_norm = float(np.linalg.norm(approach))
    if approach_norm < 1e-8:
        raise ValueError("current EE already coincides with grasp TCP")
    approach /= approach_norm

    closing = command[:3, 1] - np.dot(command[:3, 1], approach) * approach
    if np.linalg.norm(closing) < 1e-6:
        fallback_axis = np.eye(3)[int(np.argmin(np.abs(approach)))]
        closing = fallback_axis - np.dot(fallback_axis, approach) * approach
    closing /= np.linalg.norm(closing)
    lateral = np.cross(approach, closing)
    lateral /= np.linalg.norm(lateral)
    closing = np.cross(lateral, approach)

    adjusted_command = command.copy()
    adjusted_command[:3, :3] = np.column_stack(
        (approach, closing, lateral)
    )
    adjusted_command[:3, 3] = tcp - 0.12 * approach
    return adjusted_command @ np.linalg.inv(transform)


def _approach_roll_grasp_pose(
    world_grasp_pose: np.ndarray,
    grasp_to_robotwin: np.ndarray,
    angle_rad: float,
) -> np.ndarray:
    """Roll the command about its approach axis without moving its TCP.

    The calibrated transform maps M2T2 local +Z to RoboTwin command +X.
    Right-multiplying the command rotation by ``Rx`` therefore preserves the
    command origin/+X and, after mapping back, the M2T2 origin/+Z as well.
    """
    grasp = _pose_matrix(world_grasp_pose, name="world_grasp_pose")
    transform = _pose_matrix(
        grasp_to_robotwin, name="grasp_to_robotwin"
    )
    angle = float(angle_rad)
    if not np.isfinite(angle):
        raise ValueError("approach roll angle must be finite")
    command = grasp @ transform
    rolled_command = command.copy()
    rolled_command[:3, :3] = (
        command[:3, :3]
        @ t3d.axangles.axangle2mat([1.0, 0.0, 0.0], angle)
    )
    return rolled_command @ np.linalg.inv(transform)


def _closest_approach_roll_angle(
    current_rotation: np.ndarray, target_rotation: np.ndarray
) -> float:
    """Return the command-local +X roll closest to a target rotation."""
    current = np.asarray(current_rotation, dtype=np.float64)
    target = np.asarray(target_rotation, dtype=np.float64)
    for value, name in ((current, "current"), (target, "target")):
        if (
            value.shape != (3, 3)
            or not np.all(np.isfinite(value))
            or not np.allclose(value.T @ value, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(value), 1.0, atol=1e-6)
        ):
            raise ValueError(f"{name} rotation must be a rigid 3x3 matrix")
    relative = current.T @ target
    return float(np.arctan2(
        relative[2, 1] - relative[1, 2],
        relative[1, 1] + relative[2, 2],
    ))


def _place_facing_grasp_pose(
    world_grasp_pose: np.ndarray,
    world_object_pose: np.ndarray,
    desired_world_object_pose: np.ndarray,
    arm_reference_position: np.ndarray,
    grasp_to_robotwin: np.ndarray,
    *,
    world_closing_axis: np.ndarray | None = None,
) -> np.ndarray:
    """Preserve a grasp TCP while making its transported pose robot-facing."""
    transform = _pose_matrix(
        grasp_to_robotwin, name="grasp_to_robotwin"
    )
    initial_command = _pose_matrix(
        world_grasp_pose, name="world_grasp_pose"
    ) @ transform
    world_object = _pose_matrix(
        world_object_pose, name="world_object_pose"
    )
    desired_object = _pose_matrix(
        desired_world_object_pose, name="desired_world_object_pose"
    )
    arm_reference = np.asarray(arm_reference_position, dtype=np.float64)
    if arm_reference.shape != (3,) or not np.all(np.isfinite(arm_reference)):
        raise ValueError("arm_reference_position must be a finite 3-vector")

    object_delta = desired_object @ np.linalg.inv(world_object)
    initial_tcp = (
        initial_command[:3, 3] + 0.12 * initial_command[:3, 0]
    )
    final_tcp = (
        object_delta[:3, :3] @ initial_tcp + object_delta[:3, 3]
    )
    approach = final_tcp - arm_reference
    approach_norm = float(np.linalg.norm(approach))
    if approach_norm < 1e-8:
        raise ValueError("arm reference already coincides with final grasp TCP")
    approach /= approach_norm

    if world_closing_axis is None:
        transported_closing = (
            object_delta[:3, :3] @ initial_command[:3, 1]
        )
    else:
        closing_axis = np.asarray(world_closing_axis, dtype=np.float64)
        closing_norm = float(np.linalg.norm(closing_axis))
        if closing_axis.shape != (3,) or not np.isfinite(closing_norm) or closing_norm < 1e-8:
            raise ValueError("world_closing_axis must be a finite nonzero 3-vector")
        transported_closing = object_delta[:3, :3] @ (closing_axis / closing_norm)
        transported_raw = object_delta[:3, :3] @ initial_command[:3, 1]
        if np.dot(transported_closing, transported_raw) < 0.0:
            transported_closing = -transported_closing
    closing = transported_closing - np.dot(
        transported_closing, approach
    ) * approach
    if np.linalg.norm(closing) < 1e-6:
        fallback_axis = np.eye(3)[int(np.argmin(np.abs(approach)))]
        closing = fallback_axis - np.dot(fallback_axis, approach) * approach
    closing /= np.linalg.norm(closing)
    lateral = np.cross(approach, closing)
    lateral /= np.linalg.norm(lateral)
    closing = np.cross(lateral, approach)

    final_command = np.eye(4, dtype=np.float64)
    final_command[:3, :3] = np.column_stack(
        (approach, closing, lateral)
    )
    final_command[:3, 3] = final_tcp - 0.12 * approach
    adjusted_initial_command = np.linalg.inv(object_delta) @ final_command
    return adjusted_initial_command @ np.linalg.inv(transform)

def _narrow_axis_grasp_poses(
    world_grasp_pose: np.ndarray,
    world_object_pose: np.ndarray,
    desired_world_object_pose: np.ndarray,
    arm_reference_position: np.ndarray,
    grasp_to_robotwin: np.ndarray,
    world_narrow_axis: np.ndarray,
    canonical_final_approach: np.ndarray,
    *,
    max_approaches: int = 10,
) -> tuple[np.ndarray, ...]:
    """Sweep bounded approach rolls while preserving a target-narrow jaw axis."""
    if max_approaches < 1:
        raise ValueError("max_approaches must be positive")
    transform = _pose_matrix(
        grasp_to_robotwin, name="grasp_to_robotwin"
    )
    initial_command = _pose_matrix(
        world_grasp_pose, name="world_grasp_pose"
    ) @ transform
    world_object = _pose_matrix(
        world_object_pose, name="world_object_pose"
    )
    desired_object = _pose_matrix(
        desired_world_object_pose, name="desired_world_object_pose"
    )
    arm_reference = np.asarray(arm_reference_position, dtype=np.float64)
    if arm_reference.shape != (3,) or not np.all(np.isfinite(arm_reference)):
        raise ValueError("arm_reference_position must be a finite 3-vector")
    narrow = np.asarray(world_narrow_axis, dtype=np.float64)
    narrow_norm = float(np.linalg.norm(narrow))
    if narrow.shape != (3,) or not np.isfinite(narrow_norm) or narrow_norm < 1e-8:
        raise ValueError("world_narrow_axis must be a finite nonzero 3-vector")
    narrow = narrow / narrow_norm
    if np.dot(narrow, initial_command[:3, 1]) < 0.0:
        narrow = -narrow

    object_delta = desired_object @ np.linalg.inv(world_object)
    initial_tcp = (
        initial_command[:3, 3] + 0.12 * initial_command[:3, 0]
    )
    final_tcp = (
        object_delta[:3, :3] @ initial_tcp + object_delta[:3, 3]
    )
    final_narrow = object_delta[:3, :3] @ narrow

    def projected(axis: np.ndarray, normal: np.ndarray) -> np.ndarray | None:
        value = np.asarray(axis, dtype=np.float64)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            return None
        value = value - np.dot(value, normal) * normal
        norm = float(np.linalg.norm(value))
        return None if norm < 1e-8 else value / norm

    seed_axes: list[np.ndarray] = []
    initial_seed = projected(initial_tcp - arm_reference, narrow)
    if initial_seed is not None:
        seed_axes.append(initial_seed)
    final_seed = projected(final_tcp - arm_reference, final_narrow)
    if final_seed is not None:
        final_seed = projected(object_delta[:3, :3].T @ final_seed, narrow)
        if final_seed is not None:
            seed_axes.append(final_seed)
    canonical = projected(canonical_final_approach, final_narrow)
    if canonical is not None:
        canonical = projected(object_delta[:3, :3].T @ canonical, narrow)
        if canonical is not None:
            seed_axes.append(canonical)
    if not seed_axes:
        return ()

    approaches: list[np.ndarray] = []

    def add(axis: np.ndarray) -> None:
        value = projected(axis, narrow)
        if value is None:
            return
        if any(np.linalg.norm(value - existing) < 1e-4 for existing in approaches):
            return
        if len(approaches) < max_approaches:
            approaches.append(value)

    for seed in seed_axes:
        add(seed)
    primary = seed_axes[0]
    for angle in (
        np.pi,
        np.pi / 4.0,
        -np.pi / 4.0,
        np.pi / 2.0,
        -np.pi / 2.0,
        3.0 * np.pi / 4.0,
        -3.0 * np.pi / 4.0,
    ):
        add(t3d.axangles.axangle2mat(narrow, angle) @ primary)

    poses: list[np.ndarray] = []
    for approach in approaches:
        lateral = np.cross(approach, narrow)
        lateral /= np.linalg.norm(lateral)
        closing = np.cross(lateral, approach)
        command = np.eye(4, dtype=np.float64)
        command[:3, :3] = np.column_stack(
            (approach, closing, lateral)
        )
        command[:3, 3] = initial_tcp - 0.12 * approach
        poses.append(command @ np.linalg.inv(transform))
    return tuple(poses)



def _elongated_object_axis(
    scene: SceneObservation,
    target: ObjectState,
    *,
    minimum_variance_ratio: float = 2.0,
) -> np.ndarray | None:
    """Return the target cloud's principal axis only when clearly elongated."""
    if minimum_variance_ratio <= 1.0:
        raise ValueError("minimum_variance_ratio must be greater than one")
    labels = np.asarray(scene.instance_labels)
    points = np.asarray(scene.xyz, dtype=np.float64)[
        labels == target.instance_id
    ]
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 3:
        return None
    centered = points - points.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(
        centered.T @ centered / len(centered)
    )
    if eigenvalues[-1] <= minimum_variance_ratio * max(
        float(eigenvalues[-2]), 1e-12
    ):
        return None
    return eigenvectors[:, -1]


def _target_narrow_axis(
    scene: SceneObservation,
    target: ObjectState,
    desired_world_object_pose: np.ndarray,
    final_approach_axis: np.ndarray,
    *,
    maximum_approach_alignment: float = 0.35,
) -> np.ndarray | None:
    """Choose the narrowest GT object axis compatible with final approach."""
    if (
        not np.isfinite(maximum_approach_alignment)
        or maximum_approach_alignment < 0.0
        or maximum_approach_alignment >= 1.0
    ):
        raise ValueError("maximum_approach_alignment must be in [0, 1)")
    world_object = _pose_matrix(
        target.world_pose, name="target.world_pose"
    )
    desired_object = _pose_matrix(
        desired_world_object_pose, name="desired_world_object_pose"
    )
    approach = np.array(final_approach_axis, dtype=np.float64, copy=True)
    approach_norm = float(np.linalg.norm(approach))
    if approach.shape != (3,) or not np.isfinite(approach_norm) or approach_norm < 1e-8:
        raise ValueError("final_approach_axis must be a finite nonzero 3-vector")
    approach /= approach_norm

    labels = np.asarray(scene.instance_labels)
    points = np.asarray(scene.xyz, dtype=np.float64)[
        labels == target.instance_id
    ]
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 2:
        return None
    local_points = (
        points - world_object[:3, 3]
    ) @ world_object[:3, :3]
    extents = np.ptp(local_points, axis=0)
    eligible = [
        axis
        for axis in range(3)
        if abs(float(np.dot(desired_object[:3, axis], approach)))
        <= maximum_approach_alignment
    ]
    if not eligible:
        return None
    selected = min(eligible, key=lambda axis: float(extents[axis]))
    return world_object[:3, selected].copy()


def _target_width_along_axis(
    scene: SceneObservation,
    target: ObjectState,
    axis: np.ndarray,
) -> float | None:
    """Measure segmented target extent along a unit jaw-closing axis."""
    direction = np.array(axis, dtype=np.float64, copy=True)
    norm = float(np.linalg.norm(direction))
    if direction.shape != (3,) or not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("axis must be a finite nonzero 3-vector")
    direction /= norm
    labels = np.asarray(scene.instance_labels)
    points = np.asarray(scene.xyz, dtype=np.float64)[
        labels == target.instance_id
    ]
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 2:
        return None
    projections = points @ direction
    return float(np.ptp(projections))


def _estimate_target_support_plane_z(
    scene: SceneObservation,
    target: ObjectState,
    *,
    maximum_gap_m: float = 0.020,
    minimum_points: int = 24,
    mode_bin_m: float = 0.002,
    maximum_plane_spread_m: float = 0.0035,
) -> float | None:
    """Estimate a nearby horizontal support from non-target RGB-D points."""
    if (
        not np.isfinite(maximum_gap_m)
        or maximum_gap_m <= 0.0
        or minimum_points < 3
        or not np.isfinite(mode_bin_m)
        or mode_bin_m <= 0.0
        or not np.isfinite(maximum_plane_spread_m)
        or maximum_plane_spread_m <= 0.0
    ):
        raise ValueError("invalid support-plane estimator configuration")
    points = np.asarray(scene.xyz, dtype=np.float64).reshape(-1, 3)
    labels = np.asarray(scene.instance_labels).reshape(-1)
    if len(points) != len(labels):
        raise ValueError("scene points and instance labels must have equal length")
    finite = np.all(np.isfinite(points), axis=1)
    target_mask = finite & (labels == target.instance_id)
    target_points = points[target_mask]
    if len(target_points) < 4:
        return None

    lower_z = float(np.quantile(target_points[:, 2], 0.01))
    xy_low, xy_high = np.quantile(
        target_points[:, :2], [0.01, 0.99], axis=0
    )
    xy_span = np.maximum(xy_high - xy_low, 0.0)
    margin = float(
        np.clip(0.5 * max(float(np.max(xy_span)), 0.02), 0.025, 0.08)
    )
    near_xy = np.all(
        (points[:, :2] >= xy_low - margin)
        & (points[:, :2] <= xy_high + margin),
        axis=1,
    )
    below_target = (
        (points[:, 2] >= lower_z - maximum_gap_m)
        & (points[:, 2] <= lower_z + 0.001)
    )
    candidates = points[finite & ~target_mask & near_xy & below_target]
    if len(candidates) < minimum_points:
        return None

    bins = np.rint(candidates[:, 2] / mode_bin_m).astype(np.int64)
    values, counts = np.unique(bins, return_counts=True)
    peak_count = int(np.max(counts))
    peak_values = values[counts == peak_count]
    # When equal-size surfaces are present, prefer the one closest below the
    # target rather than a lower shelf or background patch.
    peak = int(np.max(peak_values))
    center_z = float(peak) * mode_bin_m
    plane_points = candidates[
        np.abs(candidates[:, 2] - center_z) <= mode_bin_m
    ]
    if len(plane_points) < minimum_points:
        return None
    spread = float(
        np.quantile(plane_points[:, 2], 0.90)
        - np.quantile(plane_points[:, 2], 0.10)
    )
    centered_xy = plane_points[:, :2] - np.mean(
        plane_points[:, :2], axis=0
    )
    if (
        spread > maximum_plane_spread_m
        or np.linalg.matrix_rank(centered_xy, tol=1e-4) < 2
    ):
        return None
    support_z = float(np.median(plane_points[:, 2]))
    gap = lower_z - support_z
    if gap < -0.001 or gap > maximum_gap_m:
        return None
    return support_z


def _target_m2t2_palm_depth(
    scene: SceneObservation,
    target: ObjectState,
    world_grasp_pose: np.ndarray,
    grasp_to_robotwin: np.ndarray,
    *,
    quantile: float = 0.01,
) -> float | None:
    """Return the target's robust minimum +Z depth in the M2T2 frame.

    M2T2's gripper geometry is defined in the source grasp frame, while Mink
    consumes ``world_grasp_pose @ grasp_to_robotwin``. Reconstructing that
    source frame from the calibrated command transform makes the frame
    convention explicit and keeps this check correct for non-default rigid
    command transforms. Only GT-segmented target points contribute.
    """
    q = float(quantile)
    if not np.isfinite(q) or not 0.0 <= q <= 0.5:
        raise ValueError("quantile must be finite and in [0, 0.5]")
    source_grasp = _pose_matrix(
        world_grasp_pose, name="world_grasp_pose"
    )
    command_transform = _pose_matrix(
        grasp_to_robotwin, name="grasp_to_robotwin"
    )
    world_command = source_grasp @ command_transform
    world_m2t2 = world_command @ np.linalg.inv(command_transform)
    labels = np.asarray(scene.instance_labels)
    points = np.asarray(scene.xyz, dtype=np.float64)[
        labels == target.instance_id
    ]
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 2:
        return None
    local_points = (
        points - world_m2t2[:3, 3]
    ) @ world_m2t2[:3, :3]
    return float(np.quantile(local_points[:, 2], q))


class BimanualQposActionBuffer:
    """Compose two independent IK paths into synchronized RoboTwin qpos rows."""

    def __init__(
        self,
        task_env: Any,
        *,
        gripper_settle_actions: int = 5,
    ) -> None:
        if gripper_settle_actions < 1:
            raise ValueError("gripper_settle_actions must be at least 1")
        self.task_env = task_env
        self.gripper_settle_actions = int(gripper_settle_actions)
        self.actions: list[np.ndarray] = []
        self.metadata: list[dict[str, Any]] = []
        self.left = np.empty(0)
        self.right = np.empty(0)
        self.left_gripper = 1.0
        self.right_gripper = 1.0

    def reset(self) -> None:
        left = _arm_joint_state(self.task_env, "left")
        right = _arm_joint_state(self.task_env, "right")
        self.left, self.left_gripper = left[:-1].copy(), float(left[-1])
        self.right, self.right_gripper = right[:-1].copy(), float(right[-1])
        self.actions = []
        self.metadata = []

    @staticmethod
    def _resample_path(
        start: np.ndarray, path: np.ndarray, count: int
    ) -> np.ndarray:
        start = np.asarray(start, dtype=np.float64)
        path = np.asarray(path, dtype=np.float64)
        if path.ndim != 2 or path.shape[1:] != start.shape or len(path) < 1:
            raise ValueError("each bimanual path must contain joint waypoints")
        source = np.vstack((start, path))
        source_progress = np.linspace(0.0, 1.0, len(source))
        target_progress = np.linspace(0.0, 1.0, count + 1)[1:]
        return np.column_stack(
            [
                np.interp(target_progress, source_progress, source[:, joint])
                for joint in range(source.shape[1])
            ]
        )

    def _append(
        self,
        *,
        phase: str,
        endpoint: bool,
        waypoint_index: int,
        waypoint_count: int,
        plans: dict[str, _BimanualArmPlan],
        command_poses: dict[str, np.ndarray | None] | None = None,
    ) -> None:
        command_poses = command_poses or {}
        self.actions.append(
            np.concatenate(
                (
                    self.left,
                    [self.left_gripper],
                    self.right,
                    [self.right_gripper],
                )
            )
        )
        arm_targets: dict[str, dict[str, Any]] = {}
        for arm in ("left", "right"):
            plan = plans[arm]
            qpos = self.left if arm == "left" else self.right
            gripper = (
                self.left_gripper if arm == "left" else self.right_gripper
            )
            command_pose = command_poses.get(arm)
            arm_targets[arm] = {
                "target_qpos": qpos.copy(),
                "target_gripper": float(gripper),
                "command_pose": (
                    None
                    if command_pose is None
                    else np.asarray(command_pose, dtype=np.float64).copy()
                ),
                "target_name": plan.target_name,
                "arm_source": plan.arm_source,
            }
        sources = {plan.arm_source for plan in plans.values()}
        self.metadata.append(
            {
                "phase": phase,
                "arm": "both",
                "endpoint": bool(endpoint),
                "waypoint_index": int(waypoint_index),
                "waypoint_count": int(waypoint_count),
                "arm_targets": arm_targets,
                "arm_source": sources.pop() if len(sources) == 1 else "mixed",
            }
        )

    def _move_phase(
        self,
        phase: str,
        path_index: int,
        plans: dict[str, _BimanualArmPlan],
    ) -> None:
        left_path = plans["left"].paths[path_index]
        right_path = plans["right"].paths[path_index]
        count = max(len(left_path), len(right_path))
        left_waypoints = self._resample_path(self.left, left_path, count)
        right_waypoints = self._resample_path(self.right, right_path, count)
        for index, (left, right) in enumerate(
            zip(left_waypoints, right_waypoints), start=1
        ):
            self.left = left.copy()
            self.right = right.copy()
            endpoint = index == count
            self._append(
                phase=phase,
                endpoint=endpoint,
                waypoint_index=index,
                waypoint_count=count,
                plans=plans,
                command_poses=(
                    {
                        arm: plans[arm].command_targets[path_index]
                        for arm in ("left", "right")
                    }
                    if endpoint
                    else None
                ),
            )

    def build(
        self,
        left_plan: _BimanualArmPlan,
        right_plan: _BimanualArmPlan,
    ) -> list[np.ndarray]:
        plans = {"left": left_plan, "right": right_plan}
        if left_plan.arm != "left" or right_plan.arm != "right":
            raise ValueError("bimanual plans must contain one plan per arm")
        for plan in plans.values():
            if len(plan.paths) != 4 or len(plan.command_targets) != 4:
                raise ValueError(
                    "bimanual plans require pregrasp/grasp/lift/transport"
                )

        self.reset()
        self.left_gripper = 1.0
        self.right_gripper = 1.0
        self._append(
            phase="open",
            endpoint=True,
            waypoint_index=1,
            waypoint_count=1,
            plans=plans,
        )
        self._move_phase("pregrasp", 0, plans)
        self._move_phase("grasp", 1, plans)

        self.left_gripper = 0.0
        self.right_gripper = 0.0
        for index in range(1, self.gripper_settle_actions + 1):
            self._append(
                phase="close",
                endpoint=index == self.gripper_settle_actions,
                waypoint_index=index,
                waypoint_count=self.gripper_settle_actions,
                plans=plans,
            )
        self._move_phase("lift", 2, plans)
        self._move_phase("transport", 3, plans)
        return [action.copy() for action in self.actions]


class ConfidenceRankedGrasps:
    """Filter target grasps and rank solely by raw M2T2 confidence."""

    def __init__(
        self,
        grasps: M2T2GraspGenerator,
        min_confidence: float = 0.0,
    ) -> None:
        self.grasps = grasps
        self.min_confidence = float(min_confidence)
        self.last_candidates: list[GraspCandidate] = []

    def propose(
        self, observation: SceneObservation, target: ObjectState
    ) -> list[GraspCandidate]:
        self.last_candidates = []
        candidates = [
            candidate
            for candidate in self.grasps.propose(observation, target)
            if candidate.confidence >= self.min_confidence
        ]

        # Python's sort is stable, so equal-confidence M2T2 candidates retain
        # their generator order. M2T2 poses are preserved exactly: neither
        # orientation nor contact position is rescored or moved here.
        self.last_candidates = sorted(
            candidates,
            key=lambda item: item.confidence,
            reverse=True,
        )
        return list(self.last_candidates)


# Backward-compatible name and call signature. The legacy arm and transform
# arguments are deliberately ignored: they cannot affect confidence ranking.
class ReachabilityRankedGrasps(ConfidenceRankedGrasps):
    def __init__(
        self,
        grasps: M2T2GraspGenerator,
        arm: Any | None = None,
        grasp_to_robotwin: np.ndarray | None = None,
        min_confidence: float = 0.0,
    ) -> None:
        del arm, grasp_to_robotwin
        super().__init__(grasps, min_confidence=min_confidence)


def _robotwin_ground_truth_arm(task_env: Any, target_name: str) -> str | None:
    """Return the unique expert arm associated with the active target."""
    plan = getattr(task_env, "heuristic_task_plan", None)
    matching_stages = [
        stage
        for stage in getattr(plan, "stages", ())
        if getattr(stage, "target", None) == target_name
    ]
    if len(matching_stages) != 1:
        return None
    arm = getattr(matching_stages[0], "arm", None)
    if arm is None:
        return None
    normalized = str(arm).strip().lower()
    return normalized if normalized in {"left", "right"} else None


def _structural_handoff_stages(
    stages: tuple[Any, ...],
) -> tuple[Any, Any, Any] | None:
    """Select one ordered Pick/Handoff/Place chain from procedural stages."""
    handoffs = [
        (index, stage)
        for index, stage in enumerate(stages)
        if all(
            hasattr(stage, name)
            for name in ("object", "from_arm", "to_arm")
        )
        and (
            hasattr(stage, "rendezvous_pose")
            or hasattr(stage, "rendezvous_pose_attr")
        )
    ]
    if not handoffs:
        return None

    def normalized_arm(stage: Any) -> str | None:
        arm = getattr(stage, "arm", None)
        if arm is None:
            return None
        value = str(arm).strip().lower()
        return value if value in {"left", "right"} else ""

    def group_id(stage: Any) -> int | None:
        value = getattr(stage, "group_id", None)
        return None if value is None else int(value)

    def nearest_grouped(
        candidates: list[tuple[int, Any]],
        handoff_stage: Any,
        *,
        preceding: bool,
    ) -> Any:
        if not candidates:
            side = "preceding Pick" if preceding else "following Place"
            raise TargetSelectionFailure(
                f"handoff task lacks a matching {side}"
            )
        reference_group = group_id(handoff_stage)
        grouped = [
            (index, stage, group_id(stage))
            for index, stage in candidates
            if group_id(stage) is not None
        ]
        if reference_group is not None and grouped:
            eligible = [
                item
                for item in grouped
                if (
                    item[2] <= reference_group
                    if preceding
                    else item[2] >= reference_group
                )
            ]
            if not eligible:
                raise TargetSelectionFailure(
                    "handoff stage groups violate procedural order"
                )
            selected_group = (
                max(item[2] for item in eligible)
                if preceding
                else min(item[2] for item in eligible)
            )
            nearest = [
                item for item in eligible if item[2] == selected_group
            ]
            if len(nearest) != 1:
                raise TargetSelectionFailure(
                    "handoff stage group is structurally ambiguous"
                )
            return nearest[0][1]
        return candidates[-1 if preceding else 0][1]

    chains: list[tuple[Any, Any, Any]] = []
    for handoff_index, handoff in handoffs:
        giver = str(handoff.from_arm).strip().lower()
        receiver = str(handoff.to_arm).strip().lower()
        if {giver, receiver} != {"left", "right"}:
            raise TargetSelectionFailure(
                "handoff requires distinct left/right giver and receiver arms"
            )
        pick_candidates = [
            (index, stage)
            for index, stage in enumerate(stages[:handoff_index])
            if getattr(stage, "target", None) == handoff.object
            and normalized_arm(stage) in {None, giver}
        ]
        place_candidates = [
            (index, stage)
            for index, stage in enumerate(
                stages[handoff_index + 1 :],
                start=handoff_index + 1,
            )
            if getattr(stage, "object", None) == handoff.object
            and (
                hasattr(stage, "destination")
                or hasattr(stage, "target_pose")
            )
            and normalized_arm(stage) in {None, receiver}
        ]
        pick = nearest_grouped(
            pick_candidates, handoff, preceding=True
        )
        place = nearest_grouped(
            place_candidates, handoff, preceding=False
        )
        chains.append((pick, handoff, place))
    if len(chains) != 1:
        raise TargetSelectionFailure(
            "task plan contains multiple structural handoff chains"
        )
    return chains[0]


class RoboTwinHeuristicRuntime:
    def __init__(
        self,
        *,
        task_env: Any,
        grasps: M2T2GraspGenerator,
        config: PolicyConfig,
        automatic_target: bool,
        automatic_arm: bool,
        grasp_to_robotwin: np.ndarray,
        max_waypoints_per_segment: int = 64,
        max_joint_step_rad: float = 0.12,
        gripper_settle_actions: int = 5,
        mink_model_path: str | Path | None = None,
        mink_config: MinkIKConfig | None = None,
        relax_orientation_on_failure: bool = False,
        canonical_seed_on_failure: bool = True,
        save_grasp_visualizations: bool = True,
        visualization_dir: str | Path | None = None,
        max_visualized_grasps: int | None = None,
        max_visualized_points: int = 30_000,
        grasp_visualization_scale: int = 4,
        bimanual_max_plans_per_arm: int = 4,
        bimanual_lift_m: float = 0.10,
        bimanual_collision_step_rad: float = 0.03,
        bimanual_max_jaw_axis_alignment: float = 0.75,
        bimanual_max_target_width_m: float = 0.10,
        support_collision_filter_enabled: bool = False,
    ) -> None:
        self.task_env = task_env
        self.grasps = grasps
        self.config = config
        self.automatic_target = automatic_target
        self.automatic_arm = automatic_arm
        self.save_grasp_visualizations = bool(save_grasp_visualizations)
        self.visualization_dir = (
            None
            if visualization_dir is None
            else Path(visualization_dir).expanduser().resolve()
        )
        self.max_visualized_grasps = (
            None
            if max_visualized_grasps is None
            else int(max_visualized_grasps)
        )
        self.max_visualized_points = int(max_visualized_points)
        self.grasp_visualization_scale = int(grasp_visualization_scale)
        self.bimanual_max_plans_per_arm = int(bimanual_max_plans_per_arm)
        self.bimanual_lift_m = float(bimanual_lift_m)
        self.bimanual_collision_step_rad = float(
            bimanual_collision_step_rad
        )
        self.bimanual_max_jaw_axis_alignment = float(
            bimanual_max_jaw_axis_alignment
        )
        self.bimanual_max_target_width_m = float(
            bimanual_max_target_width_m
        )
        self.support_collision_filter_enabled = bool(
            support_collision_filter_enabled
        )
        if (
            self.max_visualized_grasps is not None
            and self.max_visualized_grasps <= 0
        ) or (
            self.max_visualized_points <= 0
            or self.grasp_visualization_scale <= 0
        ):
            raise ValueError("visualization limits must be positive")
        if (
            self.bimanual_max_plans_per_arm <= 0
            or self.bimanual_lift_m <= 0.0
            or self.bimanual_collision_step_rad <= 0.0
            or not 0.0 < self.bimanual_max_jaw_axis_alignment < 1.0
            or not np.isfinite(self.bimanual_max_target_width_m)
            or self.bimanual_max_target_width_m <= 0.0
        ):
            raise ValueError("bimanual planning limits must be positive")
        self._visualization_index = 0
        self._grasp_attempted = False
        self._action_metadata_override: list[dict[str, Any]] | None = None
        self.simulator = SceneSnapshot()
        if mink_model_path is None:
            mink_model_path = (
                Path(__file__).resolve().parents[2]
                / "assets" / "embodiments" / "aloha-agilex"
                / "urdf" / "arx5_description_isaac.urdf"
            )
        self.ik = RoboTwinMinkIK(
            task_env,
            grasp_to_robotwin,
            model_path=mink_model_path,
            max_joint_step_rad=max_joint_step_rad,
            max_waypoints_per_segment=max_waypoints_per_segment,
            relax_orientation_on_failure=relax_orientation_on_failure,
            canonical_seed_on_failure=canonical_seed_on_failure,
            ik_config=mink_config,
        )
        self.controller = QposActionBuffer(
            task_env,
            self.ik,
            max_waypoints_per_segment=max_waypoints_per_segment,
            gripper_settle_actions=gripper_settle_actions,
        )
        self.staged_controller = StagedQposActionBuffer(
            task_env,
            max_waypoints_per_segment=max_waypoints_per_segment,
        )
        self.gripper_settle_actions = int(gripper_settle_actions)
        self.bimanual_controller = BimanualQposActionBuffer(
            task_env,
            gripper_settle_actions=gripper_settle_actions,
        )
        self.backend = grasps.backend

    def _configure_support_plane(
        self,
        scene: SceneObservation,
        target: ObjectState,
        arms: tuple[str, ...],
    ) -> float | None:
        support_z = (
            _estimate_target_support_plane_z(scene, target)
            if getattr(self, "support_collision_filter_enabled", False)
            and hasattr(scene, "xyz")
            and hasattr(scene, "instance_labels")
            else None
        )
        setter = getattr(self.ik, "set_support_plane", None)
        if setter is not None:
            for arm in arms:
                setter(arm, support_z)
        return support_z

    def _support_plane_for_arm(self, arm: str) -> float | None:
        getter = getattr(self.ik, "support_plane_z", None)
        return None if getter is None else getter(arm)

    def _target_names(self, scene: SceneObservation) -> tuple[str, ...]:
        if not self.automatic_target:
            return (self.config.object_name,)
        plan = getattr(self.task_env, "heuristic_task_plan", None)
        planned: list[str] = []
        for stage in getattr(plan, "stages", ()):
            name = getattr(stage, "target", None)
            if name is not None and name not in planned:
                planned.append(name)
        if planned:
            missing = [name for name in planned if name not in scene.objects]
            if missing:
                raise TargetSelectionFailure(
                    "RoboTwin task-plan targets are absent from RGB-D scene: "
                    + ", ".join(missing)
                )
            return tuple(planned)
        names = [name for name in scene.objects if name != "wall"]
        if len(names) != 1:
            available = ", ".join(sorted(names))
            raise TargetSelectionFailure(
                "object_name=auto requires task-plan targets or exactly one "
                "non-wall tracked object; "
                f"available: {available}"
            )
        return (names[0],)

    def _target_name(self, scene: SceneObservation) -> str:
        names = self._target_names(scene)
        if len(names) != 1:
            raise TargetSelectionFailure(
                f"single-arm policy received {len(names)} task targets"
            )
        return names[0]

    def _select_arm(self, target: ObjectState) -> tuple[str, str]:
        if not self.automatic_arm:
            return self.config.arm, "explicit_config"
        expert_arm = _robotwin_ground_truth_arm(self.task_env, target.name)
        if expert_arm is not None:
            return expert_arm, "robotwin_ground_truth"
        geometric_arm = "left" if target.world_pose[0, 3] < 0.0 else "right"
        return geometric_arm, "geometry_fallback"

    def _save_grasp_visualization(
        self,
        scene: SceneObservation,
        target: ObjectState,
        candidates: list[GraspCandidate],
        selected: GraspCandidate | None,
        arm: str,
        *,
        raw_trace_override: dict[str, Any] | None = None,
        executed_command_pose_override: np.ndarray | None = None,
        use_default_executed_pose: bool = True,
    ) -> None:
        if not self.save_grasp_visualizations or self.visualization_dir is None:
            return
        plan_index = self._visualization_index
        self._visualization_index += 1
        episode = int(getattr(self.task_env, "test_num", 0))
        seed = int(getattr(self.task_env, "episode_seed", 0))
        step = int(getattr(self.task_env, "take_action_cnt", 0))
        output_path = (
            self.visualization_dir
            / "grasp_viz"
            / f"episode{episode:04d}_seed{seed}_step{step:04d}_plan{plan_index:03d}.png"
        )
        try:
            raw_trace = (
                getattr(self.backend, "last_trace", {})
                if raw_trace_override is None
                else raw_trace_override
            )
            raw_poses = np.asarray(
                raw_trace.get("poses", np.empty((0, 4, 4))),
                dtype=np.float64,
            ).reshape(-1, 4, 4)
            raw_scores = np.asarray(
                raw_trace.get("scores", np.empty(0)), dtype=np.float64
            )
            raw_target_contacts = np.asarray(
                raw_trace.get("target_contacts", np.empty(0)), dtype=bool
            )
            if not (
                len(raw_poses) == len(raw_scores) == len(raw_target_contacts)
            ):
                raise ValueError("M2T2 visualization trace arrays are misaligned")
            m2t2_candidates = [
                GraspCandidate(pose, float(score), target.name)
                for pose, score, is_target_contact in zip(
                    raw_poses, raw_scores, raw_target_contacts
                )
                if is_target_contact
            ]
            rejected_candidates = [
                GraspCandidate(pose, float(score), target.name)
                for pose, score, is_target_contact in zip(
                    raw_poses, raw_scores, raw_target_contacts
                )
                if not is_target_contact
            ]
            head_observation = (
                getattr(self.task_env, "now_obs", {})
                .get("observation", {})
                .get("head_camera", {})
            )
            for required_key in ("rgb", "intrinsic_cv", "extrinsic_cv"):
                if required_key not in head_observation:
                    raise ValueError(
                        f"head-camera observation lacks {required_key}"
                    )
            saved_path = save_grasp_visualization(
                output_path,
                scene,
                target,
                m2t2_candidates,
                selected,
                arm=arm,
                grasp_to_robotwin=self.ik.grasp_to_robotwin,
                camera_rgb=head_observation["rgb"],
                camera_intrinsic=head_observation["intrinsic_cv"],
                camera_extrinsic=head_observation["extrinsic_cv"],
                rejected_candidates=rejected_candidates,
                policy_ranked_candidates=candidates,
                executed_command_pose=(
                    self.ik.selected_grasp_command_pose
                    if use_default_executed_pose
                    else executed_command_pose_override
                ),
                raw_trace=raw_trace,
                max_grasps=self.max_visualized_grasps,
                max_points=self.max_visualized_points,
                image_scale=self.grasp_visualization_scale,
            )
            data_path = saved_path.with_suffix(".npz")
            print(
                f"[heuristic] grasp visualization: {saved_path} "
                f"(data: {data_path})"
            )
        except Exception as exc:
            print(f"[heuristic] grasp visualization failed: {exc}")

    @staticmethod
    def _copy_backend_trace(trace: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in trace.items()
        }

    @staticmethod
    def _handoff_stages(
        task_env: Any,
    ) -> tuple[Pick, Handoff, Place] | None:
        task_plan = getattr(task_env, "heuristic_task_plan", None)
        stages = tuple(getattr(task_plan, "stages", ()))
        return _structural_handoff_stages(stages)

    @staticmethod
    def _single_arm_place_stages(task_env: Any) -> tuple[Pick, Place] | None:
        task_plan = getattr(task_env, "heuristic_task_plan", None)
        stages = tuple(getattr(task_plan, "stages", ()))
        # eval_policy attaches the plan through ``policy.heuristic_baseline``
        # while the deploy package can be imported as ``heuristic_baseline``.
        # Match the frozen stage protocol structurally so duplicate module
        # namespaces cannot make otherwise-identical dataclasses fail here.
        if (
            len(stages) == 2
            and getattr(stages[0], "target", None) is not None
            and getattr(stages[1], "object", None) == stages[0].target
            and (
                getattr(stages[1], "destination", None) is not None
                or getattr(stages[1], "target_pose", None) is not None
            )
            and hasattr(stages[1], "preplace_offset_m")
            and hasattr(stages[1], "place_offset_m")
        ):
            return stages[0], stages[1]
        return None

    @staticmethod
    def _simultaneous_same_object_pick_stages(
        task_env: Any,
    ) -> tuple[Pick, Pick] | None:
        """Return a recorded terminal two-arm Pick of one shared object."""
        task_plan = getattr(task_env, "heuristic_task_plan", None)
        stages = tuple(getattr(task_plan, "stages", ()))
        if (
            len(stages) != 2
            or any(getattr(stage, "target", None) is None for stage in stages)
            or stages[0].target != stages[1].target
        ):
            return None

        group = getattr(stages[0], "group_id", None)
        if (
            group is None
            or getattr(stages[1], "group_id", None) != group
        ):
            return None
        arms = [
            str(getattr(stage, "arm", "")).strip().lower()
            for stage in stages
        ]
        if any(arm not in {"left", "right"} for arm in arms):
            return None
        if set(arms) != {"left", "right"}:
            raise TargetSelectionFailure(
                "simultaneous same-object Pick stages require one left and "
                "one right arm"
            )
        return tuple(
            stage
            for _, stage in sorted(
                zip(arms, stages), key=lambda item: item[0]
            )
        )

    @staticmethod
    def _auxiliary_pick_place_stages(
        task_env: Any,
    ) -> tuple[Pick, Pick, Place] | None:
        """Return an auxiliary displaced Pick and one primary Pick/Place chain."""
        task_plan = getattr(task_env, "heuristic_task_plan", None)
        stages = tuple(getattr(task_plan, "stages", ()))
        picks = [
            stage for stage in stages
            if getattr(stage, "target", None) is not None
        ]
        places = [
            stage for stage in stages
            if getattr(stage, "object", None) is not None
            and hasattr(stage, "preplace_offset_m")
            and hasattr(stage, "place_offset_m")
        ]
        if len(picks) != 2 or len(places) != 1 or len(stages) != 3:
            return None
        place = places[0]
        primary = next(
            (pick for pick in picks if pick.target == place.object),
            None,
        )
        if primary is None:
            return None
        auxiliary = next(
            (pick for pick in picks if pick.target != place.object),
            None,
        )
        displacement = getattr(
            auxiliary, "postgrasp_displacement", None
        ) if auxiliary is not None else None
        if (
            auxiliary is None
            or displacement is None
            or np.linalg.norm(np.asarray(displacement, dtype=np.float64))
            < 1e-8
        ):
            return None
        auxiliary_arm = str(
            getattr(auxiliary, "arm", "")
        ).strip().lower()
        primary_arm = str(getattr(primary, "arm", "")).strip().lower()
        if (
            auxiliary_arm not in {"left", "right"}
            or primary_arm not in {"left", "right"}
            or auxiliary_arm == primary_arm
        ):
            return None
        return auxiliary, primary, place



    @staticmethod
    def _grouped_bimanual_place_stages(
        task_env: Any,
    ) -> tuple[tuple[Pick, Place], tuple[Pick, Place]] | None:
        """Return two structurally grouped Pick/Place chains, independent of names."""
        task_plan = getattr(task_env, "heuristic_task_plan", None)
        stages = tuple(getattr(task_plan, "stages", ()))
        picks = [
            stage
            for stage in stages
            if getattr(stage, "target", None) is not None
        ]
        places = [
            stage
            for stage in stages
            if getattr(stage, "object", None) is not None
            and hasattr(stage, "preplace_offset_m")
            and hasattr(stage, "place_offset_m")
        ]
        if len(picks) != 2 or len(places) != 2 or len(stages) != 4:
            return None
        pick_group = getattr(picks[0], "group_id", None)
        if pick_group is None or (
            getattr(picks[1], "group_id", None) != pick_group
        ):
            return None
        # Terminal placements may be recorded in separate expert move()
        # calls. Pair them structurally by object instead of requiring their
        # execution groups to match the simultaneous Pick group.
        place_by_object = {place.object: place for place in places}
        if len(place_by_object) != 2 or set(place_by_object) != {
            pick.target for pick in picks
        }:
            return None
        pairs = tuple((pick, place_by_object[pick.target]) for pick in picks)
        arms = []
        for pick, place in pairs:
            pick_arm = str(getattr(pick, "arm", "")).strip().lower()
            place_arm = str(getattr(place, "arm", "")).strip().lower()
            if pick_arm not in {"left", "right"}:
                return None
            if place_arm in {"left", "right"} and place_arm != pick_arm:
                raise TargetSelectionFailure(
                    "grouped Pick and Place stages assign different arms"
                )
            arms.append(pick_arm)
        if set(arms) != {"left", "right"}:
            raise TargetSelectionFailure(
                "grouped bimanual stages require one left and one right arm"
            )
        return tuple(
            pair
            for _, pair in sorted(
                zip(arms, pairs), key=lambda item: item[0]
            )
        )

    @staticmethod
    def _sequential_place_stages(
        task_env: Any,
    ) -> tuple[tuple[Pick, Place], ...] | None:
        """Return ordered, independently released Pick/Place chains."""
        task_plan = getattr(task_env, "heuristic_task_plan", None)
        stages = tuple(getattr(task_plan, "stages", ()))
        picks = [
            stage for stage in stages
            if getattr(stage, "target", None) is not None
        ]
        if len(picks) < 2 or len({pick.target for pick in picks}) != len(picks):
            return None
        terminal_places = {}
        for stage in stages:
            object_name = getattr(stage, "object", None)
            if (
                object_name is not None
                and hasattr(stage, "preplace_offset_m")
                and hasattr(stage, "place_offset_m")
                and bool(getattr(stage, "release", False))
            ):
                terminal_places[object_name] = stage
        if set(terminal_places) != {pick.target for pick in picks}:
            return None
        # Preserve expert object order while allowing intermediate Handoff or
        # transport stages between each initial Pick and terminal Place.
        return tuple((pick, terminal_places[pick.target]) for pick in picks)

    def _place_reference_poses(
        self,
        target: ObjectState,
        place: Place,
    ) -> tuple[np.ndarray, np.ndarray]:
        tracked = self.task_env.get_tracked_objects() or {}
        if target.name not in tracked:
            raise TargetSelectionFailure(
                f"placement source {target.name!r} is not tracked"
            )
        actor = tracked[target.name]
        if place.object_functional_point_id is None:
            source_reference = target.world_pose
        else:
            source_reference = actor.get_functional_point(
                place.object_functional_point_id, "matrix"
            )
            if source_reference is None:
                raise TargetSelectionFailure(
                    f"{target.name!r} lacks functional point "
                    f"{place.object_functional_point_id}"
                )

        explicit_target = getattr(place, "target_pose", None)
        if explicit_target is not None:
            destination_reference = explicit_target
        else:
            destination_name = getattr(place, "destination", None)
            if destination_name is None or destination_name not in tracked:
                raise TargetSelectionFailure(
                    "placement requires an explicit target pose or tracked "
                    f"destination; got {destination_name!r}"
                )
            destination = tracked[destination_name]
            if place.destination_functional_point_id is None:
                destination_reference = destination.get_pose()
            else:
                destination_reference = destination.get_functional_point(
                    place.destination_functional_point_id, "matrix"
                )
                if destination_reference is None:
                    raise TargetSelectionFailure(
                        f"{destination_name!r} lacks functional point "
                        f"{place.destination_functional_point_id}"
                    )
            destination_offset = getattr(place, "destination_offset", None)
            if destination_offset is not None:
                destination_reference = _pose_matrix(
                    destination_reference, name="destination_reference"
                ).copy()
                # A procedural 3-vector is converted by RoboTwin to a pose
                # with an identity quaternion.  Anchor only its translation;
                # the destination actor's randomized rotation is not part of
                # the requested place goal.
                destination_reference[:3, :3] = np.eye(3, dtype=np.float64)
                destination_reference[:3, 3] += np.asarray(
                    destination_offset, dtype=np.float64
                )
        return (
            _pose_matrix(source_reference, name="source_reference"),
            _pose_matrix(destination_reference, name="destination_reference"),
        )

    def _plan_single_arm_place(
        self,
        scene: SceneObservation,
        target: ObjectState,
        *,
        pick: Pick,
        place: Place,
        arm: str,
        arm_source: str,
        plan_limit: int = 1,
        stop_after_grasp_lift: bool = False,
    ) -> tuple[
        list[_SingleArmPlacePlan],
        list[GraspCandidate],
        dict[str, Any],
        dict[str, int],
    ]:
        if plan_limit < 1:
            raise ValueError("plan_limit must be positive")
        support_rejections_before = int(
            getattr(self.ik, "failures", {}).get("SupportClearance", 0)
        )
        source_reference, destination_reference = (
            self._place_reference_poses(target, place)
        )
        pregrasp_offset = (
            self.config.pregrasp_offset_m
            if pick.pregrasp_offset_m is None
            else float(pick.pregrasp_offset_m)
        )
        grasp_offset = float(getattr(pick, "grasp_offset_m", 0.0))
        gripper_target = float(getattr(pick, "gripper_target", 0.0))
        postgrasp = np.asarray(
            pick.postgrasp_displacement
            if pick.postgrasp_displacement is not None
            else (0.0, 0.0, self.config.retreat_offset_m),
            dtype=np.float64,
        )
        preplace_offset = float(place.preplace_offset_m)
        place_offset = float(place.place_offset_m)
        if (
            not all(
                np.isfinite(value)
                for value in (
                    pregrasp_offset,
                    grasp_offset,
                    gripper_target,
                    preplace_offset,
                    place_offset,
                )
            )
            or pregrasp_offset < 0.0
            or grasp_offset < 0.0
            or grasp_offset > pregrasp_offset
            or not 0.0 <= gripper_target <= 1.0
            or postgrasp.shape != (3,)
            or not np.all(np.isfinite(postgrasp))
        ):
            raise ValueError("invalid structured pick/place offsets")

        ranker = ConfidenceRankedGrasps(
            self.grasps, min_confidence=self.config.min_confidence
        )
        ranked = ranker.propose(scene, target)[: self.config.max_candidates]
        trace = self._copy_backend_trace(
            getattr(self.backend, "last_trace", {})
        )
        failures = {
            stage: 0
            for stage in (
                "pregrasp", "grasp", "lift", "approach_roll",
                "robot_facing", "place_facing", "narrow_facing",
                "palm_clearance", "jaw_width", "preplace", "place",
                "retreat"
            )
        }
        ee_getter = getattr(self.task_env.robot, f"get_{arm}_ee_pose")
        current_ee_position = np.asarray(
            ee_getter(), dtype=np.float64
        )[:3]

        # Preserve raw M2T2 confidence order globally. Orientation-derived
        # candidates are separate fallback tiers, so a high-confidence
        # fallback can never preempt a lower-confidence feasible raw pose.
        # The optional third tuple item freezes the raw candidate's placement
        # goal.  In particular, constrain="auto" consults the grasp frame;
        # rolling the attachment must not silently choose a different goal.
        raw_variants = [
            (candidate, "m2t2", None) for candidate in ranked
        ]
        approach_roll_variants: list[
            tuple[GraspCandidate, str, np.ndarray]
        ] = []
        robot_facing_variants: list[
            tuple[GraspCandidate, str, None]
        ] = []
        place_facing_variants: list[
            tuple[GraspCandidate, str, None]
        ] = []
        narrow_facing_variants: list[
            tuple[GraspCandidate, str, None]
        ] = []
        for raw_candidate in ranked:
            try:
                raw_goal_grasp_pose = np.asarray(
                    raw_candidate.world_grasp_pose, dtype=np.float64
                ).copy()
                raw_goal_grasp_pose[:3, 3] -= (
                    raw_goal_grasp_pose[:3, 2] * grasp_offset
                )
                raw_command = (
                    raw_goal_grasp_pose @ self.ik.grasp_to_robotwin
                )
                raw_reference = _aligned_place_reference_pose(
                    source_reference,
                    destination_reference,
                    raw_command,
                    arm=arm,
                    constrain=place.constrain,
                    z_transform=(
                        place.object_functional_point_id is None
                    ),
                )
                raw_goal_object = _desired_object_pose(
                    target.world_pose, source_reference, raw_reference
                )
                object_delta = (
                    raw_goal_object @ np.linalg.inv(target.world_pose)
                )
                final_raw_rotation = (
                    object_delta[:3, :3] @ raw_command[:3, :3]
                )
                canonical_rotation = t3d.quaternions.quat2mat(
                    CANONICAL_COMMAND_QUATERNIONS[arm]
                )
                canonical_roll = _closest_approach_roll_angle(
                    final_raw_rotation, canonical_rotation
                )
                for offset in (
                    0.0, np.pi / 8.0, -np.pi / 8.0,
                    np.pi / 4.0, -np.pi / 4.0,
                ):
                    rolled_pose = _approach_roll_grasp_pose(
                        raw_candidate.world_grasp_pose,
                        self.ik.grasp_to_robotwin,
                        canonical_roll + offset,
                    )
                    approach_roll_variants.append((
                        GraspCandidate(
                            rolled_pose,
                            float(raw_candidate.confidence),
                            raw_candidate.object_name,
                        ),
                        "approach_roll",
                        raw_reference.copy(),
                    ))
            except ValueError:
                failures["approach_roll"] += 1
            try:
                adjusted_pose = _robot_facing_grasp_pose(
                    raw_candidate.world_grasp_pose,
                    current_ee_position,
                    self.ik.grasp_to_robotwin,
                )
            except ValueError:
                failures["robot_facing"] += 1
            else:
                robot_facing_variants.append((
                    GraspCandidate(
                        adjusted_pose,
                        float(raw_candidate.confidence),
                        raw_candidate.object_name,
                    ),
                    "robot_facing",
                    None,
                ))
            try:
                raw_command = (
                    np.asarray(raw_candidate.world_grasp_pose)
                    @ self.ik.grasp_to_robotwin
                )
                variant_reference = _aligned_place_reference_pose(
                    source_reference,
                    destination_reference,
                    raw_command,
                    arm=arm,
                    constrain=place.constrain,
                    z_transform=place.object_functional_point_id is None,
                )
                variant_object = _desired_object_pose(
                    target.world_pose,
                    source_reference,
                    variant_reference,
                )
                adjusted_pose = _place_facing_grasp_pose(
                    raw_candidate.world_grasp_pose,
                    target.world_pose,
                    variant_object,
                    current_ee_position,
                    self.ik.grasp_to_robotwin,
                )
            except ValueError:
                failures["place_facing"] += 1
            else:
                place_facing_variants.append((
                    GraspCandidate(
                        adjusted_pose,
                        float(raw_candidate.confidence),
                        raw_candidate.object_name,
                    ),
                    "place_facing",
                    None,
                ))
                try:
                    canonical_approach = t3d.quaternions.quat2mat(
                        CANONICAL_COMMAND_QUATERNIONS[arm]
                    )[:, 0]
                    narrow_axis = _target_narrow_axis(
                        scene,
                        target,
                        variant_object,
                        canonical_approach,
                    )
                    narrow_poses = (
                        ()
                        if narrow_axis is None
                        else _narrow_axis_grasp_poses(
                            raw_candidate.world_grasp_pose,
                            target.world_pose,
                            variant_object,
                            current_ee_position,
                            self.ik.grasp_to_robotwin,
                            narrow_axis,
                            canonical_approach,
                            max_approaches=10,
                        )
                    )
                except ValueError:
                    failures["narrow_facing"] += 1
                else:
                    if not narrow_poses:
                        failures["narrow_facing"] += 1
                    for narrow_pose in narrow_poses:
                        narrow_facing_variants.append((
                            GraspCandidate(
                                narrow_pose,
                                float(raw_candidate.confidence),
                                raw_candidate.object_name,
                            ),
                            "narrow_facing",
                            None,
                        ))

        variants = (
            raw_variants
            + approach_roll_variants
            + robot_facing_variants
            + place_facing_variants
            + narrow_facing_variants
        )
        plans: list[_SingleArmPlacePlan] = []
        grasp_lift_plans: list[_SingleArmPlacePlan] = []
        for candidate, variant_source, raw_goal_reference in variants:
            raw_grasp_pose = np.asarray(
                candidate.world_grasp_pose, dtype=np.float64
            )
            palm_depth = _target_m2t2_palm_depth(
                scene,
                target,
                raw_grasp_pose,
                self.ik.grasp_to_robotwin,
            )
            if (
                palm_depth is not None
                and palm_depth < M2T2_MIN_TARGET_PALM_DEPTH_M
            ):
                failures["palm_clearance"] += 1
                continue
            grasp_pose = raw_grasp_pose.copy()
            grasp_pose[:3, 3] -= raw_grasp_pose[:3, 2] * grasp_offset
            pregrasp_pose = raw_grasp_pose.copy()
            pregrasp_pose[:3, 3] -= (
                raw_grasp_pose[:3, 2] * pregrasp_offset
            )
            lift_pose = grasp_pose.copy()
            lift_pose[:3, 3] += postgrasp
            solutions = [
                self.ik.solve(arm, pose)
                for pose in (pregrasp_pose, grasp_pose, lift_pose)
            ]
            if any(solution is None for solution in solutions):
                failed_stage = next(
                    stage for stage, solution in zip(
                        ("pregrasp", "grasp", "lift"), solutions
                    )
                    if solution is None
                )
                failures[failed_stage] += 1
                if variant_source != "m2t2":
                    failures[variant_source] += 1
                continue

            grasp_paths = self.ik.completed_paths
            grasp_targets = self.ik.completed_command_targets
            if len(grasp_paths) != 3 or len(grasp_targets) != 3:
                raise RuntimeError(
                    "Mink omitted a completed three-stage grasp plan"
                )
            jaw_axis = grasp_targets[1][:3, 1]
            target_width = _target_width_along_axis(
                scene, target, jaw_axis
            )
            if (
                target_width is not None
                and target_width > self.bimanual_max_target_width_m
            ):
                failures["jaw_width"] += 1
                continue
            desired_reference = (
                raw_goal_reference.copy()
                if raw_goal_reference is not None
                else _aligned_place_reference_pose(
                    source_reference,
                    destination_reference,
                    grasp_targets[1],
                    arm=arm,
                    constrain=place.constrain,
                    z_transform=(
                        place.object_functional_point_id is None
                    ),
                )
            )
            desired_object, desired_gripper = _rigid_place_command_pose(
                target.world_pose,
                source_reference,
                grasp_targets[1],
                desired_reference,
            )
            if len(grasp_lift_plans) < plan_limit:
                grasp_lift_plans.append(
                    _SingleArmPlacePlan(
                        arm=arm,
                        target_name=target.name,
                        arm_source=arm_source,
                        candidate=candidate,
                        paths=tuple(grasp_paths),
                        command_targets=tuple(grasp_targets),
                        desired_object_pose=desired_object,
                        orientation_source=variant_source,
                        completion_level="grasp_lift",
                    )
                )
            if (
                stop_after_grasp_lift
                and len(grasp_lift_plans) >= plan_limit
            ):
                break
            goal_variants = [
                (desired_object, desired_gripper, variant_source)
            ]
            if getattr(place, "target_pose", None) is not None:
                offsets = (
                    (0.0, 0.0, 0.02),
                    (0.0, 0.0, 0.04),
                    (0.02, 0.0, 0.02),
                    (-0.02, 0.0, 0.02),
                    (0.0, 0.02, 0.02),
                    (0.0, -0.02, 0.02),
                )
                for offset in offsets:
                    adjusted_object = desired_object.copy()
                    adjusted_object[:3, 3] += np.asarray(
                        offset, dtype=np.float64
                    )
                    adjusted_gripper = (
                        adjusted_object
                        @ np.linalg.inv(desired_object)
                        @ desired_gripper
                    )
                    goal_variants.append((
                        adjusted_object,
                        adjusted_gripper,
                        f"{variant_source}/goal_offset",
                    ))
                for yaw in (np.pi / 12.0, -np.pi / 12.0):
                    adjusted_object = desired_object.copy()
                    yaw_rotation = t3d.axangles.axangle2mat(
                        [0.0, 0.0, 1.0], yaw
                    )
                    adjusted_object[:3, :3] = (
                        yaw_rotation @ adjusted_object[:3, :3]
                    )
                    adjusted_object[:3, 3] += np.array(
                        [0.0, 0.0, 0.02], dtype=np.float64
                    )
                    adjusted_gripper = (
                        adjusted_object
                        @ np.linalg.inv(desired_object)
                        @ desired_gripper
                    )
                    goal_variants.append((
                        adjusted_object,
                        adjusted_gripper,
                        f"{variant_source}/goal_yaw",
                    ))

            for (
                goal_object,
                goal_gripper,
                goal_source,
            ) in goal_variants:
                offset_axis = _place_offset_axis(
                    place.preplace_axis,
                    goal_object,
                    goal_gripper,
                    destination_reference,
                )
                preplace_command = goal_gripper.copy()
                preplace_command[:3, 3] -= (
                    preplace_offset * offset_axis
                )
                place_command = goal_gripper.copy()
                place_command[:3, 3] -= place_offset * offset_axis
                retreat_command = preplace_command.copy()

                followup_paths: list[np.ndarray] = []
                followup_targets: list[np.ndarray] = []
                start = np.asarray(solutions[2], dtype=np.float64)
                complete = True
                followup_commands = [
                    ("preplace", preplace_command),
                    ("place", place_command),
                ]
                if place.release:
                    followup_commands.append(
                        ("retreat", retreat_command)
                    )
                for stage, command in followup_commands:
                    before_ik_failures = dict(self.ik.failures)
                    result = self.ik.solve_command_target(
                        arm, command, start
                    )
                    if result is None:
                        if (
                            stage == "preplace"
                            and failures[stage] == 0
                        ):
                            print(
                                "[heuristic] first rejected preplace command="
                                + np.array2string(command, precision=3)
                            )
                        failures[stage] += 1
                        for reason, count in self.ik.failures.items():
                            delta = int(count) - int(
                                before_ik_failures.get(reason, 0)
                            )
                            if delta > 0:
                                key = f"{stage}_{reason}"
                                failures[key] = (
                                    failures.get(key, 0) + delta
                                )
                        complete = False
                        break
                    start, path, accepted = result
                    followup_paths.append(path)
                    followup_targets.append(accepted)
                if not complete:
                    continue
                plans.append(
                    _SingleArmPlacePlan(
                        arm=arm,
                        target_name=target.name,
                        arm_source=arm_source,
                        candidate=candidate,
                        paths=(
                            tuple(grasp_paths)
                            + tuple(followup_paths)
                        ),
                        command_targets=(
                            tuple(grasp_targets)
                            + tuple(followup_targets)
                        ),
                        desired_object_pose=goal_object,
                        orientation_source=goal_source,
                        completion_level="place",
                    )
                )
                if len(plans) >= plan_limit:
                    break
            if len(plans) >= plan_limit:
                break
        failures["support_clearance"] = (
            int(getattr(self.ik, "failures", {}).get(
                "SupportClearance", 0
            ))
            - support_rejections_before
        )
        if not plans and getattr(place, "destination_offset", None) is None:
            plans = grasp_lift_plans
        return plans, list(ranker.last_candidates), trace, failures

    def _get_single_place_action(
        self,
        scene: SceneObservation,
        target: ObjectState,
        *,
        pick: Pick,
        place: Place,
        arm: str,
        arm_source: str,
    ) -> list[np.ndarray]:
        plans, ranked, trace, failures = self._plan_single_arm_place(
            scene,
            target,
            pick=pick,
            place=place,
            arm=arm,
            arm_source=arm_source,
        )
        selected = plans[0] if plans else None
        if selected is None:
            self._save_grasp_visualization(
                scene,
                target,
                ranked,
                None,
                arm,
                raw_trace_override=trace,
                executed_command_pose_override=None,
                use_default_executed_pose=False,
            )
            raise NoFeasiblePlanFailure(
                "M2T2/Mink produced no complete atomic placement plan; "
                f"failures={failures}; ik_failures={self.ik.failures}"
            )

        controller = self.staged_controller
        controller.reset()
        names = {arm: target.name}
        sources = {arm: arm_source}
        controller.gripper_phase(
            "open", {arm: 1.0}, 1, names, sources
        )
        for phase, index in (("pregrasp", 0), ("grasp", 1)):
            controller.move_phase(
                phase,
                {arm: selected.paths[index]},
                {arm: selected.command_targets[index]},
                names,
                sources,
            )
        controller.gripper_phase(
            "close",
            {arm: float(getattr(pick, "gripper_target", 0.0))},
            self.gripper_settle_actions,
            names,
            sources,
        )
        controller.move_phase(
            "lift",
            {arm: selected.paths[2]},
            {arm: selected.command_targets[2]},
            names,
            sources,
        )
        if selected.completion_level == "place":
            for phase, index in (("preplace", 3), ("place", 4)):
                controller.move_phase(
                    phase,
                    {arm: selected.paths[index]},
                    {arm: selected.command_targets[index]},
                    names,
                    sources,
                )
            if place.release:
                controller.gripper_phase(
                    "open",
                    {arm: 1.0},
                    self.gripper_settle_actions,
                    names,
                    sources,
                )
                controller.move_phase(
                    "retreat",
                    {arm: selected.paths[5]},
                    {arm: selected.command_targets[5]},
                    names,
                    sources,
                )
        elif selected.completion_level != "grasp_lift":
            raise ValueError(
                f"unknown placement completion level "
                f"{selected.completion_level!r}"
            )
        for record in controller.metadata:
            record["completion_level"] = selected.completion_level
            record["required_release"] = bool(place.release)
        self._action_metadata_override = list(controller.metadata)
        self._save_grasp_visualization(
            scene,
            target,
            ranked,
            selected.candidate,
            arm,
            raw_trace_override=trace,
            executed_command_pose_override=selected.command_targets[1],
            use_default_executed_pose=False,
        )
        desired = selected.desired_object_pose
        print(
            "[heuristic] placement plan selected "
            f"target={target.name} destination={place.destination} arm={arm} "
            f"confidence={selected.candidate.confidence:.3f} "
            f"orientation_source={selected.orientation_source} "
            f"completion_level={selected.completion_level} "
            f"support_plane_z={self._support_plane_for_arm(arm)} "
            f"support_clearance_rejections="
            f"{failures.get('support_clearance', 0)} "
            f"desired_xyz={np.array2string(desired[:3, 3], precision=3)}"
        )
        return [action.copy() for action in controller.actions]

    def _handoff_reference_poses(
        self,
        target: ObjectState,
        handoff: Handoff,
    ) -> tuple[np.ndarray, np.ndarray]:
        tracked = self.task_env.get_tracked_objects() or {}
        if target.name not in tracked:
            raise TargetSelectionFailure(
                f"handoff source {target.name!r} is not tracked"
            )
        actor = tracked[target.name]
        if handoff.object_functional_point_id is None:
            source_reference = target.world_pose
        else:
            source_reference = actor.get_functional_point(
                handoff.object_functional_point_id, "matrix"
            )
            if source_reference is None:
                raise TargetSelectionFailure(
                    f"{target.name!r} lacks functional point "
                    f"{handoff.object_functional_point_id}"
                )
        rendezvous = getattr(handoff, "rendezvous_pose", None)
        if rendezvous is None:
            pose_attribute = getattr(
                handoff, "rendezvous_pose_attr", None
            )
            if pose_attribute is None or not hasattr(
                self.task_env, pose_attribute
            ):
                raise TargetSelectionFailure(
                    "handoff stage lacks an explicit rendezvous pose"
                )
            rendezvous = getattr(self.task_env, pose_attribute)
        return (
            _pose_matrix(source_reference, name="handoff_source_reference"),
            _pose_matrix(rendezvous, name="handoff_rendezvous_pose"),
        )

    def _solve_handoff_command_chain(
        self,
        arm: str,
        role: str,
        stage_commands: tuple[tuple[str, np.ndarray], ...],
        failures: dict[str, int],
    ) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]] | None:
        start = _arm_joint_state(self.task_env, arm)[:-1]
        paths: list[np.ndarray] = []
        accepted_targets: list[np.ndarray] = []
        for stage, command in stage_commands:
            result = self.ik.solve_command_target(arm, command, start)
            if result is None:
                key = f"{role}_{stage}"
                failures[key] = failures.get(key, 0) + 1
                return None
            start, path, accepted = result
            paths.append(path)
            accepted_targets.append(accepted)
        return tuple(paths), tuple(accepted_targets)

    def _plan_handoff_sides(
        self,
        scene: SceneObservation,
        target: ObjectState,
        *,
        pick: Pick,
        handoff: Handoff,
        place: Place,
    ) -> tuple[
        list[_HandoffArmPlan],
        list[_HandoffArmPlan],
        list[GraspCandidate],
        dict[str, Any],
        dict[str, int],
        float,
    ]:
        giver_arm = str(handoff.from_arm).strip().lower()
        receiver_arm = str(handoff.to_arm).strip().lower()
        support_rejections_before = int(
            getattr(self.ik, "failures", {}).get("SupportClearance", 0)
        )
        for stage_arm, expected, label in (
            (pick.arm, giver_arm, "pick"),
            (place.arm, receiver_arm, "place"),
        ):
            if stage_arm is None:
                continue
            normalized = str(stage_arm).strip().lower()
            if normalized in {"left", "right"} and normalized != expected:
                raise TargetSelectionFailure(
                    f"handoff {label} arm differs from its ownership stage"
                )

        giver_pregrasp_offset = (
            self.config.pregrasp_offset_m
            if pick.pregrasp_offset_m is None
            else float(pick.pregrasp_offset_m)
        )
        receiver_pregrasp_offset = float(handoff.pregrasp_offset_m)
        giver_grasp_offset = float(getattr(pick, "grasp_offset_m", 0.0))
        receiver_grasp_offset = float(
            getattr(handoff, "grasp_offset_m", 0.0)
        )
        giver_gripper_target = float(
            getattr(pick, "gripper_target", 0.0)
        )
        receiver_gripper_target = float(
            getattr(handoff, "gripper_target", 0.0)
        )
        postgrasp = np.asarray(
            pick.postgrasp_displacement
            if pick.postgrasp_displacement is not None
            else (0.0, 0.0, self.config.retreat_offset_m),
            dtype=np.float64,
        )
        giver_release_retreat = max(
            giver_pregrasp_offset, float(np.linalg.norm(postgrasp))
        )
        preplace_offset = float(place.preplace_offset_m)
        place_offset = float(place.place_offset_m)
        if (
            not all(
                np.isfinite(value)
                for value in (
                    giver_pregrasp_offset,
                    receiver_pregrasp_offset,
                    giver_grasp_offset,
                    receiver_grasp_offset,
                    giver_gripper_target,
                    receiver_gripper_target,
                )
            )
            or giver_pregrasp_offset < 0.0
            or giver_grasp_offset < 0.0
            or giver_grasp_offset > giver_pregrasp_offset
            or receiver_pregrasp_offset < 0.0
            or receiver_grasp_offset < 0.0
            or receiver_grasp_offset > receiver_pregrasp_offset
            or not 0.0 <= giver_gripper_target <= 1.0
            or not 0.0 <= receiver_gripper_target <= 1.0
            or postgrasp.shape != (3,)
            or not np.all(np.isfinite(postgrasp))
            or not np.isfinite(preplace_offset)
            or not np.isfinite(place_offset)
        ):
            raise ValueError("invalid structured handoff offsets")

        source_reference, rendezvous_reference = (
            self._handoff_reference_poses(target, handoff)
        )
        place_source_reference, destination_reference = (
            self._place_reference_poses(target, place)
        )
        world_object = _pose_matrix(
            target.world_pose, name="handoff_world_object_pose"
        )
        desired_middle_object = _desired_object_pose(
            world_object, source_reference, rendezvous_reference
        )
        object_delta = desired_middle_object @ np.linalg.inv(world_object)
        predicted_place_source = (
            object_delta @ place_source_reference
        )

        ranker = ConfidenceRankedGrasps(
            self.grasps, min_confidence=self.config.min_confidence
        )
        ranked = ranker.propose(scene, target)[: self.config.max_candidates]
        trace = self._copy_backend_trace(
            getattr(self.backend, "last_trace", {})
        )
        failures: dict[str, int] = {
            "giver_region": 0,
            "receiver_region": 0,
            "pair_separation": 0,
        }

        world_from_object = np.linalg.inv(world_object)

        target_points = np.asarray(
            scene.xyz[scene.instance_labels == target.instance_id],
            dtype=np.float64,
        )
        local_target = (
            target_points @ world_from_object[:3, :3].T
            + world_from_object[:3, 3]
            if len(target_points)
            else np.empty((0, 3), dtype=np.float64)
        )

        def current_ee_position(arm: str) -> np.ndarray | None:
            getter = getattr(self.task_env.robot, f"get_{arm}_ee_pose", None)
            if getter is None:
                return None
            values = np.asarray(getter(), dtype=np.float64)
            if values.ndim != 1 or len(values) < 3:
                return None
            position = values[:3]
            return position if np.all(np.isfinite(position)) else None

        def grasp_is_feasible(
            world_grasp_pose: np.ndarray, role: str
        ) -> bool:
            palm_depth = _target_m2t2_palm_depth(
                scene,
                target,
                world_grasp_pose,
                self.ik.grasp_to_robotwin,
            )
            if (
                palm_depth is not None
                and palm_depth < M2T2_MIN_TARGET_PALM_DEPTH_M
            ):
                key = f"{role}_palm_clearance"
                failures[key] = failures.get(key, 0) + 1
                return False
            command = (
                np.asarray(world_grasp_pose, dtype=np.float64)
                @ self.ik.grasp_to_robotwin
            )
            width = _target_width_along_axis(
                scene, target, command[:3, 1]
            )
            if (
                width is not None
                and width > self.bimanual_max_target_width_m
            ):
                key = f"{role}_jaw_width"
                failures[key] = failures.get(key, 0) + 1
                return False
            return True

        giver_ee = current_ee_position(giver_arm)
        receiver_ee = current_ee_position(receiver_arm)

        def local_reference(
            world_reference: np.ndarray | None,
        ) -> np.ndarray | None:
            if world_reference is None:
                return None
            return (
                world_from_object[:3, :3] @ world_reference
                + world_from_object[:3, 3]
            )

        (
            giver_region,
            receiver_region,
            minimum_separation,
            contact_region_source,
        ) = _handoff_contact_regions(
            local_target,
            getattr(pick, "allowed_contact_points_local", None),
            getattr(handoff, "allowed_contact_points_local", None),
            giver_reference_local=local_reference(giver_ee),
            receiver_reference_local=local_reference(receiver_ee),
        )
        classified: dict[
            str,
            list[tuple[GraspCandidate, tuple[float, float, float]]],
        ] = {"giver": [], "receiver": []}
        for candidate in ranked:
            tcp = _grasp_command_tcp(
                candidate.world_grasp_pose, self.ik.grasp_to_robotwin
            )
            local_tcp = (
                world_from_object[:3, :3] @ tcp
                + world_from_object[:3, 3]
            )
            local_contact = tuple(float(value) for value in local_tcp)
            giver_distance = float(
                np.min(np.linalg.norm(giver_region - local_tcp, axis=1))
            )
            receiver_distance = float(
                np.min(np.linalg.norm(receiver_region - local_tcp, axis=1))
            )
            if giver_distance + 1e-6 < receiver_distance:
                classified["giver"].append((candidate, local_contact))
            else:
                failures["giver_region"] += 1
            if receiver_distance + 1e-6 < giver_distance:
                classified["receiver"].append((candidate, local_contact))
            else:
                failures["receiver_region"] += 1

        giver_raw_variants = [
            ("m2t2", candidate, local_contact,
             candidate.world_grasp_pose)
            for candidate, local_contact in classified["giver"]
        ]
        giver_fallback_variants = []
        for candidate, local_contact in classified["giver"]:
            if giver_ee is not None:
                adjusted = _robot_facing_grasp_pose(
                    candidate.world_grasp_pose,
                    giver_ee,
                    self.ik.grasp_to_robotwin,
                )
                if not np.allclose(adjusted, candidate.world_grasp_pose):
                    giver_fallback_variants.append((
                        "robot_facing_fallback",
                        candidate,
                        local_contact,
                        adjusted,
                    ))
        giver_aligned_variants = []
        for candidate in ranked:
            source_tcp = _grasp_command_tcp(
                candidate.world_grasp_pose, self.ik.grasp_to_robotwin
            )
            source_local_tcp = (
                world_from_object[:3, :3] @ source_tcp
                + world_from_object[:3, 3]
            )
            anchor = giver_region[
                int(np.argmin(
                    np.linalg.norm(giver_region - source_local_tcp, axis=1)
                ))
            ]
            giver_aligned_variants.append((
                "recorded_contact_alignment",
                candidate,
                tuple(float(value) for value in anchor),
                _align_grasp_pose_to_local_contact(
                    candidate.world_grasp_pose,
                    world_object,
                    anchor,
                    self.ik.grasp_to_robotwin,
                ),
            ))

        def limited_variant_groups(*groups: list[Any]):
            available = [group for group in groups if group]
            if not available:
                return ()
            base, extra = divmod(
                self.bimanual_max_plans_per_arm, len(available)
            )
            return tuple(
                (group, base + (index < extra))
                for index, group in enumerate(available)
                if base + (index < extra) > 0
            )

        giver_plans: list[_HandoffArmPlan] = []
        for variants, source_limit in limited_variant_groups(
            giver_raw_variants,
            giver_fallback_variants,
            giver_aligned_variants,
        ):
            source_count = 0
            for (
                orientation_source, candidate, local_contact, grasp_pose
            ) in variants:
                if not grasp_is_feasible(grasp_pose, "giver"):
                    continue
                contact_command = (
                    np.asarray(grasp_pose, dtype=np.float64)
                    @ self.ik.grasp_to_robotwin
                )
                pregrasp_command = _approach_offset_command_pose(
                    contact_command, giver_pregrasp_offset
                )
                grasp_command = _approach_offset_command_pose(
                    contact_command, giver_grasp_offset
                )
                lift_command = grasp_command.copy()
                lift_command[:3, 3] += postgrasp
                _, rendezvous_command = _rigid_place_command_pose(
                    world_object,
                    source_reference,
                    grasp_command,
                    rendezvous_reference,
                )
                retreat_command = _approach_offset_command_pose(
                    rendezvous_command, giver_release_retreat
                )
                chain = self._solve_handoff_command_chain(
                    giver_arm,
                    "giver",
                    (
                        ("pregrasp", pregrasp_command),
                        ("grasp", grasp_command),
                        ("lift", lift_command),
                        ("transport", rendezvous_command),
                        ("retreat", retreat_command),
                    ),
                    failures,
                )
                if chain is None:
                    continue
                paths, commands = chain
                planned_candidate = GraspCandidate(
                    np.asarray(grasp_pose, dtype=np.float64),
                    float(candidate.confidence),
                    candidate.object_name,
                )
                giver_plans.append(
                    _HandoffArmPlan(
                        arm=giver_arm,
                        role="giver",
                        target_name=target.name,
                        arm_source="robotwin_ground_truth",
                        candidate=planned_candidate,
                        paths=paths,
                        command_targets=commands,
                        contact_local_point=local_contact,
                        gripper_target=giver_gripper_target,
                        orientation_source=orientation_source,
                    )
                )
                source_count += 1
                if source_count >= source_limit:
                    break

        receiver_raw_variants = [
            ("m2t2", candidate, local_contact,
             candidate.world_grasp_pose)
            for candidate, local_contact in classified["receiver"]
        ]
        receiver_fallback_variants = []
        for candidate, local_contact in classified["receiver"]:
            if receiver_ee is not None:
                adjusted = _place_facing_grasp_pose(
                    candidate.world_grasp_pose,
                    world_object,
                    desired_middle_object,
                    receiver_ee,
                    self.ik.grasp_to_robotwin,
                )
                if not np.allclose(adjusted, candidate.world_grasp_pose):
                    receiver_fallback_variants.append((
                        "robot_facing_fallback",
                        candidate,
                        local_contact,
                        adjusted,
                    ))
        receiver_aligned_variants = []
        for candidate in ranked:
            source_tcp = _grasp_command_tcp(
                candidate.world_grasp_pose, self.ik.grasp_to_robotwin
            )
            source_local_tcp = (
                world_from_object[:3, :3] @ source_tcp
                + world_from_object[:3, 3]
            )
            anchor = receiver_region[
                int(np.argmin(
                    np.linalg.norm(receiver_region - source_local_tcp, axis=1)
                ))
            ]
            receiver_aligned_variants.append((
                "recorded_contact_alignment",
                candidate,
                tuple(float(value) for value in anchor),
                _align_grasp_pose_to_local_contact(
                    candidate.world_grasp_pose,
                    world_object,
                    anchor,
                    self.ik.grasp_to_robotwin,
                ),
            ))

        receiver_plans: list[_HandoffArmPlan] = []
        for variants, source_limit in limited_variant_groups(
            receiver_raw_variants,
            receiver_fallback_variants,
            receiver_aligned_variants,
        ):
            source_count = 0
            for (
                orientation_source, candidate, local_contact,
                source_grasp_pose,
            ) in variants:
                if not grasp_is_feasible(source_grasp_pose, "receiver"):
                    continue
                source_contact_command = (
                    np.asarray(source_grasp_pose, dtype=np.float64)
                    @ self.ik.grasp_to_robotwin
                )
                rendezvous_contact_command = (
                    object_delta @ source_contact_command
                )
                pregrasp_command = _approach_offset_command_pose(
                    rendezvous_contact_command, receiver_pregrasp_offset
                )
                rendezvous_command = _approach_offset_command_pose(
                    rendezvous_contact_command, receiver_grasp_offset
                )
                desired_reference = _aligned_place_reference_pose(
                    predicted_place_source,
                    destination_reference,
                    rendezvous_command,
                    arm=receiver_arm,
                    constrain=place.constrain,
                    z_transform=place.object_functional_point_id is None,
                )
                desired_object, desired_gripper = _rigid_place_command_pose(
                    desired_middle_object,
                    predicted_place_source,
                    rendezvous_command,
                    desired_reference,
                )
                offset_axis = _place_offset_axis(
                    place.preplace_axis,
                    desired_object,
                    desired_gripper,
                    destination_reference,
                )
                preplace_command = desired_gripper.copy()
                preplace_command[:3, 3] -= preplace_offset * offset_axis
                place_command = desired_gripper.copy()
                place_command[:3, 3] -= place_offset * offset_axis
                chain = self._solve_handoff_command_chain(
                    receiver_arm,
                    "receiver",
                    (
                        ("pregrasp", pregrasp_command),
                        ("grasp", rendezvous_command),
                        ("preplace", preplace_command),
                        ("place", place_command),
                    ),
                    failures,
                )
                if chain is None:
                    continue
                paths, commands = chain
                planned_candidate = GraspCandidate(
                    np.asarray(source_grasp_pose, dtype=np.float64),
                    float(candidate.confidence),
                    candidate.object_name,
                )
                receiver_plans.append(
                    _HandoffArmPlan(
                        arm=receiver_arm,
                        role="receiver",
                        target_name=target.name,
                        arm_source="robotwin_ground_truth",
                        candidate=planned_candidate,
                        paths=paths,
                        command_targets=commands,
                        contact_local_point=local_contact,
                        gripper_target=receiver_gripper_target,
                        orientation_source=orientation_source,
                    )
                )
                source_count += 1
                if source_count >= source_limit:
                    break

        failures["support_clearance"] = (
            int(getattr(self.ik, "failures", {}).get(
                "SupportClearance", 0
            ))
            - support_rejections_before
        )
        print(
            "[heuristic] handoff IK "
            f"target={target.name} giver={giver_arm}:{len(giver_plans)} "
            f"receiver={receiver_arm}:{len(receiver_plans)} "
            f"source_candidates={len(ranked)} "
            f"contact_regions={contact_region_source} failures={failures}"
        )
        return (
            giver_plans,
            receiver_plans,
            list(ranker.last_candidates),
            trace,
            failures,
            minimum_separation,
        )

    def _build_handoff_action_pair(
        self,
        giver: _HandoffArmPlan,
        receiver: _HandoffArmPlan,
        *,
        release: bool,
        receiver_pregrasp_first: bool = False,
    ) -> list[np.ndarray]:
        if (
            giver.role != "giver"
            or receiver.role != "receiver"
            or giver.arm == receiver.arm
            or len(giver.paths) != 5
            or len(receiver.paths) != 4
            or any(
                not np.isfinite(plan.gripper_target)
                or not 0.0 <= plan.gripper_target <= 1.0
                for plan in (giver, receiver)
            )
        ):
            raise ValueError("invalid giver/receiver handoff plans")
        controller = self.staged_controller
        controller.reset()
        names = {
            giver.arm: giver.target_name,
            receiver.arm: receiver.target_name,
        }
        sources = {
            giver.arm: giver.arm_source,
            receiver.arm: receiver.arm_source,
        }

        controller.gripper_phase(
            "open",
            {giver.arm: 1.0, receiver.arm: 1.0},
            1,
            names,
            sources,
        )

        def move(phase: str, plan: _HandoffArmPlan, index: int) -> None:
            controller.move_phase(
                phase,
                {plan.arm: plan.paths[index]},
                {plan.arm: plan.command_targets[index]},
                names,
                sources,
            )

        move("pregrasp", giver, 0)
        move("grasp", giver, 1)
        controller.gripper_phase(
            "close",
            {giver.arm: giver.gripper_target},
            self.gripper_settle_actions,
            names,
            sources,
        )
        move("lift", giver, 2)
        if receiver_pregrasp_first:
            move("pregrasp", receiver, 0)
            move("transport", giver, 3)
        else:
            move("transport", giver, 3)
            move("pregrasp", receiver, 0)
        move("grasp", receiver, 1)
        controller.gripper_phase(
            "close",
            {receiver.arm: receiver.gripper_target},
            self.gripper_settle_actions,
            names,
            sources,
        )
        controller.gripper_phase(
            "open",
            {giver.arm: 1.0},
            self.gripper_settle_actions,
            names,
            sources,
        )
        move("retreat", giver, 4)
        move("preplace", receiver, 2)
        move("place", receiver, 3)
        if release:
            controller.gripper_phase(
                "open",
                {receiver.arm: 1.0},
                self.gripper_settle_actions,
                names,
                sources,
            )
        return [action.copy() for action in controller.actions]

    def _get_handoff_action(
        self,
        scene: SceneObservation,
        target: ObjectState,
        *,
        pick: Pick,
        handoff: Handoff,
        place: Place,
        allow_rendezvous_fallback: bool = True,
    ) -> list[np.ndarray]:
        (
            giver_plans,
            receiver_plans,
            ranked,
            trace,
            failures,
            minimum_separation,
        ) = self._plan_handoff_sides(
            scene, target, pick=pick, handoff=handoff, place=place
        )
        if not giver_plans or not receiver_plans:
            self._save_grasp_visualization(
                scene,
                target,
                ranked,
                None,
                str(handoff.from_arm).strip().lower(),
                raw_trace_override=trace,
                executed_command_pose_override=None,
                use_default_executed_pose=False,
            )
            raise NoFeasiblePlanFailure(
                "M2T2/Mink produced no complete source-only handoff plan; "
                f"giver={len(giver_plans)} receiver={len(receiver_plans)} "
                f"failures={failures}; ik_failures={self.ik.failures}"
            )

        pairs = [
            (giver, receiver)
            for giver in giver_plans
            for receiver in receiver_plans
            if float(
                np.linalg.norm(
                    np.asarray(giver.contact_local_point)
                    - np.asarray(receiver.contact_local_point)
                )
            )
            >= minimum_separation
        ]
        failures["pair_separation"] += (
            len(giver_plans) * len(receiver_plans) - len(pairs)
        )
        pairs.sort(
            key=lambda pair: (
                all(
                    plan.orientation_source == "m2t2"
                    for plan in pair
                ),
                pair[0].candidate.confidence
                + pair[1].candidate.confidence,
            ),
            reverse=True,
        )
        selected: tuple[_HandoffArmPlan, _HandoffArmPlan] | None = None
        actions: list[np.ndarray] = []
        collision_rejections = 0
        collision_causes: dict[tuple[str, tuple[str, str]], int] = {}
        collision_check = getattr(
            self.ik,
            "handoff_path_has_self_collision",
            self.ik.full_robot_path_has_self_collision,
        )
        selected_schedule = "standard"
        for receiver_pregrasp_first in (False, True):
            for giver, receiver in pairs:
                candidate_actions = self._build_handoff_action_pair(
                    giver,
                    receiver,
                    release=place.release,
                    receiver_pregrasp_first=receiver_pregrasp_first,
                )
                if collision_check(
                    candidate_actions,
                    max_joint_step_rad=self.bimanual_collision_step_rad,
                ):
                    collision_rejections += 1
                    detail = getattr(
                        self.ik, "last_full_robot_collision", None
                    )
                    if detail is not None:
                        row_index = int(detail["row_index"])
                        metadata = self.staged_controller.metadata
                        phase = (
                            str(metadata[row_index]["phase"])
                            if 0 <= row_index < len(metadata)
                            else "unknown"
                        )
                        key = (phase, tuple(detail["body_pair"]))
                        collision_causes[key] = (
                            collision_causes.get(key, 0) + 1
                        )
                    continue
                selected = giver, receiver
                actions = candidate_actions
                selected_schedule = (
                    "receiver_pregrasp_first"
                    if receiver_pregrasp_first
                    else "standard"
                )
                break
            if selected is not None:
                break
        if selected is None and allow_rendezvous_fallback:
            _, base_rendezvous = self._handoff_reference_poses(
                target, handoff
            )
            rendezvous_offsets = (
                (0.0, 0.06, 0.0),
                (0.0, -0.06, 0.0),
                (0.06, 0.0, 0.0),
                (-0.06, 0.0, 0.0),
                (0.0, 0.0, 0.06),
            )
            fallback_failures = 0
            for offset in rendezvous_offsets:
                adjusted_rendezvous = base_rendezvous.copy()
                adjusted_rendezvous[:3, 3] += np.asarray(
                    offset, dtype=np.float64
                )
                adjusted_handoff = replace(
                    handoff, rendezvous_pose=adjusted_rendezvous
                )
                try:
                    print(
                        "[heuristic] retrying handoff rendezvous offset="
                        + np.array2string(
                            np.asarray(offset), precision=3
                        )
                    )
                    return self._get_handoff_action(
                        scene,
                        target,
                        pick=pick,
                        handoff=adjusted_handoff,
                        place=place,
                        allow_rendezvous_fallback=False,
                    )
                except NoFeasiblePlanFailure:
                    fallback_failures += 1
            print(
                "[heuristic] handoff rendezvous fallbacks exhausted "
                f"attempts={fallback_failures}"
            )
        if selected is None:
            raise NoFeasiblePlanFailure(
                "all confidence-ranked separated-region handoff pairs are "
                "infeasible; "
                f"separation_rejections={failures['pair_separation']} "
                f"collision_rejections={collision_rejections} "
                f"collision_causes={collision_causes}"
            )

        self._action_metadata_override = list(
            self.staged_controller.metadata
        )
        giver, receiver = selected
        for plan in selected:
            self._save_grasp_visualization(
                scene,
                target,
                ranked,
                plan.candidate,
                plan.arm,
                raw_trace_override=trace,
                executed_command_pose_override=plan.command_targets[1],
                use_default_executed_pose=False,
            )
        print(
            "[heuristic] handoff plan selected "
            f"giver={giver.arm} conf={giver.candidate.confidence:.3f} "
            f"contact={giver.contact_local_point} receiver={receiver.arm} "
            f"conf={receiver.candidate.confidence:.3f} "
            f"contact={receiver.contact_local_point} "
            f"orientation_sources={giver.orientation_source}/"
            f"{receiver.orientation_source} "
            f"schedule={selected_schedule} "
            f"collision_rejections={collision_rejections} "
            f"support_plane_z="
            f"{self._support_plane_for_arm(giver.arm)} "
            f"support_clearance_rejections="
            f"{failures.get('support_clearance', 0)}"
        )
        return actions

    def _plan_simultaneous_pick_sides(
        self,
        scene: SceneObservation,
        target: ObjectState,
        picks: tuple[Pick, Pick],
    ) -> tuple[
        dict[str, list[_SimultaneousPickArmPlan]],
        list[GraspCandidate],
        dict[str, Any],
        dict[str, int],
        float,
        str,
    ]:
        """Plan both recorded arms from one shared confidence-ranked pool."""
        picks_by_arm = {
            str(pick.arm).strip().lower(): pick for pick in picks
        }
        if set(picks_by_arm) != {"left", "right"}:
            raise TargetSelectionFailure(
                "simultaneous Pick requires one recorded stage per arm"
            )
        if {pick.target for pick in picks} != {target.name}:
            raise TargetSelectionFailure(
                "simultaneous Pick target differs from segmented target"
            )

        parameters: dict[str, tuple[float, float, float, np.ndarray]] = {}
        for arm, pick in picks_by_arm.items():
            pregrasp_offset = (
                self.config.pregrasp_offset_m
                if pick.pregrasp_offset_m is None
                else float(pick.pregrasp_offset_m)
            )
            grasp_offset = float(getattr(pick, "grasp_offset_m", 0.0))
            gripper_target = float(getattr(pick, "gripper_target", 0.0))
            postgrasp = np.asarray(
                pick.postgrasp_displacement
                if pick.postgrasp_displacement is not None
                else (0.0, 0.0, self.bimanual_lift_m),
                dtype=np.float64,
            )
            if (
                not np.isfinite(pregrasp_offset)
                or not np.isfinite(grasp_offset)
                or pregrasp_offset < 0.0
                or grasp_offset < 0.0
                or grasp_offset > pregrasp_offset
                or not np.isfinite(gripper_target)
                or not 0.0 <= gripper_target <= 1.0
                or postgrasp.shape != (3,)
                or not np.all(np.isfinite(postgrasp))
            ):
                raise ValueError(
                    f"invalid simultaneous Pick parameters for arm={arm}"
                )
            parameters[arm] = (
                pregrasp_offset,
                grasp_offset,
                gripper_target,
                postgrasp,
            )

        if not np.allclose(
            parameters["left"][3],
            parameters["right"][3],
            rtol=0.0,
            atol=1e-6,
        ):
            raise TargetSelectionFailure(
                "simultaneous same-object Pick requires matching world "
                "postgrasp displacements"
            )

        support_rejections_before = int(
            getattr(self.ik, "failures", {}).get("SupportClearance", 0)
        )
        ranker = ConfidenceRankedGrasps(
            self.grasps, min_confidence=self.config.min_confidence
        )
        ranked = ranker.propose(scene, target)[: self.config.max_candidates]
        trace = self._copy_backend_trace(
            getattr(self.backend, "last_trace", {})
        )
        failures: dict[str, int] = {
            "left_region": 0,
            "right_region": 0,
            "pair_separation": 0,
        }

        world_object = _pose_matrix(
            target.world_pose, name="simultaneous_pick_world_object_pose"
        )
        world_from_object = np.linalg.inv(world_object)
        points = np.asarray(scene.xyz, dtype=np.float64).reshape(-1, 3)
        labels = np.asarray(scene.instance_labels).reshape(-1)
        if len(points) != len(labels):
            raise ValueError("scene points and instance labels must align")
        target_points = points[labels == target.instance_id]
        target_points = target_points[
            np.all(np.isfinite(target_points), axis=1)
        ]
        local_target = (
            target_points @ world_from_object[:3, :3].T
            + world_from_object[:3, 3]
            if len(target_points)
            else np.empty((0, 3), dtype=np.float64)
        )

        def current_ee_position(arm: str) -> np.ndarray | None:
            getter = getattr(
                self.task_env.robot, f"get_{arm}_ee_pose", None
            )
            if getter is None:
                return None
            values = np.asarray(getter(), dtype=np.float64)
            if values.ndim != 1 or len(values) < 3:
                return None
            position = values[:3]
            return position if np.all(np.isfinite(position)) else None

        ee_positions = {
            arm: current_ee_position(arm) for arm in ("left", "right")
        }

        def local_reference(
            world_reference: np.ndarray | None,
        ) -> np.ndarray | None:
            if world_reference is None:
                return None
            return (
                world_from_object[:3, :3] @ world_reference
                + world_from_object[:3, 3]
            )

        (
            left_region,
            right_region,
            minimum_separation,
            contact_region_source,
        ) = _handoff_contact_regions(
            local_target,
            getattr(
                picks_by_arm["left"],
                "allowed_contact_points_local",
                None,
            ),
            getattr(
                picks_by_arm["right"],
                "allowed_contact_points_local",
                None,
            ),
            giver_reference_local=local_reference(ee_positions["left"]),
            receiver_reference_local=local_reference(ee_positions["right"]),
        )
        regions = {"left": left_region, "right": right_region}
        raw_region_distance = max(0.02, minimum_separation)
        classified: dict[
            str,
            list[tuple[GraspCandidate, tuple[float, float, float]]],
        ] = {"left": [], "right": []}
        for candidate in ranked:
            tcp = _grasp_command_tcp(
                candidate.world_grasp_pose, self.ik.grasp_to_robotwin
            )
            local_tcp = (
                world_from_object[:3, :3] @ tcp
                + world_from_object[:3, 3]
            )
            distances = {
                arm: float(
                    np.min(
                        np.linalg.norm(region - local_tcp, axis=1)
                    )
                )
                for arm, region in regions.items()
            }
            local_contact = tuple(float(value) for value in local_tcp)
            if (
                distances["left"] + 1e-6 < distances["right"]
                and distances["left"] <= raw_region_distance
            ):
                classified["left"].append((candidate, local_contact))
            else:
                failures["left_region"] += 1
            if (
                distances["right"] + 1e-6 < distances["left"]
                and distances["right"] <= raw_region_distance
            ):
                classified["right"].append((candidate, local_contact))
            else:
                failures["right_region"] += 1

        def grasp_is_feasible(
            world_grasp_pose: np.ndarray, arm: str
        ) -> bool:
            palm_depth = _target_m2t2_palm_depth(
                scene,
                target,
                world_grasp_pose,
                self.ik.grasp_to_robotwin,
            )
            if (
                palm_depth is not None
                and palm_depth < M2T2_MIN_TARGET_PALM_DEPTH_M
            ):
                key = f"{arm}_palm_clearance"
                failures[key] = failures.get(key, 0) + 1
                return False
            command = (
                np.asarray(world_grasp_pose, dtype=np.float64)
                @ self.ik.grasp_to_robotwin
            )
            width = _target_width_along_axis(
                scene, target, command[:3, 1]
            )
            if (
                width is not None
                and width > self.bimanual_max_target_width_m
            ):
                key = f"{arm}_jaw_width"
                failures[key] = failures.get(key, 0) + 1
                # A global target-cloud span cannot prove that a local M2T2
                # contact is outside the gripper. Keep this as telemetry, but
                # let differential IK and the paired path checks try the pose.
            return True

        plans: dict[str, list[_SimultaneousPickArmPlan]] = {
            "left": [],
            "right": [],
        }
        for arm in ("left", "right"):
            raw_variants = [
                (
                    "m2t2",
                    candidate,
                    local_contact,
                    candidate.world_grasp_pose,
                )
                for candidate, local_contact in classified[arm]
            ]
            aligned_variants = []
            aligned_robot_facing_variants = []
            for candidate in ranked:
                source_tcp = _grasp_command_tcp(
                    candidate.world_grasp_pose,
                    self.ik.grasp_to_robotwin,
                )
                source_local_tcp = (
                    world_from_object[:3, :3] @ source_tcp
                    + world_from_object[:3, 3]
                )
                region = regions[arm]
                anchor = region[
                    int(np.argmin(
                        np.linalg.norm(region - source_local_tcp, axis=1)
                    ))
                ]
                local_contact = tuple(float(value) for value in anchor)
                aligned = _align_grasp_pose_to_local_contact(
                    candidate.world_grasp_pose,
                    world_object,
                    anchor,
                    self.ik.grasp_to_robotwin,
                )
                aligned_variants.append(
                    (
                        "recorded_contact_alignment",
                        candidate,
                        local_contact,
                        aligned,
                    )
                )
                if ee_positions[arm] is None:
                    continue
                try:
                    adjusted = _robot_facing_grasp_pose(
                        aligned,
                        ee_positions[arm],
                        self.ik.grasp_to_robotwin,
                    )
                except ValueError:
                    key = f"{arm}_recorded_contact_robot_facing"
                    failures[key] = failures.get(key, 0) + 1
                    continue
                aligned_robot_facing_variants.append(
                    (
                        "recorded_contact_robot_facing",
                        candidate,
                        local_contact,
                        adjusted,
                    )
                )

            (
                pregrasp_offset,
                grasp_offset,
                gripper_target,
                postgrasp,
            ) = parameters[arm]
            for variants in (
                raw_variants,
                aligned_variants,
                aligned_robot_facing_variants,
            ):
                for (
                    orientation_source,
                    candidate,
                    local_contact,
                    grasp_pose,
                ) in variants:
                    if not grasp_is_feasible(grasp_pose, arm):
                        continue
                    contact_command = (
                        np.asarray(grasp_pose, dtype=np.float64)
                        @ self.ik.grasp_to_robotwin
                    )
                    pregrasp_command = _approach_offset_command_pose(
                        contact_command, pregrasp_offset
                    )
                    grasp_command = _approach_offset_command_pose(
                        contact_command, grasp_offset
                    )
                    lift_command = grasp_command.copy()
                    lift_command[:3, 3] += postgrasp
                    chain = self._solve_handoff_command_chain(
                        arm,
                        arm,
                        (
                            ("pregrasp", pregrasp_command),
                            ("grasp", grasp_command),
                            ("lift", lift_command),
                        ),
                        failures,
                    )
                    if chain is None:
                        continue
                    paths, commands = chain
                    planned_candidate = GraspCandidate(
                        np.asarray(grasp_pose, dtype=np.float64),
                        float(candidate.confidence),
                        candidate.object_name,
                    )
                    plans[arm].append(
                        _SimultaneousPickArmPlan(
                            arm=arm,
                            target_name=target.name,
                            arm_source="robotwin_ground_truth",
                            candidate=planned_candidate,
                            paths=paths,
                            command_targets=commands,
                            contact_local_point=local_contact,
                            gripper_target=gripper_target,
                            orientation_source=orientation_source,
                        )
                    )
                    if (
                        len(plans[arm])
                        >= self.bimanual_max_plans_per_arm
                    ):
                        break
                if (
                    len(plans[arm])
                    >= self.bimanual_max_plans_per_arm
                ):
                    break

        failures["support_clearance"] = (
            int(
                getattr(self.ik, "failures", {}).get(
                    "SupportClearance", 0
                )
            )
            - support_rejections_before
        )
        return (
            plans,
            ranked,
            trace,
            failures,
            minimum_separation,
            contact_region_source,
        )

    def _build_simultaneous_pick_action_pair(
        self,
        left: _SimultaneousPickArmPlan,
        right: _SimultaneousPickArmPlan,
    ) -> list[np.ndarray]:
        """Synchronize dual pregrasp/grasp/close/lift joint commands."""
        plans = {"left": left, "right": right}
        if left.arm != "left" or right.arm != "right":
            raise ValueError("simultaneous plans require one plan per arm")
        if left.target_name != right.target_name:
            raise ValueError("simultaneous plans must share one target")
        if any(
            len(plan.paths) != 3 or len(plan.command_targets) != 3
            for plan in plans.values()
        ):
            raise ValueError(
                "simultaneous plans require pregrasp/grasp/lift chains"
            )

        controller = self.staged_controller
        controller.reset()
        names = {arm: plan.target_name for arm, plan in plans.items()}
        sources = {arm: plan.arm_source for arm, plan in plans.items()}
        controller.gripper_phase(
            "open",
            {"left": 1.0, "right": 1.0},
            1,
            names,
            sources,
        )
        for phase, index in (
            ("pregrasp", 0),
            ("grasp", 1),
        ):
            controller.move_phase(
                phase,
                {arm: plan.paths[index] for arm, plan in plans.items()},
                {
                    arm: plan.command_targets[index]
                    for arm, plan in plans.items()
                },
                names,
                sources,
            )
        controller.gripper_phase(
            "close",
            {arm: plan.gripper_target for arm, plan in plans.items()},
            self.gripper_settle_actions,
            names,
            sources,
        )
        controller.move_phase(
            "lift",
            {arm: plan.paths[2] for arm, plan in plans.items()},
            {arm: plan.command_targets[2] for arm, plan in plans.items()},
            names,
            sources,
        )
        actions = [action.copy() for action in controller.actions]
        if any(
            action.shape != (14,) or not np.all(np.isfinite(action))
            for action in actions
        ):
            raise ValueError(
                "simultaneous Pick must compose finite 14D qpos actions"
            )
        return actions

    def _get_simultaneous_pick_action(
        self,
        scene: SceneObservation,
        target: ObjectState,
        picks: tuple[Pick, Pick],
    ) -> list[np.ndarray]:
        (
            plans,
            ranked,
            trace,
            failures,
            minimum_separation,
            contact_region_source,
        ) = self._plan_simultaneous_pick_sides(scene, target, picks)
        if not plans["left"] or not plans["right"]:
            for arm in ("left", "right"):
                self._save_grasp_visualization(
                    scene,
                    target,
                    ranked,
                    None,
                    arm,
                    raw_trace_override=trace,
                    executed_command_pose_override=None,
                    use_default_executed_pose=False,
                )
            raise NoFeasiblePlanFailure(
                "M2T2/Mink produced no complete simultaneous Pick plan; "
                f"left={len(plans['left'])} right={len(plans['right'])} "
                f"failures={failures}; ik_failures={self.ik.failures}"
            )

        pairs = [
            (left, right)
            for left in plans["left"]
            for right in plans["right"]
            if float(
                np.linalg.norm(
                    np.asarray(left.contact_local_point)
                    - np.asarray(right.contact_local_point)
                )
            )
            >= minimum_separation
        ]
        failures["pair_separation"] += (
            len(plans["left"]) * len(plans["right"]) - len(pairs)
        )
        pairs.sort(
            key=lambda pair: (
                sum(
                    plan.orientation_source == "m2t2" for plan in pair
                ),
                pair[0].candidate.confidence
                + pair[1].candidate.confidence,
            ),
            reverse=True,
        )

        selected: tuple[
            _SimultaneousPickArmPlan, _SimultaneousPickArmPlan
        ] | None = None
        actions: list[np.ndarray] = []
        collision_rejections = 0
        for left, right in pairs:
            candidate_actions = self._build_simultaneous_pick_action_pair(
                left, right
            )
            if self.ik.full_robot_path_has_self_collision(
                candidate_actions,
                max_joint_step_rad=self.bimanual_collision_step_rad,
            ):
                collision_rejections += 1
                continue
            selected = left, right
            actions = candidate_actions
            break
        if selected is None:
            raise NoFeasiblePlanFailure(
                "all confidence-ranked separated-region simultaneous Pick "
                "pairs are infeasible; "
                f"separation_rejections={failures['pair_separation']} "
                f"collision_rejections={collision_rejections}"
            )

        self._action_metadata_override = list(
            self.staged_controller.metadata
        )
        left, right = selected
        for plan in selected:
            self._save_grasp_visualization(
                scene,
                target,
                ranked,
                plan.candidate,
                plan.arm,
                raw_trace_override=trace,
                executed_command_pose_override=plan.command_targets[1],
                use_default_executed_pose=False,
            )
        print(
            "[heuristic] simultaneous Pick selected "
            f"target={target.name} left_conf={left.candidate.confidence:.3f} "
            f"left_contact={left.contact_local_point} "
            f"right_conf={right.candidate.confidence:.3f} "
            f"right_contact={right.contact_local_point} "
            f"orientation_sources={left.orientation_source}/"
            f"{right.orientation_source} "
            f"contact_regions={contact_region_source} "
            f"collision_rejections={collision_rejections} "
            f"support_plane_z={self._support_plane_for_arm('left')} "
            f"support_clearance_rejections="
            f"{failures.get('support_clearance', 0)}"
        )
        return actions

    def _get_auxiliary_pick_place_action(
        self,
        scene: SceneObservation,
        target_names: tuple[str, ...],
        stages: tuple[Pick, Pick, Place],
    ) -> list[np.ndarray]:
        """Execute a displaced auxiliary Pick before a primary placement."""
        auxiliary_pick, primary_pick, primary_place = stages
        if {
            auxiliary_pick.target, primary_pick.target
        } != set(target_names):
            raise TargetSelectionFailure(
                "auxiliary/primary Pick targets differ from segmented targets"
            )
        auxiliary_arm = str(auxiliary_pick.arm).strip().lower()
        primary_arm = str(primary_pick.arm).strip().lower()
        auxiliary_target = self.simulator.object_state(
            auxiliary_pick.target
        )
        primary_target = self.simulator.object_state(primary_pick.target)
        auxiliary_place = Place(
            object=auxiliary_pick.target,
            destination=None,
            arm=auxiliary_arm,
            target_pose=auxiliary_target.world_pose.copy(),
            preplace_offset_m=0.0,
            place_offset_m=0.0,
            constrain="free",
            release=False,
        )

        self._configure_support_plane(
            scene, auxiliary_target, (auxiliary_arm,)
        )
        auxiliary_plans, auxiliary_ranked, auxiliary_trace, auxiliary_failures = (
            self._plan_single_arm_place(
                scene,
                auxiliary_target,
                pick=auxiliary_pick,
                stop_after_grasp_lift=True,
                place=auxiliary_place,
                arm=auxiliary_arm,
                arm_source="recorded_auxiliary",
                plan_limit=1,
            )
        )
        auxiliary_plan = next(
            (plan for plan in auxiliary_plans if len(plan.paths) >= 3),
            None,
        )
        if auxiliary_plan is None:
            raise NoFeasiblePlanFailure(
                "M2T2/Mink produced no auxiliary grasp/displacement prefix; "
                f"failures={auxiliary_failures}; "
                f"ik_failures={self.ik.failures}"
            )

        starts = {
            arm: _arm_joint_state(self.task_env, arm)[:-1].copy()
            for arm in ("left", "right")
        }
        starts[auxiliary_arm] = auxiliary_plan.paths[2][-1].copy()
        for arm, joints in starts.items():
            self.ik.set_joint_start_override(arm, joints)
        self._configure_support_plane(scene, primary_target, (primary_arm,))
        try:
            primary_plans, primary_ranked, primary_trace, primary_failures = (
                self._plan_single_arm_place(
                    scene,
                    primary_target,
                    pick=primary_pick,
                    place=primary_place,
                    arm=primary_arm,
                    arm_source="recorded_primary",
                    plan_limit=self.bimanual_max_plans_per_arm,
                )
            )
        finally:
            for arm in ("left", "right"):
                self.ik.set_joint_start_override(arm, None)
        primary_plan = next(
            (
                plan for plan in primary_plans
                if plan.completion_level == "place"
                and len(plan.paths) >= (
                    6 if primary_place.release else 5
                )
            ),
            None,
        )
        if primary_plan is None:
            raise NoFeasiblePlanFailure(
                "M2T2/Mink produced no primary placement after auxiliary "
                f"interaction; failures={primary_failures}; "
                f"ik_failures={self.ik.failures}"
            )

        controller = self.staged_controller
        controller.reset()
        for target, pick, plan, arm in (
            (
                auxiliary_target,
                auxiliary_pick,
                auxiliary_plan,
                auxiliary_arm,
            ),
            (primary_target, primary_pick, primary_plan, primary_arm),
        ):
            names = {arm: target.name}
            sources = {arm: plan.arm_source}
            controller.gripper_phase(
                "open", {arm: 1.0}, 1, names, sources
            )
            for phase, index in (("pregrasp", 0), ("grasp", 1)):
                controller.move_phase(
                    phase,
                    {arm: plan.paths[index]},
                    {arm: plan.command_targets[index]},
                    names,
                    sources,
                )
            controller.gripper_phase(
                "close",
                {arm: float(getattr(pick, "gripper_target", 0.0))},
                self.gripper_settle_actions,
                names,
                sources,
            )
            controller.move_phase(
                "lift" if target is primary_target else "displace",
                {arm: plan.paths[2]},
                {arm: plan.command_targets[2]},
                names,
                sources,
            )
            if target is auxiliary_target:
                continue
            for phase, index in (("preplace", 3), ("place", 4)):
                controller.move_phase(
                    phase,
                    {arm: plan.paths[index]},
                    {arm: plan.command_targets[index]},
                    names,
                    sources,
                )
            if primary_place.release:
                controller.gripper_phase(
                    "open",
                    {arm: 1.0},
                    self.gripper_settle_actions,
                    names,
                    sources,
                )
                controller.move_phase(
                    "retreat",
                    {arm: plan.paths[5]},
                    {arm: plan.command_targets[5]},
                    names,
                    sources,
                )

        actions = [action.copy() for action in controller.actions]
        if self.ik.full_robot_path_has_self_collision(
            actions, max_joint_step_rad=self.bimanual_collision_step_rad
        ):
            raise NoFeasiblePlanFailure(
                "composed auxiliary interaction/primary placement path "
                "has a robot collision"
            )
        self._action_metadata_override = list(controller.metadata)
        for target, plan, ranked, trace in (
            (
                auxiliary_target, auxiliary_plan,
                auxiliary_ranked, auxiliary_trace,
            ),
            (primary_target, primary_plan, primary_ranked, primary_trace),
        ):
            self._save_grasp_visualization(
                scene,
                target,
                ranked,
                plan.candidate,
                plan.arm,
                raw_trace_override=trace,
                executed_command_pose_override=plan.command_targets[1],
                use_default_executed_pose=False,
            )
        print(
            "[heuristic] auxiliary interaction then placement selected "
            f"auxiliary={auxiliary_target.name}:{auxiliary_arm} "
            f"primary={primary_target.name}:{primary_arm}"
        )
        return actions


    def _get_grouped_bimanual_place_action(
        self,
        scene: SceneObservation,
        target_names: tuple[str, ...],
        stage_pairs: tuple[tuple[Pick, Place], tuple[Pick, Place]],
    ) -> list[np.ndarray]:
        """Select a collision-free pair of recorded Pick/Place chains."""
        if {pick.target for pick, _ in stage_pairs} != set(target_names):
            raise TargetSelectionFailure(
                "grouped Pick targets differ from segmented manipulation targets"
            )

        records: dict[str, tuple[Any, ...]] = {}
        for pick, place in stage_pairs:
            target = self.simulator.object_state(pick.target)
            arm, arm_source = self._select_arm(target)
            expected_arm = str(pick.arm).strip().lower()
            if arm != expected_arm:
                raise TargetSelectionFailure(
                    "recorded bimanual Pick arm differs from selected arm"
                )
            self._configure_support_plane(scene, target, (arm,))
            plans, ranked, trace, failures = self._plan_single_arm_place(
                scene,
                target,
                pick=pick,
                place=place,
                arm=arm,
                arm_source=arm_source,
                plan_limit=self.bimanual_max_plans_per_arm,
            )
            plans = [
                plan
                for plan in plans
                if plan.completion_level == "place"
                and len(plan.paths) >= (6 if place.release else 5)
                and len(plan.command_targets) >= (
                    6 if place.release else 5
                )
            ]
            if not plans:
                self._save_grasp_visualization(
                    scene,
                    target,
                    ranked,
                    None,
                    arm,
                    raw_trace_override=trace,
                    executed_command_pose_override=None,
                    use_default_executed_pose=False,
                )
                raise NoFeasiblePlanFailure(
                    "M2T2/Mink produced no complete grouped bimanual "
                    f"placement plan for arm={arm} target={target.name}; "
                    f"failures={failures}; "
                    f"ik_failures={self.ik.failures}"
                )
            if arm in records:
                raise TargetSelectionFailure(
                    f"grouped bimanual plan assigns multiple objects to {arm}"
                )
            records[arm] = (
                target,
                pick,
                place,
                plans,
                ranked,
                trace,
            )
        if set(records) != {"left", "right"}:
            raise TargetSelectionFailure(
                "grouped bimanual plan requires one chain per arm"
            )

        names = {arm: records[arm][0].name for arm in records}
        sources = {
            arm: records[arm][3][0].arm_source for arm in records
        }

        def compose(
            selected: dict[str, _SingleArmPlacePlan],
        ) -> list[np.ndarray]:
            controller = self.staged_controller
            controller.reset()
            controller.gripper_phase(
                "open",
                {arm: 1.0 for arm in selected},
                1,
                names,
                sources,
            )
            for phase, index in (("pregrasp", 0), ("grasp", 1)):
                controller.move_phase(
                    phase,
                    {
                        arm: selected[arm].paths[index]
                        for arm in selected
                    },
                    {
                        arm: selected[arm].command_targets[index]
                        for arm in selected
                    },
                    names,
                    sources,
                )
            controller.gripper_phase(
                "close",
                {
                    arm: float(records[arm][1].gripper_target)
                    for arm in selected
                },
                self.gripper_settle_actions,
                names,
                sources,
            )
            for phase, index in (
                ("lift", 2),
                ("preplace", 3),
                ("place", 4),
            ):
                controller.move_phase(
                    phase,
                    {
                        arm: selected[arm].paths[index]
                        for arm in selected
                    },
                    {
                        arm: selected[arm].command_targets[index]
                        for arm in selected
                    },
                    names,
                    sources,
                )
            release_arms = {
                arm for arm in selected if records[arm][2].release
            }
            if release_arms:
                controller.gripper_phase(
                    "open",
                    {arm: 1.0 for arm in release_arms},
                    self.gripper_settle_actions,
                    names,
                    sources,
                )
                controller.move_phase(
                    "retreat",
                    {
                        arm: selected[arm].paths[5]
                        for arm in release_arms
                    },
                    {
                        arm: selected[arm].command_targets[5]
                        for arm in release_arms
                    },
                    names,
                    sources,
                )
            return [action.copy() for action in controller.actions]

        pairs = [
            (left, right)
            for left in records["left"][3]
            for right in records["right"][3]
        ]
        pairs.sort(
            key=lambda pair: (
                all(
                    plan.orientation_source == "m2t2"
                    for plan in pair
                ),
                pair[0].candidate.confidence
                + pair[1].candidate.confidence,
            ),
            reverse=True,
        )
        selected_pair: tuple[
            _SingleArmPlacePlan, _SingleArmPlacePlan
        ] | None = None
        actions: list[np.ndarray] = []
        collision_rejections = 0
        for left, right in pairs:
            candidate_actions = compose({"left": left, "right": right})
            if self.ik.full_robot_path_has_self_collision(
                candidate_actions,
                max_joint_step_rad=self.bimanual_collision_step_rad,
            ):
                collision_rejections += 1
                continue
            selected_pair = left, right
            actions = candidate_actions
            break
        if selected_pair is None:
            def with_preplace_clearance(
                plan: _SingleArmPlacePlan,
                arm: str,
                clearance_m: float,
            ) -> _SingleArmPlacePlan | None:
                paths = list(plan.paths)
                commands = [
                    np.asarray(command, dtype=np.float64).copy()
                    for command in plan.command_targets
                ]
                raised = commands[3].copy()
                raised[2, 3] += float(clearance_m)
                start = np.asarray(paths[2][-1], dtype=np.float64)
                raised_result = self.ik.solve_command_target(
                    arm, raised, start
                )
                if raised_result is None:
                    return None
                place_result = self.ik.solve_command_target(
                    arm, commands[4], raised_result[0]
                )
                if place_result is None:
                    return None
                paths[3] = raised_result[1]
                commands[3] = raised_result[2]
                paths[4] = place_result[1]
                commands[4] = place_result[2]
                if records[arm][2].release:
                    retreat_result = self.ik.solve_command_target(
                        arm, commands[5], place_result[0]
                    )
                    if retreat_result is None:
                        return None
                    paths[5] = retreat_result[1]
                    commands[5] = retreat_result[2]
                return replace(
                    plan,
                    paths=tuple(paths),
                    command_targets=tuple(commands),
                    orientation_source=(
                        f"{plan.orientation_source}/staggered_z"
                    ),
                )

            z_assignments = (
                (0.12, 0.04),
                (0.04, 0.12),
                (0.10, 0.02),
                (0.02, 0.10),
            )
            staggered_rejections = 0
            for left_clearance, right_clearance in z_assignments:
                for left, right in pairs:
                    raised_left = with_preplace_clearance(
                        left, "left", left_clearance
                    )
                    if raised_left is None:
                        continue
                    raised_right = with_preplace_clearance(
                        right, "right", right_clearance
                    )
                    if raised_right is None:
                        continue
                    candidate_actions = compose({
                        "left": raised_left,
                        "right": raised_right,
                    })
                    if self.ik.full_robot_path_has_self_collision(
                        candidate_actions,
                        max_joint_step_rad=self.bimanual_collision_step_rad,
                    ):
                        staggered_rejections += 1
                        continue
                    selected_pair = raised_left, raised_right
                    actions = candidate_actions
                    print(
                        "[heuristic] grouped placement using staggered "
                        f"preplace_z left=+{left_clearance:.2f}m "
                        f"right=+{right_clearance:.2f}m "
                        f"collision_rejections={staggered_rejections}"
                    )
                    break
                if selected_pair is not None:
                    break
        if selected_pair is None:
            raise NoFeasiblePlanFailure(
                "all confidence-ranked grouped bimanual placement pairs "
                f"collide; rejected_pairs={collision_rejections}"
            )

        selected = {
            "left": selected_pair[0],
            "right": selected_pair[1],
        }
        self._action_metadata_override = list(
            self.staged_controller.metadata
        )
        for arm in ("left", "right"):
            target, _, _, _, ranked, trace = records[arm]
            plan = selected[arm]
            self._save_grasp_visualization(
                scene,
                target,
                ranked,
                plan.candidate,
                arm,
                raw_trace_override=trace,
                executed_command_pose_override=plan.command_targets[1],
                use_default_executed_pose=False,
            )
        print(
            "[heuristic] grouped bimanual placement selected "
            + " ".join(
                f"{arm}_target={records[arm][0].name} "
                f"{arm}_confidence={selected[arm].candidate.confidence:.3f}"
                for arm in ("left", "right")
            )
            + f" collision_rejections={collision_rejections}"
            + " support_plane_z="
            + str({
                arm: self._support_plane_for_arm(arm)
                for arm in ("left", "right")
            })
            + " support_clearance_rejections="
            + str(getattr(self.ik, "failures", {}).get(
                "SupportClearance", 0
            ))
        )
        return actions

    def _get_sequential_place_action(
        self,
        scene: SceneObservation,
        target_names: tuple[str, ...],
        stage_pairs: tuple[tuple[Pick, Place], ...],
    ) -> list[np.ndarray]:
        """Plan and compose recorded placements in expert order."""
        if len(stage_pairs) != len(target_names) or {
            pick.target for pick, _ in stage_pairs
        } != set(target_names):
            raise TargetSelectionFailure(
                "sequential manipulation requires matching Pick/Place chains"
            )

        planned = []
        home_joints = {
            arm: _arm_joint_state(self.task_env, arm)[:-1].copy()
            for arm in ("left", "right")
        }
        composed_joints = {
            arm: joints.copy() for arm, joints in home_joints.items()
        }
        for stage_index, (pick, place) in enumerate(stage_pairs):
            target = self.simulator.object_state(pick.target)
            arm, arm_source = self._select_arm(target)
            expected_arm = str(getattr(pick, "arm", "")).strip().lower()
            if expected_arm in {"left", "right"} and arm != expected_arm:
                raise TargetSelectionFailure(
                    "recorded sequential Pick arm differs from selected arm"
                )
            self._configure_support_plane(scene, target, (arm,))
            for composed_arm in ("left", "right"):
                self.ik.set_joint_start_override(
                    composed_arm, composed_joints[composed_arm]
                )
            try:
                plans, ranked, trace, failures = self._plan_single_arm_place(
                    scene,
                    target,
                    pick=pick,
                    place=place,
                    arm=arm,
                    arm_source=arm_source,
                    plan_limit=self.bimanual_max_plans_per_arm,
                )
            finally:
                for composed_arm in ("left", "right"):
                    self.ik.set_joint_start_override(composed_arm, None)
            selected = plans[0] if plans else None
            if selected is None or selected.completion_level != "place":
                raise NoFeasiblePlanFailure(
                    "M2T2/Mink produced no complete sequential placement "
                    f"for target={target.name}; failures={failures}; "
                    f"ik_failures={self.ik.failures}"
                )
            if not place.release:
                raise TargetSelectionFailure(
                    "sequential three-target placements must release each object"
                )
            clearance = None
            home_path = None
            if stage_index < len(stage_pairs) - 1:
                next_pick = stage_pairs[stage_index + 1][0]
                next_arm = str(getattr(next_pick, "arm", "")).strip().lower()
                return_home = next_arm in {"left", "right"} and next_arm != arm
                for candidate_plan in plans:
                    if candidate_plan.completion_level != "place":
                        continue
                    clearance_command = candidate_plan.command_targets[-1].copy()
                    clearance_command[:3, 3] += np.array(
                        [0.0, 0.0, self.config.retreat_offset_m],
                        dtype=np.float64,
                    )
                    for composed_arm in ("left", "right"):
                        self.ik.set_joint_start_override(
                            composed_arm, composed_joints[composed_arm]
                        )
                    try:
                        candidate_clearance = self.ik.solve_command_target(
                            arm,
                            clearance_command,
                            candidate_plan.paths[-1][-1],
                        )
                    finally:
                        for composed_arm in ("left", "right"):
                            self.ik.set_joint_start_override(composed_arm, None)
                    if candidate_clearance is None:
                        continue
                    candidate_home = None
                    if return_home:
                        for composed_arm in ("left", "right"):
                            self.ik.set_joint_start_override(
                                composed_arm, composed_joints[composed_arm]
                            )
                        try:
                            candidate_home = self.ik.plan_joint_transition(
                                arm,
                                candidate_clearance[0],
                                home_joints[arm],
                            )
                        finally:
                            for composed_arm in ("left", "right"):
                                self.ik.set_joint_start_override(
                                    composed_arm, None
                                )
                        if candidate_home is None:
                            continue
                    selected = candidate_plan
                    clearance = candidate_clearance
                    home_path = candidate_home
                    break
                if clearance is None:
                    raise NoFeasiblePlanFailure(
                        "Mink found no collision-free world-Z clearance/home "
                        "transition "
                        f"after target={target.name}"
                    )
            planned.append(
                (
                    target, pick, selected, clearance, home_path,
                    ranked, trace, failures,
                )
            )
            composed_joints[arm] = (
                home_joints[arm].copy()
                if home_path is not None
                else (
                    clearance[0].copy()
                    if clearance is not None
                    else selected.paths[-1][-1].copy()
                )
            )

        controller = self.staged_controller
        controller.reset()
        for (
            target, pick, selected, clearance, home_path, _, _, _
        ) in planned:
            arm = selected.arm
            names = {arm: target.name}
            sources = {arm: selected.arm_source}
            controller.gripper_phase("open", {arm: 1.0}, 1, names, sources)
            for phase, index in (("pregrasp", 0), ("grasp", 1)):
                controller.move_phase(
                    phase, {arm: selected.paths[index]},
                    {arm: selected.command_targets[index]}, names, sources,
                )
            controller.gripper_phase(
                "close",
                {arm: float(getattr(pick, "gripper_target", 0.0))},
                self.gripper_settle_actions,
                names,
                sources,
            )
            for phase, index in (
                ("lift", 2), ("preplace", 3), ("place", 4)
            ):
                controller.move_phase(
                    phase, {arm: selected.paths[index]},
                    {arm: selected.command_targets[index]}, names, sources,
                )
            controller.gripper_phase(
                "open", {arm: 1.0}, self.gripper_settle_actions, names, sources
            )
            controller.move_phase(
                "retreat", {arm: selected.paths[5]},
                {arm: selected.command_targets[5]}, names, sources,
            )
            if clearance is not None:
                _, clearance_path, clearance_target = clearance
                controller.move_phase(
                    "clearance_z",
                    {arm: clearance_path},
                    {arm: clearance_target},
                    names,
                    sources,
                )
            if home_path is not None:
                controller.move_phase(
                    "home",
                    {arm: home_path},
                    target_names=names,
                    arm_sources=sources,
                )

        actions = [action.copy() for action in controller.actions]
        if self.ik.full_robot_path_has_self_collision(actions):
            collision_stage = None
            for index, metadata in enumerate(controller.metadata):
                if (
                    metadata.get("endpoint")
                    and self.ik.full_robot_path_has_self_collision(
                        actions[:index + 1]
                    )
                ):
                    collision_stage = metadata
                    break
            details = ""
            if collision_stage is not None:
                details = (
                    f"; phase={collision_stage.get('phase')} "
                    f"arm={collision_stage.get('arm')} "
                    f"target={collision_stage.get('target_name')}"
                )
            raise NoFeasiblePlanFailure(
                "composed sequential multi-target path has a robot collision"
                + details
            )
        self._action_metadata_override = list(controller.metadata)
        for target, _, selected, _, _, ranked, trace, failures in planned:
            self._save_grasp_visualization(
                scene, target, ranked, selected.candidate, selected.arm,
                raw_trace_override=trace,
                executed_command_pose_override=selected.command_targets[1],
                use_default_executed_pose=False,
            )
            print(
                "[heuristic] sequential placement selected "
                f"target={target.name} arm={selected.arm} "
                f"confidence={selected.candidate.confidence:.3f} "
                f"orientation_source={selected.orientation_source} "
                f"support_clearance_rejections="
                f"{failures.get('support_clearance', 0)}"
            )
        return actions


    @property
    def action_metadata(self) -> list[dict[str, Any]]:
        if self._action_metadata_override is not None:
            return list(self._action_metadata_override)
        return list(self.controller.metadata)

    @property
    def grasp_attempted(self) -> bool:
        """Whether this episode has consumed its single grasp-planning attempt."""
        return self._grasp_attempted

    def get_action(self, *, scene: SceneObservation) -> list[np.ndarray]:
        if self._grasp_attempted:
            raise NoFeasiblePlanFailure(
                "one-shot grasp attempt already consumed for this episode"
            )
        self._grasp_attempted = True
        self._action_metadata_override = None
        clear_support = getattr(self.ik, "clear_support_planes", None)
        if clear_support is not None:
            clear_support()
        self.simulator.update(scene)
        target_names = self._target_names(scene)
        simultaneous_picks = self._simultaneous_same_object_pick_stages(
            self.task_env
        )
        if simultaneous_picks is not None:
            if (
                len(target_names) != 1
                or simultaneous_picks[0].target != target_names[0]
            ):
                raise TargetSelectionFailure(
                    "simultaneous Pick target differs from segmented "
                    "manipulation targets"
                )
            target = self.simulator.object_state(target_names[0])
            self.ik.reset_stats()
            self._configure_support_plane(
                scene, target, ("left", "right")
            )
            return self._get_simultaneous_pick_action(
                scene, target, simultaneous_picks
            )
        if len(target_names) == 2:
            self.ik.reset_stats()
            auxiliary_stages = self._auxiliary_pick_place_stages(
                self.task_env
            )
            if auxiliary_stages is not None:
                return self._get_auxiliary_pick_place_action(
                    scene, target_names, auxiliary_stages
                )
            grouped_stages = self._grouped_bimanual_place_stages(
                self.task_env
            )
            if grouped_stages is not None:
                return self._get_grouped_bimanual_place_action(
                    scene, target_names, grouped_stages
                )
            sequential_stages = self._sequential_place_stages(self.task_env)
            if sequential_stages is None:
                raise TargetSelectionFailure(
                    "two-target manipulation requires recorded grouped or "
                    "sequential Pick/Place chains"
                )
            return self._get_sequential_place_action(
                scene, target_names, sequential_stages
            )
        if len(target_names) == 3:
            sequential_stages = self._sequential_place_stages(self.task_env)
            if sequential_stages is None:
                raise TargetSelectionFailure(
                    "three-target manipulation requires three recorded "
                    "sequential Pick/Place chains"
                )
            self.ik.reset_stats()
            return self._get_sequential_place_action(
                scene, target_names, sequential_stages
            )
        if len(target_names) != 1:
            raise TargetSelectionFailure(
                f"unsupported manipulation target count: {len(target_names)}"
            )
        target_name = target_names[0]
        target = self.simulator.object_state(target_name)
        arm, arm_source = self._select_arm(target)
        print(
            f"[heuristic] arm selection target={target_name} arm={arm} "
            f"source={arm_source}"
        )

        self.controller.reset()
        self.ik.reset_stats()
        handoff_stages = self._handoff_stages(self.task_env)
        if handoff_stages is not None:
            pick, handoff, place = handoff_stages
            if pick.target != target_name:
                raise TargetSelectionFailure(
                    "structured handoff source does not match M2T2 target: "
                    f"{pick.target!r} != {target_name!r}"
                )
            self._configure_support_plane(
                scene,
                target,
                (
                    str(handoff.from_arm).strip().lower(),
                    str(handoff.to_arm).strip().lower(),
                ),
            )
            return self._get_handoff_action(
                scene,
                target,
                pick=pick,
                handoff=handoff,
                place=place,
            )
        place_stages = self._single_arm_place_stages(self.task_env)
        if place_stages is not None:
            pick, place = place_stages
            if pick.target != target_name:
                raise TargetSelectionFailure(
                    "structured place source does not match M2T2 target: "
                    f"{pick.target!r} != {target_name!r}"
                )
            if (
                place.arm is not None
                and str(place.arm).strip().lower() in {"left", "right"}
                and str(place.arm).strip().lower() != arm
            ):
                raise TargetSelectionFailure(
                    "structured place arm differs from selected grasp arm"
                )
            self._configure_support_plane(scene, target, (arm,))
            return self._get_single_place_action(
                scene,
                target,
                pick=pick,
                place=place,
                arm=arm,
                arm_source=arm_source,
            )
        self._configure_support_plane(scene, target, (arm,))
        ranker = ConfidenceRankedGrasps(
            self.grasps, min_confidence=self.config.min_confidence
        )
        policy = SimpleGraspPolicy(
            self.simulator,
            ranker,
            self.ik,
            self.controller,
            replace(self.config, object_name=target_name, arm=arm),
        )
        try:
            selected = policy.run_once()
        except RuntimeError as exc:
            self._save_grasp_visualization(
                scene, target, ranker.last_candidates, None, arm
            )
            if str(exc) != "M2T2 returned no grasp with complete feasible IK solutions":
                raise
            raise NoFeasiblePlanFailure(
                f"{exc}; Mink IK accepted {self.ik.successes}/"
                f"{self.ik.calls} stages across {self.ik.planner_attempts} attempts "
                f"({self.ik.relaxed_successes} canonical retries); "
                f"canonical_seed_successes={self.ik.canonical_seed_successes}; "
                f"failures={self.ik.failures}; "
                f"target_position={np.array2string(target.world_pose[:3, 3], precision=3)}; "
                f"first_target={np.array2string(self.ik.first_target, precision=3)}"
            ) from exc
        self._save_grasp_visualization(
            scene, target, ranker.last_candidates, selected, arm
        )
        for metadata in self.controller.metadata:
            metadata["target_name"] = target_name
            metadata["arm_source"] = arm_source
        return self.controller.actions

    def reset(self) -> None:
        self.simulator.scene = None
        self._grasp_attempted = False
        self._action_metadata_override = None
        self.controller.reset()
        staged_controller = getattr(self, "staged_controller", None)
        if staged_controller is not None:
            staged_controller.reset()
        self.bimanual_controller.reset()
        self.ik.reset_stats()
        clear_start = getattr(self.ik, "set_joint_start_override", None)
        if clear_start is not None:
            for arm in ("left", "right"):
                clear_start(arm, None)
        clear_support = getattr(self.ik, "clear_support_planes", None)
        if clear_support is not None:
            clear_support()
        self.backend.reset(int(getattr(self.task_env, "episode_seed", 0)))
        self._visualization_index = 0


def create_runtime(
    *,
    usr_args: dict,
    simple_grasp_root: str | Path,
    task_env: Any,
) -> RoboTwinHeuristicRuntime:
    root = Path(simple_grasp_root).expanduser().resolve()
    object_name = str(usr_args.get("object_name", "auto"))
    arm = str(usr_args.get("arm", "auto"))
    policy_config = PolicyConfig(
        object_name=object_name,
        arm="right" if arm == "auto" else arm,
        min_confidence=float(usr_args.get("min_confidence", 0.4)),
        max_candidates=int(usr_args.get("max_candidates", 64)),
        pregrasp_offset_m=float(usr_args.get("pregrasp_offset_m", 0.10)),
        retreat_offset_m=float(usr_args.get("retreat_offset_m", 0.12)),
    )

    backend = RoboTwinM2T2Backend(
        simple_grasp_root=root,
        checkpoint=usr_args.get("m2t2_checkpoint", "checkpoints/m2t2.pth"),
        config=usr_args.get("m2t2_config", "third_party/M2T2/config.yaml"),
        device=str(usr_args.get("device", "cuda:0")),
        num_points=int(usr_args.get("num_points", 16_384)),
        num_object_points=int(usr_args.get("num_object_points", 1_024)),
        num_runs=int(usr_args.get("num_runs", 2)),
        mask_threshold=float(usr_args.get("mask_threshold", 0.4)),
        object_threshold=float(usr_args.get("object_threshold", 0.4)),
        max_predictions=usr_args.get("max_predictions", 64),
        workspace_bounds=usr_args.get(
            "workspace_bounds", [-0.5, -0.5, 0.65, 0.5, 0.5, 1.4]
        ),
        contact_match_distance_m=float(
            usr_args.get("contact_match_distance_m", 1e-5)
        ),
        seed=int(getattr(task_env, "episode_seed", usr_args.get("seed", 0))),
    )
    grasps = M2T2GraspGenerator(backend, outputs_in_world=True)
    transform = np.asarray(
        usr_args.get("grasp_to_robotwin", M2T2_TO_ROBOTWIN), dtype=np.float64
    )
    return RoboTwinHeuristicRuntime(
        task_env=task_env,
        grasps=grasps,
        config=policy_config,
        automatic_target=object_name == "auto",
        automatic_arm=arm == "auto",
        grasp_to_robotwin=transform,
        max_waypoints_per_segment=int(
            usr_args.get("max_waypoints_per_segment", 64)
        ),
        max_joint_step_rad=float(usr_args.get("max_joint_step_rad", 0.12)),
        gripper_settle_actions=int(usr_args.get("gripper_settle_actions", 5)),
        mink_model_path=usr_args.get("mink_model_path"),
        mink_config=MinkIKConfig(
            solver=str(usr_args.get("mink_solver", "daqp")),
            dt=float(usr_args.get("mink_dt", 0.05)),
            max_iterations=int(usr_args.get("mink_max_iterations", 100)),
            position_tolerance_m=float(
                usr_args.get("mink_position_tolerance_m", 0.01)
            ),
            orientation_tolerance_rad=float(
                usr_args.get("mink_orientation_tolerance_rad", 0.10)
            ),
        ),
        relax_orientation_on_failure=bool(
            usr_args.get("relax_orientation_on_failure", False)
        ),
        canonical_seed_on_failure=bool(
            usr_args.get("canonical_seed_on_failure", True)
        ),
        save_grasp_visualizations=bool(
            usr_args.get("save_grasp_visualizations", True)
        ),
        visualization_dir=(
            usr_args.get("grasp_visualization_dir")
            or usr_args.get("eval_save_dir")
        ),
        max_visualized_grasps=(
            None
            if usr_args.get("max_visualized_grasps") in (None, "all")
            else int(usr_args["max_visualized_grasps"])
        ),
        max_visualized_points=int(
            usr_args.get("max_visualized_points", 30_000)
        ),
        grasp_visualization_scale=int(
            usr_args.get("grasp_visualization_scale", 4)
        ),
        bimanual_max_plans_per_arm=int(
            usr_args.get("bimanual_max_plans_per_arm", 4)
        ),
        bimanual_lift_m=float(usr_args.get("bimanual_lift_m", 0.10)),
        bimanual_collision_step_rad=float(
            usr_args.get("bimanual_collision_step_rad", 0.03)
        ),
        bimanual_max_jaw_axis_alignment=float(
            usr_args.get("bimanual_max_jaw_axis_alignment", 0.75)
        ),
        bimanual_max_target_width_m=float(
            usr_args.get("bimanual_max_target_width_m", 0.10)
        ),
        support_collision_filter_enabled=bool(
            usr_args.get("support_collision_filter_enabled", False)
        ),
    )

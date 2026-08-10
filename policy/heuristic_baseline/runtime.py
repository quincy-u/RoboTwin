"""Adapters connecting simple-grasp orchestration to RoboTwin."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import transforms3d as t3d

from simple_grasp.m2t2_adapter import M2T2GraspGenerator
from simple_grasp.mink_ik import MinkArmConfig, MinkIKConfig, MinkIKSolver
from simple_grasp.policy import PolicyConfig, SimpleGraspPolicy
from simple_grasp.types import GraspCandidate, ObjectState, SceneObservation

from .errors import NoFeasiblePlanFailure, TargetSelectionFailure
from .m2t2_backend import RoboTwinM2T2Backend


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
PARALLEL_JAW_ROLL_SYMMETRY = np.diag([1.0, -1.0, -1.0])
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


class RoboTwinMinkIK:
    """Mink pose IK with joint-space interpolation for RoboTwin qpos control."""

    def __init__(
        self,
        task_env: Any,
        grasp_to_robotwin: np.ndarray,
        *,
        model_path: str | Path,
        max_joint_step_rad: float = 0.12,
        max_waypoints_per_segment: int = 8,
        relax_orientation_on_failure: bool = True,
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
        self._candidate_paths: list[np.ndarray] = []
        self._completed_paths: list[np.ndarray] = []
        self.solver = MinkIKSolver.from_xml_path(
            model_path,
            arms,
            self._joint_positions,
            world_from_model=self._world_from_model,
            config=ik_config or MinkIKConfig(),
        )
        self.reset_stats()

    def _joint_positions(self, arm: str) -> np.ndarray:
        if arm in self._candidate_seed:
            return self._candidate_seed[arm].copy()
        state = getattr(self.task_env.robot, f"get_{arm}_arm_jointState")()
        return np.asarray(state[:-1], dtype=np.float64)

    def _world_from_model(self, arm: str) -> np.ndarray:
        pose = getattr(self.task_env.robot, f"{arm}_entity_origion_pose")
        return np.asarray(pose.to_transformation_matrix(), dtype=np.float64)

    def reset_stats(self) -> None:
        self.calls = 0
        self.planner_attempts = 0
        self.successes = 0
        self.relaxed_successes = 0
        self.over_limit_successes = 0
        self.failures: dict[str, int] = {}
        self.first_target: np.ndarray | None = None
        self._stage = 0
        self._candidate_seed = {}
        self._candidate_paths = []
        self._completed_paths = []
        self._candidate_command_targets: list[np.ndarray] = []
        self._completed_command_targets: list[np.ndarray] = []

    @property
    def selected_grasp_command_pose(self) -> np.ndarray | None:
        """World command pose Mink accepted for the completed grasp stage."""
        if len(self._completed_command_targets) != 3:
            return None
        return self._completed_command_targets[1].copy()

    def _path(self, start: np.ndarray, target: np.ndarray) -> np.ndarray:
        delta = float(np.max(np.abs(target - start)))
        steps = max(2, int(np.ceil(delta / self.max_joint_step_rad)))
        steps = min(steps, self.max_waypoints_per_segment)
        return np.linspace(start, target, steps + 1, dtype=np.float64)[1:]

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
            self._candidate_command_targets = []
            self._completed_command_targets = []
        if arm not in self._candidate_seed:
            return None

        target = np.asarray(world_grasp_pose, dtype=np.float64) @ self.grasp_to_robotwin
        if self.first_target is None:
            self.first_target = target.copy()
        frame_rotation = (
            np.asarray(getattr(self.task_env.robot, f"{arm}_global_trans_matrix"))
            @ np.asarray(getattr(self.task_env.robot, f"{arm}_delta_matrix"))
        )
        mink_target = target.copy()
        mink_target[:3, :3] = target[:3, :3] @ np.linalg.inv(frame_rotation)
        self.planner_attempts += 1
        joints = self.solver.solve(arm, mink_target)
        accepted_command_target = target
        if joints is None and self.relax_orientation_on_failure:
            canonical_quat = CANONICAL_COMMAND_QUATERNIONS[arm]
            canonical = target.copy()
            canonical[:3, :3] = (
                t3d.quaternions.quat2mat(canonical_quat)
                @ np.linalg.inv(frame_rotation)
            )
            self.planner_attempts += 1
            joints = self.solver.solve(arm, canonical)
            if joints is not None:
                accepted_command_target = target.copy()
                accepted_command_target[:3, :3] = t3d.quaternions.quat2mat(
                    canonical_quat
                )
                self.relaxed_successes += 1
        if joints is None:
            self.failures["NoSolution"] = self.failures.get("NoSolution", 0) + 1
            self._candidate_seed = {}
            self._candidate_paths = []
            self._candidate_command_targets = []
            return None

        joints = np.asarray(joints, dtype=np.float64)
        start = self._candidate_seed[arm]
        path = self._path(start, joints)
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
        max_waypoints_per_segment: int = 8,
    ) -> None:
        if max_waypoints_per_segment < 2:
            raise ValueError("max_waypoints_per_segment must be at least 2")
        self.task_env = task_env
        self.planner_ik = planner_ik
        self.max_waypoints_per_segment = int(max_waypoints_per_segment)
        self.actions: list[np.ndarray] = []
        self.left = np.empty(0)
        self.right = np.empty(0)
        self.left_gripper = 1.0
        self.right_gripper = 1.0

    def reset(self) -> None:
        left = np.asarray(self.task_env.robot.get_left_arm_jointState(), dtype=np.float64)
        right = np.asarray(self.task_env.robot.get_right_arm_jointState(), dtype=np.float64)
        self.left, self.left_gripper = left[:-1].copy(), float(left[-1])
        self.right, self.right_gripper = right[:-1].copy(), float(right[-1])
        self.actions = []

    def _append(self) -> None:
        self.actions.append(
            np.concatenate(
                (self.left, [self.left_gripper], self.right, [self.right_gripper])
            )
        )

    def _subsample(self, path: np.ndarray) -> np.ndarray:
        if len(path) <= self.max_waypoints_per_segment:
            return path
        indices = np.linspace(
            0, len(path) - 1, self.max_waypoints_per_segment, dtype=int
        )
        return path[indices]

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

        for waypoint in self._subsample(path):
            if arm == "left":
                self.left = waypoint[: len(self.left)].copy()
            else:
                self.right = waypoint[: len(self.right)].copy()
            self._append()

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
        self._append()


class ReachabilityRankedGrasps:
    """Rank target-contact grasps by canonical command orientation first."""

    def __init__(
        self,
        grasps: M2T2GraspGenerator,
        arm: str,
        grasp_to_robotwin: np.ndarray,
        min_confidence: float = 0.0,
    ) -> None:
        if arm not in CANONICAL_COMMAND_QUATERNIONS:
            raise ValueError(f"unknown arm {arm!r}")
        transform = np.asarray(grasp_to_robotwin, dtype=np.float64)
        if transform.shape != (4, 4):
            raise ValueError("grasp_to_robotwin must have shape (4, 4)")
        self.grasps = grasps
        self.arm = arm
        self.command_axis_map = transform[:3, :3].copy()
        self.canonical_command_rotation = t3d.quaternions.quat2mat(
            CANONICAL_COMMAND_QUATERNIONS[arm]
        )
        self.min_confidence = float(min_confidence)
        self.last_candidates: list[GraspCandidate] = []

    @staticmethod
    def _rotation_distance(first: np.ndarray, second: np.ndarray) -> float:
        relative = np.asarray(first).T @ np.asarray(second)
        cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
        return float(np.arccos(cosine))

    def _orientation_error(self, pose: np.ndarray) -> float:
        command_rotation = pose[:3, :3] @ self.command_axis_map
        rolled_rotation = command_rotation @ PARALLEL_JAW_ROLL_SYMMETRY
        return min(
            self._rotation_distance(
                self.canonical_command_rotation, command_rotation
            ),
            self._rotation_distance(
                self.canonical_command_rotation, rolled_rotation
            ),
        )

    def propose(
        self, observation: SceneObservation, target: ObjectState
    ) -> list[GraspCandidate]:
        self.last_candidates = []
        candidates = [
            candidate
            for candidate in self.grasps.propose(observation, target)
            if candidate.confidence >= self.min_confidence
        ]

        ranked = []
        for candidate in candidates:
            pose = np.array(candidate.world_grasp_pose, dtype=np.float64, copy=True)
            approximate_contact_center = pose[:3, 3] + 0.1034 * pose[:3, 2]
            target_center = target.world_pose[:3, 3]
            if np.linalg.norm(approximate_contact_center - target_center) > 0.08:
                pose[:3, 3] = target_center - 0.1034 * pose[:3, 2]

            orientation_error = self._orientation_error(pose)
            # Nearest canonical top-down orientation is the primary key. M2T2
            # confidence breaks ties; a 180-degree roll about the approach axis
            # is equivalent for the parallel-jaw gripper.
            ranking_score = (
                1.0
                + 10.0 * (np.pi - orientation_error)
                + 0.01 * candidate.confidence
            )
            ranked.append(
                GraspCandidate(pose, ranking_score, candidate.object_name)
            )
        self.last_candidates = sorted(
            ranked, key=lambda item: item.confidence, reverse=True
        )
        return list(self.last_candidates)


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
        max_waypoints_per_segment: int = 8,
        max_joint_step_rad: float = 0.12,
        mink_model_path: str | Path | None = None,
        mink_config: MinkIKConfig | None = None,
        relax_orientation_on_failure: bool = True,
        save_grasp_visualizations: bool = True,
        visualization_dir: str | Path | None = None,
        max_visualized_grasps: int | None = None,
        max_visualized_points: int = 30_000,
        grasp_visualization_scale: int = 4,
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
        if (
            self.max_visualized_grasps is not None
            and self.max_visualized_grasps <= 0
        ) or (
            self.max_visualized_points <= 0
            or self.grasp_visualization_scale <= 0
        ):
            raise ValueError("visualization limits must be positive")
        self._visualization_index = 0
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
            ik_config=mink_config,
        )
        self.controller = QposActionBuffer(
            task_env,
            self.ik,
            max_waypoints_per_segment=max_waypoints_per_segment,
        )
        self.backend = grasps.backend

    def _target_name(self, scene: SceneObservation) -> str:
        if not self.automatic_target:
            return self.config.object_name
        names = [name for name in scene.objects if name != "wall"]
        if len(names) != 1:
            available = ", ".join(sorted(names))
            raise TargetSelectionFailure(
                "object_name=auto requires exactly one non-wall tracked object; "
                f"available: {available}"
            )
        return names[0]

    def _save_grasp_visualization(
        self,
        scene: SceneObservation,
        target: ObjectState,
        candidates: list[GraspCandidate],
        selected: GraspCandidate | None,
        arm: str,
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
            raw_trace = getattr(self.backend, "last_trace", {})
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
                executed_command_pose=self.ik.selected_grasp_command_pose,
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

    def get_action(self, *, scene: SceneObservation) -> list[np.ndarray]:
        self.simulator.update(scene)
        target_name = self._target_name(scene)
        target = self.simulator.object_state(target_name)
        arm = self.config.arm
        if self.automatic_arm:
            arm = "left" if target.world_pose[0, 3] < 0.0 else "right"

        self.controller.reset()
        self.ik.reset_stats()
        ranker = ReachabilityRankedGrasps(
            self.grasps,
            arm,
            self.ik.grasp_to_robotwin,
            min_confidence=self.config.min_confidence,
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
                f"failures={self.ik.failures}; "
                f"target_position={np.array2string(target.world_pose[:3, 3], precision=3)}; "
                f"first_target={np.array2string(self.ik.first_target, precision=3)}"
            ) from exc
        self._save_grasp_visualization(
            scene, target, ranker.last_candidates, selected, arm
        )
        return self.controller.actions

    def reset(self) -> None:
        self.simulator.scene = None
        self.controller.reset()
        self.ik.reset_stats()
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
        num_runs=int(usr_args.get("num_runs", 1)),
        mask_threshold=float(usr_args.get("mask_threshold", 0.4)),
        object_threshold=float(usr_args.get("object_threshold", 0.4)),
        max_predictions=usr_args.get("max_predictions", 512),
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
            usr_args.get("max_waypoints_per_segment", 8)
        ),
        max_joint_step_rad=float(usr_args.get("max_joint_step_rad", 0.12)),
        mink_model_path=usr_args.get("mink_model_path"),
        mink_config=MinkIKConfig(
            solver=str(usr_args.get("mink_solver", "daqp")),
            dt=float(usr_args.get("mink_dt", 0.05)),
            max_iterations=int(usr_args.get("mink_max_iterations", 100)),
            position_tolerance_m=float(
                usr_args.get("mink_position_tolerance_m", 1e-3)
            ),
            orientation_tolerance_rad=float(
                usr_args.get("mink_orientation_tolerance_rad", 1e-2)
            ),
        ),
        relax_orientation_on_failure=bool(
            usr_args.get("relax_orientation_on_failure", True)
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
    )

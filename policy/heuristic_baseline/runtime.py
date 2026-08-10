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


def save_grasp_visualization(
    output_path: str | Path,
    scene: SceneObservation,
    target: ObjectState,
    candidates: list[GraspCandidate],
    selected: GraspCandidate | None,
    *,
    arm: str,
    grasp_to_robotwin: np.ndarray,
    rejected_candidates: list[GraspCandidate] | None = None,
    executed_command_pose: np.ndarray | None = None,
    raw_trace: dict[str, np.ndarray] | None = None,
    max_grasps: int = 48,
    max_points: int = 4_000,
) -> Path:
    """Save raw/target M2T2 options, source selection, and accepted IK pose."""
    if max_grasps <= 0 or max_points <= 0:
        raise ValueError("visualization limits must be positive")
    grasp_to_robotwin = np.asarray(grasp_to_robotwin, dtype=np.float64)
    if grasp_to_robotwin.shape != (4, 4):
        raise ValueError("grasp_to_robotwin must have shape (4, 4)")
    if executed_command_pose is not None:
        executed_command_pose = np.asarray(executed_command_pose, dtype=np.float64)
        if executed_command_pose.shape != (4, 4):
            raise ValueError("executed_command_pose must have shape (4, 4)")
    rejected_candidates = list(rejected_candidates or [])
    raw_trace = raw_trace or {}

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    output_path = Path(output_path).expanduser()
    target_mask = scene.instance_labels == target.instance_id
    target_points = np.asarray(scene.xyz[target_mask], dtype=np.float64)
    target_colors = np.clip(np.asarray(scene.rgb[target_mask]), 0.0, 1.0)
    context_points = np.asarray(scene.xyz[~target_mask], dtype=np.float64)
    target_visible = len(target_points) > 0
    center = np.asarray(target.world_pose[:3, 3], dtype=np.float64)

    def subsample(points: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray]:
        if len(points) <= limit:
            return points, np.arange(len(points))
        indices = np.linspace(0, len(points) - 1, limit, dtype=int)
        return points[indices], indices

    target_points, target_indices = subsample(target_points, max_points)
    target_colors = target_colors[target_indices]
    context_points, _ = subsample(context_points, max_points)
    displayed = list(candidates[:max_grasps])
    displayed_rejected = sorted(
        rejected_candidates,
        key=lambda candidate: candidate.confidence,
        reverse=True,
    )[:max_grasps]
    figure = Figure(figsize=(12, 6.5))
    FigureCanvasAgg(figure)
    world_axes = figure.add_subplot(1, 2, 1, projection="3d")
    top_axes = figure.add_subplot(1, 2, 2)
    figure.subplots_adjust(
        left=0.06, right=0.97, bottom=0.17, top=0.87, wspace=0.23
    )
    if len(context_points):
        world_axes.scatter(
            context_points[:, 0],
            context_points[:, 1],
            context_points[:, 2],
            c="#94a3b8",
            s=1,
            alpha=0.08,
            depthshade=False,
        )
        top_axes.scatter(
            context_points[:, 0],
            context_points[:, 1],
            c="#94a3b8",
            s=1,
            alpha=0.08,
        )
    if target_visible:
        world_axes.scatter(
            target_points[:, 0],
            target_points[:, 1],
            target_points[:, 2],
            c=target_colors,
            s=4,
            alpha=0.8,
            depthshade=False,
        )
        top_axes.scatter(
            target_points[:, 0],
            target_points[:, 1],
            c=target_colors,
            s=4,
            alpha=0.8,
        )
        geometry_points = [target_points]
    else:
        world_axes.scatter(*center, marker="x", c="#ef4444", s=80)
        top_axes.scatter(center[0], center[1], marker="x", c="#ef4444", s=80)
        geometry_points = [center[None, :]]

    trace_contacts = np.asarray(
        raw_trace.get("contacts", np.empty((0, 3))), dtype=np.float64
    ).reshape(-1, 3)
    trace_target_contacts = np.asarray(
        raw_trace.get("target_contacts", np.empty(0)), dtype=bool
    )
    if len(trace_contacts) == len(trace_target_contacts):
        for contact_mask, color, marker, size in (
            (~trace_target_contacts, "#ef4444", "x", 22),
            (trace_target_contacts, "#38bdf8", ".", 14),
        ):
            contact_points = trace_contacts[contact_mask]
            if len(contact_points):
                geometry_points.append(contact_points)
                world_axes.scatter(
                    contact_points[:, 0],
                    contact_points[:, 1],
                    contact_points[:, 2],
                    c=color,
                    marker=marker,
                    s=size,
                    alpha=0.9,
                    depthshade=False,
                )
                top_axes.scatter(
                    contact_points[:, 0],
                    contact_points[:, 1],
                    c=color,
                    marker=marker,
                    s=size,
                    alpha=0.9,
                )

    def draw_grasp(
        candidate: GraspCandidate, color: str, alpha: float, width: float
    ) -> None:
        pose = (
            np.asarray(candidate.world_grasp_pose, dtype=np.float64)
            @ grasp_to_robotwin
        )
        wrist = pose[:3, 3]
        approach = pose[:3, 0]
        closing = pose[:3, 1]
        pregrasp = wrist - 0.05 * approach
        contact_center = wrist + 0.12 * approach
        jaw_a = contact_center - 0.035 * closing
        jaw_b = contact_center + 0.035 * closing
        geometry_points.append(
            np.stack((pregrasp, wrist, contact_center, jaw_a, jaw_b))
        )
        for axes in (world_axes,):
            axes.plot(
                *np.stack((pregrasp, contact_center)).T,
                color=color,
                alpha=alpha,
                linewidth=width,
            )
            axes.plot(
                *np.stack((jaw_a, jaw_b)).T,
                color=color,
                alpha=alpha,
                linewidth=width,
            )
        top_axes.plot(
            [pregrasp[0], contact_center[0]],
            [pregrasp[1], contact_center[1]],
            color=color,
            alpha=alpha,
            linewidth=width,
        )
        top_axes.plot(
            [jaw_a[0], jaw_b[0]],
            [jaw_a[1], jaw_b[1]],
            color=color,
            alpha=alpha,
            linewidth=width,
        )

    rejected_strength = float(
        np.clip(6.0 / max(len(displayed_rejected), 1), 0.18, 0.65)
    )
    for candidate in reversed(displayed_rejected):
        draw_grasp(
            candidate,
            "#ef4444",
            rejected_strength,
            0.6 + rejected_strength,
        )
    rank_scale = max(len(displayed) - 1, 1)
    for rank_index in reversed(range(len(displayed))):
        importance = 1.0 - rank_index / rank_scale
        draw_grasp(
            displayed[rank_index],
            "#38bdf8",
            0.12 + 0.50 * importance,
            0.6 + importance,
        )
    if selected is not None:
        draw_grasp(selected, "#f97316", 1.0, 3.0)
    if executed_command_pose is not None:
        executed_raw_pose = executed_command_pose @ np.linalg.inv(grasp_to_robotwin)
        draw_grasp(
            GraspCandidate(executed_raw_pose, 0.0, target.name),
            "#22c55e",
            1.0,
            4.0,
        )

    frame_pose = executed_command_pose
    if frame_pose is None and selected is not None:
        frame_pose = (
            np.asarray(selected.world_grasp_pose, dtype=np.float64)
            @ grasp_to_robotwin
        )
    if frame_pose is not None:
        wrist = frame_pose[:3, 3]
        for axis_index, axis_color in enumerate(("#ef4444", "#22c55e", "#2563eb")):
            endpoint = wrist + 0.04 * frame_pose[:3, axis_index]
            world_axes.plot(
                *np.stack((wrist, endpoint)).T,
                color=axis_color,
                linewidth=2.5,
            )
            top_axes.plot(
                [wrist[0], endpoint[0]],
                [wrist[1], endpoint[1]],
                color=axis_color,
                linewidth=2.5,
            )

    all_geometry = np.concatenate(geometry_points, axis=0)
    radial_distance = np.linalg.norm(all_geometry - center, axis=1)
    radius = float(np.clip(np.quantile(radial_distance, 0.98) + 0.03, 0.12, 0.35))
    world_axes.set_xlim(center[0] - radius, center[0] + radius)
    world_axes.set_ylim(center[1] - radius, center[1] + radius)
    world_axes.set_zlim(center[2] - radius, center[2] + radius)
    world_axes.set_box_aspect((1, 1, 1))
    world_axes.view_init(elev=24, azim=-58)
    top_axes.set_xlim(center[0] - radius, center[0] + radius)
    top_axes.set_ylim(center[1] - radius, center[1] + radius)
    top_axes.set_aspect("equal", adjustable="box")
    world_axes.set_xlabel("world x [m]")
    world_axes.set_ylabel("world y [m]")
    world_axes.set_zlabel("world z [m]")
    top_axes.set_xlabel("world x [m]")
    top_axes.set_ylabel("world y [m]")
    world_axes.set_title("3D command-frame grasp glyphs")
    top_axes.set_title("Top view")
    top_axes.grid(alpha=0.2)
    if executed_command_pose is not None:
        selection_text = "Mink pose accepted"
    elif selected is not None:
        selection_text = "M2T2 source selected"
    else:
        selection_text = "no feasible IK"
    target_visibility_text = "" if target_visible else " | NO TARGET DEPTH POINTS"
    figure.suptitle(
        f"{target.name} | {arm} | target {len(displayed)}/{len(candidates)} | "
        f"off-target {len(displayed_rejected)}/{len(rejected_candidates)} | "
        f"{selection_text}{target_visibility_text}",
        y=0.96,
    )
    legend_handles = []
    if candidates:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="#38bdf8",
                lw=2,
                label="ranked target options (opacity = rank)",
            )
        )
    if rejected_candidates:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="#ef4444",
                lw=1,
                label="raw options rejected by target contact",
            )
        )
    if selected is not None:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="#f97316",
                lw=3,
                label="selected M2T2 source candidate",
            )
        )
    if executed_command_pose is not None:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="#22c55e",
                lw=4,
                label="Mink-accepted command pose",
            )
        )
    if legend_handles:
        figure.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
            ncol=min(2, len(legend_handles)),
        )

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

    ranked_poses, ranked_scores = candidate_arrays(candidates)
    rejected_poses, rejected_scores = candidate_arrays(rejected_candidates)
    selected_pose = (
        np.empty((0, 4, 4), dtype=np.float64)
        if selected is None
        else np.asarray(selected.world_grasp_pose, dtype=np.float64)[None, :, :]
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170)
    np.savez_compressed(
        output_path.with_suffix(".npz"),
        target_name=np.asarray(target.name),
        arm=np.asarray(arm),
        target_world_pose=target.world_pose,
        target_points=np.asarray(scene.xyz[target_mask], dtype=np.float32),
        target_rgb=np.asarray(scene.rgb[target_mask], dtype=np.float32),
        raw_world_grasp_poses=raw_poses,
        raw_confidences=raw_scores,
        raw_contacts=raw_contacts,
        raw_target_contacts=raw_target_contacts,
        raw_query_ids=raw_query_ids,
        ranked_world_grasp_poses=ranked_poses,
        ranking_scores=ranked_scores,
        rejected_world_grasp_poses=rejected_poses,
        rejected_confidences=rejected_scores,
        selected_world_grasp_pose=selected_pose,
        selected_source_command_pose=selected_source_command,
        mink_accepted_command_pose=accepted_command,
        grasp_to_robotwin=grasp_to_robotwin,
    )
    figure.clear()
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
        max_visualized_grasps: int = 48,
        max_visualized_points: int = 4_000,
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
        self.max_visualized_grasps = int(max_visualized_grasps)
        self.max_visualized_points = int(max_visualized_points)
        if self.max_visualized_grasps <= 0 or self.max_visualized_points <= 0:
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
            rejected_candidates = [
                GraspCandidate(pose, float(score), target.name)
                for pose, score, is_target_contact in zip(
                    raw_poses, raw_scores, raw_target_contacts
                )
                if not is_target_contact
            ]
            saved_path = save_grasp_visualization(
                output_path,
                scene,
                target,
                candidates,
                selected,
                arm=arm,
                grasp_to_robotwin=self.ik.grasp_to_robotwin,
                rejected_candidates=rejected_candidates,
                executed_command_pose=self.ik.selected_grasp_command_pose,
                raw_trace=raw_trace,
                max_grasps=self.max_visualized_grasps,
                max_points=self.max_visualized_points,
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
        max_visualized_grasps=int(
            usr_args.get("max_visualized_grasps", 48)
        ),
        max_visualized_points=int(
            usr_args.get("max_visualized_points", 4_000)
        ),
    )

"""Adapters connecting simple-grasp orchestration to RoboTwin."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import transforms3d as t3d

from simple_grasp.m2t2_adapter import M2T2GraspGenerator
from simple_grasp.policy import PolicyConfig, SimpleGraspPolicy
from simple_grasp.types import GraspCandidate, ObjectState, SceneObservation

from .errors import NoFeasiblePlanFailure, TargetSelectionFailure
from .m2t2_backend import RoboTwinM2T2Backend


M2T2_TO_ROBOTWIN = np.array(
    [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, -0.0166],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


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


class RoboTwinPlannerIK:
    """Use RoboTwin's collision-aware planner as the pose feasibility check."""

    def __init__(self, task_env: Any, grasp_to_robotwin: np.ndarray) -> None:
        self.task_env = task_env
        transform = np.asarray(grasp_to_robotwin, dtype=np.float64)
        if transform.shape != (4, 4):
            raise ValueError("grasp_to_robotwin must have shape (4, 4)")
        self.grasp_to_robotwin = transform
        self.calls = 0
        self.successes = 0
        self.failures: dict[str, int] = {}
        self.first_target: np.ndarray | None = None
        self._stage = 0
        self._candidate_seed: np.ndarray | None = None
        self._candidate_paths: list[np.ndarray] = []
        self._completed_paths: list[np.ndarray] = []

    def reset_stats(self) -> None:
        self.calls = 0
        self.successes = 0
        self.failures = {}
        self.first_target = None
        self._stage = 0
        self._candidate_seed = None
        self._candidate_paths = []
        self._completed_paths = []

    def _initial_seed(self, arm: str) -> np.ndarray:
        entity = getattr(self.task_env.robot, f"{arm}_entity")
        return np.asarray(entity.get_qpos(), dtype=np.float64).copy()

    def _arm_joint_indices(self, arm: str, arm_dim: int) -> np.ndarray:
        """Map planner arm columns into the full articulation qpos."""
        robot = self.task_env.robot
        entity = getattr(robot, f"{arm}_entity")
        arm_names = list(getattr(robot, f"{arm}_arm_joints_name", []))
        all_names: list[str] = []
        if arm_names and hasattr(entity, "get_active_joints"):
            all_names = [joint.get_name() for joint in entity.get_active_joints()]
        if not arm_names or not all_names:
            planner = getattr(robot, f"{arm}_planner", None)
            if planner is not None:
                arm_names = list(getattr(planner, "active_joints_name", []))
                all_names = list(getattr(planner, "all_joints", []))
        if not arm_names or not all_names:
            if len(entity.get_qpos()) == arm_dim:
                return np.arange(arm_dim)
            raise ValueError(f"cannot map {arm} planner joints into full qpos")
        try:
            indices = np.asarray([all_names.index(name) for name in arm_names])
        except ValueError as exc:
            raise ValueError(f"{arm} planner joint names do not match the robot") from exc
        if len(indices) != arm_dim:
            raise ValueError(
                f"{arm} planner returned {arm_dim} joints but maps {len(indices)}"
            )
        return indices

    def consume_path(self, arm: str, target: np.ndarray) -> np.ndarray | None:
        """Return a path only after all three IK stages have succeeded."""
        if not self._completed_paths:
            return None
        path = self._completed_paths.pop(0)
        arm_dim = len(getattr(self.task_env.robot, f"get_{arm}_arm_jointState")()) - 1
        if not np.allclose(path[-1, :arm_dim], target):
            raise ValueError("planner path endpoint does not match the policy target")
        return path

    def solve(self, arm: str, world_grasp_pose: np.ndarray) -> np.ndarray | None:
        self.calls += 1
        stage = self._stage
        self._stage = (self._stage + 1) % 3
        if stage == 0:
            self._candidate_seed = self._initial_seed(arm)
            self._candidate_paths = []
            self._completed_paths = []
        if self._candidate_seed is None:
            return None

        target = np.asarray(world_grasp_pose, dtype=np.float64) @ self.grasp_to_robotwin
        if self.first_target is None:
            self.first_target = target.copy()
        quat = t3d.quaternions.mat2quat(target[:3, :3])
        pose = np.concatenate((target[:3, 3], quat))
        planner = getattr(self.task_env.robot, f"{arm}_plan_path")
        result = planner(pose, last_qpos=self._candidate_seed)
        if result is None or result.get("status") != "Success":
            status = "None" if result is None else str(result.get("status"))
            self.failures[status] = self.failures.get(status, 0) + 1
            self._candidate_seed = None
            self._candidate_paths = []
            return None
        self.successes += 1

        arm_state = getattr(self.task_env.robot, f"get_{arm}_arm_jointState")()
        arm_dim = len(arm_state) - 1
        path = np.asarray(result["position"], dtype=np.float64)
        if path.ndim != 2 or path.shape[0] == 0 or path.shape[1] != arm_dim:
            raise ValueError("planner returned an invalid position path")
        if not np.all(np.isfinite(path)):
            raise ValueError("planner returned non-finite joint positions")
        self._candidate_paths.append(path.copy())
        joint_indices = self._arm_joint_indices(arm, arm_dim)
        next_seed = self._candidate_seed.copy()
        next_seed[joint_indices] = path[-1]
        self._candidate_seed = next_seed
        if stage == 2:
            self._completed_paths = [path.copy() for path in self._candidate_paths]
            self._candidate_paths = []
            self._candidate_seed = None
        return path[-1, :arm_dim].copy()


class QposActionBuffer:
    """Record controller calls as bounded full RoboTwin qpos waypoints."""

    def __init__(
        self,
        task_env: Any,
        planner_ik: RoboTwinPlannerIK,
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
    """Filter raw M2T2 confidence before applying a geometric ranking score."""

    def __init__(
        self,
        grasps: M2T2GraspGenerator,
        task_env: Any,
        arm: str,
        min_confidence: float = 0.0,
    ) -> None:
        self.grasps = grasps
        self.task_env = task_env
        self.arm = arm
        self.min_confidence = float(min_confidence)

    def propose(
        self, observation: SceneObservation, target: ObjectState
    ) -> list[GraspCandidate]:
        candidates = [
            candidate
            for candidate in self.grasps.propose(observation, target)
            if candidate.confidence >= self.min_confidence
        ]
        origin = getattr(self.task_env.robot, f"{self.arm}_entity_origion_pose").p
        toward_object = target.world_pose[:3, 3] - np.asarray(origin)
        toward_object[2] = 0.0
        norm = np.linalg.norm(toward_object)
        if norm < 1e-8:
            return candidates
        toward_object /= norm

        ranked = []
        for candidate in candidates:
            pose = candidate.world_grasp_pose
            approach_alignment = float(pose[:3, 2] @ toward_object)
            robot_side = float(
                (target.world_pose[:3, 3] - pose[:3, 3]) @ toward_object
            )
            geometric_score = (
                approach_alignment + 1.0
                + np.clip(robot_side / 0.10, -1.0, 1.0)
                + 1.0
            )
            ranked.append(
                GraspCandidate(
                    pose,
                    candidate.confidence + geometric_score,
                    candidate.object_name,
                )
            )
        return ranked


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
    ) -> None:
        self.task_env = task_env
        self.grasps = grasps
        self.config = config
        self.automatic_target = automatic_target
        self.automatic_arm = automatic_arm
        self.simulator = SceneSnapshot()
        self.ik = RoboTwinPlannerIK(task_env, grasp_to_robotwin)
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

    def get_action(self, *, scene: SceneObservation) -> list[np.ndarray]:
        self.simulator.update(scene)
        target_name = self._target_name(scene)
        target = self.simulator.object_state(target_name)
        arm = self.config.arm
        if self.automatic_arm:
            arm = "left" if target.world_pose[0, 3] < 0.0 else "right"

        self.controller.reset()
        self.ik.reset_stats()
        policy = SimpleGraspPolicy(
            self.simulator,
            ReachabilityRankedGrasps(
                self.grasps,
                self.task_env,
                arm,
                min_confidence=self.config.min_confidence,
            ),
            self.ik,
            self.controller,
            replace(self.config, object_name=target_name, arm=arm),
        )
        try:
            policy.run_once()
        except RuntimeError as exc:
            if str(exc) != "M2T2 returned no grasp with complete feasible IK solutions":
                raise
            raise NoFeasiblePlanFailure(
                f"{exc}; RoboTwin planner accepted {self.ik.successes}/"
                f"{self.ik.calls} pose checks; failures={self.ik.failures}; "
                f"first_target={np.array2string(self.ik.first_target, precision=3)}"
            ) from exc
        return self.controller.actions

    def reset(self) -> None:
        self.simulator.scene = None
        self.controller.reset()
        self.ik.reset_stats()
        self.backend.reset(int(getattr(self.task_env, "episode_seed", 0)))


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
        min_query_iou=float(usr_args.get("min_query_iou", 0.01)),
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
    )

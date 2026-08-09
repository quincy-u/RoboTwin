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
        if joints is None and self.relax_orientation_on_failure:
            canonical_quat = np.asarray(
                [-0.353523, 0.61239, -0.353524, -0.61239]
                if arm == "right"
                else [-0.61239, 0.353523, -0.61239, -0.353524],
                dtype=np.float64,
            )
            canonical = target.copy()
            canonical[:3, :3] = (
                t3d.quaternions.quat2mat(canonical_quat)
                @ np.linalg.inv(frame_rotation)
            )
            self.planner_attempts += 1
            joints = self.solver.solve(arm, canonical)
            if joints is not None:
                self.relaxed_successes += 1
        if joints is None:
            self.failures["NoSolution"] = self.failures.get("NoSolution", 0) + 1
            self._candidate_seed = {}
            self._candidate_paths = []
            return None

        joints = np.asarray(joints, dtype=np.float64)
        start = self._candidate_seed[arm]
        path = self._path(start, joints)
        self._candidate_paths.append(path)
        self._candidate_seed[arm] = joints.copy()
        self.successes += 1
        if stage == 2:
            self._completed_paths = [item.copy() for item in self._candidate_paths]
            self._candidate_paths = []
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
            pose = np.array(candidate.world_grasp_pose, dtype=np.float64, copy=True)
            approximate_contact_center = pose[:3, 3] + 0.1034 * pose[:3, 2]
            target_center = target.world_pose[:3, 3]
            if np.linalg.norm(approximate_contact_center - target_center) > 0.08:
                pose[:3, 3] = target_center - 0.1034 * pose[:3, 2]
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
        max_joint_step_rad: float = 0.12,
        mink_model_path: str | Path | None = None,
        mink_config: MinkIKConfig | None = None,
        relax_orientation_on_failure: bool = True,
    ) -> None:
        self.task_env = task_env
        self.grasps = grasps
        self.config = config
        self.automatic_target = automatic_target
        self.automatic_arm = automatic_arm
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
                f"{exc}; Mink IK accepted {self.ik.successes}/"
                f"{self.ik.calls} stages across {self.ik.planner_attempts} attempts "
                f"({self.ik.relaxed_successes} canonical retries); "
                f"failures={self.ik.failures}; "
                f"target_position={np.array2string(target.world_pose[:3, 3], precision=3)}; "
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
    )

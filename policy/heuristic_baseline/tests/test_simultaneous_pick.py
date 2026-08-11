from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from policy.heuristic_baseline.errors import (
    NoFeasiblePlanFailure,
    TargetSelectionFailure,
)
from policy.heuristic_baseline.runtime import (
    M2T2_TO_ROBOTWIN,
    RoboTwinHeuristicRuntime,
    StagedQposActionBuffer,
    _SimultaneousPickArmPlan,
    _grasp_command_tcp,
)
from policy.heuristic_baseline.task_plan import Pick, TaskPlan
from simple_grasp.types import GraspCandidate, ObjectState, SceneObservation


I = np.eye(4, dtype=np.float64)


class _Robot:
    def get_left_arm_jointState(self):
        return np.r_[np.zeros(6), 1.0]

    def get_right_arm_jointState(self):
        return np.r_[np.zeros(6), 1.0]

    def get_left_ee_pose(self):
        return np.array([-0.35, -0.20, 0.90, 1.0, 0.0, 0.0, 0.0])

    def get_right_ee_pose(self):
        return np.array([0.35, -0.20, 0.90, 1.0, 0.0, 0.0, 0.0])


class _Env:
    def __init__(self, plan: TaskPlan | None = None):
        self.robot = _Robot()
        self.heuristic_task_plan = plan


class _TrackingGrasps:
    def __init__(self, candidates):
        self.candidates = list(candidates)
        self.calls = []
        self.backend = SimpleNamespace(last_trace={})

    def propose(self, observation, target):
        self.calls.append((observation, target))
        return list(self.candidates)


class _ExactIK:
    grasp_to_robotwin = M2T2_TO_ROBOTWIN

    def __init__(self, collision_results=()):
        self.failures = {}
        self.solve_calls = []
        self.collision_calls = []
        self._collision_results = list(collision_results)

    def solve_command_target(self, arm, command, start):
        start = np.asarray(start, dtype=np.float64)
        sign = -1.0 if arm == "left" else 1.0
        goal = start + sign * 0.01
        path = np.vstack(((start + goal) / 2.0, goal))
        self.solve_calls.append((arm, np.asarray(command).copy()))
        return goal, path, np.asarray(command).copy()

    def full_robot_path_has_self_collision(
        self, actions, *, max_joint_step_rad
    ):
        self.collision_calls.append(
            ([row.copy() for row in actions], max_joint_step_rad)
        )
        if self._collision_results:
            return self._collision_results.pop(0)
        return False


def _candidate_at_tcp(
    target: ObjectState, local_tcp, confidence: float
) -> GraspCandidate:
    local_tcp = np.asarray(local_tcp, dtype=np.float64)
    world_tcp = (
        target.world_pose[:3, :3] @ local_tcp
        + target.world_pose[:3, 3]
    )
    command = I.copy()
    command[:3, 3] = world_tcp - 0.12 * command[:3, 0]
    grasp = command @ np.linalg.inv(M2T2_TO_ROBOTWIN)
    np.testing.assert_allclose(
        _grasp_command_tcp(grasp, M2T2_TO_ROBOTWIN), world_tcp,
        atol=1e-12,
    )
    return GraspCandidate(grasp, confidence, target.name)


def _scene_and_target():
    world_object = I.copy()
    # Put the shared object firmly on the geometric right. Explicit expert arm
    # tags must still retain the left-arm role.
    world_object[:3, 3] = [0.32, -0.16, 0.76]
    target = ObjectState("roller", world_object, 7)
    local = np.array([
        [x, y, z]
        for x in np.linspace(-0.14, 0.14, 15)
        for y in (-0.008, 0.008)
        for z in (-0.008, 0.008)
    ])
    points = local @ world_object[:3, :3].T + world_object[:3, 3]
    scene = SceneObservation(
        xyz=points,
        rgb=np.zeros((len(points), 3)),
        instance_labels=np.full(len(points), 7),
        camera_pose=I.copy(),
        objects={"roller": target},
    )
    return scene, target


def _picks():
    return (
        Pick(
            "roller",
            "left",
            pregrasp_offset_m=0.06,
            postgrasp_displacement=(0.01, 0.0, 0.11),
            grasp_offset_m=0.01,
            gripper_target=0.25,
            allowed_contact_points_local=((-0.11, 0.0, 0.0),),
            group_id=4,
        ),
        Pick(
            "roller",
            "right",
            pregrasp_offset_m=0.09,
            postgrasp_displacement=(0.01, 0.0, 0.11),
            grasp_offset_m=0.02,
            gripper_target=0.35,
            allowed_contact_points_local=((0.12, 0.0, 0.0),),
            group_id=4,
        ),
    )


def _arm_plan(
    arm: str,
    confidence: float,
    contact,
    *,
    gripper_target: float,
    path_lengths=(2, 2, 2),
):
    sign = -1.0 if arm == "left" else 1.0
    paths = tuple(
        np.linspace(
            np.full(6, sign * 0.01 * (index + 1)),
            np.full(6, sign * 0.02 * (index + 1)),
            count,
        )
        for index, count in enumerate(path_lengths)
    )
    pose = I.copy()
    pose[0, 3] = float(np.asarray(contact)[0])
    return _SimultaneousPickArmPlan(
        arm=arm,
        target_name="roller",
        arm_source="robotwin_ground_truth",
        candidate=GraspCandidate(pose, confidence, "roller"),
        paths=paths,
        command_targets=tuple(I.copy() for _ in range(3)),
        contact_local_point=tuple(float(value) for value in contact),
        gripper_target=gripper_target,
    )


class SimultaneousPickRuntimeTest(unittest.TestCase):
    def test_structural_detector_preserves_both_recorded_arms(self):
        left, right = _picks()
        env = _Env(TaskPlan("arbitrary_task", "pick", (right, left)))

        detected = RoboTwinHeuristicRuntime._simultaneous_same_object_pick_stages(
            env
        )

        self.assertEqual(detected, (left, right))
        duplicate_arm = TaskPlan(
            "arbitrary_task",
            "pick",
            (left, Pick("roller", "left", group_id=left.group_id)),
        )
        env.heuristic_task_plan = duplicate_arm
        with self.assertRaisesRegex(TargetSelectionFailure, "one left and one right"):
            RoboTwinHeuristicRuntime._simultaneous_same_object_pick_stages(env)

    def test_get_action_routes_shared_target_before_geometric_arm_selection(self):
        scene, target = _scene_and_target()
        left, right = _picks()
        env = _Env(
            TaskPlan("generic_shared_pick", "pick", (right, left))
        )

        class SimulatorSpy:
            def __init__(self):
                self.update_calls = 0

            def update(self, actual_scene):
                self.update_calls += 1
                self.last_scene = actual_scene

            def object_state(self, name):
                if name != "roller":
                    raise AssertionError(f"unexpected target {name!r}")
                return target

        runtime = RoboTwinHeuristicRuntime.__new__(
            RoboTwinHeuristicRuntime
        )
        runtime.task_env = env
        runtime._grasp_attempted = False
        runtime.automatic_target = True
        runtime.config = SimpleNamespace(object_name="auto")
        runtime.simulator = SimulatorSpy()
        runtime.ik = SimpleNamespace(
            clear_support_planes=lambda: None,
            reset_stats=lambda: None,
            failures={},
        )

        with patch.object(
            runtime, "_configure_support_plane", return_value=None
        ) as configure_support, patch.object(
            runtime,
            "_get_simultaneous_pick_action",
            return_value=[np.zeros(14)],
        ) as simultaneous, patch.object(
            runtime,
            "_select_arm",
            side_effect=AssertionError(
                "geometry fallback must not collapse dual-arm intent"
            ),
        ):
            actions = runtime.get_action(scene=scene)

        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].shape, (14,))
        self.assertEqual(simultaneous.call_count, 1)
        actual_scene, actual_target, actual_picks = (
            simultaneous.call_args.args
        )
        self.assertIs(actual_scene, scene)
        self.assertIs(actual_target, target)
        self.assertEqual(actual_picks, (left, right))
        configure_support.assert_called_once_with(
            scene, target, ("left", "right")
        )
        self.assertEqual(runtime.simulator.update_calls, 1)
        self.assertIs(runtime.simulator.last_scene, scene)
        self.assertTrue(runtime.grasp_attempted)

    def test_planner_uses_one_inference_and_keeps_gt_roles_and_contacts(self):
        scene, target = _scene_and_target()
        candidates = [
            _candidate_at_tcp(target, (-0.11, 0.0, 0.0), 0.90),
            _candidate_at_tcp(target, (0.12, 0.0, 0.0), 0.80),
            _candidate_at_tcp(target, (-0.10, 0.0, 0.0), 0.70),
            _candidate_at_tcp(target, (0.11, 0.0, 0.0), 0.60),
        ]
        grasps = _TrackingGrasps(candidates)
        runtime = RoboTwinHeuristicRuntime.__new__(RoboTwinHeuristicRuntime)
        runtime.task_env = _Env()
        runtime.grasps = grasps
        runtime.backend = grasps.backend
        runtime.ik = _ExactIK()
        runtime.config = SimpleNamespace(
            min_confidence=0.0,
            max_candidates=8,
            pregrasp_offset_m=0.07,
        )
        runtime.bimanual_lift_m = 0.10
        runtime.bimanual_max_plans_per_arm = 2
        runtime.bimanual_max_target_width_m = 0.10
        runtime.bimanual_max_jaw_axis_alignment = 0.75

        with patch(
            "policy.heuristic_baseline.runtime._target_m2t2_palm_depth",
            return_value=0.08,
        ):
            plans, ranked, _, failures, separation, region_source = (
                runtime._plan_simultaneous_pick_sides(
                    scene, target, _picks()
                )
            )

        self.assertEqual(len(grasps.calls), 1)
        self.assertEqual([item.confidence for item in ranked], [0.90, 0.80, 0.70, 0.60])
        self.assertEqual(
            [plan.candidate.confidence for plan in plans["left"]],
            [0.90, 0.70],
        )
        self.assertEqual(
            [plan.candidate.confidence for plan in plans["right"]],
            [0.80, 0.60],
        )
        self.assertTrue(
            all(
                plan.arm_source == "robotwin_ground_truth"
                for arm_plans in plans.values()
                for plan in arm_plans
            )
        )
        np.testing.assert_allclose(
            plans["left"][0].contact_local_point, (-0.11, 0.0, 0.0),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            plans["right"][0].contact_local_point, (0.12, 0.0, 0.0),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            plans["left"][0].command_targets[2][:3, 3]
            - plans["left"][0].command_targets[1][:3, 3],
            (0.01, 0.0, 0.11),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            plans["right"][0].command_targets[2][:3, 3]
            - plans["right"][0].command_targets[1][:3, 3],
            (0.01, 0.0, 0.11),
            atol=1e-12,
        )
        self.assertGreater(separation, 0.0)
        self.assertEqual(region_source, "recorded_regions")
        self.assertEqual(failures["pair_separation"], 0)

    def test_shared_object_rejects_inconsistent_postgrasp_motion(self):
        scene, target = _scene_and_target()
        left, right = _picks()
        runtime = RoboTwinHeuristicRuntime.__new__(RoboTwinHeuristicRuntime)
        runtime.task_env = _Env()
        runtime.config = SimpleNamespace(pregrasp_offset_m=0.07)
        runtime.bimanual_lift_m = 0.10

        with self.assertRaisesRegex(
            TargetSelectionFailure, "matching world postgrasp displacements"
        ):
            runtime._plan_simultaneous_pick_sides(
                scene,
                target,
                (
                    left,
                    replace(
                        right,
                        postgrasp_displacement=(0.0, 0.0, 0.12),
                    ),
                ),
            )

    def test_action_pair_synchronizes_dual_close_and_latches_through_lift(self):
        runtime = RoboTwinHeuristicRuntime.__new__(RoboTwinHeuristicRuntime)
        runtime.staged_controller = StagedQposActionBuffer(
            _Env(), max_waypoints_per_segment=16
        )
        runtime.gripper_settle_actions = 2
        left = _arm_plan(
            "left", 0.9, (-0.11, 0.0, 0.0),
            gripper_target=0.25, path_lengths=(3, 1, 4),
        )
        right = _arm_plan(
            "right", 0.8, (0.12, 0.0, 0.0),
            gripper_target=0.35, path_lengths=(1, 3, 2),
        )

        actions = runtime._build_simultaneous_pick_action_pair(left, right)
        metadata = runtime.staged_controller.metadata

        self.assertTrue(actions)
        self.assertTrue(all(action.shape == (14,) for action in actions))
        endpoints = [record for record in metadata if record["endpoint"]]
        self.assertEqual(
            [record["phase"] for record in endpoints],
            ["open", "pregrasp", "grasp", "close", "lift"],
        )
        self.assertTrue(all(record["arm"] == "both" for record in endpoints))
        close_start = next(
            index for index, record in enumerate(metadata)
            if record["phase"] == "close"
        )
        self.assertTrue(
            all(
                row[6] == 0.25 and row[13] == 0.35
                for row in actions[close_start:]
            )
        )
        self.assertEqual(actions[-1][6], 0.25)
        self.assertEqual(actions[-1][13], 0.35)
        self.assertFalse(
            any(record["phase"] == "open" for record in metadata[close_start:])
        )
        np.testing.assert_allclose(actions[-1][:6], left.paths[2][-1])
        np.testing.assert_allclose(actions[-1][7:13], right.paths[2][-1])

    def test_confidence_pairing_applies_separation_then_collision_fallback(self):
        left_high = _arm_plan(
            "left", 0.90, (-0.10, 0.0, 0.0), gripper_target=0.25
        )
        left_low = _arm_plan(
            "left", 0.70, (-0.12, 0.0, 0.0), gripper_target=0.25
        )
        right_too_close = _arm_plan(
            "right", 0.99, (-0.09, 0.0, 0.0), gripper_target=0.35
        )
        right_high = _arm_plan(
            "right", 0.80, (0.10, 0.0, 0.0), gripper_target=0.35
        )
        right_low = _arm_plan(
            "right", 0.60, (0.12, 0.0, 0.0), gripper_target=0.35
        )
        runtime = RoboTwinHeuristicRuntime.__new__(RoboTwinHeuristicRuntime)
        runtime.task_env = _Env()
        runtime.staged_controller = StagedQposActionBuffer(
            runtime.task_env, max_waypoints_per_segment=16
        )
        runtime.gripper_settle_actions = 2
        runtime.bimanual_collision_step_rad = 0.025
        runtime.ik = _ExactIK((True, False))
        runtime._action_metadata_override = None
        planned = (
            {
                "left": [left_high, left_low],
                "right": [right_too_close, right_high, right_low],
            },
            [],
            {},
            {"pair_separation": 0},
            0.05,
            "recorded_contact_regions",
        )

        with patch.object(
            runtime, "_plan_simultaneous_pick_sides", return_value=planned
        ), patch.object(runtime, "_save_grasp_visualization") as visualizer:
            actions = runtime._get_simultaneous_pick_action(
                object(), ObjectState("roller", I.copy(), 7), _picks()
            )

        self.assertTrue(actions)
        self.assertEqual(len(runtime.ik.collision_calls), 2)
        selected = [call.args[3] for call in visualizer.call_args_list]
        self.assertEqual(
            [candidate.confidence for candidate in selected], [0.90, 0.60]
        )
        self.assertEqual(actions[-1][6], 0.25)
        self.assertEqual(actions[-1][13], 0.35)
        self.assertTrue(
            all(
                row.shape == (14,)
                for rows, _ in runtime.ik.collision_calls
                for row in rows
            )
        )

    def test_all_colliding_pairs_fail_atomically_without_exposed_actions(self):
        left = _arm_plan(
            "left", 0.9, (-0.11, 0.0, 0.0), gripper_target=0.25
        )
        right = _arm_plan(
            "right", 0.8, (0.12, 0.0, 0.0), gripper_target=0.35
        )
        runtime = RoboTwinHeuristicRuntime.__new__(RoboTwinHeuristicRuntime)
        runtime.task_env = _Env()
        runtime.staged_controller = StagedQposActionBuffer(
            runtime.task_env, max_waypoints_per_segment=16
        )
        runtime.controller = SimpleNamespace(metadata=[])
        runtime.gripper_settle_actions = 2
        runtime.bimanual_collision_step_rad = 0.025
        runtime.ik = _ExactIK((True,))
        runtime._action_metadata_override = None
        planned = (
            {"left": [left], "right": [right]},
            [],
            {},
            {"pair_separation": 0},
            0.05,
            "recorded_contact_regions",
        )

        with patch.object(
            runtime, "_plan_simultaneous_pick_sides", return_value=planned
        ), patch.object(runtime, "_save_grasp_visualization"):
            with self.assertRaisesRegex(
                NoFeasiblePlanFailure, "simultaneous Pick"
            ):
                runtime._get_simultaneous_pick_action(
                    object(), ObjectState("roller", I.copy(), 7), _picks()
                )

        self.assertIsNone(runtime._action_metadata_override)
        self.assertEqual(runtime.action_metadata, [])
        self.assertEqual(len(runtime.ik.collision_calls), 1)


if __name__ == "__main__":
    unittest.main()

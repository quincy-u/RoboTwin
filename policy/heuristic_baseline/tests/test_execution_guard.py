"""CPU-only tests for heuristic execution endpoint validation."""
from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from policy.heuristic_baseline import deploy_policy


class _Model:
    simple_grasp_root = Path("/unused")

    def __init__(self, metadata):
        self.usr_args = {
            "execution_telemetry": False,
            "execution_guard_enabled": True,
        }
        self.last_action_metadata = metadata
        self.execution_batch_index = 0

    def get_action(self, **_kwargs):
        return [np.zeros(14) for _ in self.last_action_metadata]


class _TaskEnv:
    step_lim = 50

    def __init__(self, *, report_success: bool = False):
        self.take_action_cnt = 0
        self.eval_success = False
        self.report_success = report_success
        self.actions = []

    def take_action(self, action, *, action_type):
        self.actions.append((action, action_type))
        self.take_action_cnt += 1
        if self.report_success:
            self.eval_success = True


def _metadata(phase: str, *, endpoint: bool) -> dict:
    return {
        "phase": phase,
        "arm": "right",
        "endpoint": endpoint,
        "waypoint_index": 1,
        "waypoint_count": 1,
        "target_name": "bottle",
    }


def _arm_gripper_metadata(
    phase: str, arm: str, *, endpoint: bool
) -> dict:
    metadata = _metadata(phase, endpoint=endpoint)
    metadata["arm"] = arm
    metadata["target_gripper"] = 1.0 if phase == "open" else 0.0
    metadata["gripper_arms"] = (arm,)
    return metadata


def _dual_metadata(phase: str, *, endpoint: bool) -> dict:
    return {
        "phase": phase,
        "arm": "both",
        "endpoint": endpoint,
        "waypoint_index": 1,
        "waypoint_count": 1,
        "arm_source": {"left": "task_plan", "right": "task_plan"},
        "arm_targets": {
            "left": {
                "target_qpos": np.zeros(6),
                "target_gripper": 0.0,
                "command_pose": np.eye(4),
                "target_name": "bottle1",
            },
            "right": {
                "target_qpos": np.zeros(6),
                "target_gripper": 0.0,
                "command_pose": np.eye(4),
                "target_name": "bottle2",
            },
        },
    }


class ExecutionGuardTest(unittest.TestCase):
    def test_failure_reasons_use_raw_orientation(self) -> None:
        failures = deploy_policy._execution_guard_failures(
            {
                "qpos_max_error_rad": 0.11,
                "ee_position_error_m": 0.031,
                "ee_orientation_error_raw_rad": 0.21,
                # A symmetry-adjusted error must not hide the raw mismatch.
                "ee_orientation_error_rad": 0.0,
            },
            qpos_tolerance_rad=0.10,
            ee_position_tolerance_m=0.03,
            ee_orientation_tolerance_rad=0.20,
        )

        self.assertEqual(len(failures), 3)
        self.assertTrue(any("qpos_max_error_rad" in item for item in failures))
        self.assertTrue(any("ee_position_error_m" in item for item in failures))
        self.assertTrue(
            any("ee_orientation_error_raw_rad" in item for item in failures)
        )

    def test_measurements_do_not_hide_half_roll_error(self) -> None:
        class Robot:
            def get_right_arm_real_jointState(self):
                return np.r_[np.zeros(6), 1.0]

            def get_right_gripper_val(self):
                return 1.0

            def get_right_ee_pose(self):
                return np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])

        class Env:
            robot = Robot()

        measurements = deploy_policy._robot_measurements(
            Env(),
            {
                "arm": "right",
                "target_qpos": np.zeros(6),
                "command_pose": np.eye(4),
            },
        )

        self.assertAlmostEqual(
            measurements["ee_orientation_error_raw_rad"], np.pi
        )
        self.assertAlmostEqual(measurements["ee_orientation_error_rad"], np.pi)

    def test_bad_pregrasp_endpoint_aborts_before_close(self) -> None:
        model = _Model(
            [
                _metadata("pregrasp", endpoint=True),
                _metadata("close", endpoint=True),
            ]
        )
        model.usr_args["execution_telemetry"] = True
        env = _TaskEnv(report_success=True)
        measured = {
            "qpos_max_error_rad": 1.548,
            "ee_position_error_m": 0.189,
            "ee_orientation_error_raw_rad": 2.952,
        }

        records = []
        output = io.StringIO()
        with (
            patch.object(deploy_policy, "encode_obs", return_value=object()),
            patch.object(deploy_policy, "_robot_measurements", return_value=measured),
            patch.object(deploy_policy, "_execution_trace_path", return_value=None),
            patch.object(
                deploy_policy,
                "_write_execution_record",
                side_effect=lambda _path, record: records.append(record),
            ),
            redirect_stdout(output),
        ):
            deploy_policy.eval(env, model, {})

        self.assertEqual(len(env.actions), 1)
        self.assertFalse(env.eval_success)
        self.assertEqual(env.take_action_cnt, env.step_lim)
        self.assertIn("guard failed phase=pregrasp", output.getvalue())
        self.assertEqual(records[0]["status"], "execution_guard_failed")
        self.assertEqual(len(records[0]["execution_guard_failures"]), 4)
        self.assertIn("eval_success_before_close", records[0]["execution_guard_failures"])
        self.assertFalse(records[0]["eval_success"])

    def test_premature_success_at_intermediate_waypoint_is_guarded(self) -> None:
        model = _Model(
            [
                _metadata("pregrasp", endpoint=False),
                _metadata("pregrasp", endpoint=True),
            ]
        )
        env = _TaskEnv(report_success=True)
        measured = {
            "qpos_max_error_rad": 0.858,
            "ee_position_error_m": 0.10,
            "ee_orientation_error_raw_rad": 1.0,
        }

        with (
            patch.object(deploy_policy, "encode_obs", return_value=object()),
            patch.object(deploy_policy, "_robot_measurements", return_value=measured),
        ):
            deploy_policy.eval(env, model, {})

        self.assertEqual(len(env.actions), 1)
        self.assertFalse(env.eval_success)
        self.assertEqual(env.take_action_cnt, env.step_lim)

    def test_endpoint_within_tolerances_proceeds(self) -> None:
        model = _Model(
            [
                _metadata("pregrasp", endpoint=True),
                _metadata("action", endpoint=True),
            ]
        )
        env = _TaskEnv()
        measured = {
            "qpos_max_error_rad": 0.01,
            "ee_position_error_m": 0.002,
            "ee_orientation_error_raw_rad": 0.02,
        }

        with (
            patch.object(deploy_policy, "encode_obs", return_value=object()),
            patch.object(deploy_policy, "_robot_measurements", return_value=measured),
        ):
            deploy_policy.eval(env, model, {})

        self.assertEqual(len(env.actions), 2)
        self.assertFalse(env.eval_success)
        self.assertEqual(env.take_action_cnt, env.step_lim)

    def test_calibrated_wide_handoff_response_passes(self) -> None:
        model = _Model([])
        settings = deploy_policy._execution_guard_settings(model)
        self.assertAlmostEqual(settings[4], 0.08)

        failures = deploy_policy._gripper_execution_guard_failures(
            initial_state=1.0,
            states=[0.9100, 0.9065],
            min_delta=settings[4],
            settle_delta_max=settings[5],
        )
        self.assertEqual(failures, [])

    def test_wide_object_physical_response_passes(self) -> None:
        failures = deploy_policy._gripper_execution_guard_failures(
            initial_state=1.0,
            states=[0.84, 0.82],
            min_delta=0.10,
            settle_delta_max=0.05,
        )
        self.assertEqual(failures, [])

    def test_close_without_physical_response_fails(self) -> None:
        failures = deploy_policy._gripper_execution_guard_failures(
            initial_state=1.0,
            states=[0.96, 0.95],
            min_delta=0.08,
            settle_delta_max=0.05,
        )
        self.assertTrue(any("gripper_closure_delta" in item for item in failures))

    def test_release_physical_response_passes(self) -> None:
        failures = deploy_policy._gripper_release_execution_guard_failures(
            initial_state=0.30,
            states=[0.96, 0.98],
            min_delta=0.10,
            settle_delta_max=0.05,
        )
        self.assertEqual(failures, [])

    def test_release_without_physical_response_fails(self) -> None:
        failures = deploy_policy._gripper_release_execution_guard_failures(
            initial_state=0.30,
            states=[0.34, 0.35],
            min_delta=0.10,
            settle_delta_max=0.05,
        )
        self.assertTrue(any("gripper_release_delta" in item for item in failures))

    def test_initial_open_is_not_mistaken_for_release(self) -> None:
        model = _Model(
            [
                _arm_gripper_metadata("open", "right", endpoint=True),
                _metadata("action", endpoint=True),
            ]
        )
        env = _TaskEnv()

        with patch.object(deploy_policy, "encode_obs", return_value=object()):
            deploy_policy.eval(env, model, {})

        self.assertEqual(len(env.actions), 2)
        self.assertEqual(env.take_action_cnt, env.step_lim)

    def test_bimanual_gripper_phase_filters_unchanged_held_arm(self) -> None:
        metadata = _dual_metadata("open", endpoint=True)
        metadata["arm_targets"]["left"]["target_gripper"] = 1.0
        metadata["arm_targets"]["right"]["target_gripper"] = 0.0

        self.assertEqual(
            deploy_policy._gripper_phase_arms(metadata, "open"), ("left",)
        )

    def test_explicit_gripper_arms_excludes_latched_giver(self) -> None:
        metadata = _dual_metadata("close", endpoint=True)
        metadata["arm_targets"]["left"]["target_gripper"] = 0.0
        metadata["arm_targets"]["right"]["target_gripper"] = 0.0
        metadata["gripper_arms"] = ("right",)

        self.assertEqual(
            deploy_policy._gripper_phase_arms(metadata, "close"), ("right",)
        )

    def test_sequential_close_events_capture_independent_baselines(self) -> None:
        model = _Model(
            [
                *[
                    _arm_gripper_metadata("close", "left", endpoint=index == 2)
                    for index in range(3)
                ],
                *[
                    _arm_gripper_metadata("close", "right", endpoint=index == 2)
                    for index in range(3)
                ],
                _metadata("action", endpoint=True),
            ]
        )
        env = _TaskEnv()
        measurement_calls: list[tuple[str, int]] = []

        def measured(_task_env, action_metadata):
            arm = action_metadata["arm"]
            index = len(env.actions)
            measurement_calls.append((arm, index))
            values = {
                "left": {0: 1.0, 1: 0.84, 2: 0.82, 3: 0.82},
                "right": {3: 1.0, 4: 0.84, 5: 0.82, 6: 0.82},
            }
            return {
                "gripper_physical_state": values[arm].get(index, 0.82)
            }

        with (
            patch.object(deploy_policy, "encode_obs", return_value=object()),
            patch.object(
                deploy_policy, "_robot_measurements", side_effect=measured
            ),
        ):
            deploy_policy.eval(env, model, {})

        self.assertEqual(len(env.actions), 7)
        self.assertEqual(measurement_calls.count(("left", 0)), 1)
        self.assertEqual(measurement_calls.count(("right", 3)), 1)
        self.assertEqual(env.take_action_cnt, env.step_lim)

    def test_handoff_release_is_independent_of_receiver_close(self) -> None:
        model = _Model(
            [
                *[
                    _arm_gripper_metadata("close", "left", endpoint=index == 2)
                    for index in range(3)
                ],
                *[
                    _arm_gripper_metadata("close", "right", endpoint=index == 2)
                    for index in range(3)
                ],
                *[
                    _arm_gripper_metadata("open", "left", endpoint=index == 2)
                    for index in range(3)
                ],
                _metadata("action", endpoint=True),
            ]
        )
        env = _TaskEnv()
        baseline_calls: list[tuple[str, str, int]] = []

        def measured(_task_env, action_metadata):
            arm = action_metadata["arm"]
            phase = action_metadata["phase"]
            index = len(env.actions)
            baseline_calls.append((phase, arm, index))
            if arm == "left" and phase == "close":
                value = {0: 1.0, 1: 0.84, 2: 0.82, 3: 0.82}[index]
            elif arm == "right":
                value = {3: 1.0, 4: 0.84, 5: 0.82, 6: 0.82}[index]
            else:
                value = {6: 0.82, 7: 0.96, 8: 0.99, 9: 1.0}[index]
            return {"gripper_physical_state": value}

        with (
            patch.object(deploy_policy, "encode_obs", return_value=object()),
            patch.object(
                deploy_policy, "_robot_measurements", side_effect=measured
            ),
        ):
            deploy_policy.eval(env, model, {})

        self.assertEqual(len(env.actions), 10)
        self.assertEqual(baseline_calls.count(("open", "left", 6)), 1)
        self.assertEqual(env.take_action_cnt, env.step_lim)

    def test_release_suppresses_internal_early_success_until_settled(self) -> None:
        model = _Model(
            [
                *[
                    _arm_gripper_metadata("close", "right", endpoint=index == 2)
                    for index in range(3)
                ],
                *[
                    _arm_gripper_metadata("open", "right", endpoint=index == 4)
                    for index in range(5)
                ],
            ]
        )

        class EarlyExitReleaseEnv(_TaskEnv):
            def __init__(self):
                super().__init__()
                self.physical_gripper = 1.0
                self.release_active = False
                self.internal_open_steps = 0
                self.real_success_checks = 0

            def check_success(self):
                self.real_success_checks += 1
                return self.release_active

            def take_action(self, action, *, action_type):
                self.actions.append((action, action_type))
                self.take_action_cnt += 1
                action_count = len(self.actions)
                if action_count <= 3:
                    self.physical_gripper = {
                        1: 0.84,
                        2: 0.82,
                        3: 0.82,
                    }[action_count]
                    return
                self.release_active = True
                for _ in range(5):
                    self.internal_open_steps += 1
                    self.physical_gripper = min(
                        1.0, self.physical_gripper + 0.008
                    )
                    if self.check_success():
                        self.eval_success = True
                        return

        env = EarlyExitReleaseEnv()

        def measured(_task_env, _action_metadata):
            return {
                "gripper_physical_state": env.physical_gripper,
            }

        with (
            patch.object(deploy_policy, "encode_obs", return_value=object()),
            patch.object(
                deploy_policy, "_robot_measurements", side_effect=measured
            ),
        ):
            deploy_policy.eval(env, model, {})

        self.assertEqual(len(env.actions), 8)
        self.assertEqual(env.internal_open_steps, 25)
        self.assertEqual(env.real_success_checks, 1)
        self.assertNotIn("check_success", env.__dict__)
        self.assertTrue(env.eval_success)
        self.assertLess(env.take_action_cnt, env.step_lim)

    def test_success_on_settled_final_release_is_preserved(self) -> None:
        model = _Model(
            [
                *[
                    _arm_gripper_metadata("close", "right", endpoint=index == 2)
                    for index in range(3)
                ],
                *[
                    _arm_gripper_metadata("open", "right", endpoint=index == 2)
                    for index in range(3)
                ],
            ]
        )

        class ReleaseSuccessEnv(_TaskEnv):
            def take_action(self, action, *, action_type):
                super().take_action(action, action_type=action_type)
                self.eval_success = len(self.actions) >= 4

        env = ReleaseSuccessEnv()

        def measured(_task_env, action_metadata):
            index = len(env.actions)
            if action_metadata["phase"] == "close":
                value = {0: 1.0, 1: 0.84, 2: 0.82, 3: 0.82}[index]
            else:
                value = {3: 0.82, 4: 0.96, 5: 0.99, 6: 1.0}[index]
            return {"gripper_physical_state": value}

        with (
            patch.object(deploy_policy, "encode_obs", return_value=object()),
            patch.object(
                deploy_policy, "_robot_measurements", side_effect=measured
            ),
        ):
            deploy_policy.eval(env, model, {})

        self.assertEqual(len(env.actions), 6)
        self.assertTrue(env.eval_success)
        self.assertLess(env.take_action_cnt, env.step_lim)

    def test_close_success_does_not_skip_planned_release(self) -> None:
        model = _Model(
            [
                *[
                    _arm_gripper_metadata("close", "right", endpoint=index == 2)
                    for index in range(3)
                ],
                _metadata("lift", endpoint=True),
                _metadata("preplace", endpoint=True),
                _metadata("place", endpoint=True),
                *[
                    _arm_gripper_metadata("open", "right", endpoint=index == 2)
                    for index in range(3)
                ],
            ]
        )

        class CloseAndReleaseSuccessEnv(_TaskEnv):
            def take_action(self, action, *, action_type):
                super().take_action(action, action_type=action_type)
                self.eval_success = len(self.actions) in {3, 9}

        env = CloseAndReleaseSuccessEnv()
        good_motion = {
            "qpos_max_error_rad": 0.01,
            "ee_position_error_m": 0.002,
            "ee_orientation_error_raw_rad": 0.02,
        }

        def measured(_task_env, action_metadata):
            index = len(env.actions)
            if action_metadata["phase"] == "close":
                value = {0: 1.0, 1: 0.84, 2: 0.82, 3: 0.82}[index]
            elif action_metadata["phase"] == "open":
                value = {6: 0.82, 7: 0.96, 8: 0.99, 9: 1.0}[index]
            else:
                value = 0.82
            return {**good_motion, "gripper_physical_state": value}

        with (
            patch.object(deploy_policy, "encode_obs", return_value=object()),
            patch.object(
                deploy_policy, "_robot_measurements", side_effect=measured
            ),
        ):
            deploy_policy.eval(env, model, {})

        self.assertEqual(len(env.actions), 9)
        self.assertTrue(env.eval_success)
        self.assertLess(env.take_action_cnt, env.step_lim)

    def test_close_settle_actions_ignore_intermediate_success(self) -> None:
        model = _Model(
            [
                _metadata("close", endpoint=False),
                _metadata("close", endpoint=False),
                _metadata("close", endpoint=True),
                _metadata("retreat", endpoint=True),
            ]
        )

        class CloseSuccessEnv(_TaskEnv):
            def take_action(self, action, *, action_type):
                super().take_action(action, action_type=action_type)
                self.eval_success = len(self.actions) in {1, 2}

        env = CloseSuccessEnv()
        good = {
            "qpos_max_error_rad": 0.01,
            "ee_position_error_m": 0.002,
            "ee_orientation_error_raw_rad": 0.02,
        }

        def measured(_task_env, _metadata):
            physical = {0: 1.0, 1: 0.84, 2: 0.82, 3: 0.82}
            return {
                **good,
                "gripper_physical_state": physical.get(len(env.actions), 0.82),
            }

        with (
            patch.object(deploy_policy, "encode_obs", return_value=object()),
            patch.object(deploy_policy, "_robot_measurements", side_effect=measured),
        ):
            deploy_policy.eval(env, model, {})

        self.assertEqual(len(env.actions), 4)
        self.assertFalse(env.eval_success)
        self.assertEqual(env.take_action_cnt, env.step_lim)

    def test_close_guard_aborts_before_retreat_with_telemetry_off(self) -> None:
        model = _Model(
            [
                _metadata("close", endpoint=False),
                _metadata("close", endpoint=False),
                _metadata("close", endpoint=True),
                _metadata("retreat", endpoint=True),
            ]
        )
        env = _TaskEnv()

        def measured(_task_env, _metadata):
            physical = {0: 1.0, 1: 0.98, 2: 0.96, 3: 0.95}
            return {
                "gripper_physical_state": physical.get(len(env.actions), 0.95)
            }

        with (
            patch.object(deploy_policy, "encode_obs", return_value=object()),
            patch.object(deploy_policy, "_robot_measurements", side_effect=measured),
        ):
            deploy_policy.eval(env, model, {})

        self.assertEqual(len(env.actions), 3)
        self.assertFalse(env.eval_success)
        self.assertEqual(env.take_action_cnt, env.step_lim)

    def test_good_tracking_success_before_close_is_still_premature(self) -> None:
        model = _Model(
            [
                _metadata("pregrasp", endpoint=False),
                _metadata("pregrasp", endpoint=True),
            ]
        )
        env = _TaskEnv(report_success=True)
        measured = {
            "qpos_max_error_rad": 0.01,
            "ee_position_error_m": 0.002,
            "ee_orientation_error_raw_rad": 0.02,
        }

        with (
            patch.object(deploy_policy, "encode_obs", return_value=object()),
            patch.object(deploy_policy, "_robot_measurements", return_value=measured),
        ):
            deploy_policy.eval(env, model, {})

        self.assertEqual(len(env.actions), 1)
        self.assertFalse(env.eval_success)
        self.assertEqual(env.take_action_cnt, env.step_lim)

    def test_success_during_retreat_after_close_is_preserved(self) -> None:
        model = _Model(
            [
                _metadata("close", endpoint=False),
                _metadata("close", endpoint=False),
                _metadata("close", endpoint=True),
                _metadata("retreat", endpoint=False),
                _metadata("retreat", endpoint=True),
            ]
        )

        class RetreatSuccessEnv(_TaskEnv):
            def take_action(self, action, *, action_type):
                super().take_action(action, action_type=action_type)
                self.eval_success = len(self.actions) == 4

        env = RetreatSuccessEnv()
        good = {
            "qpos_max_error_rad": 0.01,
            "ee_position_error_m": 0.002,
            "ee_orientation_error_raw_rad": 0.02,
        }

        def measured(_task_env, _metadata):
            physical = {0: 1.0, 1: 0.84, 2: 0.82, 3: 0.82}
            return {
                **good,
                "gripper_physical_state": physical.get(len(env.actions), 0.82),
            }

        with (
            patch.object(deploy_policy, "encode_obs", return_value=object()),
            patch.object(deploy_policy, "_robot_measurements", side_effect=measured),
        ):
            deploy_policy.eval(env, model, {})

        self.assertEqual(len(env.actions), 4)
        self.assertTrue(env.eval_success)
        self.assertLess(env.take_action_cnt, env.step_lim)

    def test_final_close_success_executes_one_closed_retreat_command(self) -> None:
        model = _Model(
            [
                _metadata("close", endpoint=False),
                _metadata("close", endpoint=False),
                _metadata("close", endpoint=True),
                _metadata("retreat", endpoint=True),
            ]
        )

        class FinalCloseSuccessEnv(_TaskEnv):
            def take_action(self, action, *, action_type):
                super().take_action(action, action_type=action_type)
                self.eval_success = len(self.actions) == 3

        env = FinalCloseSuccessEnv()
        good = {
            "qpos_max_error_rad": 0.01,
            "ee_position_error_m": 0.002,
            "ee_orientation_error_raw_rad": 0.02,
        }

        def measured(_task_env, _metadata):
            physical = {0: 1.0, 1: 0.84, 2: 0.82, 3: 0.82}
            return {
                **good,
                "gripper_physical_state": physical.get(len(env.actions), 0.82),
            }

        with (
            patch.object(deploy_policy, "encode_obs", return_value=object()),
            patch.object(deploy_policy, "_robot_measurements", side_effect=measured),
        ):
            deploy_policy.eval(env, model, {})

        self.assertEqual(len(env.actions), 4)
        self.assertEqual(env.actions[-1][0][13], 0.0)
        self.assertTrue(env.eval_success)
        self.assertLess(env.take_action_cnt, env.step_lim)

    def test_robot_measurements_never_queries_ground_truth_contacts(self) -> None:
        class Robot:
            def get_right_arm_real_jointState(self):
                return np.r_[np.zeros(6), 1.0]

            def get_right_gripper_val(self):
                return 1.0

        class Env:
            robot = Robot()

            def get_gripper_actor_contact_position(self, *_args, **_kwargs):
                raise AssertionError("policy must not query simulator contacts")

        measurements = deploy_policy._robot_measurements(
            Env(),
            {
                "arm": "right",
                "target_qpos": np.zeros(6),
                "target_name": None,
            },
        )

        self.assertEqual(measurements["qpos_max_error_rad"], 0.0)
        self.assertNotIn("target_gripper_contact_count", measurements)

    def test_bimanual_measurements_are_nested_and_contact_free(self) -> None:
        class Pose:
            def __init__(self, x):
                self.p = np.array([x, 0.0, 0.8])
                self.q = np.array([1.0, 0.0, 0.0, 0.0])

        class Actor:
            def __init__(self, x):
                self.pose = Pose(x)

            def get_pose(self):
                return self.pose

        class Robot:
            def get_left_arm_real_jointState(self):
                return np.r_[np.zeros(6), 1.0]

            def get_right_arm_real_jointState(self):
                return np.r_[np.full(6, 0.2), 1.0]

            def get_left_gripper_val(self):
                return 0.3

            def get_right_gripper_val(self):
                return 0.4

            def get_left_ee_pose(self):
                return np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

            def get_right_ee_pose(self):
                return np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

        class Env:
            robot = Robot()

            def get_tracked_objects(self):
                return {"bottle1": Actor(-0.2), "bottle2": Actor(0.2)}

            def get_gripper_actor_contact_position(self, *_args, **_kwargs):
                raise AssertionError("policy must not query simulator contacts")

        metadata = _dual_metadata("transport", endpoint=True)
        metadata["arm_targets"]["right"]["target_qpos"] = np.full(6, 0.25)
        measurements = deploy_policy._robot_measurements(Env(), metadata)

        self.assertEqual(set(measurements["arm_measurements"]), {"left", "right"})
        left = measurements["arm_measurements"]["left"]
        right = measurements["arm_measurements"]["right"]
        self.assertEqual(left["target_name"], "bottle1")
        self.assertEqual(right["target_name"], "bottle2")
        self.assertAlmostEqual(left["qpos_max_error_rad"], 0.0)
        self.assertAlmostEqual(right["qpos_max_error_rad"], 0.05)
        self.assertAlmostEqual(left["target_z_m"], 0.8)
        self.assertAlmostEqual(right["target_z_m"], 0.8)
        self.assertNotIn("target_gripper_contact_count", left)
        self.assertNotIn("target_gripper_contact_count", right)

        trace_targets = deploy_policy._trace_arm_targets(metadata)
        self.assertEqual(trace_targets["left"]["target_name"], "bottle1")
        self.assertEqual(trace_targets["right"]["target_name"], "bottle2")
        self.assertEqual(len(trace_targets["right"]["target_qpos"]), 6)

    def test_bimanual_motion_guard_labels_the_failing_arm(self) -> None:
        good = {
            "qpos_max_error_rad": 0.01,
            "ee_position_error_m": 0.002,
            "ee_orientation_error_raw_rad": 0.02,
        }
        failures = deploy_policy._execution_guard_failures(
            {
                "arm_measurements": {
                    "left": good,
                    "right": {**good, "ee_position_error_m": 0.20},
                }
            },
            qpos_tolerance_rad=0.10,
            ee_position_tolerance_m=0.03,
            ee_orientation_tolerance_rad=0.20,
        )

        self.assertEqual(failures, ["right.ee_position_error_m=0.2000>0.0300"])

    def test_bimanual_close_guard_rejects_one_unresponsive_gripper(self) -> None:
        model = _Model(
            [
                _dual_metadata("close", endpoint=False),
                _dual_metadata("close", endpoint=False),
                _dual_metadata("close", endpoint=True),
                _dual_metadata("transport", endpoint=True),
            ]
        )
        env = _TaskEnv()
        good_motion = {
            "qpos_max_error_rad": 0.01,
            "ee_position_error_m": 0.002,
            "ee_orientation_error_raw_rad": 0.02,
        }

        def measured(_task_env, _metadata):
            index = len(env.actions)
            left = {0: 1.0, 1: 0.84, 2: 0.82, 3: 0.82}
            right = {0: 1.0, 1: 0.98, 2: 0.96, 3: 0.95}
            return {
                "arm_measurements": {
                    "left": {
                        **good_motion,
                        "gripper_physical_state": left.get(index, 0.82),
                    },
                    "right": {
                        **good_motion,
                        "gripper_physical_state": right.get(index, 0.95),
                    },
                }
            }

        output = io.StringIO()
        with (
            patch.object(deploy_policy, "encode_obs", return_value=object()),
            patch.object(deploy_policy, "_robot_measurements", side_effect=measured),
            redirect_stdout(output),
        ):
            deploy_policy.eval(env, model, {})

        self.assertEqual(len(env.actions), 3)
        self.assertFalse(env.eval_success)
        self.assertEqual(env.take_action_cnt, env.step_lim)
        self.assertIn("right.gripper_closure_delta", output.getvalue())

    def test_bimanual_success_during_transport_is_preserved(self) -> None:
        model = _Model(
            [
                _dual_metadata("close", endpoint=False),
                _dual_metadata("close", endpoint=False),
                _dual_metadata("close", endpoint=True),
                _dual_metadata("transport", endpoint=True),
            ]
        )
        model.usr_args["execution_telemetry"] = True

        class TransportSuccessEnv(_TaskEnv):
            def take_action(self, action, *, action_type):
                super().take_action(action, action_type=action_type)
                self.eval_success = len(self.actions) == 4

        env = TransportSuccessEnv()
        good_motion = {
            "qpos_max_error_rad": 0.01,
            "ee_position_error_m": 0.002,
            "ee_orientation_error_raw_rad": 0.02,
        }

        def measured(_task_env, action_metadata):
            index = len(env.actions)
            physical = {0: 1.0, 1: 0.84, 2: 0.82, 3: 0.82}
            motion = good_motion
            if action_metadata["phase"] == "transport":
                motion = {
                    "qpos_max_error_rad": 1.0,
                    "ee_position_error_m": 1.0,
                    "ee_orientation_error_raw_rad": 1.0,
                }
            return {
                "arm_measurements": {
                    arm: {
                        **motion,
                        "gripper_physical_state": physical.get(index, 0.82),
                    }
                    for arm in ("left", "right")
                }
            }

        with (
            patch.object(deploy_policy, "encode_obs", return_value=object()),
            patch.object(deploy_policy, "_robot_measurements", side_effect=measured),
        ):
            deploy_policy.eval(env, model, {})

        self.assertEqual(len(env.actions), 4)
        self.assertEqual(env.actions[-1][0][6], 0.0)
        self.assertEqual(env.actions[-1][0][13], 0.0)
        self.assertTrue(env.eval_success)
        self.assertLess(env.take_action_cnt, env.step_lim)

    def test_required_release_grasp_lift_runs_guards_but_cannot_succeed(
        self,
    ) -> None:
        metadata = [
            _arm_gripper_metadata("open", "right", endpoint=True),
            _metadata("pregrasp", endpoint=True),
            _metadata("grasp", endpoint=True),
            *[
                _arm_gripper_metadata(
                    "close", "right", endpoint=index == 2
                )
                for index in range(3)
            ],
            _metadata("lift", endpoint=True),
        ]
        for record in metadata:
            record["completion_level"] = "grasp_lift"
            record["required_release"] = True
        model = _Model(metadata)

        class PositionSuccessEnv(_TaskEnv):
            def take_action(self, action, *, action_type):
                super().take_action(action, action_type=action_type)
                self.eval_success = len(self.actions) >= 6

        env = PositionSuccessEnv()
        good_motion = {
            "qpos_max_error_rad": 0.01,
            "ee_position_error_m": 0.002,
            "ee_orientation_error_raw_rad": 0.02,
        }

        def measured(_task_env, _action_metadata):
            physical = {3: 1.0, 4: 0.84, 5: 0.82, 6: 0.82}
            return {
                **good_motion,
                "gripper_physical_state": physical.get(
                    len(env.actions), 0.82
                ),
            }

        output = io.StringIO()
        with (
            patch.object(deploy_policy, "encode_obs", return_value=object()),
            patch.object(
                deploy_policy, "_robot_measurements", side_effect=measured
            ),
            patch.object(
                deploy_policy,
                "_execution_guard_failures",
                wraps=deploy_policy._execution_guard_failures,
            ) as motion_guard,
            patch.object(
                deploy_policy,
                "_gripper_execution_guard_failures",
                wraps=deploy_policy._gripper_execution_guard_failures,
            ) as gripper_guard,
            redirect_stdout(output),
        ):
            deploy_policy.eval(env, model, {})

        self.assertEqual(len(env.actions), len(metadata))
        self.assertGreaterEqual(motion_guard.call_count, 3)
        self.assertEqual(gripper_guard.call_count, 1)
        self.assertFalse(env.eval_success)
        self.assertEqual(env.take_action_cnt, env.step_lim)
        self.assertIn(
            "one-shot rollout complete without task success",
            output.getvalue(),
        )

    def test_nonrelease_grasp_lift_preserves_terminal_pick_success(self) -> None:
        metadata = [
            _arm_gripper_metadata("open", "right", endpoint=True),
            _metadata("pregrasp", endpoint=True),
            _metadata("grasp", endpoint=True),
            *[
                _arm_gripper_metadata(
                    "close", "right", endpoint=index == 2
                )
                for index in range(3)
            ],
            _metadata("lift", endpoint=True),
        ]
        for record in metadata:
            record["completion_level"] = "grasp_lift"
            record["required_release"] = False
        model = _Model(metadata)

        class PickSuccessEnv(_TaskEnv):
            def take_action(self, action, *, action_type):
                super().take_action(action, action_type=action_type)
                self.eval_success = len(self.actions) == len(metadata)

        env = PickSuccessEnv()
        good_motion = {
            "qpos_max_error_rad": 0.01,
            "ee_position_error_m": 0.002,
            "ee_orientation_error_raw_rad": 0.02,
        }

        def measured(_task_env, _action_metadata):
            physical = {3: 1.0, 4: 0.84, 5: 0.82, 6: 0.82}
            return {
                **good_motion,
                "gripper_physical_state": physical.get(
                    len(env.actions), 0.82
                ),
            }

        with (
            patch.object(deploy_policy, "encode_obs", return_value=object()),
            patch.object(
                deploy_policy, "_robot_measurements", side_effect=measured
            ),
        ):
            deploy_policy.eval(env, model, {})

        self.assertEqual(len(env.actions), len(metadata))
        self.assertTrue(env.eval_success)
        self.assertLess(env.take_action_cnt, env.step_lim)

if __name__ == "__main__":
    unittest.main()

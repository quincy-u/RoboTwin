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
        self.assertEqual(len(records[0]["execution_guard_failures"]), 3)
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
                _metadata("close", endpoint=True),
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


if __name__ == "__main__":
    unittest.main()

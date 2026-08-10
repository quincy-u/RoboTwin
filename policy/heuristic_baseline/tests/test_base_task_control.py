"""CPU-only checks for RoboTwin's qpos fallback and eval-video capture."""
from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

import numpy as np

from envs._base_task import Base_Task


class _RaisingPlanner:
    def TOPP(self, *_args, **_kwargs):
        raise RuntimeError("synthetic TOPP failure")


class _EmptyPlanner:
    def TOPP(self, path, *_args, **_kwargs):
        width = np.asarray(path).shape[1]
        empty = np.empty((0, width), dtype=np.float64)
        return None, empty, empty.copy(), None, None


class _FrameSink:
    def __init__(self) -> None:
        self.writes: list[bytes] = []

    def write(self, value: bytes) -> None:
        self.writes.append(value)


class _FreshFrameCamera:
    def __init__(self) -> None:
        self.update_calls = 0
        self.frame = np.zeros((2, 3, 3), dtype=np.uint8)

    def update_picture(self) -> None:
        self.update_calls += 1
        self.frame = np.full((2, 3, 3), 173, dtype=np.uint8)

    def get_rgb(self):
        return {"head_camera": {"rgb": self.frame.copy()}}


class BaseTaskControlTest(unittest.TestCase):
    def task_with_planner(self, arm: str, planner) -> Base_Task:
        task = Base_Task.__new__(Base_Task)
        task.robot = SimpleNamespace(
            **{f"{arm}_mplib_planner": planner}
        )
        return task

    def test_nonzero_topp_failure_uses_bounded_linear_fallback(self) -> None:
        task = self.task_with_planner("left", _RaisingPlanner())
        path = np.array([[0.0, 0.0], [0.12, -0.04]])

        output = io.StringIO()
        with redirect_stdout(output):
            result, num_steps, moves_arm = task._qpos_arm_trajectory(
                "left", path
            )

        self.assertTrue(moves_arm)
        self.assertEqual(num_steps, 50)
        np.testing.assert_allclose(result["position"][-1], path[-1])
        increments = np.diff(
            np.vstack((path[0], result["position"])), axis=0
        )
        self.assertLessEqual(np.max(np.abs(increments)), 0.02 + 1e-12)
        self.assertIn("left arm TOPP failed", output.getvalue())
        self.assertIn("synthetic TOPP failure", output.getvalue())

    def test_nonzero_zero_step_topp_result_uses_fallback(self) -> None:
        task = self.task_with_planner("right", _EmptyPlanner())
        path = np.array([[0.0, 0.0], [0.03, 0.0]])

        output = io.StringIO()
        with redirect_stdout(output):
            result, num_steps, moves_arm = task._qpos_arm_trajectory(
                "right", path
            )

        self.assertTrue(moves_arm)
        self.assertEqual(num_steps, 50)
        np.testing.assert_allclose(result["position"][-1], path[-1])
        self.assertIn("returned zero steps", output.getvalue())

    def test_zero_delta_topp_failure_remains_silent_noop(self) -> None:
        task = self.task_with_planner("left", _RaisingPlanner())
        output = io.StringIO()
        with redirect_stdout(output):
            result, num_steps, moves_arm = task._qpos_arm_trajectory(
                "left", np.zeros((2, 2))
            )

        self.assertIsNone(result)
        self.assertEqual(num_steps, 50)
        self.assertFalse(moves_arm)
        self.assertEqual(output.getvalue(), "")

    def test_eval_video_frame_refreshes_camera_before_write(self) -> None:
        task = Base_Task.__new__(Base_Task)
        task.eval_video_path = "enabled"
        task.cameras = _FreshFrameCamera()
        sink = _FrameSink()
        task.eval_video_ffmpeg = SimpleNamespace(stdin=sink)

        task._write_eval_video_frame()

        self.assertEqual(task.cameras.update_calls, 1)
        self.assertEqual(len(sink.writes), 1)
        expected = np.full((2, 3, 3), 173, dtype=np.uint8).tobytes()
        self.assertEqual(sink.writes[0], expected)


if __name__ == "__main__":
    unittest.main()

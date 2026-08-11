"""CPU-only tests for simulator RGB-D and ground-truth scene encoding."""
from __future__ import annotations

import unittest

import numpy as np

from policy.heuristic_baseline.observation import (
    _head_depth_m,
    _world_pointcloud,
    encode_obs,
)
from policy.heuristic_baseline.task_plan import Pick, TaskPlan


class _Pose:
    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = matrix

    def to_transformation_matrix(self) -> np.ndarray:
        return self.matrix


class _Actor:
    per_scene_id = 42

    def __init__(self) -> None:
        self.matrix = np.eye(4)
        self.matrix[:3, 3] = [0.1, 0.2, 0.3]

    def get_pose(self) -> _Pose:
        return _Pose(self.matrix)


class _Camera:
    def get_picture(self, name: str) -> np.ndarray:
        if name != "Position":
            raise AssertionError(name)
        return np.array(
            [[[1.0, 2.0, 3.0, 0.0], [4.0, 5.0, 6.0, 0.0]]],
            dtype=np.float32,
        )


class _Cameras:
    static_camera_name = ["head_camera"]
    static_camera_list = [_Camera()]

    def get_object_masks(self, tracked):
        if set(tracked) != {"bottle"}:
            raise AssertionError(f"expected target-only mask request, got {set(tracked)}")
        return {
            "head_camera": {
                "bottle": np.array([[True, False]], dtype=bool),
            }
        }


class _TaskEnv:
    cameras = _Cameras()

    def __init__(self) -> None:
        self.actor = _Actor()

    def get_tracked_objects(self):
        return {"bottle": self.actor, "wall": self.actor}


class _DualActor:
    def __init__(self, instance_id: int, translation: tuple[float, ...]) -> None:
        self.per_scene_id = instance_id
        self.matrix = np.eye(4)
        self.matrix[:3, 3] = translation

    def get_pose(self) -> _Pose:
        return _Pose(self.matrix)


class _DualCamera:
    def get_picture(self, name: str) -> np.ndarray:
        if name != "Position":
            raise AssertionError(name)
        return np.array(
            [
                [
                    [0.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, 1.0, 0.0],
                    [2.0, 0.0, 1.0, 0.0],
                ]
            ],
            dtype=np.float32,
        )


class _DualCameras:
    static_camera_name = ["head_camera"]
    static_camera_list = [_DualCamera()]

    def __init__(self) -> None:
        self.mask_requests: list[set[str]] = []

    def get_object_masks(self, tracked):
        self.mask_requests.append(set(tracked))
        return {
            "head_camera": {
                "bottle1": np.array([[True, False, False]], dtype=bool),
                "bottle2": np.array([[False, True, False]], dtype=bool),
            }
        }


class _DualTaskEnv:
    def __init__(self) -> None:
        self.cameras = _DualCameras()
        self.actors = {
            "bottle1": _DualActor(41, (-0.1, 0.1, 0.8)),
            "bottle2": _DualActor(42, (0.1, 0.1, 0.8)),
            "distractor": _DualActor(43, (0.0, -0.1, 0.8)),
        }
        self.heuristic_task_plan = TaskPlan(
            "pick_dual_bottles",
            "pick_place",
            (Pick("bottle1", "left"), Pick("bottle2", "right")),
        )

    def get_tracked_objects(self):
        return self.actors



class ObservationTest(unittest.TestCase):
    def test_world_pointcloud_accepts_three_by_four_extrinsic(self) -> None:
        xyz, valid, camera_to_world = _world_pointcloud(
            np.array([[2.0]], dtype=np.float32),
            np.eye(3),
            np.eye(4)[:3],
        )

        np.testing.assert_allclose(xyz, [[0.0, 0.0, 2.0]])
        self.assertEqual(valid.tolist(), [True])
        np.testing.assert_allclose(camera_to_world, np.eye(4))

    def test_encode_obs_includes_metric_depth_labels_and_six_d_pose(self) -> None:
        camera_to_world = np.eye(4)
        camera_to_world[0, 3] = 10.0
        observation = {
            "observation": {
                "head_camera": {
                    "rgb": np.array(
                        [[[255, 0, 0], [0, 128, 255]]], dtype=np.uint8
                    ),
                    "depth": np.array([[1000.0, 2000.0]], dtype=np.float32),
                    "cam2world_gl": camera_to_world,
                }
            }
        }

        task_env = _TaskEnv()
        depth_m = _head_depth_m(
            task_env, observation["observation"]["head_camera"]
        )
        scene = encode_obs(task_env, observation)

        np.testing.assert_allclose(depth_m, [[1.0, 2.0]])
        np.testing.assert_allclose(scene.xyz, [[11, 2, 3], [14, 5, 6]])
        np.testing.assert_allclose(
            scene.rgb,
            [[1.0, 0.0, 0.0], [0.0, 128.0 / 255.0, 1.0]],
        )
        self.assertEqual(scene.instance_labels.tolist(), [42, -1])
        self.assertEqual(scene.objects["bottle"].instance_id, 42)
        np.testing.assert_allclose(
            scene.objects["bottle"].world_pose[:3, 3], [0.1, 0.2, 0.3]
        )
        np.testing.assert_allclose(scene.camera_pose, camera_to_world)

    def test_auto_dual_plan_encodes_exactly_both_targets(self) -> None:
        observation = {
            "observation": {
                "head_camera": {
                    "rgb": np.zeros((1, 3, 3), dtype=np.uint8),
                    "depth": np.full((1, 3), 1000.0, dtype=np.float32),
                    "cam2world_gl": np.eye(4),
                }
            }
        }

        task_env = _DualTaskEnv()
        scene = encode_obs(task_env, observation)

        self.assertEqual(
            task_env.cameras.mask_requests,
            [{"bottle1", "bottle2"}],
        )
        self.assertEqual(set(scene.objects), {"bottle1", "bottle2"})
        self.assertEqual(scene.instance_labels.tolist(), [41, 42, -1])
        self.assertEqual(
            {state.instance_id for state in scene.objects.values()},
            {41, 42},
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
import transforms3d as t3d

from policy.heuristic_baseline.runtime import (
    CANONICAL_COMMAND_QUATERNIONS,
    ConfidenceRankedGrasps,
    M2T2_TO_ROBOTWIN,
    M2T2_GRIPPER_POLYLINE,
    QposActionBuffer,
    RoboTwinMinkIK,
    SELECTED_GRASP_COLOR,
    _aloha_self_collision_config,
    _grasp_wireframes,
    _project_world_points_cv,
    save_grasp_visualization,
)
from simple_grasp.types import (
    GraspCandidate,
    ObjectState,
    SceneObservation,
)

I = np.eye(4)


class Pose:
    def to_transformation_matrix(self):
        return I.copy()


class Robot:
    left_arm_joints_name = tuple(f"fl_joint{i}" for i in range(1, 7))
    right_arm_joints_name = tuple(f"fr_joint{i}" for i in range(1, 7))
    left_entity_origion_pose = Pose()
    right_entity_origion_pose = Pose()
    left_global_trans_matrix = np.diag([1.0, -1.0, -1.0])
    right_global_trans_matrix = np.diag([1.0, -1.0, -1.0])
    left_delta_matrix = np.eye(3)
    right_delta_matrix = np.eye(3)

    def get_left_arm_jointState(self):
        return np.r_[np.zeros(6), 1.0]

    def get_right_arm_jointState(self):
        return np.r_[np.zeros(6), 1.0]


class Env:
    robot = Robot()


class FakeSolver:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def solve(self, arm, target):
        self.calls.append((arm, np.asarray(target).copy()))
        return self.results.pop(0)


class FakeGrasps:
    def __init__(self, candidates):
        self.candidates = candidates

    def propose(self, observation, target):
        return list(self.candidates)


class MinkRuntimeTest(unittest.TestCase):
    def make_ik(self, results):
        fake = FakeSolver(results)
        with patch(
            "policy.heuristic_baseline.runtime.MinkIKSolver.from_xml_path",
            return_value=fake,
        ):
            ik = RoboTwinMinkIK(
                Env(), I, model_path="unused.urdf", max_joint_step_rad=0.1
            )
        return ik, fake

    def test_chains_mink_solutions_and_builds_bounded_paths(self):
        goals = [np.full(6, value) for value in (0.95, 1.05, 1.15)]
        ik, fake = self.make_ik(goals)

        actual = [ik.solve("right", I) for _ in range(3)]

        for result, expected in zip(actual, goals):
            np.testing.assert_allclose(result, expected)
        self.assertEqual(len(fake.calls), 3)
        np.testing.assert_allclose(fake.calls[0][1][:3, :3], np.eye(3))
        previous = np.zeros(6)
        paths = []
        for goal in goals:
            path = ik.consume_path("right", goal)
            paths.append(path)
            self.assertGreaterEqual(len(path), 2)
            np.testing.assert_allclose(path[-1], goal)
            with_start = np.vstack((previous, path))
            self.assertLessEqual(
                float(np.max(np.abs(np.diff(with_start, axis=0)))),
                ik.max_joint_step_rad + 1e-12,
            )
            previous = goal
        self.assertGreater(len(paths[0]), 8)

    def test_retries_with_canonical_orientation_and_preserves_tcp(self):
        goal = np.arange(6, dtype=np.float64) * 0.01
        ik, fake = self.make_ik([None, goal])

        result = ik.solve("right", I)

        np.testing.assert_allclose(result, goal)
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(ik.relaxed_successes, 1)
        canonical = fake.calls[1][1]
        np.testing.assert_allclose(
            canonical[:3, :3],
            t3d.quaternions.quat2mat(
                CANONICAL_COMMAND_QUATERNIONS["right"]
            ),
        )
        np.testing.assert_allclose(
            canonical[:3, 3] + 0.12 * canonical[:3, 0],
            np.array([0.12, 0.0, 0.0]),
        )

    def test_latches_mink_accepted_orientation_across_all_stages(self):
        goals = [np.full(6, value) for value in (0.1, 0.2, 0.3)]
        ik, fake = self.make_ik([None, goals[0], goals[1], goals[2]])

        actual = [ik.solve("right", I) for _ in range(3)]

        for result, expected in zip(actual, goals):
            np.testing.assert_allclose(result, expected)
        self.assertEqual(len(fake.calls), 4)
        canonical_rotation = t3d.quaternions.quat2mat(
            CANONICAL_COMMAND_QUATERNIONS["right"]
        )
        for _, target in fake.calls[1:]:
            np.testing.assert_allclose(target[:3, :3], canonical_rotation)
            np.testing.assert_allclose(
                target[:3, 3] + 0.12 * target[:3, 0],
                np.array([0.12, 0.0, 0.0]),
            )
        accepted = ik.selected_grasp_command_pose
        self.assertIsNotNone(accepted)
        np.testing.assert_allclose(accepted, fake.calls[2][1])

    def test_wraps_revolute_solution_to_nearest_safe_branch(self):
        raw = np.array(
            [
                2.0 * np.pi + 0.25,
                -2.0 * np.pi - 0.30,
                0.4,
                -0.5,
                4.0 * np.pi + 0.6,
                -4.0 * np.pi - 0.7,
            ]
        )
        ik, _ = self.make_ik([raw])

        actual = ik.solve("right", I)

        np.testing.assert_allclose(
            actual, np.array([0.25, -0.30, 0.4, -0.5, 0.6, -0.7])
        )
        self.assertTrue(np.all(np.abs(actual) <= np.pi))

    def test_aloha_collision_config_contains_both_arm_groups(self):
        model_path = (
            Path(__file__).resolve().parents[3]
            / "assets"
            / "embodiments"
            / "aloha-agilex"
            / "urdf"
            / "arx5_description_isaac.urdf"
        )
        model = mujoco.MjModel.from_xml_path(str(model_path))

        config = _aloha_self_collision_config(model)

        self.assertEqual(len(config.geom_pairs), 2)
        for geom_group, same_group in config.geom_pairs:
            self.assertEqual(geom_group, same_group)
            self.assertGreaterEqual(len(geom_group), 8)

    def test_failed_trace_uses_collision_free_tcp_preserving_chain(self):
        class WorldPose:
            def to_transformation_matrix(self):
                pose = I.copy()
                pose[:3, :3] = t3d.quaternions.quat2mat(
                    [0.707, 0.0, 0.0, 0.707]
                )
                pose[:3, 3] = [0.0, -0.65, 0.0]
                return pose

        class RealRobot(Robot):
            left_entity_origion_pose = WorldPose()
            right_entity_origion_pose = WorldPose()

            def get_left_arm_real_jointState(self):
                return np.r_[np.zeros(6), 1.0]

            def get_right_arm_real_jointState(self):
                return np.r_[np.zeros(6), 1.0]

        class RealEnv:
            robot = RealRobot()

        rotation = np.array(
            [
                [0.0580159845, 0.9808343830, 0.1860055338],
                [-0.3003783012, 0.1948358701, -0.9337086590],
                [-0.9520541065, -0.0017019992, 0.3059249606],
            ]
        )
        poses = []
        for position in (
            [0.0948227233, -0.0688552869, 0.9386393998],
            [0.0977235225, -0.0838742020, 0.8910366944],
            [0.0977235225, -0.0838742020, 0.9410366944],
        ):
            pose = I.copy()
            pose[:3, :3] = rotation
            pose[:3, 3] = position
            poses.append(pose)
        model_path = (
            Path(__file__).resolve().parents[3]
            / "assets"
            / "embodiments"
            / "aloha-agilex"
            / "urdf"
            / "arx5_description_isaac.urdf"
        )
        ik = RoboTwinMinkIK(
            RealEnv(),
            I,
            model_path=model_path,
            max_waypoints_per_segment=64,
            relax_orientation_on_failure=True,
        )

        joints = [ik.solve("right", pose) for pose in poses]

        self.assertTrue(all(item is not None for item in joints))
        self.assertGreaterEqual(ik.failures.get("SelfCollision", 0), 1)
        self.assertEqual(ik.relaxed_successes, 1)
        previous = np.zeros(6)
        for target in joints:
            path = ik.consume_path("right", target)
            self.assertFalse(ik._path_has_self_collision("right", path))
            self.assertLessEqual(
                np.max(np.abs(np.diff(np.vstack((previous, path)), axis=0))),
                0.12 + 1e-12,
            )
            previous = target
        accepted = ik.selected_grasp_command_pose
        np.testing.assert_allclose(
            accepted[:3, 3] + 0.12 * accepted[:3, 0],
            poses[1][:3, 3] + 0.12 * poses[1][:3, 0],
        )

    def test_higher_confidence_wins_regardless_of_orientation(self):
        axis_map = M2T2_TO_ROBOTWIN[:3, :3]
        canonical_command = t3d.quaternions.quat2mat(
            CANONICAL_COMMAND_QUATERNIONS["right"]
        )
        quarter_turn = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        canonical_pose = I.copy()
        canonical_pose[:3, :3] = canonical_command @ axis_map.T
        sideways_pose = I.copy()
        sideways_pose[:3, :3] = canonical_command @ quarter_turn @ axis_map.T
        candidates = [
            GraspCandidate(canonical_pose, 0.40, "target"),
            GraspCandidate(sideways_pose, 0.99, "target"),
        ]
        ranker = ConfidenceRankedGrasps(
            FakeGrasps(candidates), min_confidence=0.0
        )
        target = ObjectState("target", I, 1)

        ranked = ranker.propose(None, target)

        self.assertEqual([item.confidence for item in ranked], [0.99, 0.40])
        np.testing.assert_allclose(
            ranked[0].world_grasp_pose[:3, :3], sideways_pose[:3, :3]
        )
        self.assertEqual(len(ranker.last_candidates), 2)
        self.assertIs(ranker.last_candidates[0], ranked[0])

    def test_equal_confidence_preserves_generator_order(self):
        first_pose = I.copy()
        second_pose = I.copy()
        second_pose[:3, :3] = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        candidates = [
            GraspCandidate(first_pose, 0.75, "target"),
            GraspCandidate(second_pose, 0.75, "target"),
        ]
        ranker = ConfidenceRankedGrasps(FakeGrasps(candidates))
        target_pose = I.copy()
        target_pose[2, 3] = 0.1034
        target = ObjectState("target", target_pose, 1)

        ranked = ranker.propose(None, target)

        np.testing.assert_allclose(ranked[0].world_grasp_pose, first_pose)
        np.testing.assert_allclose(ranked[1].world_grasp_pose, second_pose)
        self.assertEqual([item.confidence for item in ranked], [0.75, 0.75])

    def test_m2t2_wireframe_projects_with_robotwin_cv_camera(self):
        poses = np.repeat(I[None, :, :], 2, axis=0)
        poses[0, :3, 3] = [0.0, 0.0, 1.0]
        poses[1, :3, 3] = [0.1, -0.2, 1.5]

        wireframes = _grasp_wireframes(poses)

        self.assertEqual(wireframes.shape, (2, 7, 3))
        np.testing.assert_allclose(
            wireframes[0], M2T2_GRIPPER_POLYLINE + [0.0, 0.0, 1.0]
        )
        intrinsic = np.array(
            [[200.0, 0.0, 160.0], [0.0, 200.0, 120.0], [0.0, 0.0, 1.0]]
        )
        extrinsic = np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)
        pixels, camera = _project_world_points_cv(
            np.array([[[0.0, 0.0, 1.0], [0.1, 0.2, 1.0]]]),
            intrinsic,
            extrinsic,
        )
        np.testing.assert_allclose(pixels[0], [[160.0, 120.0], [180.0, 160.0]])
        np.testing.assert_allclose(camera[0, :, 2], [1.0, 1.0])

        pixels_4x4, _ = _project_world_points_cv(
            np.array([[[0.0, 0.0, 1.0]]]), intrinsic, I
        )
        np.testing.assert_allclose(pixels_4x4[0, 0], [160.0, 120.0])
        behind, _ = _project_world_points_cv(
            np.array([[[0.0, 0.0, 0.05]]]), intrinsic, extrinsic
        )
        self.assertTrue(np.isnan(behind).all())

    def test_grasp_visualization_writes_png(self):
        center = np.array([0.0, 0.0, 0.80])
        offsets = np.linspace(-0.025, 0.025, 5)
        target_points = np.array(
            [center + [x, y, z] for x in offsets for y in offsets for z in offsets]
        )
        context_points = np.array([[0.35, 0.20, 1.10], [-0.30, -0.15, 0.95]])
        points = np.concatenate((target_points, context_points), axis=0)
        colors = np.concatenate(
            (
                np.tile([[0.15, 0.65, 0.95]], (len(target_points), 1)),
                np.array([[0.95, 0.20, 0.10], [0.10, 0.90, 0.25]]),
            ),
            axis=0,
        )
        labels = np.concatenate(
            (np.full(len(target_points), 7), np.full(len(context_points), -1))
        )
        target_pose = I.copy()
        target_pose[:3, 3] = center
        target = ObjectState("bottle", target_pose, 7)
        scene = SceneObservation(
            xyz=points,
            rgb=colors,
            instance_labels=labels,
            camera_pose=I,
            objects={"bottle": target},
        )
        candidates = []
        for index, angle in enumerate(np.linspace(-1.2, 1.2, 73)):
            rotation_z = np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0.0],
                    [np.sin(angle), np.cos(angle), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            pose = I.copy()
            pose[:3, :3] = rotation_z
            pose[:3, 3] = center + [
                0.025 * np.cos(angle),
                0.025 * np.sin(angle),
                0.0,
            ]
            candidates.append(
                GraspCandidate(pose, 1.0 - index / 100.0, "bottle")
            )

        camera_rgb = np.zeros((240, 320, 3), dtype=np.uint8)
        camera_rgb[..., 0] = np.linspace(25, 80, 320, dtype=np.uint8)
        camera_rgb[..., 1] = 35
        camera_rgb[..., 2] = np.linspace(70, 20, 240, dtype=np.uint8)[:, None]
        intrinsic = np.array(
            [[250.0, 0.0, 160.0], [0.0, 250.0, 120.0], [0.0, 0.0, 1.0]]
        )
        extrinsic = np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)
        selected_index = 36

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "grasp_viz.png"
            raw_trace = {
                "poses": np.asarray(
                    [candidate.world_grasp_pose for candidate in candidates]
                ),
                "scores": np.asarray(
                    [candidate.confidence for candidate in candidates]
                ),
                "contacts": np.tile(center, (len(candidates), 1)),
                "target_contacts": np.ones(len(candidates), dtype=bool),
                "query_ids": np.zeros((len(candidates), 2), dtype=int),
            }
            executed_command = (
                candidates[selected_index].world_grasp_pose @ M2T2_TO_ROBOTWIN
            )
            saved = save_grasp_visualization(
                output,
                scene,
                target,
                candidates,
                candidates[selected_index],
                arm="right",
                grasp_to_robotwin=M2T2_TO_ROBOTWIN,
                camera_rgb=camera_rgb,
                camera_intrinsic=intrinsic,
                camera_extrinsic=extrinsic,
                executed_command_pose=executed_command,
                raw_trace=raw_trace,
            )

            self.assertEqual(saved, output)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 10_000)
            self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            from PIL import Image

            rendered = np.asarray(Image.open(output).convert("RGB"))
            self.assertEqual(rendered.shape, (960, 1280, 3))
            self.assertTrue(
                np.any(np.all(rendered == np.array([255, 45, 149]), axis=2))
            )

            data_path = output.with_suffix(".npz")
            self.assertTrue(data_path.is_file())
            with np.load(data_path, allow_pickle=False) as data:
                self.assertEqual(data["raw_world_grasp_poses"].shape, (73, 4, 4))
                self.assertEqual(data["ranked_world_grasp_poses"].shape, (73, 4, 4))
                self.assertEqual(
                    data["rendered_grasp_wireframes_world"].shape, (73, 7, 3)
                )
                self.assertEqual(
                    data["projected_grasp_polylines"].shape, (73, 7, 2)
                )
                self.assertTrue(np.all(data["grasp_edges_drawn"] == 6))
                self.assertEqual(
                    data["selected_grasp_wireframe_world"].shape, (1, 7, 3)
                )
                self.assertEqual(int(data["selected_edges_drawn"]), 6)
                self.assertEqual(
                    int(data["selected_grasp_index"]), selected_index
                )
                self.assertEqual(
                    str(data["selected_grasp_color"]), SELECTED_GRASP_COLOR
                )
                np.testing.assert_array_equal(data["camera_rgb"], camera_rgb)
                np.testing.assert_allclose(data["camera_intrinsic_cv"], intrinsic)
                np.testing.assert_allclose(data["camera_extrinsic_cv"], extrinsic)
                self.assertEqual(int(data["raw_target_contacts"].sum()), 73)

    def test_qpos_buffer_keeps_inactive_arm_and_uses_joint_waypoints(self):
        goals = [np.full(6, value) for value in (0.1, 0.2, 0.3)]
        env = Env()
        ik, _ = self.make_ik(goals)
        buffer = QposActionBuffer(env, ik, max_waypoints_per_segment=8)
        buffer.reset()
        solved = [ik.solve("right", I) for _ in range(3)]

        buffer.open_gripper("right")
        buffer.move_joints("right", solved[0])
        buffer.move_joints("right", solved[1])
        buffer.close_gripper("right")
        buffer.move_joints("right", solved[2])

        self.assertTrue(buffer.actions)
        self.assertTrue(all(action.shape == (14,) for action in buffer.actions))
        self.assertTrue(all(np.allclose(action[:6], 0.0) for action in buffer.actions))
        np.testing.assert_allclose(buffer.actions[-1][7:13], goals[-1])
        self.assertEqual(len(buffer.metadata), len(buffer.actions))
        endpoints = [item for item in buffer.metadata if item["endpoint"]]
        self.assertEqual(
            [item["phase"] for item in endpoints],
            ["open", "pregrasp", "grasp", "close", "retreat"],
        )
        self.assertTrue(
            all(item["command_pose"] is not None for item in endpoints[1:3])
        )
        self.assertIsNotNone(endpoints[-1]["command_pose"])

    def test_mink_and_qpos_buffer_prefer_measured_joint_state(self):
        class MeasuredRobot(Robot):
            def get_left_arm_real_jointState(self):
                return np.r_[np.full(6, 0.25), 0.75]

            def get_right_arm_real_jointState(self):
                return np.r_[np.full(6, -0.5), 0.60]

        class MeasuredEnv:
            robot = MeasuredRobot()

        fake = FakeSolver([])
        with patch(
            "policy.heuristic_baseline.runtime.MinkIKSolver.from_xml_path",
            return_value=fake,
        ):
            ik = RoboTwinMinkIK(MeasuredEnv(), I, model_path="unused.urdf")

        np.testing.assert_allclose(ik._joint_positions("left"), np.full(6, 0.25))
        np.testing.assert_allclose(ik._joint_positions("right"), np.full(6, -0.5))
        buffer = QposActionBuffer(MeasuredEnv(), ik)
        buffer.reset()
        np.testing.assert_allclose(buffer.left, np.full(6, 0.25))
        np.testing.assert_allclose(buffer.right, np.full(6, -0.5))
        self.assertEqual(buffer.left_gripper, 0.75)
        self.assertEqual(buffer.right_gripper, 0.60)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np
import transforms3d as t3d

from policy.heuristic_baseline.runtime import (
    BimanualQposActionBuffer,
    CANONICAL_COMMAND_QUATERNIONS,
    ConfidenceRankedGrasps,
    M2T2_TO_ROBOTWIN,
    M2T2_GRIPPER_POLYLINE,
    QposActionBuffer,
    RoboTwinHeuristicRuntime,
    RoboTwinMinkIK,
    SELECTED_GRASP_COLOR,
    _BimanualArmPlan,
    _aloha_self_collision_config,
    _elongated_object_axis,
    _grasp_wireframes,
    _project_world_points_cv,
    _rigid_transport_command_pose,
    _robot_facing_grasp_pose,
    _target_width_along_axis,
    save_grasp_visualization,
)
from policy.heuristic_baseline.errors import NoFeasiblePlanFailure
from policy.heuristic_baseline.task_plan import Pick, TaskPlan
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

    def test_canonical_bridge_ends_at_exact_selected_orientation(self):
        bridge = np.full(6, -0.10)
        exact_pregrasp = np.full(6, 0.20)
        exact_grasp = np.full(6, 0.25)
        exact_retreat = np.full(6, 0.30)
        fake = FakeSolver(
            [None, bridge, exact_pregrasp, exact_grasp, exact_retreat]
        )
        with patch(
            "policy.heuristic_baseline.runtime.MinkIKSolver.from_xml_path",
            return_value=fake,
        ):
            ik = RoboTwinMinkIK(
                Env(),
                I,
                model_path="unused.urdf",
                max_joint_step_rad=0.1,
                relax_orientation_on_failure=False,
                canonical_seed_on_failure=True,
            )

        actual = [ik.solve("right", I) for _ in range(3)]

        np.testing.assert_allclose(actual[0], exact_pregrasp)
        np.testing.assert_allclose(actual[1], exact_grasp)
        np.testing.assert_allclose(actual[2], exact_retreat)
        self.assertEqual(len(fake.calls), 5)
        np.testing.assert_allclose(fake.calls[0][1], I)
        np.testing.assert_allclose(fake.calls[2][1], I)
        np.testing.assert_allclose(
            fake.calls[1][1][:3, :3],
            t3d.quaternions.quat2mat(
                CANONICAL_COMMAND_QUATERNIONS["right"]
            ),
        )
        self.assertEqual(ik.canonical_seed_successes, 1)
        self.assertEqual(ik.relaxed_successes, 0)
        np.testing.assert_allclose(ik.selected_grasp_command_pose, I)
        pregrasp_path = ik.consume_path("right", exact_pregrasp)
        self.assertGreaterEqual(len(pregrasp_path), 2)
        self.assertTrue(np.all(pregrasp_path >= 0.0))


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

    def test_full_robot_collision_check_catches_cross_arm_collision(self):
        model_path = (
            Path(__file__).resolve().parents[3]
            / "assets"
            / "embodiments"
            / "aloha-agilex"
            / "urdf"
            / "arx5_description_isaac.urdf"
        )
        ik = RoboTwinMinkIK(Env(), I, model_path=model_path)
        # Each arm is collision-free in isolation at this configuration, but
        # fl_link4 penetrates fr_link6 when both states are composed.
        paired_joints = np.array(
            [
                -1.0340855482545677,
                2.0142694645012504,
                1.8972693918087504,
                0.5356564445993564,
                0.4042366364583674,
                2.9317180441036665,
                0.2117960552245357,
                -3.088287664171324,
                -1.820309030507989,
                1.2497323938933018,
                1.2763808736218287,
                2.694677242155173,
            ],
            dtype=np.float64,
        )

        self.assertFalse(
            ik._path_has_self_collision("left", paired_joints[None, :6])
        )
        self.assertFalse(
            ik._path_has_self_collision("right", paired_joints[None, 6:])
        )
        full_action = np.r_[
            paired_joints[:6], 0.0, paired_joints[6:], 0.0
        ]
        self.assertTrue(ik.full_robot_path_has_self_collision([full_action]))

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

    def test_confidence_ranking_preserves_off_origin_world_grasp_pose(self):
        target_pose = I.copy()
        target_pose[:3, 3] = [0.31, -0.24, 0.82]
        low_pose = I.copy()
        low_pose[:3, 3] = [-0.18, 0.27, 1.04]
        high_pose = I.copy()
        high_pose[:3, :3] = np.array(
            [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]
        )
        high_pose[:3, 3] = [0.07, -0.11, 0.96]
        candidates = [
            GraspCandidate(low_pose, 0.51, "target"),
            GraspCandidate(high_pose, 0.97, "target"),
        ]
        ranker = ConfidenceRankedGrasps(FakeGrasps(candidates))
        target = ObjectState("target", target_pose, 1)

        ranked = ranker.propose(None, target)

        self.assertEqual([item.confidence for item in ranked], [0.97, 0.51])
        np.testing.assert_array_equal(ranked[0].world_grasp_pose, high_pose)
        np.testing.assert_array_equal(ranked[1].world_grasp_pose, low_pose)

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

    def test_elongated_object_axis_uses_only_target_labeled_points(self):
        expected_axis = np.array([2.0, -1.0, 0.5], dtype=np.float64)
        expected_axis /= np.linalg.norm(expected_axis)
        center = np.array([0.13, -0.21, 0.84])
        target_points = (
            center
            + np.linspace(-0.12, 0.12, 9)[:, None] * expected_axis
        )
        distractor_points = np.column_stack(
            (
                np.full(17, -0.35),
                np.linspace(-4.0, 4.0, 17),
                np.full(17, 1.20),
            )
        )
        target_pose = I.copy()
        target_pose[:3, 3] = center
        target = ObjectState("target", target_pose, 7)

        def make_scene(points, labels):
            return SceneObservation(
                xyz=np.asarray(points, dtype=np.float64),
                rgb=np.zeros((len(points), 3), dtype=np.float64),
                instance_labels=np.asarray(labels, dtype=np.int64),
                camera_pose=I,
                objects={"target": target},
            )

        target_only = make_scene(target_points, np.full(len(target_points), 7))
        mixed = make_scene(
            np.vstack((target_points, distractor_points)),
            np.r_[
                np.full(len(target_points), 7),
                np.full(len(distractor_points), 99),
            ],
        )

        target_axis = _elongated_object_axis(target_only, target)
        mixed_axis = _elongated_object_axis(mixed, target)

        self.assertIsNotNone(target_axis)
        self.assertIsNotNone(mixed_axis)
        self.assertAlmostEqual(float(np.linalg.norm(mixed_axis)), 1.0, places=12)
        self.assertAlmostEqual(
            abs(float(np.dot(mixed_axis, expected_axis))), 1.0, places=12
        )
        self.assertAlmostEqual(
            abs(float(np.dot(mixed_axis, target_axis))), 1.0, places=12
        )

    def test_elongated_object_axis_rejects_isotropic_target_cloud(self):
        center = np.array([0.13, -0.21, 0.84])
        offsets = 0.03 * np.array(
            [
                [x, y, z]
                for x in (-1.0, 1.0)
                for y in (-1.0, 1.0)
                for z in (-1.0, 1.0)
            ]
        )
        points = center + offsets
        target_pose = I.copy()
        target_pose[:3, 3] = center
        target = ObjectState("target", target_pose, 7)
        scene = SceneObservation(
            xyz=points,
            rgb=np.zeros((len(points), 3), dtype=np.float64),
            instance_labels=np.full(len(points), 7, dtype=np.int64),
            camera_pose=I,
            objects={"target": target},
        )

        self.assertIsNone(_elongated_object_axis(scene, target))

    def test_bimanual_jaw_axis_alignment_threshold_is_validated(self):
        common = dict(
            task_env=Env(),
            grasps=SimpleNamespace(backend=SimpleNamespace()),
            config=SimpleNamespace(),
            automatic_target=True,
            automatic_arm=True,
            grasp_to_robotwin=I,
            mink_model_path="unused.urdf",
        )
        for invalid in (0.0, 1.0, -0.1, 1.1, np.nan):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "bimanual planning limits"):
                    RoboTwinHeuristicRuntime(
                        **common,
                        bimanual_max_jaw_axis_alignment=invalid,
                    )

        with patch(
            "policy.heuristic_baseline.runtime.MinkIKSolver.from_xml_path",
            return_value=FakeSolver([]),
        ):
            runtime = RoboTwinHeuristicRuntime(
                **common,
                bimanual_max_jaw_axis_alignment=0.65,
            )
        self.assertEqual(runtime.bimanual_max_jaw_axis_alignment, 0.65)

    def test_target_width_uses_target_labels_and_does_not_mutate_axis(self):
        unit_axis = np.array([2.0, -1.0, 0.5], dtype=np.float64)
        unit_axis /= np.linalg.norm(unit_axis)
        input_axis = 7.5 * unit_axis
        original_axis = input_axis.copy()
        center = np.array([0.13, -0.21, 0.84])
        target_points = (
            center + np.array([-0.03, 0.01, 0.09])[:, None] * unit_axis
        )
        distractor_points = center + np.array([-5.0, 5.0])[:, None] * unit_axis
        points = np.vstack((target_points, distractor_points))
        labels = np.r_[np.full(len(target_points), 7), [99, 99]]
        target_pose = I.copy()
        target_pose[:3, 3] = center
        target = ObjectState("target", target_pose, 7)
        scene = SceneObservation(
            xyz=points,
            rgb=np.zeros((len(points), 3), dtype=np.float64),
            instance_labels=labels,
            camera_pose=I,
            objects={"target": target},
        )

        width = _target_width_along_axis(scene, target, input_axis)

        self.assertAlmostEqual(width, 0.12, places=6)
        np.testing.assert_array_equal(input_axis, original_axis)

    def test_target_width_returns_none_with_fewer_than_two_target_points(self):
        target_pose = I.copy()
        target = ObjectState("target", target_pose, 7)
        scene = SceneObservation(
            xyz=np.array(
                [[0.1, 0.2, 0.3], [-5.0, 0.0, 0.0], [5.0, 0.0, 0.0]]
            ),
            rgb=np.zeros((3, 3), dtype=np.float64),
            instance_labels=np.array([7, 99, 99]),
            camera_pose=I,
            objects={"target": target},
        )

        self.assertIsNone(
            _target_width_along_axis(scene, target, np.array([3.0, 0.0, 0.0]))
        )

    def test_bimanual_max_target_width_is_validated(self):
        common = dict(
            task_env=Env(),
            grasps=SimpleNamespace(backend=SimpleNamespace()),
            config=SimpleNamespace(),
            automatic_target=True,
            automatic_arm=True,
            grasp_to_robotwin=I,
            mink_model_path="unused.urdf",
        )
        for invalid in (0.0, -0.01, np.inf, np.nan):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "bimanual planning limits"):
                    RoboTwinHeuristicRuntime(
                        **common,
                        bimanual_max_target_width_m=invalid,
                    )

        with patch(
            "policy.heuristic_baseline.runtime.MinkIKSolver.from_xml_path",
            return_value=FakeSolver([]),
        ):
            runtime = RoboTwinHeuristicRuntime(
                **common,
                bimanual_max_target_width_m=0.085,
            )
        self.assertEqual(runtime.bimanual_max_target_width_m, 0.085)

    def test_rigid_transport_preserves_grasp_rotation_and_moves_fp_xyz(self):
        object_pose = I.copy()
        object_pose[:3, :3] = t3d.euler.euler2mat(-0.31, 0.22, 0.48)
        object_pose[:3, 3] = [0.16, -0.20, 0.79]
        current_functional = I.copy()
        current_functional[:3, :3] = t3d.euler.euler2mat(0.52, -0.18, 0.09)
        current_functional[:3, 3] = [0.19, -0.14, 0.86]
        grasp_command = I.copy()
        grasp_command[:3, :3] = t3d.euler.euler2mat(0.27, 0.41, -0.63)
        grasp_command[:3, 3] = [0.08, -0.26, 0.97]
        desired_xyz = np.array([-0.07, -0.105, 0.94])
        desired_identity = np.r_[desired_xyz, [1.0, 0.0, 0.0, 0.0]]
        desired_rotated = np.r_[desired_xyz, [0.0, 0.6, 0.8, 0.0]]
        expected_delta = desired_xyz - current_functional[:3, 3]

        transported_identity = _rigid_transport_command_pose(
            object_pose,
            current_functional,
            grasp_command,
            desired_identity,
        )
        transported_rotated = _rigid_transport_command_pose(
            object_pose,
            current_functional,
            grasp_command,
            desired_rotated,
        )

        np.testing.assert_array_equal(
            transported_identity[:3, :3], grasp_command[:3, :3]
        )
        np.testing.assert_allclose(
            transported_identity[:3, 3] - grasp_command[:3, 3],
            expected_delta,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            current_functional[:3, 3]
            + transported_identity[:3, 3]
            - grasp_command[:3, 3],
            desired_xyz,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            transported_rotated, transported_identity, atol=1e-12
        )

    def test_robot_facing_grasp_preserves_tcp_and_projected_closing_axis(self):
        grasp_pose = I.copy()
        grasp_pose[:3, :3] = t3d.euler.euler2mat(0.43, -0.28, 0.71)
        grasp_pose[:3, 3] = [0.14, -0.19, 0.91]
        current_ee = np.array([-0.27, 0.08, 0.68])
        original_command = grasp_pose @ M2T2_TO_ROBOTWIN
        original_tcp = (
            original_command[:3, 3] + 0.12 * original_command[:3, 0]
        )
        expected_approach = original_tcp - current_ee
        expected_approach /= np.linalg.norm(expected_approach)
        projected_closing = original_command[:3, 1] - np.dot(
            original_command[:3, 1], expected_approach
        ) * expected_approach
        self.assertGreater(np.linalg.norm(projected_closing), 1e-3)
        projected_closing /= np.linalg.norm(projected_closing)

        adjusted_pose = _robot_facing_grasp_pose(
            grasp_pose, current_ee, M2T2_TO_ROBOTWIN
        )
        adjusted_command = adjusted_pose @ M2T2_TO_ROBOTWIN
        adjusted_tcp = (
            adjusted_command[:3, 3] + 0.12 * adjusted_command[:3, 0]
        )

        np.testing.assert_allclose(
            adjusted_pose[:3, :3].T @ adjusted_pose[:3, :3],
            np.eye(3),
            atol=1e-12,
        )
        self.assertAlmostEqual(
            float(np.linalg.det(adjusted_pose[:3, :3])), 1.0, places=12
        )
        np.testing.assert_allclose(adjusted_pose[3], [0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(adjusted_tcp, original_tcp, atol=1e-12)
        np.testing.assert_allclose(
            adjusted_command[:3, 0], expected_approach, atol=1e-12
        )
        np.testing.assert_allclose(
            adjusted_command[:3, 1], projected_closing, atol=1e-12
        )

    def test_bimanual_buffer_synchronizes_paths_and_latches_both_grippers(self):
        def make_plan(arm, target_name, sign, lengths):
            paths = []
            start = np.zeros(6, dtype=np.float64)
            command_targets = []
            for phase_index, (endpoint_value, count) in enumerate(
                zip((0.10, 0.20, 0.30, 0.40), lengths)
            ):
                endpoint = np.full(6, sign * endpoint_value)
                paths.append(
                    np.linspace(start, endpoint, count + 1, dtype=np.float64)[1:]
                )
                start = endpoint
                command = I.copy()
                command[0, 3] = sign * (phase_index + 1) * 0.01
                command_targets.append(command)
            candidate_pose = I.copy()
            candidate_pose[0, 3] = sign * 0.10
            return _BimanualArmPlan(
                arm=arm,
                target_name=target_name,
                arm_source="robotwin_ground_truth",
                candidate=GraspCandidate(candidate_pose, 0.9, target_name),
                paths=tuple(paths),
                command_targets=tuple(command_targets),
            )

        left_plan = make_plan("left", "bottle1", -1.0, (2, 4, 1, 3))
        right_plan = make_plan("right", "bottle2", 1.0, (5, 2, 4, 1))
        buffer = BimanualQposActionBuffer(
            Env(), gripper_settle_actions=3
        )

        actions = buffer.build(left_plan, right_plan)

        self.assertTrue(actions)
        self.assertTrue(all(action.shape == (14,) for action in actions))
        self.assertEqual(len(actions), len(buffer.metadata))
        self.assertTrue(all(item["arm"] == "both" for item in buffer.metadata))
        phase_counts = {
            phase: sum(item["phase"] == phase for item in buffer.metadata)
            for phase in (
                "open",
                "pregrasp",
                "grasp",
                "close",
                "lift",
                "transport",
            )
        }
        self.assertEqual(
            phase_counts,
            {
                "open": 1,
                "pregrasp": 5,
                "grasp": 4,
                "close": 3,
                "lift": 4,
                "transport": 3,
            },
        )
        endpoints = [item for item in buffer.metadata if item["endpoint"]]
        self.assertEqual(
            [item["phase"] for item in endpoints],
            ["open", "pregrasp", "grasp", "close", "lift", "transport"],
        )
        for endpoint in endpoints[1:3] + endpoints[-2:]:
            self.assertIsNotNone(
                endpoint["arm_targets"]["left"]["command_pose"]
            )
            self.assertIsNotNone(
                endpoint["arm_targets"]["right"]["command_pose"]
            )

        first_close = next(
            index
            for index, item in enumerate(buffer.metadata)
            if item["phase"] == "close"
        )
        self.assertTrue(
            all(
                action[6] == 0.0 and action[13] == 0.0
                for action in actions[first_close:]
            )
        )
        for metadata in buffer.metadata[first_close:]:
            self.assertEqual(
                metadata["arm_targets"]["left"]["target_gripper"], 0.0
            )
            self.assertEqual(
                metadata["arm_targets"]["right"]["target_gripper"], 0.0
            )
        self.assertTrue(
            all(
                actions[index][6] == 1.0 and actions[index][13] == 1.0
                for index in range(first_close)
            )
        )
        np.testing.assert_allclose(actions[-1][:6], np.full(6, -0.40))
        np.testing.assert_allclose(actions[-1][7:13], np.full(6, 0.40))
        self.assertEqual(
            endpoints[-1]["arm_targets"]["left"]["target_name"],
            "bottle1",
        )
        self.assertEqual(
            endpoints[-1]["arm_targets"]["right"]["target_name"],
            "bottle2",
        )

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
        close_actions = [
            item for item in buffer.metadata if item["phase"] == "close"
        ]
        self.assertEqual(len(close_actions), 5)
        self.assertEqual(
            [item["endpoint"] for item in close_actions],
            [False, False, False, False, True],
        )
        self.assertEqual(
            [item["waypoint_index"] for item in close_actions],
            [1, 2, 3, 4, 5],
        )
        self.assertTrue(
            all(item["waypoint_count"] == 5 for item in close_actions)
        )
        self.assertTrue(
            all(item["target_gripper"] == 0.0 for item in close_actions)
        )

        retreat_actions = [
            item for item in buffer.metadata if item["phase"] == "retreat"
        ]
        self.assertTrue(retreat_actions)
        self.assertTrue(
            all(item["target_gripper"] == 0.0 for item in retreat_actions)
        )
        self.assertEqual(buffer.actions[-1][13], 0.0)

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

    def test_auto_arm_prefers_robotwin_ground_truth_over_geometry(self):
        pose = I.copy()
        pose[0, 3] = 0.25
        target = ObjectState("bottle", pose, 7)
        runtime = RoboTwinHeuristicRuntime.__new__(RoboTwinHeuristicRuntime)
        runtime.automatic_arm = True
        runtime.config = SimpleNamespace(arm="right")
        runtime.task_env = SimpleNamespace(
            heuristic_task_plan=TaskPlan(
                "test_task",
                "pick",
                (Pick("bottle", "left"),),
            )
        )

        self.assertEqual(
            runtime._select_arm(target),
            ("left", "robotwin_ground_truth"),
        )

    def test_auto_arm_uses_geometry_when_ground_truth_is_unavailable_or_ambiguous(self):
        pose = I.copy()
        pose[0, 3] = 0.25
        target = ObjectState("bottle", pose, 7)
        plans = (
            None,
            TaskPlan("test_task", "pick", (Pick("bottle", None),)),
            TaskPlan(
                "test_task",
                "pick",
                (Pick("bottle", "left"), Pick("bottle", "right")),
            ),
            TaskPlan("test_task", "pick", (Pick("other", "left"),)),
        )

        for plan in plans:
            with self.subTest(plan=plan):
                runtime = RoboTwinHeuristicRuntime.__new__(
                    RoboTwinHeuristicRuntime
                )
                runtime.automatic_arm = True
                runtime.config = SimpleNamespace(arm="left")
                runtime.task_env = SimpleNamespace(
                    heuristic_task_plan=plan
                )
                self.assertEqual(
                    runtime._select_arm(target),
                    ("right", "geometry_fallback"),
                )

    def test_explicit_arm_overrides_ground_truth_and_geometry(self):
        pose = I.copy()
        pose[0, 3] = 0.25
        target = ObjectState("bottle", pose, 7)
        runtime = RoboTwinHeuristicRuntime.__new__(RoboTwinHeuristicRuntime)
        runtime.automatic_arm = False
        runtime.config = SimpleNamespace(arm="left")
        runtime.task_env = SimpleNamespace(
            heuristic_task_plan=TaskPlan(
                "test_task",
                "pick",
                (Pick("bottle", "right"),),
            )
        )

        self.assertEqual(
            runtime._select_arm(target),
            ("left", "explicit_config"),
        )

    def test_runtime_rejects_second_attempt_before_planning(self):
        class SimulatorSpy:
            def __init__(self):
                self.update_calls = 0

            def update(self, _scene):
                self.update_calls += 1

        runtime = RoboTwinHeuristicRuntime.__new__(RoboTwinHeuristicRuntime)
        runtime._grasp_attempted = True
        runtime.simulator = SimulatorSpy()

        with patch(
            "policy.heuristic_baseline.runtime.SimpleGraspPolicy",
            side_effect=AssertionError("a second attempt must not plan"),
        ):
            with self.assertRaisesRegex(NoFeasiblePlanFailure, "one-shot"):
                runtime.get_action(scene=object())

        self.assertEqual(runtime.simulator.update_calls, 0)

    def test_bimanual_collision_failure_is_atomic_and_consumes_one_shot(self):
        def make_plan(arm, target_name, sign):
            candidate_pose = I.copy()
            candidate_pose[0, 3] = sign * 0.10
            candidate = GraspCandidate(candidate_pose, 0.9, target_name)
            paths = tuple(
                np.full((1, 6), sign * value, dtype=np.float64)
                for value in (0.10, 0.20, 0.30, 0.40)
            )
            return _BimanualArmPlan(
                arm=arm,
                target_name=target_name,
                arm_source="robotwin_ground_truth",
                candidate=candidate,
                paths=paths,
                command_targets=(I.copy(), I.copy(), I.copy(), I.copy()),
            )

        class SimulatorSpy:
            def __init__(self, targets):
                self.targets = targets
                self.update_calls = 0

            def update(self, _scene):
                self.update_calls += 1

            def object_state(self, name):
                return self.targets[name]

        class CollisionIK:
            def __init__(self):
                self.failures = {}
                self.checked_actions = []
                self.reset_calls = 0

            def reset_stats(self):
                self.reset_calls += 1

            def full_robot_path_has_self_collision(
                self, actions, *, max_joint_step_rad
            ):
                self.checked_actions.append(
                    ([action.copy() for action in actions], max_joint_step_rad)
                )
                return True

        left_pose = I.copy()
        left_pose[0, 3] = -0.10
        right_pose = I.copy()
        right_pose[0, 3] = 0.10
        targets = {
            "bottle1": ObjectState("bottle1", left_pose, 1),
            "bottle2": ObjectState("bottle2", right_pose, 2),
        }
        plans = {
            "left": make_plan("left", "bottle1", -1.0),
            "right": make_plan("right", "bottle2", 1.0),
        }
        runtime = RoboTwinHeuristicRuntime.__new__(RoboTwinHeuristicRuntime)
        runtime._grasp_attempted = False
        runtime._action_metadata_override = None
        runtime.simulator = SimulatorSpy(targets)
        runtime.task_env = SimpleNamespace(
            get_tracked_objects=lambda: {"bottle1": object(), "bottle2": object()}
        )
        runtime.controller = SimpleNamespace(metadata=[])
        runtime.bimanual_controller = BimanualQposActionBuffer(
            Env(), gripper_settle_actions=1
        )
        runtime.ik = CollisionIK()
        runtime.bimanual_collision_step_rad = 0.025

        def select_arm(target):
            return (
                ("left", "robotwin_ground_truth")
                if target.name == "bottle1"
                else ("right", "robotwin_ground_truth")
            )

        def plan_arm(_scene, target, *, arm, arm_source, actor):
            del target, arm_source, actor
            plan = plans[arm]
            return [plan], [plan.candidate], {}

        with patch.object(
            runtime,
            "_target_names",
            return_value=("bottle1", "bottle2"),
        ), patch.object(runtime, "_select_arm", side_effect=select_arm), patch.object(
            runtime, "_plan_bimanual_arm", side_effect=plan_arm
        ) as planner:
            with self.assertRaisesRegex(
                NoFeasiblePlanFailure,
                "all confidence-ranked bimanual grasp pairs self-collide",
            ):
                runtime.get_action(scene=object())

            self.assertTrue(runtime.grasp_attempted)
            self.assertEqual(runtime.action_metadata, [])
            self.assertIsNone(runtime._action_metadata_override)
            self.assertEqual(planner.call_count, 2)
            self.assertEqual(len(runtime.ik.checked_actions), 1)
            checked_actions, collision_step = runtime.ik.checked_actions[0]
            self.assertTrue(
                checked_actions
                and all(action.shape == (14,) for action in checked_actions)
            )
            self.assertEqual(collision_step, 0.025)

            with self.assertRaisesRegex(NoFeasiblePlanFailure, "one-shot"):
                runtime.get_action(scene=object())

        self.assertEqual(runtime.simulator.update_calls, 1)
        self.assertEqual(planner.call_count, 2)

    def test_runtime_reset_reenables_one_attempt(self):
        class ResetSpy:
            def __init__(self):
                self.calls = 0

            def reset(self):
                self.calls += 1

            def reset_stats(self):
                self.calls += 1

        class BackendSpy:
            def __init__(self):
                self.seeds = []

            def reset(self, seed):
                self.seeds.append(seed)

        runtime = RoboTwinHeuristicRuntime.__new__(RoboTwinHeuristicRuntime)
        runtime._grasp_attempted = True
        runtime.simulator = type("Simulator", (), {"scene": object()})()
        runtime.controller = ResetSpy()
        runtime.bimanual_controller = ResetSpy()
        runtime.ik = ResetSpy()
        runtime.backend = BackendSpy()
        runtime.task_env = type("Env", (), {"episode_seed": 23})()
        runtime._visualization_index = 9

        runtime.reset()

        self.assertFalse(runtime.grasp_attempted)
        self.assertIsNone(runtime.simulator.scene)
        self.assertEqual(runtime.controller.calls, 1)
        self.assertEqual(runtime.bimanual_controller.calls, 1)
        self.assertEqual(runtime.ik.calls, 1)
        self.assertEqual(runtime.backend.seeds, [23])
        self.assertEqual(runtime._visualization_index, 0)


if __name__ == "__main__":
    unittest.main()

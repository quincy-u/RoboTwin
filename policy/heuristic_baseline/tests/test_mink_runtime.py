from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
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
    M2T2_MIN_TARGET_PALM_DEPTH_M,
    M2T2_TO_ROBOTWIN,
    M2T2_GRIPPER_POLYLINE,
    QposActionBuffer,
    StagedQposActionBuffer,
    RoboTwinHeuristicRuntime,
    RoboTwinMinkIK,
    SELECTED_GRASP_COLOR,
    _BimanualArmPlan,
    _HandoffArmPlan,
    _SingleArmPlacePlan,
    _handoff_contact_regions,
    _aligned_place_reference_pose,
    _approach_roll_grasp_pose,
    _estimate_target_support_plane_z,
    _grasp_command_tcp,
    _rigid_place_command_pose,
    _aloha_self_collision_config,
    _elongated_object_axis,
    _narrow_axis_grasp_poses,
    _place_facing_grasp_pose,
    _grasp_wireframes,
    _project_world_points_cv,
    _rigid_transport_command_pose,
    _robot_facing_grasp_pose,
    _target_m2t2_palm_depth,
    _target_width_along_axis,
    _target_narrow_axis,
    save_grasp_visualization,
)
from policy.heuristic_baseline.errors import NoFeasiblePlanFailure
from policy.heuristic_baseline.task_plan import Handoff, Pick, Place, TaskPlan
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
    def test_position_relative_place_anchors_translation_not_rotation(self):
        destination_pose = t3d.affines.compose(
            [0.2, -0.1, 0.75],
            t3d.axangles.axangle2mat([0.0, 0.0, 1.0], np.pi / 3.0),
            [1.0, 1.0, 1.0],
        )

        class Actor:
            def __init__(self, matrix):
                self.matrix = matrix

            def get_pose(self):
                return SimpleNamespace(
                    to_transformation_matrix=lambda: self.matrix.copy()
                )

        source_pose = I.copy()
        source = Actor(source_pose)
        destination = Actor(destination_pose)
        runtime = object.__new__(RoboTwinHeuristicRuntime)
        runtime.task_env = SimpleNamespace(
            get_tracked_objects=lambda: {
                "object": source,
                "target_object": destination,
            }
        )
        target = ObjectState("object", source_pose, 1)
        place = Place(
            "object",
            "target_object",
            "left",
            destination_offset=(-0.13, 0.0, 0.0),
        )

        _, reference = runtime._place_reference_poses(target, place)

        np.testing.assert_allclose(reference[:3, :3], np.eye(3), atol=1e-12)
        np.testing.assert_allclose(reference[:3, 3], [0.07, -0.1, 0.75])

    def make_ik(self, results, **kwargs):
        fake = FakeSolver(results)
        with patch(
            "policy.heuristic_baseline.runtime.MinkIKSolver.from_xml_path",
            return_value=fake,
        ):
            ik = RoboTwinMinkIK(
                Env(), I, model_path="unused.urdf", max_joint_step_rad=0.1,
                **kwargs,
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

    def test_followup_canonical_seed_still_returns_exact_place_pose(self):
        bridge = np.full(6, -0.10)
        exact = np.full(6, 0.20)
        ik, fake = self.make_ik(
            [None, bridge, exact],
            canonical_seed_on_failure=True,
        )
        target = I.copy()
        target[:3, :3] = t3d.axangles.axangle2mat(
            [0.0, 0.0, 1.0], 0.35
        )
        target[:3, 3] = [0.05, -0.08, 0.82]

        result = ik.solve_command_target(
            "right", target, np.zeros(6, dtype=np.float64)
        )

        self.assertIsNotNone(result)
        joints, path, accepted = result
        np.testing.assert_allclose(joints, exact)
        np.testing.assert_allclose(path[-1], exact)
        np.testing.assert_allclose(accepted, target)
        self.assertEqual(len(fake.calls), 3)
        np.testing.assert_allclose(fake.calls[0][1], target)
        np.testing.assert_allclose(fake.calls[2][1], target)
        self.assertFalse(np.allclose(fake.calls[1][1], target))
        self.assertTrue(np.all(path >= 0.0))
        self.assertEqual(ik.canonical_seed_successes, 1)



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

    def test_aloha_collision_config_covers_arm_and_fixed_geometry(self):
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

        self.assertEqual(len(config.geom_pairs), 5)
        left_self, right_self, cross_arm, left_fixed, right_fixed = (
            config.geom_pairs
        )
        self.assertEqual(left_self[0], left_self[1])
        self.assertEqual(right_self[0], right_self[1])
        self.assertEqual(cross_arm, (left_self[0], right_self[0]))

        def body_names(geom_group):
            return {
                mujoco.mj_id2name(
                    model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    int(model.geom_bodyid[geom_id]),
                )
                or "world"
                for geom_id in geom_group
            }

        for pair, prefix in (
            (left_fixed, "fl_link"),
            (right_fixed, "fr_link"),
        ):
            movable_names = body_names(pair[0])
            fixed_names = body_names(pair[1])
            self.assertTrue(movable_names)
            self.assertTrue(
                all(
                    name.startswith(prefix) and name != f"{prefix}1"
                    for name in movable_names
                )
            )
            self.assertIn("world", fixed_names)
            self.assertTrue(
                all(
                    not name.startswith(("fl_link", "fr_link"))
                    for name in fixed_names
                )
            )

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

    def test_handoff_collision_filter_allows_only_named_body_pair(self):
        ik = RoboTwinMinkIK.__new__(RoboTwinMinkIK)
        ik._neutral_contact_depths = {}
        model = SimpleNamespace(geom_bodyid=np.array([4, 9, 12]))
        data = SimpleNamespace(
            ncon=1,
            contact=[SimpleNamespace(
                dist=-0.01, geom1=0, geom2=1
            )],
        )

        self.assertTrue(ik._has_disallowed_arm_contact(
            model, data, {4, 9, 12}
        ))
        self.assertFalse(ik._has_disallowed_arm_contact(
            model, data, {4, 9, 12}, frozenset({(4, 9)})
        ))
        self.assertTrue(ik._has_disallowed_arm_contact(
            model, data, {4, 9, 12}, frozenset({(4, 12)})
        ))

    def test_collision_checks_reject_failed_handoff_arm_world_contact(self):
        model_path = (
            Path(__file__).resolve().parents[3]
            / "assets"
            / "embodiments"
            / "aloha-agilex"
            / "urdf"
            / "arx5_description_isaac.urdf"
        )
        ik = RoboTwinMinkIK(Env(), I, model_path=model_path)
        giver_transport = np.array(
            [
                -0.6095460362515137,
                1.7238051558415757,
                1.1811273449350868,
                0.4402887769782051,
                0.1732808011187723,
                0.12696789194998104,
            ]
        )
        failed_receiver_pregrasp = np.array(
            [
                -2.5809608189136912,
                -0.7349881436853356,
                0.8964625949136673,
                1.4742128555279983,
                0.2642966092378525,
                -0.0439599438235625,
            ]
        )

        # qpos0 contains intentional shoulder/base overlap; it stays valid.
        self.assertFalse(
            ik._path_has_self_collision("right", np.zeros((1, 6)))
        )
        self.assertFalse(
            ik.full_robot_path_has_self_collision([np.zeros(14)])
        )
        # This exact live-run target adds world--fr_link2 penetration of
        # -0.03281 m, which stalled SAPIEN while the old arm-only check passed.
        self.assertTrue(
            ik._path_has_self_collision(
                "right", failed_receiver_pregrasp[None, :]
            )
        )
        failed_action = np.r_[
            giver_transport,
            0.0,
            failed_receiver_pregrasp,
            1.0,
        ]
        self.assertTrue(
            ik.full_robot_path_has_self_collision([failed_action])
        )

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
    def test_gt_axis_skips_thinnest_axis_parallel_to_final_approach(self):
        world_object = I.copy()
        world_object[:3, :3] = t3d.euler.euler2mat(0.29, -0.36, 0.47)
        world_object[:3, 3] = [0.14, -0.16, 0.78]
        local_points = np.array(
            [
                [x, y, z]
                for x in (-0.04, 0.0, 0.04)
                for y in (-0.01, 0.0, 0.01)
                for z in (-0.09, 0.0, 0.09)
            ],
            dtype=np.float64,
        )
        target_points = (
            world_object[:3, 3]
            + local_points @ world_object[:3, :3].T
        )
        distractors = world_object[:3, 3] + np.array(
            [[-4.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
        ) @ world_object[:3, :3].T
        target = ObjectState("target", world_object, 7)
        scene = SceneObservation(
            xyz=np.vstack((target_points, distractors)),
            rgb=np.zeros((len(target_points) + len(distractors), 3)),
            instance_labels=np.r_[
                np.full(len(target_points), 7),
                np.full(len(distractors), 99),
            ],
            camera_pose=I,
            objects={"target": target},
        )
        desired_object = I.copy()
        desired_object[:3, :3] = t3d.euler.euler2mat(-0.41, 0.23, 0.62)
        desired_object[:3, 3] = [-0.07, -0.05, 0.74]
        final_approach = desired_object[:3, 1].copy()
        original_approach = final_approach.copy()

        selected = _target_narrow_axis(
            scene,
            target,
            desired_object,
            final_approach,
            maximum_approach_alignment=0.1,
        )

        np.testing.assert_array_equal(final_approach, original_approach)
        np.testing.assert_allclose(selected, world_object[:3, 0], atol=1e-12)
        self.assertAlmostEqual(
            _target_width_along_axis(scene, target, world_object[:3, 1]),
            0.02,
            places=6,
        )
        self.assertAlmostEqual(
            _target_width_along_axis(scene, target, selected),
            0.08,
            places=6,
        )


    def test_narrow_place_facing_preserves_tcp_and_uses_target_width(self):
        shape_rotation = t3d.euler.euler2mat(0.31, -0.42, 0.57)
        center = np.array([0.16, -0.18, 0.79])
        local_points = np.array(
            [
                [x, y, z]
                for x in (-0.09, 0.0, 0.09)
                for y in (-0.04, 0.0, 0.04)
                for z in (-0.012, 0.0, 0.012)
            ],
            dtype=np.float64,
        )
        target_points = center + local_points @ shape_rotation.T
        distractors = center + np.array([-5.0, 5.0])[:, None] * shape_rotation[:, 2]
        world_object = I.copy()
        world_object[:3, :3] = shape_rotation
        world_object[:3, 3] = center
        target = ObjectState("shoe", world_object, 7)
        scene = SceneObservation(
            xyz=np.vstack((target_points, distractors)),
            rgb=np.zeros((len(target_points) + len(distractors), 3)),
            instance_labels=np.r_[
                np.full(len(target_points), 7),
                np.full(len(distractors), 99),
            ],
            camera_pose=I,
            objects={"shoe": target},
        )
        desired_object = I.copy()
        desired_object[:3, :3] = t3d.euler.euler2mat(0.48, -0.17, 0.62)
        desired_object[:3, 3] = [-0.08, -0.06, 0.75]
        object_delta = desired_object @ np.linalg.inv(world_object)

        grasp_pose = I.copy()
        grasp_pose[:3, :3] = t3d.euler.euler2mat(0.27, -0.51, 0.38)
        grasp_pose[:3, 3] = [0.11, -0.23, 0.88]
        original_command = grasp_pose @ M2T2_TO_ROBOTWIN
        original_tcp = (
            original_command[:3, 3] + 0.12 * original_command[:3, 0]
        )
        final_tcp = (
            object_delta[:3, :3] @ original_tcp + object_delta[:3, 3]
        )
        expected_approach = object_delta[:3, :3] @ shape_rotation[:, 0]
        expected_approach /= np.linalg.norm(expected_approach)
        arm_reference = final_tcp - 0.35 * expected_approach

        narrow_axis = _target_narrow_axis(
            scene, target, desired_object, expected_approach
        )
        self.assertIsNotNone(narrow_axis)
        self.assertAlmostEqual(
            abs(float(np.dot(narrow_axis, shape_rotation[:, 2]))),
            1.0,
            places=12,
        )
        adjusted_pose = _place_facing_grasp_pose(
            grasp_pose,
            world_object,
            desired_object,
            arm_reference,
            M2T2_TO_ROBOTWIN,
            world_closing_axis=narrow_axis,
        )
        adjusted_command = adjusted_pose @ M2T2_TO_ROBOTWIN
        adjusted_tcp = (
            adjusted_command[:3, 3] + 0.12 * adjusted_command[:3, 0]
        )
        final_command = object_delta @ adjusted_command
        final_command_tcp = (
            final_command[:3, 3] + 0.12 * final_command[:3, 0]
        )
        expected_closing = object_delta[:3, :3] @ narrow_axis
        transported_raw = object_delta[:3, :3] @ original_command[:3, 1]
        if np.dot(expected_closing, transported_raw) < 0.0:
            expected_closing = -expected_closing
        expected_closing -= (
            np.dot(expected_closing, expected_approach) * expected_approach
        )
        expected_closing /= np.linalg.norm(expected_closing)

        np.testing.assert_allclose(adjusted_tcp, original_tcp, atol=1e-12)
        np.testing.assert_allclose(final_command_tcp, final_tcp, atol=1e-12)
        np.testing.assert_allclose(
            final_command[:3, 0], expected_approach, atol=1e-12
        )
        np.testing.assert_allclose(
            final_command[:3, 1], expected_closing, atol=1e-12
        )
        self.assertAlmostEqual(
            _target_width_along_axis(scene, target, adjusted_command[:3, 1]),
            0.024,
            places=6,
        )
        self.assertLess(
            _target_width_along_axis(scene, target, adjusted_command[:3, 1]),
            0.10,
        )


    def test_narrow_axis_sweep_preserves_tcp_jaw_and_seed_directions(self):
        grasp_pose = I.copy()
        grasp_pose[:3, :3] = t3d.euler.euler2mat(0.24, -0.37, 0.51)
        grasp_pose[:3, 3] = [0.12, -0.17, 0.86]
        world_object = I.copy()
        world_object[:3, :3] = t3d.euler.euler2mat(-0.13, 0.28, -0.41)
        world_object[:3, 3] = [0.16, -0.12, 0.77]
        desired_object = I.copy()
        desired_object[:3, :3] = t3d.euler.euler2mat(0.45, -0.21, 0.63)
        desired_object[:3, 3] = [-0.06, -0.03, 0.74]
        narrow = np.array([0.18, -0.91, 0.37])
        narrow /= np.linalg.norm(narrow)
        arm_reference = np.array([-0.31, 0.04, 0.69])
        canonical_final = np.array([0.61, 0.29, -0.74])
        canonical_final /= np.linalg.norm(canonical_final)

        poses = _narrow_axis_grasp_poses(
            grasp_pose,
            world_object,
            desired_object,
            arm_reference,
            M2T2_TO_ROBOTWIN,
            narrow,
            canonical_final,
            max_approaches=10,
        )

        self.assertGreaterEqual(len(poses), 8)
        self.assertLessEqual(len(poses), 10)
        raw_command = grasp_pose @ M2T2_TO_ROBOTWIN
        raw_tcp = raw_command[:3, 3] + 0.12 * raw_command[:3, 0]
        object_delta = desired_object @ np.linalg.inv(world_object)
        expected_final_tcp = (
            object_delta[:3, :3] @ raw_tcp + object_delta[:3, 3]
        )
        approaches = []
        for pose in poses:
            command = pose @ M2T2_TO_ROBOTWIN
            tcp = command[:3, 3] + 0.12 * command[:3, 0]
            final_command = object_delta @ command
            final_tcp = (
                final_command[:3, 3] + 0.12 * final_command[:3, 0]
            )
            np.testing.assert_allclose(tcp, raw_tcp, atol=1e-12)
            np.testing.assert_allclose(
                final_tcp, expected_final_tcp, atol=1e-12
            )
            self.assertAlmostEqual(
                abs(float(np.dot(command[:3, 1], narrow))), 1.0, places=12
            )
            self.assertAlmostEqual(
                float(np.dot(command[:3, 0], command[:3, 1])),
                0.0,
                places=12,
            )
            approaches.append(command[:3, 0])
        for index, approach in enumerate(approaches):
            self.assertTrue(
                all(
                    np.linalg.norm(approach - other) > 1e-4
                    for other in approaches[:index]
                )
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

    def test_rigid_place_preserves_attachment_and_aligns_functional_frame(self):
        world_object = I.copy()
        world_object[:3, :3] = t3d.axangles.axangle2mat(
            [0.0, 0.0, 1.0], 0.4
        )
        world_object[:3, 3] = [0.2, -0.1, 0.75]
        object_from_source = I.copy()
        object_from_source[:3, :3] = t3d.axangles.axangle2mat(
            [1.0, 0.0, 0.0], np.pi / 2.0
        )
        object_from_source[:3, 3] = [0.01, 0.03, -0.02]
        object_from_gripper = I.copy()
        object_from_gripper[:3, :3] = t3d.axangles.axangle2mat(
            [0.0, 1.0, 0.0], -0.3
        )
        object_from_gripper[:3, 3] = [-0.04, 0.02, 0.12]
        world_source = world_object @ object_from_source
        world_gripper = world_object @ object_from_gripper
        desired_source = I.copy()
        desired_source[:3, :3] = t3d.axangles.axangle2mat(
            [0.0, 0.0, 1.0], -0.8
        )
        desired_source[:3, 3] = [-0.05, -0.08, 0.74]

        desired_object, desired_gripper = _rigid_place_command_pose(
            world_object,
            world_source,
            world_gripper,
            desired_source,
        )

        np.testing.assert_allclose(
            desired_object @ object_from_source,
            desired_source,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.linalg.inv(desired_object) @ desired_gripper,
            object_from_gripper,
            atol=1e-12,
        )

    def test_auto_place_alignment_makes_can_upright(self):
        source = I.copy()
        source[:3, 3] = [0.25, 0.10, 0.75]
        destination = I.copy()
        destination[:3, 3] = [0.18, 0.0, 0.741]
        grasp = I.copy()
        grasp[:3, 3] = source[:3, 3] - [0.10, 0.0, 0.0]

        desired_reference = _aligned_place_reference_pose(
            source,
            destination,
            grasp,
            arm="right",
            constrain="auto",
            z_transform=True,
        )
        desired_object, _ = _rigid_place_command_pose(
            source,
            source,
            grasp,
            desired_reference,
        )
        roll, pitch, _ = np.degrees(
            t3d.euler.mat2euler(desired_object[:3, :3])
        )

        self.assertAlmostEqual(abs(roll), 90.0, places=5)
        self.assertAlmostEqual(pitch, 0.0, places=5)
        np.testing.assert_allclose(
            desired_object[:3, 3], destination[:3, 3], atol=1e-7
        )

    def test_staged_buffer_releases_selected_arm_after_place(self):
        buffer = StagedQposActionBuffer(
            Env(), max_waypoints_per_segment=8
        )
        buffer.reset()
        names = {"right": "shoe"}
        sources = {"right": "robotwin_ground_truth"}
        buffer.gripper_phase("open", {"right": 1.0}, 1, names, sources)
        start = np.zeros(6)
        paths = []
        commands = []
        for index in range(6):
            goal = np.full(6, 0.05 * (index + 1))
            paths.append(np.linspace(start, goal, 3)[1:])
            command = I.copy()
            command[2, 3] = 0.75 + 0.01 * index
            commands.append(command)
            start = goal

        for phase, index in (("pregrasp", 0), ("grasp", 1)):
            buffer.move_phase(
                phase, {"right": paths[index]},
                {"right": commands[index]}, names, sources
            )
        buffer.gripper_phase("close", {"right": 0.0}, 3, names, sources)
        for phase, index in (
            ("lift", 2), ("preplace", 3), ("place", 4)
        ):
            buffer.move_phase(
                phase, {"right": paths[index]},
                {"right": commands[index]}, names, sources
            )
        buffer.gripper_phase("open", {"right": 1.0}, 2, names, sources)
        buffer.move_phase(
            "retreat", {"right": paths[5]},
            {"right": commands[5]}, names, sources
        )

        self.assertTrue(all(action.shape == (14,) for action in buffer.actions))
        self.assertTrue(all(np.allclose(action[:6], 0.0) for action in buffer.actions))
        self.assertTrue(all(action[6] == 1.0 for action in buffer.actions))
        endpoints = [item for item in buffer.metadata if item["endpoint"]]
        self.assertEqual(
            [item["phase"] for item in endpoints],
            [
                "open", "pregrasp", "grasp", "close", "lift",
                "preplace", "place", "open", "retreat",
            ],
        )
        gripper_records = [
            item for item in buffer.metadata if "gripper_arms" in item
        ]
        self.assertTrue(gripper_records)
        self.assertTrue(
            all(item["gripper_arms"] == ["right"] for item in gripper_records)
        )
        close_index = next(
            index for index, item in enumerate(buffer.metadata)
            if item["phase"] == "close"
        )
        release_index = next(
            index for index, item in enumerate(buffer.metadata)
            if item["phase"] == "open" and index > close_index
        )
        self.assertTrue(
            all(action[13] == 0.0 for action in buffer.actions[
                close_index:release_index
            ])
        )
        self.assertTrue(
            all(action[13] == 1.0 for action in buffer.actions[release_index:])
        )

    def test_m2t2_logical_tcp_recovers_predicted_contact(self):
        grasp = I.copy()
        grasp[:3, :3] = t3d.euler.euler2mat(0.2, -0.3, 0.4)
        grasp[:3, 3] = [0.12, -0.08, 0.71]

        actual = _grasp_command_tcp(grasp, M2T2_TO_ROBOTWIN)

        np.testing.assert_allclose(
            actual,
            grasp[:3, 3] + 0.1034 * grasp[:3, 2],
            atol=1e-12,
        )

    def test_handoff_dispatch_is_structural_across_module_namespaces(self):
        foreign_pick = SimpleNamespace(target="box", arm="left")
        foreign_handoff = SimpleNamespace(
            object="box",
            from_arm="left",
            to_arm="right",
            rendezvous_pose_attr="block_middle_pose",
        )
        foreign_place = SimpleNamespace(
            object="box", destination="target_box", arm="right"
        )
        env = SimpleNamespace(
            heuristic_task_plan=SimpleNamespace(
                stages=(foreign_pick, foreign_handoff, foreign_place)
            )
        )

        actual = RoboTwinHeuristicRuntime._handoff_stages(env)

        self.assertEqual(
            actual, (foreign_pick, foreign_handoff, foreign_place)
        )

    def test_handoff_buffer_transfers_ownership_and_releases_receiver(self):
        def make_plan(
            arm, role, confidence, local_point, path_count, gripper_target
        ):
            sign = -1.0 if arm == "left" else 1.0
            paths = tuple(
                np.full((2, 6), sign * 0.05 * (index + 1))
                for index in range(path_count)
            )
            commands = []
            for index in range(path_count):
                command = I.copy()
                command[0, 3] = sign * 0.02 * (index + 1)
                commands.append(command)
            return _HandoffArmPlan(
                arm=arm,
                role=role,
                target_name="box",
                arm_source="robotwin_ground_truth",
                candidate=GraspCandidate(I.copy(), confidence, "box"),
                paths=paths,
                command_targets=tuple(commands),
                contact_local_point=local_point,
                gripper_target=gripper_target,
            )

        giver = make_plan(
            "left", "giver", 0.9, (-0.02, 0.01, 0.07), 5, 0.25
        )
        receiver = make_plan(
            "right", "receiver", 0.8, (0.03, -0.01, -0.07), 4, 0.35
        )
        runtime = RoboTwinHeuristicRuntime.__new__(RoboTwinHeuristicRuntime)
        runtime.staged_controller = StagedQposActionBuffer(
            Env(), max_waypoints_per_segment=8
        )
        runtime.gripper_settle_actions = 2

        actions = runtime._build_handoff_action_pair(
            giver, receiver, release=True
        )
        metadata = runtime.staged_controller.metadata

        self.assertTrue(actions and all(row.shape == (14,) for row in actions))
        endpoints = [record for record in metadata if record["endpoint"]]
        self.assertEqual(
            [record["phase"] for record in endpoints],
            [
                "open",
                "pregrasp",
                "grasp",
                "close",
                "lift",
                "transport",
                "pregrasp",
                "grasp",
                "close",
                "open",
                "retreat",
                "preplace",
                "place",
                "open",
            ],
        )
        gripper_events = [
            (record["phase"], tuple(record["gripper_arms"]))
            for record in endpoints
            if "gripper_arms" in record
        ]
        self.assertEqual(
            gripper_events,
            [
                ("open", ("left", "right")),
                ("close", ("left",)),
                ("close", ("right",)),
                ("open", ("left",)),
                ("open", ("right",)),
            ],
        )
        receiver_close = next(
            index for index, record in enumerate(metadata)
            if record.get("gripper_arms") == ["right"]
            and record["phase"] == "close"
        )
        giver_release = next(
            index for index, record in enumerate(metadata)
            if record.get("gripper_arms") == ["left"]
            and record["phase"] == "open"
        )
        receiver_release = next(
            index for index, record in enumerate(metadata)
            if record.get("gripper_arms") == ["right"]
            and record["phase"] == "open"
        )
        self.assertTrue(
            all(row[6] == 0.25 and row[13] == 0.35
                for row in actions[receiver_close:giver_release])
        )
        self.assertTrue(
            all(row[6] == 1.0 and row[13] == 0.35
                for row in actions[giver_release:receiver_release])
        )
        self.assertEqual(actions[-1][6], 1.0)
        self.assertEqual(actions[-1][13], 1.0)
        self.assertEqual(giver.contact_local_point, (-0.02, 0.01, 0.07))
        self.assertEqual(receiver.contact_local_point, (0.03, -0.01, -0.07))

    def test_handoff_uses_one_source_inference_and_full_rigid_rendezvous(self):
        class Actor:
            def __init__(self, functional_points):
                self.functional_points = functional_points

            def get_functional_point(self, index, representation):
                self.assert_representation = representation
                return self.functional_points[index].copy()

        class TrackingGrasps(FakeGrasps):
            def __init__(self, candidates):
                super().__init__(candidates)
                self.calls = []
                self.backend = SimpleNamespace(last_trace={})

            def propose(self, observation, target):
                self.calls.append((observation, target))
                return super().propose(observation, target)

        class ExactIK:
            grasp_to_robotwin = M2T2_TO_ROBOTWIN

            def __init__(self):
                self.failures = {}
                self.solve_calls = []
                self.collision_calls = []

            def solve_command_target(self, arm, command, start):
                start = np.asarray(start, dtype=np.float64)
                goal = start + (0.01 if arm == "left" else -0.01)
                path = np.vstack(((start + goal) / 2.0, goal))
                self.solve_calls.append((arm, np.asarray(command).copy()))
                return goal, path, np.asarray(command).copy()

            def full_robot_path_has_self_collision(
                self, actions, *, max_joint_step_rad
            ):
                self.collision_calls.append(
                    ([row.copy() for row in actions], max_joint_step_rad)
                )
                return len(self.collision_calls) == 1

        def candidate(contact_z, confidence):
            pose = I.copy()
            pose[2, 3] = contact_z - 0.1034
            return GraspCandidate(pose, confidence, "box")

        giver_best = candidate(0.07, 0.90)
        giver_second = candidate(0.06, 0.80)
        receiver_best = candidate(-0.07, 0.70)
        receiver_second = candidate(-0.06, 0.69)
        grasps = TrackingGrasps(
            [giver_best, giver_second, receiver_best, receiver_second]
        )
        middle = I.copy()
        middle[:3, :3] = t3d.axangles.axangle2mat(
            [0.0, 1.0, 0.0], 0.45
        )
        middle[:3, 3] = [0.0, 0.0, 0.9]
        destination = I.copy()
        destination[:3, :3] = t3d.axangles.axangle2mat(
            [0.0, 0.0, 1.0], -0.3
        )
        destination[:3, 3] = [0.18, 0.16, 0.84]
        target = ObjectState("box", I.copy(), 7)
        scene = SceneObservation(
            xyz=np.array([[0.0, 0.0, -0.1], [0.0, 0.0, 0.1]]),
            rgb=np.zeros((2, 3)),
            instance_labels=np.array([7, 7]),
            camera_pose=I.copy(),
            objects={"box": target},
        )
        env = Env()
        env.block_middle_pose = middle
        env.get_tracked_objects = lambda: {
            "box": Actor({0: I.copy()}),
            "target_box": Actor({1: destination}),
        }
        runtime = RoboTwinHeuristicRuntime.__new__(RoboTwinHeuristicRuntime)
        runtime.task_env = env
        runtime.grasps = grasps
        runtime.backend = grasps.backend
        runtime.config = SimpleNamespace(
            min_confidence=0.0,
            max_candidates=8,
            pregrasp_offset_m=0.07,
            retreat_offset_m=0.10,
        )
        runtime.ik = ExactIK()
        runtime.bimanual_max_plans_per_arm = 4
        runtime.bimanual_collision_step_rad = 0.025
        runtime.bimanual_max_jaw_axis_alignment = 0.75
        runtime.bimanual_max_target_width_m = 0.10
        runtime.gripper_settle_actions = 2
        runtime.staged_controller = StagedQposActionBuffer(
            env, max_waypoints_per_segment=8
        )
        runtime._action_metadata_override = None
        pick = Pick(
            "box", "left", pregrasp_offset_m=0.07,
            postgrasp_displacement=(0.0, 0.0, 0.10),
            gripper_target=0.25,
            allowed_contact_points_local=(
                (0.0, 0.0, 0.07), (0.0, 0.0, 0.06)
            ),
        )
        handoff = Handoff(
            "box", "left", "right",
            object_functional_point_id=0,
            pregrasp_offset_m=0.07,
            constrain="free",
            rendezvous_pose=tuple(
                tuple(float(value) for value in row) for row in middle
            ),
            gripper_target=0.35,
            allowed_contact_points_local=(
                (0.0, 0.0, -0.07), (0.0, 0.0, -0.06)
            ),
        )
        place = Place(
            "box", "target_box", "right",
            object_functional_point_id=0,
            destination_functional_point_id=1,
            preplace_offset_m=0.05,
            place_offset_m=0.0,
            constrain="align",
            preplace_axis="fp",
            release=True,
        )

        with patch.object(
            runtime, "_save_grasp_visualization"
        ) as visualizer, patch(
            "policy.heuristic_baseline.runtime._target_m2t2_palm_depth",
            return_value=0.08,
        ):
            actions = runtime._get_handoff_action(
                scene, target, pick=pick, handoff=handoff, place=place
            )

        self.assertTrue(actions)
        self.assertEqual(len(grasps.calls), 1)
        self.assertEqual(len(runtime.ik.collision_calls), 2)
        self.assertTrue(
            all(row.shape == (14,)
                for row in runtime.ik.collision_calls[0][0])
        )
        selected_candidates = [
            call.args[3] for call in visualizer.call_args_list
        ]
        self.assertEqual(
            [item.confidence for item in selected_candidates],
            [giver_best.confidence, receiver_second.confidence],
        )
        endpoints = [
            record for record in runtime.action_metadata
            if record["endpoint"]
        ]
        giver_transport = next(
            record for record in endpoints
            if record["phase"] == "transport" and record["arm"] == "left"
        )
        receiver_grasp = next(
            record for record in endpoints
            if record["phase"] == "grasp" and record["arm"] == "right"
        )
        np.testing.assert_allclose(
            giver_transport["command_pose"],
            middle @ giver_best.world_grasp_pose @ M2T2_TO_ROBOTWIN,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            receiver_grasp["command_pose"],
            middle @ receiver_second.world_grasp_pose @ M2T2_TO_ROBOTWIN,
            atol=1e-12,
        )
        giver_close = next(
            record for record in endpoints
            if record["phase"] == "close"
            and record.get("gripper_arms") == ["left"]
        )
        receiver_close = next(
            record for record in endpoints
            if record["phase"] == "close"
            and record.get("gripper_arms") == ["right"]
        )
        self.assertEqual(giver_close["target_gripper"], 0.25)
        self.assertEqual(receiver_close["target_gripper"], 0.35)

        grasps.candidates = [receiver_best, receiver_second]
        runtime.ik = ExactIK()
        with patch(
            "policy.heuristic_baseline.runtime._target_m2t2_palm_depth",
            return_value=0.08,
        ):
            giver_plans, receiver_plans, *_ = runtime._plan_handoff_sides(
                scene, target, pick=pick, handoff=handoff, place=place
            )

        self.assertTrue(giver_plans)
        self.assertTrue(receiver_plans)
        self.assertTrue(all(
            plan.orientation_source == "recorded_contact_alignment"
            for plan in giver_plans
        ))
        self.assertTrue(all(
            plan.contact_local_point[2] > 0.0 for plan in giver_plans
        ))

    def test_handoff_pca_fallback_is_axis_sign_invariant(self):
        local_target = np.array([
            [-0.12, 0.00, 0.00], [-0.10, 0.01, 0.00],
            [0.10, -0.01, 0.00], [0.12, 0.00, 0.00],
        ])

        giver, receiver, separation, source = _handoff_contact_regions(
            local_target, None, None,
            giver_reference_local=None,
            receiver_reference_local=None,
        )

        centroids = np.stack((giver.mean(axis=0), receiver.mean(axis=0)))
        self.assertEqual(source, "segmented_pca_fallback")
        self.assertGreater(separation, 0.0)
        self.assertGreater(np.linalg.norm(centroids[0] - centroids[1]), 0.20)
        self.assertEqual(
            {np.sign(value) for value in centroids[:, 0]}, {-1.0, 1.0}
        )

    def test_handoff_collision_rejection_exposes_no_partial_actions(self):
        def make_plan(arm, role, count, local_point):
            return _HandoffArmPlan(
                arm=arm,
                role=role,
                target_name="box",
                arm_source="robotwin_ground_truth",
                candidate=GraspCandidate(I.copy(), 0.9, "box"),
                paths=tuple(np.full((1, 6), 0.05) for _ in range(count)),
                command_targets=tuple(I.copy() for _ in range(count)),
                contact_local_point=local_point,
            )

        class CollisionIK:
            failures = {}

            def __init__(self):
                self.checked = []

            def full_robot_path_has_self_collision(
                self, actions, *, max_joint_step_rad
            ):
                self.checked.append(
                    ([row.copy() for row in actions], max_joint_step_rad)
                )
                return True

        giver = make_plan(
            "left", "giver", 5, (-0.03, 0.01, 0.07)
        )
        receiver = make_plan(
            "right", "receiver", 4, (0.02, -0.01, -0.07)
        )
        runtime = RoboTwinHeuristicRuntime.__new__(RoboTwinHeuristicRuntime)
        runtime.task_env = Env()
        runtime.staged_controller = StagedQposActionBuffer(
            runtime.task_env, max_waypoints_per_segment=8
        )
        runtime.controller = SimpleNamespace(metadata=[])
        runtime.gripper_settle_actions = 2
        runtime.bimanual_collision_step_rad = 0.02
        runtime._action_metadata_override = None
        runtime.ik = CollisionIK()
        target = ObjectState("box", I.copy(), 7)
        place = Place("box", "target_box", "right", release=True)
        planned = (
            [giver], [receiver], [giver.candidate, receiver.candidate],
            {}, {"pair_separation": 0}, 0.02,
        )

        with patch.object(
            runtime, "_plan_handoff_sides", return_value=planned
        ):
            with self.assertRaisesRegex(
                NoFeasiblePlanFailure, "separated-region handoff pairs"
            ):
                runtime._get_handoff_action(
                    object(),
                    target,
                    pick=Pick("box", "left"),
                    handoff=Handoff(
                        "box", "left", "right",
                        rendezvous_pose=tuple(
                            tuple(float(value) for value in row) for row in I
                        ),
                    ),
                    place=place,
                )

        self.assertEqual(len(runtime.ik.checked), 2)
        checked_actions, step = runtime.ik.checked[0]
        self.assertEqual(step, 0.02)
        self.assertTrue(
            runtime.ik.checked[1][0]
            and all(row.shape == (14,) for row in runtime.ik.checked[1][0])
        )
        self.assertTrue(
            checked_actions and all(row.shape == (14,) for row in checked_actions)
        )
        self.assertIsNone(runtime._action_metadata_override)
        self.assertEqual(runtime.action_metadata, [])

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

    def test_mink_joint_start_override_preempts_measured_state(self):
        fake = FakeSolver([])
        with patch(
            "policy.heuristic_baseline.runtime.MinkIKSolver.from_xml_path",
            return_value=fake,
        ):
            ik = RoboTwinMinkIK(Env(), I, model_path="unused.urdf")

        override = np.linspace(0.1, 0.6, 6)
        ik.set_joint_start_override("left", override)
        np.testing.assert_allclose(ik._joint_positions("left"), override)
        np.testing.assert_allclose(ik._joint_positions("right"), np.zeros(6))

        ik.set_joint_start_override("left", None)
        np.testing.assert_allclose(ik._joint_positions("left"), np.zeros(6))

    def test_mink_joint_start_override_rejects_invalid_shape(self):
        fake = FakeSolver([])
        with patch(
            "policy.heuristic_baseline.runtime.MinkIKSolver.from_xml_path",
            return_value=fake,
        ):
            ik = RoboTwinMinkIK(Env(), I, model_path="unused.urdf")

        with self.assertRaisesRegex(ValueError, "finite 6-vector"):
            ik.set_joint_start_override("left", np.zeros(5))

    def test_mink_joint_transition_reaches_home_with_bounded_steps(self):
        fake = FakeSolver([])
        with patch(
            "policy.heuristic_baseline.runtime.MinkIKSolver.from_xml_path",
            return_value=fake,
        ):
            ik = RoboTwinMinkIK(
                Env(),
                I,
                model_path="unused.urdf",
                max_joint_step_rad=0.1,
            )

        path = ik.plan_joint_transition(
            "left", np.full(6, 0.25), np.zeros(6)
        )

        self.assertIsNotNone(path)
        np.testing.assert_allclose(path[-1], np.zeros(6))
        self.assertLessEqual(
            np.max(np.abs(np.diff(
                np.vstack((np.full(6, 0.25), path)), axis=0
            ))),
            0.1,
        )

    def test_place_dispatch_is_structural_across_module_namespaces(self):
        foreign_pick = SimpleNamespace(target="shoe", arm="right")
        foreign_place = SimpleNamespace(
            object="shoe",
            destination="target_block",
            preplace_offset_m=0.12,
            place_offset_m=0.02,
        )
        env = SimpleNamespace(
            heuristic_task_plan=SimpleNamespace(
                stages=(foreign_pick, foreign_place)
            )
        )

        actual = RoboTwinHeuristicRuntime._single_arm_place_stages(env)

        self.assertIsNotNone(actual)
        self.assertIs(actual[0], foreign_pick)
        self.assertIs(actual[1], foreign_place)

    def test_three_sequential_places_are_task_name_independent(self):
        pairs = tuple(
            (
                Pick(f"box_{index}", "left" if index == 0 else "right"),
                Place(f"box_{index}", None, target_pose=I),
            )
            for index in range(3)
        )
        env = SimpleNamespace(
            heuristic_task_plan=TaskPlan(
                "blocks_ranking_size",
                "pick_place",
                tuple(stage for pair in pairs for stage in pair),
            )
        )

        actual = RoboTwinHeuristicRuntime._sequential_place_stages(env)

        self.assertEqual(actual, pairs)

    def test_two_separate_bimanual_places_are_sequential(self):
        pairs = (
            (
                Pick("object_left", "left"),
                Place("object_left", None, "left", target_pose=I),
            ),
            (
                Pick("object_right", "right"),
                Place("object_right", None, "right", target_pose=I),
            ),
        )
        env = SimpleNamespace(
            heuristic_task_plan=TaskPlan(
                "separate_bimanual",
                "pick_place",
                tuple(stage for pair in pairs for stage in pair),
            )
        )

        self.assertEqual(
            RoboTwinHeuristicRuntime._sequential_place_stages(env),
            pairs,
        )

    def test_three_sequential_places_reject_mismatched_objects(self):
        stages = (
            Pick("red", "left"), Place("red", None, target_pose=I),
            Pick("green", "right"), Place("blue", None, target_pose=I),
            Pick("blue", "right"), Place("blue", None, target_pose=I),
        )
        env = SimpleNamespace(
            heuristic_task_plan=TaskPlan(
                "blocks_ranking_rgb", "pick_place", stages
            )
        )

        self.assertIsNone(
            RoboTwinHeuristicRuntime._sequential_place_stages(env)
        )


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
        def make_plan(arm, target_name, sign, confidence):
            candidate_pose = I.copy()
            candidate_pose[0, 3] = sign * 0.10
            candidate = GraspCandidate(
                candidate_pose, confidence, target_name
            )
            paths = tuple(
                np.full((1, 6), sign * value, dtype=np.float64)
                for value in (0.10, 0.20, 0.30, 0.40, 0.50)
            )
            return _SingleArmPlacePlan(
                arm=arm,
                target_name=target_name,
                arm_source="robotwin_ground_truth",
                candidate=candidate,
                paths=paths,
                command_targets=tuple(I.copy() for _ in paths),
                desired_object_pose=I.copy(),
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

        targets = {
            name: ObjectState(name, I.copy(), instance_id)
            for instance_id, name in enumerate(("object_a", "object_b"), 1)
        }
        picks = (
            Pick("object_a", "left", group_id=0),
            Pick("object_b", "right", group_id=0),
        )
        pose = tuple(tuple(float(value) for value in row) for row in I)
        places = (
            Place(
                "object_a", None, "left", target_pose=pose,
                release=False, group_id=2,
            ),
            Place(
                "object_b", None, "right", target_pose=pose,
                release=False, group_id=2,
            ),
        )
        task_plan = TaskPlan(
            "generic_dual", "pick_place", picks + places
        )
        plans = {
            "left": [
                make_plan("left", "object_a", -1.0, 0.90),
                make_plan("left", "object_a", -1.0, 0.80),
            ],
            "right": [
                make_plan("right", "object_b", 1.0, 0.85),
                make_plan("right", "object_b", 1.0, 0.75),
            ],
        }
        plans["left"].insert(
            0,
            replace(
                plans["left"][0],
                paths=plans["left"][0].paths[:3],
                command_targets=plans["left"][0].command_targets[:3],
                completion_level="grasp_lift",
            ),
        )
        runtime = RoboTwinHeuristicRuntime.__new__(RoboTwinHeuristicRuntime)
        runtime._grasp_attempted = False
        runtime._action_metadata_override = None
        runtime.simulator = SimulatorSpy(targets)
        runtime.task_env = SimpleNamespace(heuristic_task_plan=task_plan)
        runtime.controller = SimpleNamespace(metadata=[])
        runtime.staged_controller = StagedQposActionBuffer(
            Env(), max_waypoints_per_segment=8
        )
        runtime.ik = CollisionIK()
        runtime.bimanual_collision_step_rad = 0.025
        runtime.bimanual_max_plans_per_arm = 2
        runtime.gripper_settle_actions = 1

        def select_arm(target):
            return (
                ("left", "robotwin_ground_truth")
                if target.name == "object_a"
                else ("right", "robotwin_ground_truth")
            )

        def plan_arm(
            _scene,
            target,
            *,
            pick,
            place,
            arm,
            arm_source,
            plan_limit,
        ):
            del target, pick, place, arm_source
            self.assertEqual(plan_limit, 2)
            return plans[arm], [item.candidate for item in plans[arm]], {}, {}

        with patch.object(
            runtime,
            "_target_names",
            return_value=("object_a", "object_b"),
        ), patch.object(
            runtime, "_select_arm", side_effect=select_arm
        ), patch.object(
            runtime, "_plan_single_arm_place", side_effect=plan_arm
        ) as planner, patch.object(
            runtime, "_save_grasp_visualization"
        ):
            with self.assertRaisesRegex(
                NoFeasiblePlanFailure,
                "all confidence-ranked grouped bimanual placement pairs",
            ):
                runtime.get_action(scene=object())

            self.assertTrue(runtime.grasp_attempted)
            self.assertEqual(runtime.action_metadata, [])
            self.assertIsNone(runtime._action_metadata_override)
            self.assertEqual(planner.call_count, 2)
            self.assertEqual(len(runtime.ik.checked_actions), 4)
            self.assertTrue(
                all(
                    actions
                    and all(action.shape == (14,) for action in actions)
                    and collision_step == 0.025
                    for actions, collision_step in runtime.ik.checked_actions
                )
            )
            with self.assertRaisesRegex(NoFeasiblePlanFailure, "one-shot"):
                runtime.get_action(scene=object())

        self.assertEqual(runtime.simulator.update_calls, 1)

    def test_single_place_raw_pose_preempts_higher_confidence_fallback(self):
        class Actor:
            def get_pose(self):
                return I.copy()

        class TierIK:
            grasp_to_robotwin = I.copy()

            def __init__(self):
                self.failures = {}
                self.solve_count = 0
                self.completed_paths = ()
                self.completed_command_targets = ()

            def solve(self, arm, pose):
                del arm
                stage = self.solve_count % 3
                if stage == 0:
                    self.completed_paths = ()
                    self.completed_command_targets = ()
                self.solve_count += 1
                pose = np.asarray(pose, dtype=np.float64)
                if np.isclose(pose[0, 3], 0.20):
                    return None
                goal = np.full(6, 0.01 * self.solve_count)
                self.completed_paths += (goal[None, :],)
                self.completed_command_targets += (pose.copy(),)
                return goal

            def solve_command_target(self, arm, command, start):
                del arm
                goal = np.asarray(start, dtype=np.float64) + 0.01
                return goal, goal[None, :], np.asarray(command).copy()

        high_pose = I.copy()
        high_pose[0, 3] = 0.20
        low_pose = I.copy()
        low_pose[0, 3] = 0.40
        candidates = [
            GraspCandidate(high_pose, 0.90, "target"),
            GraspCandidate(low_pose, 0.40, "target"),
        ]
        grasps = FakeGrasps(candidates)
        grasps.backend = SimpleNamespace(last_trace={})
        actor = Actor()
        env = SimpleNamespace(
            robot=SimpleNamespace(
                get_right_ee_pose=lambda: np.array(
                    [0.5, -0.4, 0.9, 1.0, 0.0, 0.0, 0.0]
                )
            ),
            get_tracked_objects=lambda: {"target": actor},
        )
        target = ObjectState("target", I.copy(), 7)
        y = np.linspace(-0.04, 0.04, 9)
        points = np.column_stack((
            0.002 * np.sin(np.arange(len(y))),
            y,
            0.08 + 0.001 * np.cos(np.arange(len(y))),
        ))
        scene = SceneObservation(
            xyz=points,
            rgb=np.zeros((len(points), 3)),
            instance_labels=np.full(len(points), 7),
            camera_pose=I.copy(),
            objects={"target": target},
        )
        runtime = RoboTwinHeuristicRuntime.__new__(
            RoboTwinHeuristicRuntime
        )
        runtime.task_env = env
        runtime.grasps = grasps
        runtime.backend = grasps.backend
        runtime.config = SimpleNamespace(
            min_confidence=0.0,
            max_candidates=8,
            pregrasp_offset_m=0.07,
            retreat_offset_m=0.10,
        )
        runtime.ik = TierIK()
        runtime.bimanual_max_target_width_m = 0.10
        pose_tuple = tuple(
            tuple(float(value) for value in row) for row in I
        )
        pick = Pick("target", "right", pregrasp_offset_m=0.07)
        place = Place(
            "target",
            None,
            "right",
            target_pose=pose_tuple,
            preplace_offset_m=0.05,
            place_offset_m=0.0,
            constrain="free",
            preplace_axis=(0.0, 0.0, 1.0),
        )

        def robot_facing(pose, *_args):
            adjusted = np.asarray(pose).copy()
            adjusted[0, 3] += 1.0
            return adjusted

        def aligned(_source, destination, _grasp, **_kwargs):
            return np.asarray(destination).copy()

        with patch(
            "policy.heuristic_baseline.runtime._target_m2t2_palm_depth",
            return_value=0.08,
        ), patch(
            "policy.heuristic_baseline.runtime._robot_facing_grasp_pose",
            side_effect=robot_facing,
        ), patch(
            "policy.heuristic_baseline.runtime._place_facing_grasp_pose",
            side_effect=ValueError("disabled in tier-order fixture"),
        ), patch(
            "policy.heuristic_baseline.runtime._aligned_place_reference_pose",
            side_effect=aligned,
        ):
            plans, _, _, failures = runtime._plan_single_arm_place(
                scene,
                target,
                pick=pick,
                place=place,
                arm="right",
                arm_source="robotwin_ground_truth",
                plan_limit=2,
            )

        self.assertEqual(
            [plan.orientation_source for plan in plans],
            ["m2t2", "approach_roll"],
        )
        self.assertEqual(
            [plan.candidate.confidence for plan in plans], [0.40, 0.40]
        )
        np.testing.assert_allclose(
            plans[0].candidate.world_grasp_pose, low_pose
        )
        elongated = _elongated_object_axis(scene, target)
        self.assertIsNotNone(elongated)
        self.assertGreater(abs(float(np.dot(elongated, [0.0, 1.0, 0.0]))), 0.99)
        self.assertLessEqual(
            _target_width_along_axis(scene, target, [0.0, 1.0, 0.0]),
            runtime.bimanual_max_target_width_m,
        )
        self.assertEqual(failures["jaw_width"], 0)
        self.assertNotIn("jaw_axis", failures)

    def test_target_palm_depth_rejects_intruding_rotated_fallback(self):
        contact = np.zeros(3)
        raw = I.copy()
        raw[:3, 3] = contact - 0.1034 * raw[:3, 2]
        fallback = I.copy()
        fallback[:3, :3] = t3d.axangles.axangle2mat(
            [0.0, 1.0, 0.0], np.pi / 2.0
        )
        fallback[:3, 3] = contact - 0.1034 * fallback[:3, 2]
        target_points = np.array([
            [x, y, z]
            for x in np.linspace(-0.0806, -0.0600, 8)
            for y in (-0.003, 0.003)
            for z in np.linspace(-0.0439, -0.0350, 6)
        ])
        distractors = np.array([
            [-10.0, 0.0, -10.0], [10.0, 0.0, 10.0]
        ])
        points = np.vstack((target_points, distractors))
        target = ObjectState("target", I.copy(), 7)
        scene = SceneObservation(
            xyz=points,
            rgb=np.zeros((len(points), 3)),
            instance_labels=np.r_[
                np.full(len(target_points), 7), [99, 99]
            ],
            camera_pose=I.copy(),
            objects={"target": target},
        )

        raw_depth = _target_m2t2_palm_depth(
            scene, target, raw, M2T2_TO_ROBOTWIN
        )
        fallback_depth = _target_m2t2_palm_depth(
            scene, target, fallback, M2T2_TO_ROBOTWIN
        )

        np.testing.assert_allclose(
            _grasp_command_tcp(raw, M2T2_TO_ROBOTWIN), contact,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            _grasp_command_tcp(fallback, M2T2_TO_ROBOTWIN), contact,
            atol=1e-12,
        )
        self.assertGreaterEqual(
            raw_depth, M2T2_MIN_TARGET_PALM_DEPTH_M
        )
        self.assertLess(
            fallback_depth, M2T2_MIN_TARGET_PALM_DEPTH_M
        )

    def test_handoff_raw_pair_preempts_higher_confidence_fallbacks(self):
        def make_plan(arm, role, confidence, source, contact):
            count = 5 if role == "giver" else 4
            return _HandoffArmPlan(
                arm=arm,
                role=role,
                target_name="target",
                arm_source="robotwin_ground_truth",
                candidate=GraspCandidate(I.copy(), confidence, "target"),
                paths=tuple(np.zeros((1, 6)) for _ in range(count)),
                command_targets=tuple(I.copy() for _ in range(count)),
                contact_local_point=contact,
                orientation_source=source,
            )

        giver_fallback = make_plan(
            "left", "giver", 0.99, "robot_facing_fallback",
            (-0.10, 0.0, 0.0),
        )
        giver_raw = make_plan(
            "left", "giver", 0.35, "m2t2", (-0.10, 0.0, 0.0)
        )
        receiver_fallback = make_plan(
            "right", "receiver", 0.98, "robot_facing_fallback",
            (0.10, 0.0, 0.0),
        )
        receiver_raw = make_plan(
            "right", "receiver", 0.30, "m2t2", (0.10, 0.0, 0.0)
        )
        runtime = RoboTwinHeuristicRuntime.__new__(
            RoboTwinHeuristicRuntime
        )
        runtime.ik = SimpleNamespace(
            failures={},
            full_robot_path_has_self_collision=(
                lambda _actions, *, max_joint_step_rad: False
            ),
        )
        runtime.bimanual_collision_step_rad = 0.025
        runtime.staged_controller = SimpleNamespace(metadata=[])
        runtime._action_metadata_override = None
        planned = (
            [giver_fallback, giver_raw],
            [receiver_fallback, receiver_raw],
            [],
            {},
            {"pair_separation": 0},
            0.05,
        )
        target = ObjectState("target", I.copy(), 7)
        handoff = Handoff(
            "target", "left", "right",
            rendezvous_pose=tuple(
                tuple(float(value) for value in row) for row in I
            ),
        )
        place = Place("target", None, "right", release=True)

        with patch.object(
            runtime, "_plan_handoff_sides", return_value=planned
        ), patch.object(
            runtime, "_build_handoff_action_pair",
            return_value=[np.zeros(14)],
        ) as builder, patch.object(
            runtime, "_save_grasp_visualization"
        ):
            actions = runtime._get_handoff_action(
                object(),
                target,
                pick=Pick("target", "left"),
                handoff=handoff,
                place=place,
            )

        self.assertTrue(actions)
        selected_giver, selected_receiver = builder.call_args.args[:2]
        self.assertEqual(selected_giver.orientation_source, "m2t2")
        self.assertEqual(selected_receiver.orientation_source, "m2t2")
        self.assertLess(
            selected_giver.candidate.confidence
            + selected_receiver.candidate.confidence,
            giver_fallback.candidate.confidence
            + receiver_fallback.candidate.confidence,
        )


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


    def test_approach_roll_preserves_command_and_source_geometry(self):
        grasp = I.copy()
        grasp[:3, :3] = t3d.euler.euler2mat(0.31, -0.42, 0.57)
        grasp[:3, 3] = [0.13, -0.19, 0.84]
        command = grasp @ M2T2_TO_ROBOTWIN
        tcp = command[:3, 3] + 0.12 * command[:3, 0]

        rolled = _approach_roll_grasp_pose(
            grasp, M2T2_TO_ROBOTWIN, np.pi / 4.0
        )
        rolled_command = rolled @ M2T2_TO_ROBOTWIN
        rolled_tcp = (
            rolled_command[:3, 3]
            + 0.12 * rolled_command[:3, 0]
        )

        np.testing.assert_allclose(rolled[:3, 3], grasp[:3, 3], atol=1e-12)
        np.testing.assert_allclose(rolled[:3, 2], grasp[:3, 2], atol=1e-12)
        np.testing.assert_allclose(
            rolled_command[:3, 3], command[:3, 3], atol=1e-12
        )
        np.testing.assert_allclose(
            rolled_command[:3, 0], command[:3, 0], atol=1e-12
        )
        np.testing.assert_allclose(rolled_tcp, tcp, atol=1e-12)
        np.testing.assert_allclose(
            rolled[:3, :3].T @ rolled[:3, :3], np.eye(3), atol=1e-12
        )
        self.assertAlmostEqual(
            float(np.linalg.det(rolled[:3, :3])), 1.0, places=12
        )


    def test_support_plane_estimator_excludes_target_bottom_points(self):
        target_pose = I.copy()
        target_pose[:3, 3] = [0.02, -0.03, 0.79]
        target = ObjectState("target", target_pose, 7)
        target_points = np.array([
            [0.02 + x, -0.03 + y, z]
            for x in np.linspace(-0.02, 0.02, 5)
            for y in np.linspace(-0.015, 0.015, 5)
            for z in (0.755, 0.785, 0.820)
        ])
        plane_points = np.array([
            [0.02 + x, -0.03 + y, 0.740 + 0.0001 * ((i + j) % 3)]
            for i, x in enumerate(np.linspace(-0.04, 0.04, 7))
            for j, y in enumerate(np.linspace(-0.04, 0.04, 7))
        ])

        def make_scene(points, labels):
            return SceneObservation(
                xyz=np.asarray(points, dtype=np.float64),
                rgb=np.zeros((len(points), 3), dtype=np.float64),
                instance_labels=np.asarray(labels, dtype=np.int64),
                camera_pose=I.copy(),
                objects={"target": target},
            )

        mixed = make_scene(
            np.vstack((target_points, plane_points)),
            np.r_[
                np.full(len(target_points), 7),
                np.full(len(plane_points), 99),
            ],
        )
        target_only = make_scene(
            target_points, np.full(len(target_points), 7)
        )

        self.assertAlmostEqual(
            _estimate_target_support_plane_z(mixed, target),
            float(np.median(plane_points[:, 2])),
            places=6,
        )
        self.assertIsNone(
            _estimate_target_support_plane_z(target_only, target)
        )

    def test_support_collision_includes_fixed_descendant_and_dense_midpoint(self):
        model = mujoco.MjModel.from_xml_string(
            """
            <mujoco>
              <worldbody>
                <body name="controlled_root" pos="0 0 1">
                  <joint name="right_joint" type="hinge" axis="0 1 0"/>
                  <geom name="root_collision" type="sphere" size="0.02"/>
                  <body name="fixed_wrist_sensor" pos="0.4 0 0">
                    <geom name="sensor_collision" type="box"
                          size="0.04 0.03 0.02"/>
                  </body>
                </body>
              </worldbody>
            </mujoco>
            """
        )

        class OneJointRobot:
            right_arm_joints_name = ("right_joint",)
            right_entity_origion_pose = Pose()

            def get_right_arm_jointState(self):
                return np.array([0.0, 1.0])

        ik = RoboTwinMinkIK.__new__(RoboTwinMinkIK)
        ik.solver = SimpleNamespace(model=model)
        ik.task_env = SimpleNamespace(robot=OneJointRobot())
        ik.support_collision_checks = 0
        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "fixed_wrist_sensor"
        )
        geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "sensor_collision"
        )

        self.assertIn(body_id, ik.active_arm_body_ids("right"))
        self.assertIn(geom_id, ik.active_arm_geom_ids("right"))

        def minimum_z(joint):
            data = mujoco.MjData(model)
            data.qpos[0] = joint
            mujoco.mj_forward(model, data)
            return min(
                ik._geom_minimum_world_z(model, data, item, I)
                for item in ik.active_arm_geom_ids("right")
            )

        # Both endpoints clear the support. The welded sensor box dips below
        # it near pi/2, so only dense path validation catches this motion.
        self.assertGreater(minimum_z(0.0), 0.753)
        self.assertGreater(minimum_z(np.pi), 0.753)
        self.assertTrue(
            ik.path_has_support_collision(
                "right",
                np.array([[np.pi]]),
                0.750,
                start=np.array([0.0]),
            )
        )
        self.assertFalse(
            ik.path_has_support_collision(
                "right",
                np.array([[-np.pi / 2.0]]),
                0.750,
                start=np.array([0.0]),
            )
        )


    def test_saved_style_support_collision_rejects_bad_and_allows_safe_pose(self):
        class WorldPose:
            def to_transformation_matrix(self):
                pose = I.copy()
                pose[:3, :3] = t3d.quaternions.quat2mat(
                    [0.707, 0.0, 0.0, 0.707]
                )
                pose[:3, 3] = [0.0, -0.65, 0.0]
                return pose

        class WorldRobot(Robot):
            left_entity_origion_pose = WorldPose()
            right_entity_origion_pose = WorldPose()

        model_path = (
            Path(__file__).resolve().parents[3]
            / "assets"
            / "embodiments"
            / "aloha-agilex"
            / "urdf"
            / "arx5_description_isaac.urdf"
        )
        ik = RoboTwinMinkIK(
            SimpleNamespace(robot=WorldRobot()), I, model_path=model_path
        )
        bad = np.array([
            0.3155142192501402,
            2.430038960022039,
            1.2861609031780952,
            0.14376985254571506,
            1.3626009970739368,
            -2.233478929904455,
        ])
        safe = np.array([
            -0.027226163863374175,
            2.1285174328107104,
            0.9029227655007368,
            0.9928067097744266,
            0.018112636133269078,
            0.29531992859857503,
        ])

        self.assertTrue(
            ik.path_has_support_collision("left", bad[None], 0.740)
        )
        self.assertFalse(
            ik.path_has_support_collision("left", safe[None], 0.740)
        )


    def test_narrow_fallback_considers_candidates_beyond_rank_sixteen(self):
        class Actor:
            def get_pose(self):
                return I.copy()

        marker_x = 0.116

        class NarrowOnlyIK:
            grasp_to_robotwin = I.copy()

            def __init__(self):
                self.failures = {}
                self.solve_count = 0
                self.completed_paths = ()
                self.completed_command_targets = ()

            def solve(self, arm, pose):
                del arm
                stage = self.solve_count % 3
                if stage == 0:
                    self.completed_paths = ()
                    self.completed_command_targets = ()
                self.solve_count += 1
                pose = np.asarray(pose, dtype=np.float64)
                if not (
                    np.isclose(pose[0, 3], marker_x)
                    and np.isclose(pose[1, 3], 0.50)
                ):
                    return None
                goal = np.full(6, 0.01 * (stage + 1))
                self.completed_paths += (goal[None, :],)
                self.completed_command_targets += (pose.copy(),)
                return goal

            def solve_command_target(self, arm, command, start):
                del arm
                goal = np.asarray(start, dtype=np.float64) + 0.01
                return goal, goal[None, :], np.asarray(command).copy()

        candidates = []
        for index in range(17):
            pose = I.copy()
            pose[0, 3] = 0.100 + 0.001 * index
            candidates.append(
                GraspCandidate(
                    pose, 1.0 - 0.01 * index, "target"
                )
            )
        grasps = FakeGrasps(candidates)
        grasps.backend = SimpleNamespace(last_trace={})
        env = SimpleNamespace(
            robot=SimpleNamespace(
                get_right_ee_pose=lambda: np.array(
                    [0.5, -0.4, 0.9, 1.0, 0.0, 0.0, 0.0]
                )
            ),
            get_tracked_objects=lambda: {"target": Actor()},
        )
        target = ObjectState("target", I.copy(), 7)
        scene = SceneObservation(
            xyz=np.array([
                [-0.02, -0.01, 0.75], [0.02, -0.01, 0.75],
                [-0.02, 0.01, 0.80], [0.02, 0.01, 0.80],
            ]),
            rgb=np.zeros((4, 3)),
            instance_labels=np.full(4, 7),
            camera_pose=I.copy(),
            objects={"target": target},
        )
        runtime = RoboTwinHeuristicRuntime.__new__(
            RoboTwinHeuristicRuntime
        )
        runtime.task_env = env
        runtime.grasps = grasps
        runtime.backend = grasps.backend
        runtime.config = SimpleNamespace(
            min_confidence=0.0,
            max_candidates=32,
            pregrasp_offset_m=0.07,
            retreat_offset_m=0.10,
        )
        runtime.ik = NarrowOnlyIK()
        runtime.bimanual_max_target_width_m = 0.10
        pose_tuple = tuple(tuple(float(value) for value in row) for row in I)
        narrow_calls = []

        def narrow_pose(pose, *_args, **_kwargs):
            narrow_calls.append(float(pose[0, 3]))
            result = np.asarray(pose, dtype=np.float64).copy()
            result[1, 3] = 0.50
            return (result,)

        with patch(
            "policy.heuristic_baseline.runtime._target_m2t2_palm_depth",
            return_value=0.08,
        ), patch(
            "policy.heuristic_baseline.runtime._target_width_along_axis",
            return_value=0.05,
        ), patch(
            "policy.heuristic_baseline.runtime._robot_facing_grasp_pose",
            side_effect=ValueError("disabled in narrow-tier fixture"),
        ), patch(
            "policy.heuristic_baseline.runtime._place_facing_grasp_pose",
            side_effect=lambda pose, *_args: np.asarray(pose).copy(),
        ), patch(
            "policy.heuristic_baseline.runtime._target_narrow_axis",
            return_value=np.array([0.0, 1.0, 0.0]),
        ), patch(
            "policy.heuristic_baseline.runtime._narrow_axis_grasp_poses",
            side_effect=narrow_pose,
        ), patch(
            "policy.heuristic_baseline.runtime._aligned_place_reference_pose",
            side_effect=lambda _source, destination, *_args, **_kwargs:
            np.asarray(destination).copy(),
        ):
            plans, _, _, _ = runtime._plan_single_arm_place(
                scene,
                target,
                pick=Pick("target", "right", pregrasp_offset_m=0.07),
                place=Place(
                    "target",
                    None,
                    "right",
                    target_pose=pose_tuple,
                    preplace_offset_m=0.05,
                    place_offset_m=0.0,
                    constrain="free",
                    preplace_axis=(0.0, 0.0, 1.0),
                ),
                arm="right",
                arm_source="robotwin_ground_truth",
            )

        self.assertEqual(len(narrow_calls), 17)
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0].orientation_source, "narrow_facing")
        self.assertAlmostEqual(
            plans[0].candidate.confidence, candidates[16].confidence
        )


if __name__ == "__main__":
    unittest.main()

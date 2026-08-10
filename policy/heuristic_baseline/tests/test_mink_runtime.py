from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import transforms3d as t3d

from policy.heuristic_baseline.runtime import (
    CANONICAL_COMMAND_QUATERNIONS,
    M2T2_TO_ROBOTWIN,
    PARALLEL_JAW_ROLL_SYMMETRY,
    QposActionBuffer,
    ReachabilityRankedGrasps,
    RoboTwinMinkIK,
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
        goals = [np.full(6, value) for value in (0.2, 0.3, 0.4)]
        ik, fake = self.make_ik(goals)

        actual = [ik.solve("right", I) for _ in range(3)]

        for result, expected in zip(actual, goals):
            np.testing.assert_allclose(result, expected)
        self.assertEqual(len(fake.calls), 3)
        expected_rotation = np.diag([1.0, -1.0, -1.0])
        np.testing.assert_allclose(fake.calls[0][1][:3, :3], expected_rotation)
        for goal in goals:
            path = ik.consume_path("right", goal)
            self.assertGreaterEqual(len(path), 2)
            self.assertLessEqual(len(path), 8)
            np.testing.assert_allclose(path[-1], goal)

    def test_retries_with_canonical_orientation(self):
        goal = np.arange(6, dtype=np.float64) * 0.01
        ik, fake = self.make_ik([None, goal])

        result = ik.solve("right", I)

        np.testing.assert_allclose(result, goal)
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(ik.relaxed_successes, 1)

    def test_records_exact_mink_accepted_grasp_command(self):
        goals = [np.full(6, value) for value in (0.1, 0.2, 0.3)]
        ik, _ = self.make_ik([goals[0], None, goals[1], goals[2]])

        actual = [ik.solve("right", I) for _ in range(3)]

        for result, expected in zip(actual, goals):
            np.testing.assert_allclose(result, expected)
        accepted = ik.selected_grasp_command_pose
        self.assertIsNotNone(accepted)
        np.testing.assert_allclose(
            accepted[:3, :3],
            t3d.quaternions.quat2mat(
                CANONICAL_COMMAND_QUATERNIONS["right"]
            ),
        )
        np.testing.assert_allclose(accepted[:3, 3], np.zeros(3))

    def test_canonical_orientation_outranks_higher_confidence_sideways_grasp(self):
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
            GraspCandidate(sideways_pose, 0.99, "target"),
            GraspCandidate(canonical_pose, 0.40, "target"),
        ]
        ranker = ReachabilityRankedGrasps(
            FakeGrasps(candidates), "right", M2T2_TO_ROBOTWIN
        )
        target = ObjectState("target", I, 1)

        ranked = ranker.propose(None, target)

        np.testing.assert_allclose(
            ranked[0].world_grasp_pose[:3, :3], canonical_pose[:3, :3]
        )
        self.assertEqual(len(ranker.last_candidates), 2)
        self.assertIs(ranker.last_candidates[0], ranked[0])

    def test_parallel_jaw_half_roll_has_same_orientation_error(self):
        axis_map = M2T2_TO_ROBOTWIN[:3, :3]
        canonical_command = t3d.quaternions.quat2mat(
            CANONICAL_COMMAND_QUATERNIONS["left"]
        )
        canonical_pose = I.copy()
        canonical_pose[:3, :3] = canonical_command @ axis_map.T
        rolled_pose = I.copy()
        rolled_pose[:3, :3] = (
            canonical_command @ PARALLEL_JAW_ROLL_SYMMETRY @ axis_map.T
        )
        ranker = ReachabilityRankedGrasps(
            FakeGrasps([]), "left", M2T2_TO_ROBOTWIN
        )

        self.assertAlmostEqual(ranker._orientation_error(canonical_pose), 0.0)
        self.assertAlmostEqual(ranker._orientation_error(rolled_pose), 0.0)

    def test_grasp_visualization_writes_png(self):
        center = np.array([0.02, -0.10, 0.80])
        offsets = np.linspace(-0.025, 0.025, 5)
        points = np.array(
            [center + [x, y, z] for x in offsets for y in offsets for z in offsets]
        )
        colors = np.tile([[0.15, 0.65, 0.95]], (len(points), 1))
        target_pose = I.copy()
        target_pose[:3, 3] = center
        target = ObjectState("bottle", target_pose, 7)
        scene = SceneObservation(
            xyz=points,
            rgb=colors,
            instance_labels=np.full(len(points), 7),
            camera_pose=I,
            objects={"bottle": target},
        )
        axis_map = M2T2_TO_ROBOTWIN[:3, :3]
        canonical_command = t3d.quaternions.quat2mat(
            CANONICAL_COMMAND_QUATERNIONS["right"]
        )
        candidates = []
        for angle in np.linspace(-0.8, 0.8, 9):
            rotation_z = np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0.0],
                    [np.sin(angle), np.cos(angle), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            pose = I.copy()
            pose[:3, :3] = canonical_command @ rotation_z @ axis_map.T
            pose[:3, 3] = center - 0.1034 * pose[:3, 2]
            candidates.append(GraspCandidate(pose, 1.0, "bottle"))

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "grasp_viz.png"
            raw_trace = {
                "poses": np.asarray(
                    [candidate.world_grasp_pose for candidate in candidates]
                ),
                "scores": np.linspace(0.4, 0.8, len(candidates)),
                "contacts": np.tile(center, (len(candidates), 1)),
                "target_contacts": np.array(
                    [True] * 6 + [False] * 3
                ),
                "query_ids": np.zeros((len(candidates), 2), dtype=int),
            }
            executed_command = (
                candidates[4].world_grasp_pose @ M2T2_TO_ROBOTWIN
            )
            saved = save_grasp_visualization(
                output,
                scene,
                target,
                candidates[:6],
                candidates[4],
                arm="right",
                grasp_to_robotwin=M2T2_TO_ROBOTWIN,
                rejected_candidates=candidates[6:],
                executed_command_pose=executed_command,
                raw_trace=raw_trace,
            )

            self.assertEqual(saved, output)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 10_000)
            self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            data_path = output.with_suffix(".npz")
            self.assertTrue(data_path.is_file())
            with np.load(data_path, allow_pickle=False) as data:
                self.assertEqual(data["raw_world_grasp_poses"].shape, (9, 4, 4))
                self.assertEqual(data["ranked_world_grasp_poses"].shape, (6, 4, 4))
                self.assertEqual(data["rejected_world_grasp_poses"].shape, (3, 4, 4))
                self.assertEqual(data["selected_world_grasp_pose"].shape, (1, 4, 4))
                self.assertEqual(data["mink_accepted_command_pose"].shape, (1, 4, 4))
                self.assertEqual(int(data["raw_target_contacts"].sum()), 6)

    def test_qpos_buffer_keeps_inactive_arm_and_uses_joint_waypoints(self):
        goals = [np.full(6, value) for value in (0.1, 0.2, 0.3)]
        env = Env()
        ik, _ = self.make_ik(goals)
        buffer = QposActionBuffer(env, ik, max_waypoints_per_segment=8)
        buffer.reset()
        solved = [ik.solve("right", I) for _ in range(3)]

        buffer.open_gripper("right")
        for goal in solved:
            buffer.move_joints("right", goal)

        self.assertTrue(buffer.actions)
        self.assertTrue(all(action.shape == (14,) for action in buffer.actions))
        self.assertTrue(all(np.allclose(action[:6], 0.0) for action in buffer.actions))
        np.testing.assert_allclose(buffer.actions[-1][7:13], goals[-1])


if __name__ == "__main__":
    unittest.main()

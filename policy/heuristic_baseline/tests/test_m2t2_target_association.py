"""CPU-only tests for M2T2 target-query association helpers."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from policy.heuristic_baseline.errors import NoVisibleTargetFailure
from policy.heuristic_baseline.m2t2_backend import RoboTwinM2T2Backend


class M2T2TargetAssociationTest(unittest.TestCase):
    def test_query_alignment_rejects_misaligned_collections(self) -> None:
        masks = torch.zeros((2, 4), dtype=torch.bool)
        poses = [np.zeros((1, 4, 4)), np.zeros((1, 4, 4))]
        scores = [np.zeros(1)]
        contacts = [np.zeros((1, 3)), np.zeros((1, 3))]

        with self.assertRaisesRegex(ValueError, "grasp_confidence has 1 queries"):
            RoboTwinM2T2Backend._validate_query_alignment(
                masks, poses, scores, contacts
            )

    def test_contact_filter_uses_ground_truth_target_points(self) -> None:
        contacts = np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 9e-6], [0.0, 0.0, 2e-5]],
            dtype=np.float64,
        )
        target_points = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

        keep = RoboTwinM2T2Backend._target_contact_membership(
            contacts, target_points, tolerance_m=1e-5
        )

        self.assertEqual(keep.tolist(), [True, True, False])

    def test_rigid_grasp_pose_projects_m2t2_float_drift(self) -> None:
        pose = np.eye(4)
        pose[:3, :3] = np.array(
            [
                [0.1125166267, -0.6069677472, 0.7867243290],
                [-0.9931591153, -0.0438179709, 0.1082058400],
                [-0.0312226918, -0.7935174108, -0.6077468395],
            ]
        )
        pose[:3, 3] = [0.1, -0.2, 0.3]

        rigid = RoboTwinM2T2Backend._rigid_grasp_pose(pose)

        self.assertIsNotNone(rigid)
        np.testing.assert_allclose(rigid[:3, :3].T @ rigid[:3, :3], np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(rigid[:3, :3]), 1.0)
        np.testing.assert_array_equal(rigid[:3, 3], pose[:3, 3])
        np.testing.assert_array_equal(rigid[3], pose[3])

    def test_rigid_grasp_pose_rejects_grossly_invalid_frames(self) -> None:
        invalid = []
        reflection = np.eye(4)
        reflection[0, 0] = -1.0
        invalid.append(reflection)
        scaled = np.eye(4)
        scaled[0, 0] = 2.0
        invalid.append(scaled)
        degenerate = np.eye(4)
        degenerate[2, :3] = 0.0
        invalid.append(degenerate)
        nonfinite = np.eye(4)
        nonfinite[0, 0] = np.nan
        invalid.append(nonfinite)

        for pose in invalid:
            with self.subTest(pose=pose):
                self.assertIsNone(RoboTwinM2T2Backend._rigid_grasp_pose(pose))

    def test_reset_replays_private_sampling_without_global_rng(self) -> None:
        backend = object.__new__(RoboTwinM2T2Backend)
        backend.num_points = 8
        backend._seed = 17
        backend.reset()
        xyz = np.zeros((8, 3), dtype=np.float32)
        xyz[2:] = 1.0
        target_pose = np.eye(4)
        global_state = np.random.get_state()

        first = backend._sample_indices(np.array([True, True, False, False, False, False, False, False]))
        backend._sample_indices(np.array([True, True, False, False, False, False, False, False]))
        backend.reset()
        replayed = backend._sample_indices(np.array([True, True, False, False, False, False, False, False]))
        after = np.random.get_state()

        np.testing.assert_array_equal(first, replayed)
        self.assertEqual(global_state[0], after[0])
        np.testing.assert_array_equal(global_state[1], after[1])
        self.assertEqual(global_state[2:], after[2:])

    def test_sampling_is_exactly_half_target_and_half_context(self) -> None:
        backend = object.__new__(RoboTwinM2T2Backend)
        backend.num_points = 8
        backend._seed = 3
        backend.reset()
        membership = np.array(
            [True, True, False, False, False, False, False, False]
        )

        indices = backend._sample_indices(membership)

        self.assertEqual(len(indices), 8)
        self.assertEqual(int(membership[indices].sum()), 4)
        self.assertEqual(int((~membership[indices]).sum()), 4)

    def test_sampling_absent_target_is_a_visible_target_failure(self) -> None:
        backend = object.__new__(RoboTwinM2T2Backend)
        backend.num_points = 4
        backend._seed = 0
        backend.reset()

        with self.assertRaises(NoVisibleTargetFailure):
            backend._sample_indices(np.zeros(2, dtype=bool))

    def test_predict_ignores_query_masks_and_uses_target_contact(self) -> None:
        backend = object.__new__(RoboTwinM2T2Backend)
        backend.num_points = 4
        backend.num_runs = 1
        backend._seed = 4
        backend.reset()
        backend.workspace_bounds = None
        backend.contact_match_distance_m = 1e-5
        backend.torch = torch
        backend.device = torch.device("cpu")
        backend.cfg = SimpleNamespace(eval=object())
        backend._input_batch = lambda xyz, rgb, indices: {}

        wrong_pose = torch.eye(4).unsqueeze(0)
        wrong_pose[0, 0, 3] = 99.0
        matched_pose = torch.eye(4).unsqueeze(0)
        matched_pose[0, 0, 3] = 1.0

        class FakeModel:
            def infer(self, batch, config):
                randomized_pose = matched_pose.clone()
                randomized_pose[0, 0, 1] = 5e-5
                randomized_pose[0, 1, 3] = torch.randperm(100)[0].float()
                return {
                    "grasping_masks": [
                        torch.tensor([[False] * 4, [True] * 4])
                    ],
                    "grasps": [[randomized_pose, wrong_pose]],
                    "grasp_confidence": [
                        [torch.tensor([0.9]), torch.tensor([0.1])]
                    ],
                    "grasp_contacts": [
                        [torch.zeros((1, 3)), torch.tensor([[1.0, 0.0, 0.0]])]
                    ],
                }

        backend.model = FakeModel()
        xyz = np.zeros((4, 3), dtype=np.float32)
        torch_state = torch.random.get_rng_state()
        poses, scores = backend.predict(
            xyz,
            np.zeros((4, 3), dtype=np.float32),
            np.eye(4),
            xyz,
            np.ones(4, dtype=bool),
        )
        np.testing.assert_array_equal(torch.random.get_rng_state(), torch_state)

        torch.randperm(1000)
        perturbed_state = torch.random.get_rng_state()
        backend.reset(4)
        replayed_poses, replayed_scores = backend.predict(
            xyz,
            np.zeros((4, 3), dtype=np.float32),
            np.eye(4),
            xyz,
            np.ones(4, dtype=bool),
        )
        np.testing.assert_array_equal(
            torch.random.get_rng_state(), perturbed_state
        )

        self.assertEqual(len(poses), 1)
        np.testing.assert_allclose(
            poses[0][:3, :3].T @ poses[0][:3, :3], np.eye(3), atol=1e-12
        )
        self.assertAlmostEqual(np.linalg.det(poses[0][:3, :3]), 1.0)
        anchor = -0.1034 * poses[0][:3, 2]
        self.assertLessEqual(
            np.linalg.norm(poses[0][:3, 3] - anchor), 0.045 + 1e-9
        )
        np.testing.assert_allclose(scores, [0.9])
        np.testing.assert_allclose(replayed_poses, poses)
        np.testing.assert_allclose(replayed_scores, scores)


if __name__ == "__main__":
    unittest.main()

"""CPU-only boundary tests for the heuristic deployment adapter."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from policy.heuristic_baseline import deploy_policy
from policy.heuristic_baseline.errors import (
    HeuristicEpisodeFailure,
    NoFeasiblePlanFailure,
    NoObjectQueryFailure,
    NoVisibleTargetFailure,
    TargetSelectionFailure,
)
from policy.heuristic_baseline.model import HeuristicPolicy


class _Runtime:
    def __init__(self, actions: list[np.ndarray]) -> None:
        self.actions = actions

    def get_action(self, *, scene: object) -> list[np.ndarray]:
        return self.actions

    def reset(self) -> None:
        pass


class _TaskEnv:
    step_lim = 17

    def __init__(self) -> None:
        self.take_action_cnt = 0
        self.actions: list[tuple[np.ndarray, str]] = []

    def take_action(self, action: np.ndarray, *, action_type: str) -> None:
        self.actions.append((action, action_type))


class BoundaryContractTests(unittest.TestCase):
    def test_configured_root_precedes_runtime_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            received: dict[str, object] = {}

            def factory(**kwargs: object) -> _Runtime:
                received.update(kwargs)
                return _Runtime([np.array([1.0])])

            policy = HeuristicPolicy(
                {"simple_grasp_root": str(root)}, runtime_factory=factory
            )
            self.assertEqual(sys.path[0], str(root / "src"))
            policy.get_action(scene=object(), task_env=object())
            self.assertEqual(received["simple_grasp_root"], root.resolve())

    def test_empty_runtime_actions_are_an_episode_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            policy = HeuristicPolicy({}, runtime_factory=lambda **_: _Runtime([]), simple_grasp_root=root)
            with self.assertRaises(NoFeasiblePlanFailure):
                policy.get_action(scene=object(), task_env=object())

    def test_deploy_terminates_only_expected_episode_failures(self) -> None:
        env = _TaskEnv()

        class FailingModel:
            simple_grasp_root = Path("/unused")

            def get_action(self, **kwargs: object) -> list[np.ndarray]:
                raise NoObjectQueryFailure("no matched query")

        with patch.object(deploy_policy, "encode_obs", return_value=object()):
            deploy_policy.eval(env, FailingModel(), {})
        self.assertEqual(env.take_action_cnt, env.step_lim)
        self.assertEqual(env.actions, [])

    def test_failure_types_share_the_episode_boundary(self) -> None:
        for failure in (TargetSelectionFailure, NoVisibleTargetFailure, NoObjectQueryFailure, NoFeasiblePlanFailure):
            self.assertTrue(issubclass(failure, HeuristicEpisodeFailure))


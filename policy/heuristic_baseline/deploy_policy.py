"""RoboTwin entrypoints for the heuristic grasp baseline."""
from __future__ import annotations

from typing import Any

from .errors import HeuristicEpisodeFailure
from .model import HeuristicPolicy
from .observation import encode_obs


def get_model(usr_args: dict) -> HeuristicPolicy:
    return HeuristicPolicy(usr_args)


def eval(task_env: Any, model: HeuristicPolicy, observation: dict) -> None:
    try:
        actions = model.get_action(
            scene=encode_obs(
                task_env,
                observation,
                simple_grasp_root=model.simple_grasp_root,
            ),
            task_env=task_env,
        )
    except HeuristicEpisodeFailure as exc:
        print(f"[heuristic] rollout failed: {exc}")
        task_env.take_action_cnt = task_env.step_lim
        return
    for action in actions:
        task_env.take_action(action, action_type="qpos")


def reset_model(model: HeuristicPolicy) -> None:
    model.reset()
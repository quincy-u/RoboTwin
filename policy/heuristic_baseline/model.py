"""Model wrapper for the heuristic grasp baseline."""
from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from ._simple_grasp import ensure_simple_grasp_importable
from .errors import NoFeasiblePlanFailure

if TYPE_CHECKING:
    from simple_grasp.types import SceneObservation


class HeuristicRuntime(Protocol):

    def get_action(self, *, scene: "SceneObservation") -> Iterable[np.ndarray]: ...

    def reset(self) -> None: ...


RuntimeFactory = Callable[..., HeuristicRuntime]


class HeuristicPolicy:
    """Lazy wrapper around the concrete M2T2, IK, and control runtime."""

    def __init__(
        self,
        usr_args: dict,
        *,
        runtime_factory: RuntimeFactory | None = None,
        simple_grasp_root: str | Path | None = None,
    ) -> None:
        self.usr_args = dict(usr_args)
        configured_root = simple_grasp_root or self.usr_args.get("simple_grasp_root")
        self.simple_grasp_root = ensure_simple_grasp_importable(configured_root)
        self._runtime_factory = runtime_factory
        self._runtime: HeuristicRuntime | None = None

    def _default_runtime_factory(self) -> RuntimeFactory:
        # ``runtime`` imports simple_grasp at module scope, so configure this
        # policy's checkout immediately before importing it.
        ensure_simple_grasp_importable(self.simple_grasp_root)
        from .runtime import create_runtime

        return create_runtime

    def _ensure_runtime(self, task_env: Any) -> HeuristicRuntime:
        if self._runtime is None:
            factory = self._runtime_factory or self._default_runtime_factory()
            self._runtime = factory(
                usr_args=self.usr_args,
                simple_grasp_root=self.simple_grasp_root,
                task_env=task_env,
            )
        return self._runtime

    def get_action(
        self,
        *,
        scene: "SceneObservation",
        task_env: Any,
    ) -> list[np.ndarray]:
        runtime = self._ensure_runtime(task_env)
        actions = [
            np.asarray(action, dtype=np.float64)
            for action in runtime.get_action(scene=scene)
        ]
        if not actions:
            raise NoFeasiblePlanFailure("no feasible grasp plan")
        return actions

    def reset(self) -> None:
        if self._runtime is not None:
            self._runtime.reset()

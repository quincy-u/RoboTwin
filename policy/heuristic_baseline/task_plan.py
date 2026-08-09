"""Structured pick/place plans derived from RoboTwin task code and metadata."""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias


@dataclass(frozen=True)
class Pick:
    target: str
    arm: str | None = None


@dataclass(frozen=True)
class Place:
    object: str
    destination: str
    arm: str | None = None


TaskStage: TypeAlias = Pick | Place


@dataclass(frozen=True)
class TaskPlan:
    task_name: str
    family: str
    stages: tuple[TaskStage, ...]

    @property
    def primary_target(self) -> str | None:
        return self.stages[0].target if self.stages and isinstance(self.stages[0], Pick) else None


def _task_source(task_name: str) -> str:
    path = Path(__file__).resolve().parents[2] / "envs" / f"{task_name}.py"
    return path.read_text() if path.is_file() else ""


def task_primitives(task_name: str) -> frozenset[str]:
    """Find expert manipulation primitives from the task module AST."""
    source = _task_source(task_name)
    if not source:
        return frozenset()
    tree = ast.parse(source)
    calls: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Attribute):
            calls.add(function.attr)
        elif isinstance(function, ast.Name):
            calls.add(function.id)
    return frozenset(calls)


def task_family(task_name: str) -> str:
    primitives = task_primitives(task_name)
    if "place_actor" in primitives:
        return "pick_place"
    if "grasp_actor" in primitives:
        return "pick"
    return "other"


def _actor_dataset_label(actor: Any) -> str | None:
    modelname = getattr(actor, "modelname", None)
    model_id = getattr(actor, "model_id", None)
    if not modelname:
        return None
    suffix = 0 if model_id is None or model_id == "" else model_id
    return f"{modelname}/base{suffix}"


def _plain_object_name(label: str) -> str:
    model = label.split("/", 1)[0]
    return re.sub(r"^\d+_", "", model)


def _resolve_object_name(label: str, tracked: dict[str, Any]) -> str:
    matches = [name for name, actor in tracked.items() if _actor_dataset_label(actor) == label]
    if len(matches) == 1:
        return matches[0]
    return _plain_object_name(label)


def task_plan_from_task(task_env: Any, task_name: str, info: dict[str, str]) -> TaskPlan:
    """Derive a minimal plan from expert code and resolve metadata to actor names."""
    family = task_family(task_name)
    if family == "other":
        return TaskPlan(task_name, family, ())
    primary_label = info.get("{A}")
    if not primary_label:
        raise ValueError("RoboTwin task metadata does not define primary target {A}")
    tracked = task_env.get_tracked_objects() or {}
    primary = _resolve_object_name(primary_label, tracked)
    arm = info.get("{a}")
    stages: list[TaskStage] = [Pick(primary, arm)]
    destination_label = info.get("{B}")
    if family == "pick_place" and destination_label:
        destination = _resolve_object_name(destination_label, tracked)
        stages.append(Place(primary, destination, arm))
    return TaskPlan(task_name, family, tuple(stages))


__all__ = [
    "Pick", "Place", "TaskPlan", "TaskStage", "task_family",
    "task_plan_from_task", "task_primitives",
]

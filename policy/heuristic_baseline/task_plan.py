"""Task plans recorded from RoboTwin's own procedural expert calls."""
from __future__ import annotations

import inspect
from dataclasses import dataclass, replace
from typing import Any, TypeAlias

import numpy as np
import transforms3d as t3d


Point3: TypeAlias = tuple[float, float, float]
PoseMatrix: TypeAlias = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]


@dataclass(frozen=True)
class Pick:
    target: str
    arm: str | None = None
    pregrasp_offset_m: float | None = None
    postgrasp_displacement: Point3 | None = None
    grasp_offset_m: float = 0.0
    gripper_target: float = 0.0
    allowed_contact_points_local: tuple[Point3, ...] | None = None
    group_id: int | None = None


@dataclass(frozen=True)
class Place:
    object: str
    destination: str | None
    arm: str | None = None
    object_functional_point_id: int | None = None
    destination_functional_point_id: int | None = None
    target_pose_attr: str | None = None
    preplace_offset_m: float = 0.1
    place_offset_m: float = 0.02
    constrain: str = "auto"
    preplace_axis: str | Point3 = "grasp"
    release: bool = True
    target_pose: PoseMatrix | None = None
    destination_offset: Point3 | None = None
    group_id: int | None = None


@dataclass(frozen=True)
class Handoff:
    object: str
    from_arm: str
    to_arm: str
    rendezvous_pose_attr: str | None = None
    object_functional_point_id: int | None = 0
    pregrasp_offset_m: float = 0.07
    constrain: str = "free"
    rendezvous_pose: PoseMatrix | None = None
    grasp_offset_m: float = 0.0
    allowed_contact_points_local: tuple[Point3, ...] | None = None
    group_id: int | None = None
    gripper_target: float = 0.0


TaskStage: TypeAlias = Pick | Place | Handoff


@dataclass(frozen=True)
class TaskPlan:
    task_name: str
    family: str
    stages: tuple[TaskStage, ...]

    @property
    def primary_target(self) -> str | None:
        return (
            self.stages[0].target
            if self.stages and isinstance(self.stages[0], Pick)
            else None
        )

    @property
    def manipulation_targets(self) -> tuple[str, ...]:
        """Objects that require grasp proposals and target segmentation masks."""
        targets: list[str] = []
        for stage in self.stages:
            if isinstance(stage, Pick) and stage.target not in targets:
                targets.append(stage.target)
        return tuple(targets)

    @property
    def pose_objects(self) -> tuple[str, ...]:
        """Tracked objects whose GT 6D poses are required by this plan."""
        names = list(self.manipulation_targets)
        for stage in self.stages:
            if (
                isinstance(stage, Place)
                and stage.destination is not None
                and stage.destination not in names
            ):
                names.append(stage.destination)
        return tuple(names)


@dataclass(frozen=True)
class PrimitiveCall:
    """Immutable data captured before one RoboTwin expert primitive runs."""

    kind: str
    group_id: int | None = None
    object_name: str | None = None
    arm: str | None = None
    pre_offset_m: float | None = None
    final_offset_m: float | None = None
    gripper_target: float | None = None
    allowed_contact_points_local: tuple[Point3, ...] | None = None
    displacement: Point3 | None = None
    move_axis: str | None = None
    target_pose: PoseMatrix | None = None
    destination: str | None = None
    object_functional_point_id: int | None = None
    destination_functional_point_id: int | None = None
    destination_offset: Point3 | None = None
    constrain: str | None = None
    preplace_axis: str | Point3 | None = None
    release: bool | None = None


def _pose_matrix(pose: Any) -> np.ndarray:
    if hasattr(pose, "to_transformation_matrix"):
        matrix = np.asarray(pose.to_transformation_matrix(), dtype=np.float64)
    else:
        values = np.asarray(pose, dtype=np.float64)
        if values.shape == (4, 4):
            matrix = values.copy()
        elif values.shape == (7,):
            matrix = np.eye(4, dtype=np.float64)
            matrix[:3, 3] = values[:3]
            matrix[:3, :3] = t3d.quaternions.quat2mat(values[3:])
        else:
            raise ValueError("procedural pose must be a pose7 or 4x4 matrix")
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("procedural pose must be a finite 4x4 transform")
    return matrix


def _freeze_pose(pose: Any) -> PoseMatrix:
    matrix = _pose_matrix(pose)
    return tuple(tuple(float(value) for value in row) for row in matrix)  # type: ignore[return-value]


def _freeze_axis(axis: Any) -> str | Point3:
    if isinstance(axis, str):
        return axis
    values = np.asarray(axis, dtype=np.float64)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError("pre-placement axis must be a finite 3-vector")
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def _normalize_arm(arm: Any) -> str:
    normalized = str(arm).strip().lower()
    if normalized not in {"left", "right"}:
        raise ValueError(f"unknown RoboTwin arm {arm!r}")
    return normalized


def _functional_point_count(actor: Any) -> int:
    config = getattr(actor, "config", None)
    points = getattr(actor, "POINTS", None)
    if not isinstance(config, dict) or not isinstance(points, dict):
        return 0
    values = config.get(points.get("functional", ""), ())
    try:
        return len(values)
    except TypeError:
        return 0


def _match_destination(
    tracked: dict[str, Any], source_actor: Any, target_pose: Any
) -> tuple[str | None, int | None]:
    target = _pose_matrix(target_pose)
    matches: list[tuple[str, int | None]] = []
    for name, actor in tracked.items():
        if actor is source_actor:
            continue
        try:
            actor_pose = _pose_matrix(actor.get_pose())
        except (AttributeError, TypeError, ValueError):
            actor_pose = None
        if actor_pose is not None and np.allclose(
            target, actor_pose, atol=1e-6, rtol=0.0
        ):
            matches.append((name, None))
        getter = getattr(actor, "get_functional_point", None)
        if getter is None:
            continue
        for index in range(_functional_point_count(actor)):
            try:
                candidate = getter(index, "matrix")
                if candidate is not None and np.allclose(
                    target, _pose_matrix(candidate), atol=1e-6, rtol=0.0
                ):
                    matches.append((name, index))
            except (IndexError, KeyError, TypeError, ValueError):
                continue
    if not matches:
        return None, None
    matches.sort(key=lambda item: (item[0], item[1] is None, item[1] or 0))
    return matches[0]


def _match_position_destination(
    tracked: dict[str, Any], source_actor: Any, target_position: Any
) -> tuple[str, Point3]:
    """Anchor a position-only placement on the nearest tracked object."""
    position = np.asarray(target_position, dtype=np.float64)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("procedural position must be a finite 3-vector")
    candidates: list[tuple[float, str, np.ndarray]] = []
    for name, actor in tracked.items():
        if actor is source_actor:
            continue
        try:
            anchor = _pose_matrix(actor.get_pose())[:3, 3]
        except (AttributeError, TypeError, ValueError):
            continue
        candidates.append((float(np.linalg.norm(position - anchor)), name, anchor))
    if not candidates:
        raise ValueError(
            "procedural position requires another tracked object as its anchor"
        )
    _, name, anchor = min(candidates, key=lambda item: (item[0], item[1]))
    offset = position - anchor
    return name, tuple(float(value) for value in offset)  # type: ignore[return-value]


def _allowed_contacts_local(
    actor: Any, contact_point_id: Any
) -> tuple[Point3, ...] | None:
    if contact_point_id is None:
        return None
    if isinstance(contact_point_id, (list, tuple, set, frozenset, np.ndarray)):
        indices = [int(index) for index in contact_point_id]
    else:
        indices = [int(contact_point_id)]
    world_object = _pose_matrix(actor.get_pose())
    object_from_world = np.linalg.inv(world_object)
    points: list[Point3] = []
    for index in indices:
        world_contact = actor.get_contact_point(index, "matrix")
        if world_contact is None:
            continue
        local = object_from_world @ _pose_matrix(world_contact)
        points.append(tuple(float(value) for value in local[:3, 3]))
    return tuple(points)


class ProceduralTaskRecorder:
    """Record expert primitives without changing their execution behavior."""

    _METHODS = (
        "grasp_actor",
        "place_actor",
        "move_by_displacement",
        "move_to_pose",
        "open_gripper",
        "move",
    )

    def __init__(self, task_env: Any) -> None:
        self.task_env = task_env
        self.calls: list[PrimitiveCall] = []
        self._tracked: dict[str, Any] = {}
        self._actor_names: dict[int, str] = {}
        self._action_call_indices: dict[int, int] = {}
        self._action_results: dict[int, Any] = {}
        self._execution_order: list[int] = []
        self._group_id = 0
        self._restore: dict[str, tuple[bool, Any]] = {}
        self._held_object_by_arm: dict[str, str] = {}

    @property
    def trace(self) -> tuple[PrimitiveCall, ...]:
        return tuple(self.calls[index] for index in self._execution_order)

    def _object_name(self, actor: Any) -> str:
        try:
            return self._actor_names[id(actor)]
        except KeyError as exc:
            raise ValueError(
                "expert primitive references an object absent from "
                "get_tracked_objects()"
            ) from exc

    @staticmethod
    def _arguments(method: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
        bound = inspect.signature(method).bind_partial(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)

    def _append(self, call: PrimitiveCall) -> int:
        self.calls.append(call)
        return len(self.calls) - 1

    def _remember_result(self, index: int, result: Any) -> Any:
        self._action_call_indices[id(result)] = index
        self._action_results[id(result)] = result
        return result

    def _patch(self, name: str, wrapper: Any) -> None:
        attributes = vars(self.task_env)
        self._restore[name] = (name in attributes, attributes.get(name))
        setattr(self.task_env, name, wrapper)

    def __enter__(self) -> "ProceduralTaskRecorder":
        if self._restore:
            raise RuntimeError("procedural recorder cannot be re-entered")
        self._tracked = dict(self.task_env.get_tracked_objects() or {})
        self._actor_names = {}
        for name, actor in self._tracked.items():
            self._actor_names.setdefault(id(actor), name)

        grasp = getattr(self.task_env, "grasp_actor")
        place = getattr(self.task_env, "place_actor")
        displacement = getattr(self.task_env, "move_by_displacement")
        move_to_pose = getattr(self.task_env, "move_to_pose")
        open_gripper = getattr(self.task_env, "open_gripper")
        move = getattr(self.task_env, "move")

        def record_grasp(*args: Any, **kwargs: Any) -> Any:
            values = self._arguments(grasp, args, kwargs)
            actor = values["actor"]
            arm = _normalize_arm(values["arm_tag"])
            object_name = self._object_name(actor)
            self._held_object_by_arm[arm] = object_name
            index = self._append(
                PrimitiveCall(
                    "grasp",
                    object_name=object_name,
                    arm=arm,
                    pre_offset_m=float(values["pre_grasp_dis"]),
                    final_offset_m=float(values["grasp_dis"]),
                    gripper_target=float(values["gripper_pos"]),
                    allowed_contact_points_local=_allowed_contacts_local(
                        actor, values["contact_point_id"]
                    ),
                )
            )
            return self._remember_result(index, grasp(*args, **kwargs))

        def record_place(*args: Any, **kwargs: Any) -> Any:
            values = self._arguments(place, args, kwargs)
            actor = values["actor"]
            source_name = self._object_name(actor)
            extras = values.get("args", {})
            if not isinstance(extras, dict):
                extras = {}
            target_pose = values["target_pose"]
            target_values = None
            if not hasattr(target_pose, "to_transformation_matrix"):
                target_values = np.asarray(target_pose, dtype=np.float64)
            if target_values is not None and target_values.shape == (3,):
                destination, destination_offset = _match_position_destination(
                    self._tracked, actor, target_values
                )
                destination_fp = None
                frozen_target_pose = None
            else:
                destination, destination_fp = _match_destination(
                    self._tracked, actor, target_pose
                )
                destination_offset = None
                frozen_target_pose = _freeze_pose(target_pose)
            index = self._append(
                PrimitiveCall(
                    "place",
                    object_name=source_name,
                    arm=_normalize_arm(values["arm_tag"]),
                    pre_offset_m=float(values["pre_dis"]),
                    final_offset_m=float(values["dis"]),
                    target_pose=frozen_target_pose,
                    destination=destination,
                    object_functional_point_id=values["functional_point_id"],
                    destination_functional_point_id=destination_fp,
                    destination_offset=destination_offset,
                    constrain=str(extras.get("constrain", "auto")),
                    preplace_axis=_freeze_axis(
                        extras.get("pre_dis_axis", "grasp")
                    ),
                    release=bool(values["is_open"]),
                )
            )
            return self._remember_result(index, place(*args, **kwargs))

        def record_displacement(*args: Any, **kwargs: Any) -> Any:
            values = self._arguments(displacement, args, kwargs)
            index = self._append(
                PrimitiveCall(
                    "displacement",
                    arm=_normalize_arm(values["arm_tag"]),
                    displacement=(
                        float(values["x"]),
                        float(values["y"]),
                        float(values["z"]),
                    ),
                    move_axis=str(values["move_axis"]),
                )
            )
            return self._remember_result(
                index, displacement(*args, **kwargs)
            )

        def record_move_to_pose(*args: Any, **kwargs: Any) -> Any:
            values = self._arguments(move_to_pose, args, kwargs)
            arm_tag = values["arm_tag"]
            arm = _normalize_arm(arm_tag)
            object_name = self._held_object_by_arm.get(arm)
            if object_name is None:
                return move_to_pose(*args, **kwargs)
            held = self._tracked[object_name]
            world_object = _pose_matrix(held.get_pose())
            world_gripper = _pose_matrix(self.task_env.get_arm_pose(arm_tag))
            desired_gripper = _pose_matrix(values["target_pose"])
            desired_object = (
                desired_gripper @ np.linalg.inv(world_gripper) @ world_object
            )
            index = self._append(
                PrimitiveCall(
                    "place",
                    object_name=object_name,
                    arm=arm,
                    pre_offset_m=0.0,
                    final_offset_m=0.0,
                    target_pose=_freeze_pose(desired_object),
                    constrain="free",
                    preplace_axis="grasp",
                    release=False,
                )
            )
            return self._remember_result(
                index, move_to_pose(*args, **kwargs)
            )

        def record_open(*args: Any, **kwargs: Any) -> Any:
            values = self._arguments(open_gripper, args, kwargs)
            arm = _normalize_arm(values["arm_tag"])
            object_name = self._held_object_by_arm.get(arm)
            target_pose = (
                None
                if object_name is None
                else _freeze_pose(self._tracked[object_name].get_pose())
            )
            index = self._append(
                PrimitiveCall(
                    "open",
                    arm=arm,
                    object_name=object_name,
                    target_pose=target_pose,
                    gripper_target=float(values["pos"]),
                )
            )
            return self._remember_result(index, open_gripper(*args, **kwargs))

        def record_move(*args: Any, **kwargs: Any) -> Any:
            group_indices: list[int] = []
            for action in (*args, *kwargs.values()):
                action_id = id(action)
                index = self._action_call_indices.pop(action_id, None)
                if index is not None and index not in group_indices:
                    group_indices.append(index)
                self._action_results.pop(action_id, None)
            if group_indices:
                for index in group_indices:
                    self.calls[index] = replace(
                        self.calls[index], group_id=self._group_id
                    )
                self._execution_order.extend(group_indices)
            self._group_id += 1
            return move(*args, **kwargs)

        self._patch("grasp_actor", record_grasp)
        self._patch("place_actor", record_place)
        self._patch("move_by_displacement", record_displacement)
        self._patch("move_to_pose", record_move_to_pose)
        self._patch("open_gripper", record_open)
        self._patch("move", record_move)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        for name in reversed(self._METHODS):
            had_override, previous = self._restore[name]
            if had_override:
                setattr(self.task_env, name, previous)
            else:
                delattr(self.task_env, name)
        self._restore = {}
        self._action_call_indices = {}
        self._action_results = {}


def _place_from_call(call: PrimitiveCall, *, release: bool | None = None) -> Place:
    if (
        call.object_name is None
        or call.arm is None
        or (call.target_pose is None and call.destination is None)
    ):
        raise ValueError("incomplete recorded place primitive")
    return Place(
        object=call.object_name,
        destination=call.destination,
        arm=call.arm,
        object_functional_point_id=call.object_functional_point_id,
        destination_functional_point_id=call.destination_functional_point_id,
        preplace_offset_m=float(call.pre_offset_m or 0.0),
        place_offset_m=float(call.final_offset_m or 0.0),
        constrain=str(call.constrain or "auto"),
        preplace_axis=call.preplace_axis or "grasp",
        release=bool(call.release if release is None else release),
        target_pose=call.target_pose,
        destination_offset=call.destination_offset,
        group_id=call.group_id,
    )


def task_plan_from_trace(
    task_env: Any,
    task_name: str,
    trace: tuple[PrimitiveCall, ...] | list[PrimitiveCall],
) -> TaskPlan:
    """Reduce recorded expert calls into Pick/Place/Handoff stages."""
    tracked = dict(task_env.get_tracked_objects() or {})
    calls = tuple(trace)
    for call in calls:
        if call.object_name is not None and call.object_name not in tracked:
            raise ValueError(
                f"recorded object {call.object_name!r} is not tracked in replay"
            )

    stages: list[TaskStage] = []
    active_object_by_arm: dict[str, str] = {}
    owner_by_object: dict[str, str] = {}
    active_pick_by_arm: dict[str, int] = {}
    active_handoff_by_arm: dict[str, int] = {}
    terminal_place_by_arm: dict[str, int] = {}
    pending_places: dict[str, tuple[int, PrimitiveCall]] = {}

    def flush_pending(object_name: str, *, release: bool | None = None) -> None:
        pending = pending_places.pop(object_name, None)
        if pending is not None:
            _, call = pending
            stages.append(_place_from_call(call, release=release))

    for call_index, call in enumerate(calls):
        if call.kind == "grasp":
            if call.object_name is None or call.arm is None:
                raise ValueError("recorded grasp lacks object or arm")
            previous_owner = owner_by_object.get(call.object_name)
            pending = pending_places.get(call.object_name)
            if (
                previous_owner is not None
                and previous_owner != call.arm
                and pending is not None
            ):
                _, rendezvous = pending_places.pop(call.object_name)
                if rendezvous.target_pose is None:
                    raise ValueError("recorded handoff lacks rendezvous pose")
                stages.append(
                    Handoff(
                        object=call.object_name,
                        from_arm=previous_owner,
                        to_arm=call.arm,
                        object_functional_point_id=(
                            rendezvous.object_functional_point_id
                        ),
                        pregrasp_offset_m=float(call.pre_offset_m or 0.0),
                        constrain=str(rendezvous.constrain or "free"),
                        rendezvous_pose=rendezvous.target_pose,
                        grasp_offset_m=float(call.final_offset_m or 0.0),
                        allowed_contact_points_local=(
                            call.allowed_contact_points_local
                        ),
                        group_id=rendezvous.group_id,
                        gripper_target=float(call.gripper_target or 0.0),
                    )
                )
                owner_by_object[call.object_name] = call.arm
                active_object_by_arm[call.arm] = call.object_name
                active_pick_by_arm.pop(call.arm, None)
                active_handoff_by_arm[call.arm] = len(stages) - 1
                terminal_place_by_arm.pop(call.arm, None)
                continue

            if pending is not None:
                flush_pending(call.object_name)
            pick = Pick(
                target=call.object_name,
                arm=call.arm,
                pregrasp_offset_m=call.pre_offset_m,
                grasp_offset_m=float(call.final_offset_m or 0.0),
                gripper_target=float(call.gripper_target or 0.0),
                allowed_contact_points_local=call.allowed_contact_points_local,
                group_id=call.group_id,
            )
            stages.append(pick)
            active_object_by_arm[call.arm] = call.object_name
            owner_by_object[call.object_name] = call.arm
            active_pick_by_arm[call.arm] = len(stages) - 1
            active_handoff_by_arm.pop(call.arm, None)
            terminal_place_by_arm.pop(call.arm, None)
            continue

        if call.kind == "displacement" and call.arm is not None:
            object_name = active_object_by_arm.get(call.arm)
            pick_index = active_pick_by_arm.get(call.arm)
            if (
                object_name is not None
                and pick_index is not None
                and object_name not in pending_places
                and call.move_axis == "world"
                and call.displacement is not None
            ):
                pick = stages[pick_index]
                if not isinstance(pick, Pick):
                    raise RuntimeError("recorded pick index is invalid")
                previous = pick.postgrasp_displacement or (0.0, 0.0, 0.0)
                stages[pick_index] = replace(
                    pick,
                    postgrasp_displacement=tuple(
                        float(first + second)
                        for first, second in zip(previous, call.displacement)
                    ),
                )
            handoff_index = active_handoff_by_arm.get(call.arm)
            if (
                object_name is not None
                and owner_by_object.get(object_name) == call.arm
                and handoff_index is not None
                and call.move_axis == "world"
                and call.displacement is not None
            ):
                handoff = stages[handoff_index]
                if not isinstance(handoff, Handoff):
                    raise RuntimeError("recorded handoff index is invalid")
                terminal_index = terminal_place_by_arm.get(call.arm)
                if terminal_index is None:
                    rendezvous = handoff.rendezvous_pose
                    if rendezvous is None:
                        raise ValueError("recorded handoff lacks rendezvous pose")
                    target = np.asarray(rendezvous, dtype=np.float64).copy()
                else:
                    terminal = stages[terminal_index]
                    if not isinstance(terminal, Place):
                        raise RuntimeError(
                            "recorded terminal Place index is invalid"
                        )
                    target = np.asarray(
                        terminal.target_pose, dtype=np.float64
                    ).copy()
                target[:3, 3] += np.asarray(call.displacement, dtype=np.float64)
                terminal = Place(
                    object=object_name,
                    destination=None,
                    arm=call.arm,
                    object_functional_point_id=(
                        handoff.object_functional_point_id
                    ),
                    target_pose=_freeze_pose(target),
                    preplace_offset_m=0.0,
                    place_offset_m=0.0,
                    constrain="free",
                    release=False,
                    group_id=call.group_id,
                )
                if terminal_index is None:
                    stages.append(terminal)
                    terminal_place_by_arm[call.arm] = len(stages) - 1
                else:
                    stages[terminal_index] = terminal
            continue

        if call.kind == "place":
            if call.object_name is None or call.arm is None:
                raise ValueError("recorded place lacks object or arm")
            flush_pending(call.object_name)
            if call.release:
                stages.append(_place_from_call(call))
                active_object_by_arm.pop(call.arm, None)
                active_pick_by_arm.pop(call.arm, None)
                active_handoff_by_arm.pop(call.arm, None)
                terminal_place_by_arm.pop(call.arm, None)
                if owner_by_object.get(call.object_name) == call.arm:
                    owner_by_object.pop(call.object_name, None)
            else:
                pending_places[call.object_name] = (call_index, call)
            continue

        if call.kind == "open" and call.arm is not None:
            object_name = active_object_by_arm.pop(call.arm, None)
            active_pick_by_arm.pop(call.arm, None)
            active_handoff_by_arm.pop(call.arm, None)
            terminal_place_by_arm.pop(call.arm, None)
            if object_name is None:
                continue
            pending = pending_places.get(object_name)
            if pending is not None and pending[1].arm == call.arm:
                flush_pending(object_name, release=True)
            elif (
                owner_by_object.get(object_name) == call.arm
                and call.target_pose is not None
            ):
                stages.append(
                    Place(
                        object=object_name,
                        destination=None,
                        arm=call.arm,
                        target_pose=call.target_pose,
                        preplace_offset_m=0.0,
                        place_offset_m=0.0,
                        constrain="free",
                        release=True,
                        group_id=call.group_id,
                    )
                )
            if owner_by_object.get(object_name) == call.arm:
                owner_by_object.pop(object_name, None)

    for object_name, _ in sorted(
        pending_places.items(), key=lambda item: item[1][0]
    ):
        flush_pending(object_name)

    if any(isinstance(stage, Handoff) for stage in stages):
        family = "handoff"
    elif any(isinstance(stage, Place) for stage in stages):
        family = "pick_place"
    elif any(isinstance(stage, Pick) for stage in stages):
        family = "pick"
    else:
        family = "other"
    return TaskPlan(task_name, family, tuple(stages))


__all__ = [
    "Handoff",
    "Pick",
    "Place",
    "PoseMatrix",
    "PrimitiveCall",
    "ProceduralTaskRecorder",
    "TaskPlan",
    "TaskStage",
    "task_plan_from_trace",
]

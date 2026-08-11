"""RoboTwin entrypoints for the heuristic grasp baseline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from transforms3d.quaternions import quat2mat

from .errors import HeuristicEpisodeFailure
from .model import HeuristicPolicy
from .observation import encode_obs

_MOTION_ENDPOINT_PHASES = frozenset(
    {"pregrasp", "grasp", "retreat", "lift", "transport", "place"}
)
_POST_CLOSE_MOTION_PHASES = frozenset({"retreat", "lift", "transport", "place"})
_ARM_NAMES = ("left", "right")


def _metadata_arms(metadata: dict[str, Any]) -> tuple[str, ...]:
    """Return the arms commanded by one action-metadata record."""
    arm = metadata.get("arm")
    if arm == "both":
        return _ARM_NAMES
    if arm in _ARM_NAMES:
        return (arm,)
    return ()


def _arm_measurements(
    measurements: dict[str, Any], arm: str, *, bimanual: bool
) -> dict[str, Any]:
    if not bimanual:
        return measurements
    nested = measurements.get("arm_measurements", {})
    if not isinstance(nested, dict):
        return {}
    arm_result = nested.get(arm, {})
    return arm_result if isinstance(arm_result, dict) else {}


def _execution_guard_failures(
    measurements: dict[str, Any],
    *,
    qpos_tolerance_rad: float,
    ee_position_tolerance_m: float,
    ee_orientation_tolerance_rad: float,
) -> list[str]:
    """Return concise reasons why measured execution missed its command."""
    nested = measurements.get("arm_measurements")
    if isinstance(nested, dict):
        failures: list[str] = []
        for arm in _ARM_NAMES:
            arm_measurements = nested.get(arm, {})
            if not isinstance(arm_measurements, dict):
                arm_measurements = {}
            failures.extend(
                f"{arm}.{failure}"
                for failure in _execution_guard_failures(
                    arm_measurements,
                    qpos_tolerance_rad=qpos_tolerance_rad,
                    ee_position_tolerance_m=ee_position_tolerance_m,
                    ee_orientation_tolerance_rad=ee_orientation_tolerance_rad,
                )
            )
        return failures

    limits = (
        ("qpos_max_error_rad", qpos_tolerance_rad),
        ("ee_position_error_m", ee_position_tolerance_m),
        ("ee_orientation_error_raw_rad", ee_orientation_tolerance_rad),
    )
    failures: list[str] = []
    for name, limit in limits:
        if not np.isfinite(limit) or limit <= 0.0:
            raise ValueError(f"{name} tolerance must be finite and positive")
        value = measurements.get(name)
        if value is None or not np.isfinite(value):
            failures.append(f"{name}=missing")
        elif float(value) > limit:
            failures.append(f"{name}={float(value):.4f}>{limit:.4f}")
    return failures


def _execution_guard_settings(
    model: HeuristicPolicy,
) -> tuple[bool, float, float, float, float, float]:
    args = getattr(model, "usr_args", {})
    return (
        bool(args.get("execution_guard_enabled", True)),
        float(args.get("execution_guard_qpos_tolerance_rad", 0.10)),
        float(args.get("execution_guard_ee_position_tolerance_m", 0.03)),
        float(args.get("execution_guard_ee_orientation_tolerance_rad", 0.20)),
        float(args.get("execution_guard_gripper_min_delta", 0.10)),
        float(args.get("execution_guard_gripper_settle_delta_max", 0.05)),
    )


def _gripper_execution_guard_failures(
    *,
    initial_state: float | None,
    states: list[float],
    min_delta: float,
    settle_delta_max: float,
) -> list[str]:
    """Check that closure responded and stopped moving before retreat."""
    if not np.isfinite(min_delta) or not 0.0 < min_delta <= 1.0:
        raise ValueError("execution_guard_gripper_min_delta must be in (0, 1]")
    if not np.isfinite(settle_delta_max) or not 0.0 < settle_delta_max <= 1.0:
        raise ValueError(
            "execution_guard_gripper_settle_delta_max must be in (0, 1]"
        )
    if initial_state is None or not np.isfinite(initial_state):
        return ["gripper_initial_physical_state=missing"]
    if len(states) < 2 or not all(np.isfinite(value) for value in states[-2:]):
        return ["gripper_settle_samples<2"]
    failures: list[str] = []
    closure_delta = float(initial_state) - float(states[-1])
    if closure_delta < min_delta:
        failures.append(f"gripper_closure_delta={closure_delta:.4f}<{min_delta:.4f}")
    settle_delta = abs(float(states[-1]) - float(states[-2]))
    if settle_delta > settle_delta_max:
        failures.append(
            f"gripper_settle_delta={settle_delta:.4f}>{settle_delta_max:.4f}"
        )
    return failures


def _robot_measurements(task_env: Any, metadata: dict[str, Any]) -> dict[str, Any]:
    arm = metadata.get("arm")
    if arm == "both":
        targets = metadata.get("arm_targets", {})
        if not isinstance(targets, dict):
            targets = {}
        arm_measurements: dict[str, dict[str, Any]] = {}
        for side in _ARM_NAMES:
            target_metadata = targets.get(side, {})
            if not isinstance(target_metadata, dict):
                target_metadata = {}
            arm_measurements[side] = _robot_measurements(
                task_env, {**target_metadata, "arm": side}
            )
        return {"arm_measurements": arm_measurements}
    robot = task_env.robot
    result: dict[str, Any] = {}
    if arm in {"left", "right"}:
        getter = getattr(robot, f"get_{arm}_arm_real_jointState", None)
        if getter is None:
            getter = getattr(robot, f"get_{arm}_arm_jointState", None)
        if getter is not None:
            state = np.asarray(getter(), dtype=np.float64)
            real_qpos = state[:-1]
            target_qpos = np.asarray(metadata.get("target_qpos", []), dtype=np.float64)
            result["real_qpos"] = real_qpos.tolist()
            if target_qpos.shape == real_qpos.shape:
                error = target_qpos - real_qpos
                result["qpos_max_error_rad"] = float(np.max(np.abs(error)))
                result["qpos_l2_error_rad"] = float(np.linalg.norm(error))

        gripper_getter = getattr(robot, f"get_{arm}_gripper_val", None)
        if gripper_getter is not None:
            result["gripper_state"] = float(gripper_getter())
        try:
            entity = getattr(robot, f"{arm}_entity")
            active_joints = list(entity.get_active_joints())
            qpos = np.asarray(entity.get_qpos(), dtype=np.float64)
            gripper_entries = list(getattr(robot, f"{arm}_gripper"))
            scale = np.asarray(getattr(robot, f"{arm}_gripper_scale"), dtype=np.float64)
            normalized = []
            for joint, multiplier, offset in gripper_entries:
                if joint not in active_joints or float(multiplier) == 0.0:
                    continue
                raw = (qpos[active_joints.index(joint)] - float(offset)) / float(multiplier)
                normalized.append((raw - scale[0]) / (scale[1] - scale[0]))
            if normalized:
                result["gripper_physical_state"] = float(
                    np.clip(np.median(normalized), 0.0, 1.0)
                )
        except (AttributeError, IndexError, TypeError, ValueError, ZeroDivisionError):
            pass

        command_pose = metadata.get("command_pose")
        ee_getter = getattr(robot, f"get_{arm}_ee_pose", None)
        if command_pose is not None and ee_getter is not None:
            actual = np.asarray(ee_getter(), dtype=np.float64)
            target = np.asarray(command_pose, dtype=np.float64)
            if actual.shape == (7,) and target.shape == (4, 4):
                actual_rotation = quat2mat(actual[3:])
                delta_rotation = target[:3, :3].T @ actual_rotation
                raw_cosine = np.clip(
                    (np.trace(delta_rotation) - 1.0) / 2.0, -1.0, 1.0
                )
                raw_orientation_error = float(np.arccos(raw_cosine))
                result["actual_ee_pose"] = actual.tolist()
                result["ee_position_error_m"] = float(
                    np.linalg.norm(target[:3, 3] - actual[:3])
                )
                result["ee_orientation_error_raw_rad"] = raw_orientation_error
                result["ee_orientation_error_rad"] = raw_orientation_error

    target_name = metadata.get("target_name")
    result["target_name"] = target_name
    if target_name:
        try:
            actor = (task_env.get_tracked_objects() or {})[target_name]
            pose = actor.get_pose()
            target_pose = np.concatenate((np.asarray(pose.p), np.asarray(pose.q)))
            result["target_pose"] = target_pose.astype(float).tolist()
            result["target_z_m"] = float(target_pose[2])
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
    return result


def _execution_trace_path(task_env: Any, model: HeuristicPolicy) -> Path | None:
    save_dir = getattr(model, "usr_args", {}).get("eval_save_dir")
    if not save_dir:
        return None
    episode = int(getattr(task_env, "test_num", 0))
    seed = int(getattr(task_env, "episode_seed", 0))
    trace_dir = Path(save_dir) / "execution_trace"
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"episode{episode:04d}_seed{seed}.jsonl"
    episode_key = (episode, seed)
    if getattr(model, "_execution_trace_episode", None) != episode_key:
        path.write_text("", encoding="utf-8")
        model._execution_trace_episode = episode_key
        print(f"[heuristic] execution trace: {path}")
    return path


def _trace_arm_targets(metadata: dict[str, Any]) -> dict[str, Any] | None:
    if metadata.get("arm") != "both":
        return None
    targets = metadata.get("arm_targets", {})
    if not isinstance(targets, dict):
        targets = {}
    result: dict[str, Any] = {}
    for side in _ARM_NAMES:
        target = targets.get(side, {})
        if not isinstance(target, dict):
            target = {}
        target_qpos = target.get("target_qpos")
        command_pose = target.get("command_pose")
        result[side] = {
            "target_qpos": (
                None
                if target_qpos is None
                else np.asarray(target_qpos, dtype=np.float64).tolist()
            ),
            "target_gripper": target.get("target_gripper"),
            "command_pose": (
                None
                if command_pose is None
                else np.asarray(command_pose, dtype=np.float64).tolist()
            ),
            "target_name": target.get("target_name"),
        }
    return result


def _write_execution_record(path: Path | None, record: dict[str, Any]) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def _print_endpoint(record: dict[str, Any]) -> None:
    if not record.get("endpoint") or record["status"] != "executed":
        return
    if record.get("arm") == "both":
        targets = record.get("arm_targets") or {}
        measurements = record.get("arm_measurements") or {}
        arm_source = record.get("arm_source")
        for side in _ARM_NAMES:
            side_target = targets.get(side, {})
            side_measurement = measurements.get(side, {})
            side_record = {**record, **side_measurement}
            side_record["arm"] = side
            side_record["gripper_target"] = side_target.get("target_gripper")
            if isinstance(arm_source, dict):
                side_record["arm_source"] = arm_source.get(side)
            _print_endpoint(side_record)
        return

    def value(name: str, unit: str = "") -> str:
        item = record.get(name)
        return "n/a" if item is None else f"{item:.4f}{unit}"

    print(
        f"[heuristic][exec] phase={record['phase']} arm={record.get('arm')} "
        f"arm_source={record.get('arm_source', 'n/a')} "
        f"qerr_max={value('qpos_max_error_rad', 'rad')} "
        f"ee_pos={value('ee_position_error_m', 'm')} "
        f"ee_rot={value('ee_orientation_error_rad', 'rad')} "
        f"gripper={record.get('gripper_target')}/"
        f"{record.get('gripper_state', 'n/a')}/"
        f"{record.get('gripper_physical_state', 'n/a')} "
        f"target_z={value('target_z_m', 'm')}"
    )


def get_model(usr_args: dict) -> HeuristicPolicy:
    return HeuristicPolicy(usr_args)


def eval(task_env: Any, model: HeuristicPolicy, observation: dict) -> None:
    try:
        actions = model.get_action(
            scene=encode_obs(
                task_env,
                observation,
                simple_grasp_root=model.simple_grasp_root,
                target_name=str(getattr(model, "usr_args", {}).get("object_name", "auto")),
            ),
            task_env=task_env,
        )
    except HeuristicEpisodeFailure as exc:
        print(f"[heuristic] rollout failed: {exc}")
        task_env.take_action_cnt = task_env.step_lim
        return
    telemetry_enabled = bool(
        getattr(model, "usr_args", {}).get("execution_telemetry", True)
    )
    (
        guard_enabled,
        guard_qpos_tolerance,
        guard_position_tolerance,
        guard_orientation_tolerance,
        guard_gripper_min_delta,
        guard_gripper_settle_delta_max,
    ) = _execution_guard_settings(model)
    metadata = list(getattr(model, "last_action_metadata", []))
    if not metadata:
        metadata = [
            {
                "phase": "action",
                "arm": None,
                "endpoint": index == len(actions) - 1,
                "waypoint_index": index + 1,
                "waypoint_count": len(actions),
            }
            for index in range(len(actions))
        ]
    batch_index = int(getattr(model, "execution_batch_index", 0))
    model.execution_batch_index = batch_index + 1
    trace_path = None
    if telemetry_enabled:
        try:
            trace_path = _execution_trace_path(task_env, model)
        except OSError as exc:
            print(f"[heuristic] execution telemetry disabled: {exc}")

    close_completed = False
    close_initial_physical_states: dict[str, float] = {}
    close_physical_states: dict[str, list[float]] = {arm: [] for arm in _ARM_NAMES}
    deferred_close_success = False
    for action_index, (action, action_metadata) in enumerate(
        zip(actions, metadata)
    ):
        phase = action_metadata.get("phase", "action")
        endpoint = bool(action_metadata.get("endpoint", False))
        action_arms = _metadata_arms(action_metadata)
        bimanual = action_metadata.get("arm") == "both"

        # A task-level collision can set eval_success during an intermediate
        # close pulse. Clear it so the remaining settle pulses are actually
        # executed; validate the final physical response below.
        if (
            phase == "close"
            and not close_completed
            and task_env.take_action_cnt < task_env.step_lim
        ):
            try:
                before_close = _robot_measurements(task_env, action_metadata)
                for arm in action_arms:
                    arm_values = _arm_measurements(before_close, arm, bimanual=bimanual)
                    value = arm_values.get("gripper_physical_state")
                    if value is not None and np.isfinite(value):
                        close_initial_physical_states.setdefault(arm, float(value))
            except Exception:
                pass
            task_env.eval_success = False

        if getattr(task_env, "eval_success", False):
            status = "skipped_eval_success"
        elif task_env.take_action_cnt >= task_env.step_lim:
            status = "skipped_step_limit"
        else:
            task_env.take_action(action, action_type="qpos")
            status = "executed"

        eval_success_reported = bool(getattr(task_env, "eval_success", False))
        terminal_post_close_success = (
            eval_success_reported
            and close_completed
            and phase in _POST_CLOSE_MOTION_PHASES
        )
        premature_success = (
            eval_success_reported and not close_completed and phase != "close"
        )
        motion_guard_required = (
            guard_enabled
            and status == "executed"
            and phase in _MOTION_ENDPOINT_PHASES
            and (endpoint or premature_success)
            and not terminal_post_close_success
        )
        close_guard_required = (
            guard_enabled
            and status == "executed"
            and phase == "close"
            and endpoint
        )
        close_endpoint_executed = (
            status == "executed" and phase == "close" and endpoint
        )
        premature_guard_required = (
            guard_enabled and status == "executed" and premature_success
        )
        guard_required = (
            motion_guard_required
            or close_guard_required
            or premature_guard_required
        )
        measurement_required = (
            telemetry_enabled
            or guard_required
            or (guard_enabled and status == "executed" and phase == "close")
        )
        measurements: dict[str, Any] = {}
        if status == "executed" and measurement_required:
            try:
                measurements = _robot_measurements(task_env, action_metadata)
            except Exception as exc:
                measurements["telemetry_error"] = f"{type(exc).__name__}: {exc}"

        if status == "executed" and phase == "close":
            for arm in action_arms:
                arm_values = _arm_measurements(measurements, arm, bimanual=bimanual)
                physical_state = arm_values.get("gripper_physical_state")
                if physical_state is not None and np.isfinite(physical_state):
                    close_physical_states[arm].append(float(physical_state))

        guard_failures: list[str] = []
        close_validation_failures: list[str] = []
        if motion_guard_required:
            guard_failures.extend(
                _execution_guard_failures(
                    measurements,
                    qpos_tolerance_rad=guard_qpos_tolerance,
                    ee_position_tolerance_m=guard_position_tolerance,
                    ee_orientation_tolerance_rad=guard_orientation_tolerance,
                )
            )
        if premature_guard_required:
            guard_failures.append("eval_success_before_close")
        if close_endpoint_executed and guard_enabled:
            for arm in action_arms:
                arm_failures = _gripper_execution_guard_failures(
                    initial_state=close_initial_physical_states.get(arm),
                    states=close_physical_states[arm],
                    min_delta=guard_gripper_min_delta,
                    settle_delta_max=guard_gripper_settle_delta_max,
                )
                close_validation_failures.extend(
                    f"{arm}.{failure}" if bimanual else failure
                    for failure in arm_failures
                )
        if close_guard_required:
            guard_failures.extend(close_validation_failures)
        if close_endpoint_executed and not close_validation_failures:
            close_completed = True

        if phase == "close" and not endpoint and eval_success_reported:
            task_env.eval_success = False

        if guard_failures:
            task_env.eval_success = False
            task_env.take_action_cnt = task_env.step_lim
            status = "execution_guard_failed"
            print(
                f"[heuristic][exec] guard failed phase={phase} "
                f"arm={action_metadata.get('arm')}: "
                + ", ".join(guard_failures)
            )

        # If RoboTwin reports task success partway through the final close
        # interpolation, take_action() can return before the requested 0.0
        # gripper target is fully applied. Defer that success until one
        # post-close motion command executes with the closed target, leaving
        # the gripper drive closed without a retry.
        if (
            close_endpoint_executed
            and not guard_failures
            and eval_success_reported
        ):
            deferred_close_success = True
            task_env.eval_success = False

        if (
            deferred_close_success
            and status == "executed"
            and phase in _POST_CLOSE_MOTION_PHASES
            and not guard_failures
        ):
            task_env.eval_success = True
            deferred_close_success = False

        if not telemetry_enabled:
            if status != "executed":
                break
            continue

        target_qpos = action_metadata.get("target_qpos")
        command_pose = action_metadata.get("command_pose")
        record = {
            "episode": int(getattr(task_env, "test_num", 0)),
            "seed": int(getattr(task_env, "episode_seed", 0)),
            "batch_index": batch_index,
            "action_index": action_index,
            "phase": phase,
            "arm": action_metadata.get("arm"),
            "arm_source": action_metadata.get("arm_source"),
            "arm_targets": _trace_arm_targets(action_metadata),
            "endpoint": endpoint,
            "waypoint_index": int(action_metadata.get("waypoint_index", 1)),
            "waypoint_count": int(action_metadata.get("waypoint_count", 1)),
            "status": status,
            "sim_step": int(getattr(task_env, "take_action_cnt", 0)),
            "eval_success": bool(getattr(task_env, "eval_success", False)),
            "eval_success_reported": eval_success_reported,
            "target_qpos": (
                None
                if target_qpos is None
                else np.asarray(target_qpos, dtype=np.float64).tolist()
            ),
            "gripper_target": action_metadata.get("target_gripper"),
            "command_pose": (
                None
                if command_pose is None
                else np.asarray(command_pose, dtype=np.float64).tolist()
            ),
            "execution_guard_failures": guard_failures,
        }
        record.update(measurements)
        try:
            _write_execution_record(trace_path, record)
        except OSError as exc:
            print(f"[heuristic] execution trace write failed: {exc}")
            trace_path = None
        _print_endpoint(record)
        if guard_failures:
            break

    if (
        not bool(getattr(task_env, "eval_success", False))
        and task_env.take_action_cnt < task_env.step_lim
    ):
        print("[heuristic][exec] one-shot rollout complete without task success")
        task_env.take_action_cnt = task_env.step_lim


def reset_model(model: HeuristicPolicy) -> None:
    model.reset()

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


def _robot_measurements(task_env: Any, metadata: dict[str, Any]) -> dict[str, Any]:
    arm = metadata.get("arm")
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
                # ALOHA's SAPIEN and MuJoCo link frames differ by a fixed
                # half-roll about local X. Report both the raw frame error and
                # the error after accounting for that representation change.
                frame_equivalence = np.diag([1.0, -1.0, -1.0])
                equivalent_rotation = (
                    target[:3, :3] @ frame_equivalence
                ).T @ actual_rotation
                equivalent_cosine = np.clip(
                    (np.trace(equivalent_rotation) - 1.0) / 2.0,
                    -1.0,
                    1.0,
                )
                result["actual_ee_pose"] = actual.tolist()
                result["ee_position_error_m"] = float(
                    np.linalg.norm(target[:3, 3] - actual[:3])
                )
                result["ee_orientation_error_raw_rad"] = raw_orientation_error
                result["ee_orientation_error_rad"] = min(
                    raw_orientation_error, float(np.arccos(equivalent_cosine))
                )

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


def _write_execution_record(path: Path | None, record: dict[str, Any]) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def _print_endpoint(record: dict[str, Any]) -> None:
    if not record.get("endpoint") or record["status"] != "executed":
        return

    def value(name: str, unit: str = "") -> str:
        item = record.get(name)
        return "n/a" if item is None else f"{item:.4f}{unit}"

    print(
        f"[heuristic][exec] phase={record['phase']} arm={record.get('arm')} "
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

    for action_index, (action, action_metadata) in enumerate(
        zip(actions, metadata)
    ):
        if getattr(task_env, "eval_success", False):
            status = "skipped_eval_success"
        elif task_env.take_action_cnt >= task_env.step_lim:
            status = "skipped_step_limit"
        else:
            task_env.take_action(action, action_type="qpos")
            status = "executed"

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
            "phase": action_metadata.get("phase", "action"),
            "arm": action_metadata.get("arm"),
            "endpoint": bool(action_metadata.get("endpoint", False)),
            "waypoint_index": int(action_metadata.get("waypoint_index", 1)),
            "waypoint_count": int(action_metadata.get("waypoint_count", 1)),
            "status": status,
            "sim_step": int(getattr(task_env, "take_action_cnt", 0)),
            "eval_success": bool(getattr(task_env, "eval_success", False)),
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
        }
        try:
            record.update(_robot_measurements(task_env, action_metadata))
        except Exception as exc:
            record["telemetry_error"] = f"{type(exc).__name__}: {exc}"
        try:
            _write_execution_record(trace_path, record)
        except OSError as exc:
            print(f"[heuristic] execution trace write failed: {exc}")
            trace_path = None
        _print_endpoint(record)


def reset_model(model: HeuristicPolicy) -> None:
    model.reset()
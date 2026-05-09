"""RoboTwin deploy entrypoint for the OBB-aware pi0.5 policy.

The OBB visualization mirrors the dino-wam recipe in
``vjepa2/evals/robotwin/eval.py``: rather than writing a parallel video at
inference time, we let RoboTwin record its clean ``episode<N>.mp4`` first,
cache the predicted OBB *chunks* together with the env-step counter at the
time of each inference, and then post-process the file by re-reading every
frame and drawing the per-frame OBB.

Chunk math (same as dino-wam):
    A chunk predicted at env-step ``S`` carries ``H`` future-OBB rows
    (``H`` = action_horizon = pi0_step). It covers video frames
    ``[S+1, S+H]``. For video frame ``F`` we pick the latest ``S`` with
    ``1 <= F-S <= H`` and draw OBB row ``F-S-1`` of that chunk.
"""
from __future__ import annotations

import atexit
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)

from pi_model import PI0OBB  # noqa: E402


# -- OBB drawing (dino-wam style) ----------------------------------------------------------------

_BOX_EDGES = (
    (0, 1), (1, 3), (3, 2), (2, 0),  # face x=-
    (4, 5), (5, 7), (7, 6), (6, 4),  # face x=+
    (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
)
# BGR; indexed by object slot.
_OBB_COLORS = ((0, 220, 0), (0, 180, 255))

# Tasks that supervise BOTH OBB slots at training. Every other task masks dim 16:32
# in the OBB loss, so slot 1's prediction there is unsupervised — don't draw it.
_DUAL_OBB_TASKS: set[str] = {"pick_dual_bottles"}


def _num_obb_for_task(task_name: str | None) -> int:
    return 2 if str(task_name).strip() in _DUAL_OBB_TASKS else 1


def _draw_obb_on_frame(frame_bgr: np.ndarray, obb_step: np.ndarray, num_obb: int = 1) -> None:
    """Draw the OBB(s) for a single timestep onto a BGR frame in place.

    ``obb_step``: ``[max_objects, 8, 2]`` (or flat ``[D]``) in [0, 1] image fractions.
    ``num_obb``: cap on how many object slots to draw (1 for single-object tasks,
    2 for ``pick_dual_bottles``). Slots that are exactly zero are also skipped.
    """
    import cv2  # local import — keeps module importable without OpenCV.

    h, w = frame_bgr.shape[:2]
    flat = np.asarray(obb_step, dtype=np.float32).reshape(-1)
    n_avail = flat.size // 16
    n_draw = min(int(num_obb), n_avail)
    for oi in range(n_draw):
        sl = flat[oi * 16 : (oi + 1) * 16]
        if not np.any(sl):
            continue
        corners = sl.reshape(8, 2).copy()
        corners[:, 0] *= float(w)
        corners[:, 1] *= float(h)
        corners_int = corners.astype(np.int32)
        color = _OBB_COLORS[oi % len(_OBB_COLORS)]
        for a, b in _BOX_EDGES:
            cv2.line(frame_bgr, tuple(corners_int[a]), tuple(corners_int[b]), color, 2)
        for cx, cy in corners_int:
            cv2.circle(frame_bgr, (int(cx), int(cy)), 2, color, -1)


def _write_h264(frames: list, out_path: Path, fps: float, w: int, h: int) -> None:
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pixel_format", "bgr24",
        "-video_size", f"{w}x{h}",
        "-framerate", str(fps),
        "-i", "-",
        "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "23",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert proc.stdin is not None
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg H.264 encode failed for {out_path}")


def _overlay_obb_on_video(
    video_path: Path,
    step_indices: np.ndarray,
    obbs: np.ndarray,
    num_obb: int = 1,
) -> None:
    """Read ``video_path``, draw per-frame OBBs, write back in place.

    ``obbs``: ``[N, H, max_objects, 8, 2]`` in [0, 1] image fractions.
    ``step_indices``: ``[N]`` int — env-step counter at each inference.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[pi05_obb] OBB overlay: could not open {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        return

    horizon = int(obbs.shape[1])
    si = np.asarray(step_indices, dtype=np.int64)
    for frame_idx, frame in enumerate(frames):
        offsets = frame_idx - si  # [N]
        valid = (offsets >= 1) & (offsets <= horizon)
        if not valid.any():
            continue
        candidates = np.where(valid)[0]
        pred_idx = int(candidates[-1])  # most recent covering chunk
        within = int(offsets[pred_idx] - 1)  # 0-based row in the chunk
        _draw_obb_on_frame(frame, obbs[pred_idx, within], num_obb=num_obb)

    tmp = video_path.with_suffix(".obb_tmp.mp4")
    _write_h264(frames, tmp, fps, vw, vh)
    tmp.replace(video_path)


# -- Per-episode state on the model ----------------------------------------------------------------

def _reset_obb_state(model: PI0OBB) -> None:
    model._env_step_count = 0
    model._step_indices = []
    model._obb_chunks = []
    model._pending_video_path = None
    model._pending_task_name = None


def _normalize_obb_chunk(obb: np.ndarray) -> np.ndarray | None:
    """Coerce the model's OBB output to ``[H, max_objects, 8, 2]`` in [0, 1]."""
    if obb is None:
        return None
    obb = np.asarray(obb, dtype=np.float32)
    if obb.ndim == 4:
        # Already [H, max_objects, 8, 2] (RobotwinObbOutputs path).
        return obb
    if obb.ndim == 3:
        # [H, max_objects, 16] in [-1, 1] (raw model output, no output transform applied).
        h, mo, d = obb.shape
        if d != 16:
            return None
        x01 = (obb + 1.0) * 0.5
        return x01.reshape(h, mo, 8, 2)
    if obb.ndim == 2:
        # [H, max_objects * 16] flat — shouldn't normally happen.
        h, total = obb.shape
        if total % 16:
            return None
        mo = total // 16
        x01 = (obb + 1.0) * 0.5
        return x01.reshape(h, mo, 8, 2)
    return None


def _finalize_episode(model: PI0OBB) -> None:
    """Overlay all cached predictions onto the just-finished episode video."""
    video_path = getattr(model, "_pending_video_path", None)
    step_indices = getattr(model, "_step_indices", None)
    obb_chunks = getattr(model, "_obb_chunks", None)
    if not video_path or not step_indices or not obb_chunks:
        if hasattr(model, "_pending_video_path"):
            _reset_obb_state(model)
        return
    if not Path(video_path).exists():
        _reset_obb_state(model)
        return

    obbs = np.stack(obb_chunks, axis=0).astype(np.float32)
    si = np.asarray(step_indices, dtype=np.int64)
    num_obb = _num_obb_for_task(getattr(model, "_pending_task_name", None))
    try:
        _overlay_obb_on_video(Path(video_path), si, obbs, num_obb=num_obb)
    except Exception as e:  # noqa: BLE001
        print(f"[pi05_obb] OBB overlay failed for {video_path}: {e}")
    _reset_obb_state(model)


_EXIT_HOOK_REGISTERED = False


def _ensure_exit_hook(model: PI0OBB) -> None:
    """Finalize the LAST episode (reset_model isn't called after the final one)."""
    global _EXIT_HOOK_REGISTERED
    if _EXIT_HOOK_REGISTERED:
        return
    _EXIT_HOOK_REGISTERED = True
    atexit.register(_finalize_episode, model)


# -- RoboTwin entrypoints ------------------------------------------------------------------------

def encode_obs(observation: dict):
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    input_state = observation["joint_action"]["vector"]
    return input_rgb_arr, input_state


def get_model(usr_args: dict) -> PI0OBB:
    model = PI0OBB(
        train_config_name=usr_args["train_config_name"],
        model_name=usr_args["model_name"],
        checkpoint_id=usr_args["checkpoint_id"],
        pi0_step=usr_args["pi0_step"],
    )
    _reset_obb_state(model)
    _ensure_exit_hook(model)
    return model


def eval(TASK_ENV, model: PI0OBB, observation: dict):
    if model.observation_window is None:
        instruction = TASK_ENV.get_instruction()
        model.set_language(instruction)

    # Inference at the current env-step counter.
    input_rgb_arr, input_state = encode_obs(observation)
    model.update_observation_window(input_rgb_arr, input_state)
    actions = model.get_action()[: model.pi0_step]

    # Cache predicted OBB chunk for post-processing this episode.
    obb = _normalize_obb_chunk(model.last_future_obb)
    if obb is not None:
        model._obb_chunks.append(obb)
        model._step_indices.append(int(model._env_step_count))

    # Remember which video file belongs to the current episode (RoboTwin's ffmpeg
    # writes ``episode<test_num>.mp4`` under ``eval_video_path``). ``test_num`` is
    # incremented *after* the episode finishes, so capture it here while it's stable.
    eval_video_path = getattr(TASK_ENV, "eval_video_path", None)
    if eval_video_path is not None:
        model._pending_video_path = Path(eval_video_path) / f"episode{TASK_ENV.test_num}.mp4"
    # Drives per-task num_obb (1 for single-object tasks; 2 for pick_dual_bottles).
    model._pending_task_name = getattr(TASK_ENV, "task_name", None)

    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        # RoboTwin renders + writes one video frame per ``take_action``; advance our
        # per-episode counter in lockstep so chunk math matches frame indices.
        model._env_step_count += 1
        input_rgb_arr, input_state = encode_obs(observation)
        model.update_observation_window(input_rgb_arr, input_state)


def reset_model(model: PI0OBB):
    # Finalize the just-finished episode's video (the final episode is handled by atexit).
    _finalize_episode(model)
    model.reset_obsrvationwindows()

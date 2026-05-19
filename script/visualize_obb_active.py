"""Render one H.264 MP4 per episode showing OBB only for objects currently
being manipulated by the grippers (up to 2 simultaneous OBBs).

An object is considered "active for arm X" at frame t when:
  - Arm X's gripper is closing (gripper_val < OPEN_THRESH), AND
  - Arm X's TCP is within DIST_THRESH meters of the object's center.
The frame's active set is the union over both arms (so 0–2 OBBs are drawn).

Output: a single 4-camera grid MP4 per episode, encoded with H.264 so it
plays in browsers / VS Code's preview.

Usage:
  python script/visualize_obb_active.py episode.hdf5 --out vis.mp4
  python script/visualize_obb_active.py --all data/  # batch mode
"""
import argparse
import io
import json
from pathlib import Path

import cv2
import h5py
import imageio
import numpy as np
from PIL import Image

CAMERAS = ["head_camera", "front_camera", "left_camera", "right_camera"]
BOX_EDGES = [
    (0, 1), (1, 3), (3, 2), (2, 0),
    (4, 5), (5, 7), (7, 6), (6, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

# Two distinct colors for left/right arm objects (RGB). High-saturation so the
# OBB is visible against busy scenes (matches dino-wam's overlay style).
COLOR_LEFT = (0, 255, 0)      # pure green
COLOR_RIGHT = (255, 140, 0)   # bright orange

LINE_THICKNESS = 3            # bumped from 2 for visibility in 2x2 grid
CORNER_RADIUS = 4             # filled dots at each of the 8 OBB corners

OPEN_THRESH = 0.85       # gripper_val above this = open (treated as not gripping)
DIST_THRESH = 0.35       # meters; arm TCP-to-object proximity (TCP is ~wrist, allow finger reach)
MOVE_WINDOW = 6          # frames; window over which to measure object motion
MOVE_EPS = 0.003         # meters; min xyz position change in MOVE_WINDOW
QPOS_EPS = 0.01          # radians; min joint angle change for articulated motion


def decode_rgb(raw_bytes: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(bytes(raw_bytes))))


def _decode_target_array(arr) -> list[str]:
    """HDF5 vlen-string dataset → python list of str (decoded)."""
    out = []
    for v in arr[()]:
        if isinstance(v, bytes):
            out.append(v.decode("utf-8"))
        elif v is None:
            out.append("")
        else:
            out.append(str(v))
    return out


def per_frame_active(f: h5py.File) -> list[list[tuple[str, tuple[int, int, int]]]]:
    """Return list of length T: each element is [(obj_name, color_rgb), ...].

    Source-of-truth lookup order:
      1. ``object_target/left`` and ``object_target/right`` HDF5 datasets if
         present (written by Base_Task.save_obb2d during collection or by
         script/label_object_targets_all.py post-hoc). One target string per
         frame per arm — empty string means "no target".
      2. Live heuristic fallback (same algorithm as the labeler) for legacy
         HDF5s that pre-date the object_target datasets.

    Strategy: ALWAYS draw at least one OBB so the user can see what the robot
    is about to manipulate (approach phase) in addition to what it is currently
    grasping. We do this in two passes:

    1) **Detect engagement events** per arm: contiguous runs of frames where
       the gripper is closed and a tracked object is within DIST_THRESH (or
       moving — for articulated/press tasks). Each event picks the
       most-frequent nearest object as its target.

    2) **Fill the timeline** per arm using the events:
       - Frames before the first event       → target = first event's object  (initial approach)
       - Frames during event i                → target = event i's object       (active manipulation)
       - Frames between events i and i+1      → target = event i+1's object     (approach to next)
       - Frames after the last event          → target = last event's object    (release / linger)

    3) **Fallback** for arms with zero events: pick the object that is closest
       to that arm averaged across the whole episode (only used to keep some
       OBB visible — covers tasks where no gripper closes, e.g. press_stapler
       on the older convention, or arms that never move).
    """
    T = f["endpose/left_endpose"].shape[0]

    # Preferred path: HDF5 carries the canonical per-frame, per-arm targets.
    if "object_target" in f and "object_target/left" in f and "object_target/right" in f:
        L_tgt = _decode_target_array(f["object_target/left"])
        R_tgt = _decode_target_array(f["object_target/right"])

        # Per-arm: index where the *current* run (same target as frame t) began.
        # Used to disambiguate handovers: when both arms target the same object,
        # the arm that started its run later is the one that just took control.
        def _run_starts(seq: list[str]) -> list[int]:
            starts = [0] * len(seq)
            for i in range(1, len(seq)):
                starts[i] = i if seq[i] != seq[i - 1] else starts[i - 1]
            return starts

        # Back-fill the initial-reach phase for whichever arm engages FIRST.
        # The other arm is genuinely idle during that window (it hasn't acted
        # yet), so we don't paint a phantom OBB for it. This matches the
        # state-machine semantics: only the arm that is approaching/grasping
        # carries the object's target label.
        L_first = next((i for i, v in enumerate(L_tgt) if v), -1)
        R_first = next((i for i, v in enumerate(R_tgt) if v), -1)
        L_full, R_full = list(L_tgt), list(R_tgt)
        if L_first >= 0 and (R_first < 0 or L_first <= R_first):
            for i in range(L_first):
                L_full[i] = L_tgt[L_first]
        elif R_first >= 0:
            for i in range(R_first):
                R_full[i] = R_tgt[R_first]

        # Run starts on the back-filled sequences feed the dual-target
        # tiebreak below (later-started run = arm that just took control).
        L_start = _run_starts(L_full)
        R_start = _run_starts(R_full)

        actives = []
        for t in range(T):
            L, R = L_full[t], R_full[t]
            frame: list[tuple[str, tuple[int, int, int]]] = []
            if L and R and L == R:
                # Same object held / approached by both arms. The arm whose
                # current run started later is the one that just took control
                # (e.g. the receiving hand mid-handover) — use its color.
                if R_start[t] >= L_start[t]:
                    frame.append((R, COLOR_RIGHT))
                else:
                    frame.append((L, COLOR_LEFT))
            else:
                if L:
                    frame.append((L, COLOR_LEFT))
                if R:
                    frame.append((R, COLOR_RIGHT))
            actives.append(frame)
        return actives

    L_pose = f["endpose/left_endpose"][:]   # (T, 7)  [xyz + quat]
    R_pose = f["endpose/right_endpose"][:]
    L_grip = f["endpose/left_gripper"][:]
    R_grip = f["endpose/right_gripper"][:]

    pose_keys = list(f["object_pose"].keys()) if "object_pose" in f else []
    cams_with_rgb = [c for c in CAMERAS if f"observation/{c}/rgb" in f]
    valid_objs = []
    for o in pose_keys:
        if any(f"observation/{c}/proj_3d_obb/{o}" in f for c in cams_with_rgb):
            valid_objs.append(o)
    if not valid_objs:
        return [[] for _ in range(T)]

    obj_xyz = {o: f[f"object_pose/{o}"][:, :3] for o in valid_objs}

    obj_qpos = {}
    if "object_qpos" in f:
        for o in valid_objs:
            k = f"object_qpos/{o}"
            if k in f:
                q = np.asarray(f[k][:])
                if q.ndim == 1:
                    q = q[:, None]
                obj_qpos[o] = q

    # Per-frame "moving" flag per object (xyz or qpos delta over a small window)
    obj_moving = {}
    for o in valid_objs:
        xyz = obj_xyz[o]
        m = np.zeros(T, dtype=bool)
        for t in range(T):
            t0 = max(0, t - MOVE_WINDOW)
            if float(np.linalg.norm(xyz[t] - xyz[t0])) >= MOVE_EPS:
                m[t] = True
                continue
            if o in obj_qpos and float(np.linalg.norm(obj_qpos[o][t] - obj_qpos[o][t0])) >= QPOS_EPS:
                m[t] = True
        obj_moving[o] = m

    def _detect_events(arm_pose, arm_grip):
        """Return list of (t_start, t_end, target_obj). Each event is one
        contiguous block where the arm is engaged with an object.
        """
        # Per-frame "engaged" boolean + nearest object name
        engaged = np.zeros(T, dtype=bool)
        nearest = [None] * T
        for t in range(T):
            tcp = arm_pose[t, :3]
            grip_closed = arm_grip[t] < OPEN_THRESH
            best_obj, best_d = None, float("inf")
            for o in valid_objs:
                d = float(np.linalg.norm(obj_xyz[o][t] - tcp))
                if d > DIST_THRESH:
                    continue
                moving = bool(obj_moving[o][t])
                if not (grip_closed or moving):
                    continue
                if d < best_d:
                    best_obj, best_d = o, d
            nearest[t] = best_obj
            engaged[t] = best_obj is not None
        # Coalesce contiguous engaged spans (≥3 frames to drop spurious blips)
        events = []
        t = 0
        while t < T:
            if not engaged[t]:
                t += 1
                continue
            j = t
            while j < T and engaged[j]:
                j += 1
            if j - t >= 3:
                # Most common nearest object within the span = the target
                names = [n for n in nearest[t:j] if n is not None]
                target = max(set(names), key=names.count) if names else None
                if target is not None:
                    events.append((t, j - 1, target))
            t = j
        return events

    def _fill_timeline(events):
        """Map events → per-frame target object (length T). Returns None when
        the arm has no events (caller falls back to whole-episode nearest).
        """
        if not events:
            return None
        targets = [None] * T
        # During each event, set the target.
        for t0, t1, obj in events:
            for tt in range(t0, t1 + 1):
                targets[tt] = obj
        # Before first event: approach to first event's target.
        first_t0 = events[0][0]
        first_obj = events[0][2]
        for tt in range(0, first_t0):
            targets[tt] = first_obj
        # Between events i and i+1: approach to next event's target.
        for i in range(len(events) - 1):
            _, t_end_i, _ = events[i]
            t_start_next, _, obj_next = events[i + 1]
            for tt in range(t_end_i + 1, t_start_next):
                targets[tt] = obj_next
        # After last event: linger on last event's target.
        last_t1, _, _ = events[-1][1], events[-1][0], events[-1][2]  # noqa
        last_t1 = events[-1][1]
        last_obj = events[-1][2]
        for tt in range(last_t1 + 1, T):
            targets[tt] = last_obj
        return targets

    L_events = _detect_events(L_pose, L_grip)
    R_events = _detect_events(R_pose, R_grip)
    L_targets = _fill_timeline(L_events)
    R_targets = _fill_timeline(R_events)

    # Fallback: if an arm has no events but moves substantially during the
    # episode, assign it the object that was closest to it on average — keeps
    # some OBB visible for arms that don't grasp.
    def _arm_overall_nearest(arm_pose):
        # Average distance per object across the whole episode
        if not valid_objs:
            return None
        avg = {}
        for o in valid_objs:
            ds = np.linalg.norm(obj_xyz[o] - arm_pose[:, :3], axis=1)
            avg[o] = float(ds.mean())
        return min(avg, key=avg.get)

    if L_targets is None and R_targets is None:
        # Nothing engaged on either arm; pick whichever arm moves more and
        # use its nearest object across the episode. If both arms equally
        # static (unusual), fall back to the most-moving object in the scene.
        L_motion = float(np.linalg.norm(L_pose[1:, :3] - L_pose[:-1, :3], axis=1).sum())
        R_motion = float(np.linalg.norm(R_pose[1:, :3] - R_pose[:-1, :3], axis=1).sum())
        if max(L_motion, R_motion) > 0.05:
            arm_pose = L_pose if L_motion >= R_motion else R_pose
            color = COLOR_LEFT if L_motion >= R_motion else COLOR_RIGHT
            obj = _arm_overall_nearest(arm_pose)
            if obj is not None:
                return [[(obj, color)] for _ in range(T)]
        # Last-resort fallback: most-moving object in scene with neutral color
        moves_total = {o: int(obj_moving[o].sum()) for o in valid_objs}
        obj = max(moves_total, key=moves_total.get)
        return [[(obj, COLOR_LEFT)] for _ in range(T)]

    actives = []
    for t in range(T):
        frame = []
        used = set()
        for tgt, color in [(L_targets, COLOR_LEFT), (R_targets, COLOR_RIGHT)]:
            if tgt is None:
                continue
            obj = tgt[t]
            if obj is None or obj in used:
                continue
            frame.append((obj, color))
            used.add(obj)
        # Guarantee at least one OBB per frame: if both arms ended up with
        # None at this frame (shouldn't happen given the fill rule, but be
        # defensive), reuse the nearer arm's overall-nearest object.
        if not frame:
            arm_pose = L_pose if L_targets is None else R_pose
            color = COLOR_LEFT if L_targets is None else COLOR_RIGHT
            obj = _arm_overall_nearest(arm_pose)
            if obj is not None:
                frame.append((obj, color))
        actives.append(frame)
    return actives


def make_grid(frames: list[np.ndarray], labels: list[str]) -> np.ndarray:
    """2x2 (or 1x2 / 1x1) tile of camera frames with text labels."""
    n = len(frames)
    if n == 0:
        return np.zeros((240, 320, 3), dtype=np.uint8)
    out = []
    for f, lab in zip(frames, labels):
        cv2.putText(f, lab, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2, cv2.LINE_AA)
        out.append(f)
    if n == 1:
        return out[0]
    if n == 2:
        return np.concatenate(out, axis=1)
    if n == 3:
        out.append(np.zeros_like(out[0]))
    row1 = np.concatenate(out[:2], axis=1)
    row2 = np.concatenate(out[2:4], axis=1)
    return np.concatenate([row1, row2], axis=0)


def render_episode(hdf5_path: Path, out_path: Path, fps: int = 15) -> dict:
    """Render one MP4. Returns small report dict."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(hdf5_path, "r") as f:
        cams = [c for c in CAMERAS if f"observation/{c}/rgb" in f]
        T = f[f"observation/{cams[0]}/rgb"].shape[0]
        rgb = {c: f[f"observation/{c}/rgb"][:] for c in cams}
        # OBB pixel coords saved during data collection.
        obb = {}  # obb[obj][cam] -> (T, 16)
        for c in cams:
            grp = f.get(f"observation/{c}/proj_3d_obb")
            if grp is None:
                continue
            for obj in grp.keys():
                obb.setdefault(obj, {})[c] = grp[obj][:]

        actives_per_frame = per_frame_active(f)

    # Use imageio + ffmpeg → broadly compatible H.264 yuv420p MP4.
    writer = imageio.get_writer(
        str(out_path), fps=fps, codec="libx264", quality=8,
        pixelformat="yuv420p", macro_block_size=1,
    )

    n_active_total = 0
    obj_frame_count = {}
    try:
        for t in range(T):
            actives = actives_per_frame[t]
            n_active_total += len(actives)
            for o, _ in actives:
                obj_frame_count[o] = obj_frame_count.get(o, 0) + 1

            tiles = []
            labels = []
            for c in cams:
                img = decode_rgb(rgb[c][t]).copy()
                for obj_name, color in actives:
                    cam_obb = obb.get(obj_name, {}).get(c)
                    if cam_obb is None:
                        continue
                    pts = cam_obb[t].reshape(8, 2).astype(int)
                    # Image is RGB (PIL); cv2.line writes channels in order so
                    # passing the RGB tuple draws the intended color.
                    for a, b in BOX_EDGES:
                        pa = (int(pts[a, 0]), int(pts[a, 1]))
                        pb = (int(pts[b, 0]), int(pts[b, 1]))
                        cv2.line(img, pa, pb, color, LINE_THICKNESS, cv2.LINE_AA)
                    # Corner dots (matches dino-wam style: easier to read at
                    # small frame sizes than lines alone).
                    for cx, cy in pts:
                        cv2.circle(img, (int(cx), int(cy)), CORNER_RADIUS,
                                   color, -1, cv2.LINE_AA)
                # HUD line at bottom-left. Use the active object's color so
                # the user can correlate label ↔ box at a glance.
                hud_color = actives[0][1] if actives else (220, 220, 220)
                if len(actives) == 2:
                    hud = f"t={t}  {actives[0][0]} | {actives[1][0]}"
                elif actives:
                    hud = f"t={t}  {actives[0][0]}"
                else:
                    hud = f"t={t}  (idle)"
                cv2.putText(img, hud, (4, img.shape[0] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 0, 0), 3, cv2.LINE_AA)  # black outline
                cv2.putText(img, hud, (4, img.shape[0] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            hud_color, 1, cv2.LINE_AA)
                tiles.append(img)
                labels.append(c)
            grid = make_grid(tiles, labels)
            writer.append_data(grid)
    finally:
        writer.close()

    return {
        "frames": T,
        "active_total": n_active_total,
        "per_object_frames": obj_frame_count,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("hdf5", nargs="?", help="Single episode HDF5 (omit with --all)")
    p.add_argument("--out", help="Output mp4 (single-episode mode)")
    p.add_argument("--all", help="Batch mode: data root (e.g. data/)")
    p.add_argument("--config", default="demo_obb_vis", help="Demo dir under <task>/")
    p.add_argument("--out-root", default="obb_videos_active",
                   help="Where to write batch outputs")
    p.add_argument("--fps", type=int, default=15)
    args = p.parse_args()

    if args.all:
        root = Path(args.all)
        out_root = Path(args.out_root) / args.config
        out_root.mkdir(parents=True, exist_ok=True)
        ok, fail = 0, 0
        for task_dir in sorted(root.iterdir()):
            ep_dir = task_dir / args.config / "data"
            if not ep_dir.is_dir():
                continue
            episodes = sorted(ep_dir.glob("episode*.hdf5"))
            if not episodes:
                continue
            ep = episodes[0]
            out = out_root / f"{task_dir.name}.mp4"
            try:
                rep = render_episode(ep, out, fps=args.fps)
                summary = ", ".join(f"{o}={n}f" for o, n in rep["per_object_frames"].items())
                print(f"  ok  {task_dir.name:35s}  T={rep['frames']:4d}  active={rep['active_total']:4d}  [{summary}]")
                ok += 1
            except Exception as e:
                print(f"  ERR {task_dir.name}: {e}")
                fail += 1
        print(f"\nDone. ok={ok} fail={fail} → {out_root}")
    else:
        if not args.hdf5 or not args.out:
            p.error("provide hdf5 and --out, or use --all")
        rep = render_episode(Path(args.hdf5), Path(args.out), fps=args.fps)
        print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()

"""Render per-episode mp4s overlaying the left/right OBB stored in a LeRobot v2 dataset.

Reads ``observation.obb_head`` (flat 32-float [left_16 | right_16] in native head-camera
pixel space) from each episode's parquet and decodes ``observation.images.cam_high`` frames
directly from the on-disk mp4. For each episode emits one mp4 with the 8-corner OBB drawn
on top of the cam_high frame; left hand is green, right hand is cyan, zero-OBB frames
render the bare image (means "no target for that hand").

Reads files directly rather than going through ``LeRobotDataset.__getitem__`` so it does
not depend on a working torchcodec install.

Usage:
    python script/visualize_obb_lerobot.py \\
        --lerobot-root /home/qinxiyu2/dev/RoboTwin/data_randomized_lerobot \\
        --out-dir /home/qinxiyu2/dev/RoboTwin/obb_videos
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd

# cv2.VideoWriter's mp4v output doesn't preview in VS Code / browsers reliably, so write
# H.264 by piping raw BGR24 to a system ffmpeg subprocess. Fall back to cv2's mp4v only
# if ffmpeg isn't on PATH.
FFMPEG_BIN = shutil.which("ffmpeg")

# Corner ordering matches Base_Task.save_obb2d / script/visualize_obb.py:
#   signs = [(-1,-1,-1),(-1,-1,1),(-1,1,-1),(-1,1,1),(1,-1,-1),(1,-1,1),(1,1,-1),(1,1,1)]
BOX_EDGES = [
    (0, 1), (1, 3), (3, 2), (2, 0),  # bottom face
    (4, 5), (5, 7), (7, 6), (6, 4),  # top face
    (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
]

LEFT_COLOR_BGR = (0, 255, 0)      # green
RIGHT_COLOR_BGR = (255, 255, 0)   # cyan
LINE_THICKNESS = 2
CORNER_RADIUS = 3
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_tasks_index(root: Path) -> dict[int, str]:
    """task_index -> task prompt string (from meta/tasks.jsonl)."""
    out: dict[int, str] = {}
    with (root / "meta" / "tasks.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)
            out[int(j["task_index"])] = j["task"]
    return out


def load_episodes_index(root: Path) -> list[dict]:
    """List of episode dicts (from meta/episodes.jsonl), ordered by episode_index."""
    rows: list[dict] = []
    with (root / "meta" / "episodes.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    rows.sort(key=lambda r: int(r["episode_index"]))
    return rows


def load_info(root: Path) -> dict:
    return json.loads((root / "meta" / "info.json").read_text())


def episode_file(info: dict, root: Path, ep_idx: int, kind: str, video_key: str | None = None) -> Path:
    """Resolve the on-disk path for an episode's parquet/mp4 using info.json templates."""
    chunk = ep_idx // int(info["chunks_size"])
    if kind == "parquet":
        rel = info["data_path"].format(episode_chunk=chunk, episode_index=ep_idx)
    else:
        rel = info["video_path"].format(episode_chunk=chunk, episode_index=ep_idx, video_key=video_key)
    return root / rel


def draw_obb(canvas: np.ndarray, corners_2d: np.ndarray, color, label: str | None) -> None:
    """corners_2d: (8, 2) int pixel coords."""
    if not np.any(corners_2d):
        return
    h, w = canvas.shape[:2]
    pts = corners_2d.astype(int)
    for a, b in BOX_EDGES:
        cv2.line(canvas, tuple(pts[a]), tuple(pts[b]), color, LINE_THICKNESS, cv2.LINE_AA)
    for p in pts:
        cv2.circle(canvas, tuple(p), CORNER_RADIUS, color, -1, cv2.LINE_AA)
    if label:
        anchor = pts[np.argmax(pts[:, 1])]
        x = int(np.clip(anchor[0], 4, w - 80))
        y = int(np.clip(anchor[1] + 14, 14, h - 4))
        cv2.putText(canvas, label, (x, y), LABEL_FONT, 0.45, color, 1, cv2.LINE_AA)


def render_episode(
    info: dict,
    root: Path,
    ep_idx: int,
    task_prompt: str,
    out_path: Path,
) -> tuple[int, int]:
    """Write one mp4 for episode ep_idx. Returns (n_frames, n_target_frames)."""
    parquet_path = episode_file(info, root, ep_idx, kind="parquet")
    video_path = episode_file(info, root, ep_idx, kind="video", video_key="observation.images.cam_high")

    df = pd.read_parquet(parquet_path)
    df = df.sort_values("frame_index").reset_index(drop=True)
    n_frames = len(df)
    obbs = np.stack(df["observation.obb_head"].to_numpy()).astype(np.float32)
    obbs = obbs.reshape(n_frames, 2, 8, 2)

    # OpenCV can't decode the AV1 streams that LeRobot writes by default, so decode with
    # PyAV (libdav1d under the hood) and re-encode the overlay output as plain mp4v.
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    w, h = stream.codec_context.width, stream.codec_context.height
    fps_video = float(stream.average_rate) if stream.average_rate else float(info.get("fps", 25))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if FFMPEG_BIN:
        ff = subprocess.Popen(
            [
                FFMPEG_BIN, "-y", "-loglevel", "error",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-s", f"{w}x{h}", "-pix_fmt", "bgr24",
                "-r", f"{fps_video}", "-i", "-",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(out_path),
            ],
            stdin=subprocess.PIPE,
        )
        writer = None
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps_video, (w, h))
        if not writer.isOpened():
            container.close()
            raise RuntimeError(f"cv2.VideoWriter failed to open {out_path}")
        ff = None

    n_target = 0
    i = 0
    try:
        for frame in container.decode(video=0):
            if i >= n_frames:
                break  # parquet/video length mismatch: stop at the shorter of the two
            rgb = frame.to_ndarray(format="rgb24")
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            obb_left = obbs[i, 0]
            obb_right = obbs[i, 1]
            has_left = bool(np.any(obb_left))
            has_right = bool(np.any(obb_right))
            if has_left or has_right:
                n_target += 1
            if has_left:
                draw_obb(bgr, obb_left, LEFT_COLOR_BGR, "L")
            if has_right:
                draw_obb(bgr, obb_right, RIGHT_COLOR_BGR, "R")
            footer = f"ep={ep_idx} t={i}/{n_frames - 1}"
            cv2.putText(bgr, footer, (4, h - 6), LABEL_FONT, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            if task_prompt:
                cv2.putText(bgr, task_prompt[:48], (4, 14), LABEL_FONT, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            if ff is not None:
                ff.stdin.write(bgr.tobytes())
            else:
                writer.write(bgr)
            i += 1
    finally:
        if ff is not None:
            ff.stdin.close()
            rc = ff.wait()
            if rc != 0:
                raise RuntimeError(f"ffmpeg exited {rc} for {out_path}")
        if writer is not None:
            writer.release()
        container.close()
    return i, n_target


def slugify(prompt: str) -> str:
    out = "".join(c if c.isalnum() else "_" for c in prompt.lower()).strip("_")
    return out or "episode"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lerobot-root", type=str, required=True,
                    help="On-disk root of the LeRobot dataset (same value passed to the converter's --lerobot-root).")
    ap.add_argument("--out-dir", type=str, required=True,
                    help="Where to write the mp4s.")
    ap.add_argument("--limit", type=int, default=0,
                    help=">0 to render only the first N episodes (smoke test).")
    args = ap.parse_args(argv)

    root = Path(args.lerobot_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    info = load_info(root)
    tasks_idx = load_tasks_index(root)
    episodes = load_episodes_index(root)
    print(f"Loaded {len(episodes)} episodes, fps={info.get('fps')}, root={root}")

    n_done = 0
    for ep_row in episodes:
        ep_idx = int(ep_row["episode_index"])
        if args.limit and n_done >= args.limit:
            break
        # episodes.jsonl stores 'tasks' as a list; first row's task_index is also in the parquet.
        prompt = ep_row.get("tasks", [""])[0] if ep_row.get("tasks") else ""
        if not prompt:
            df_head = pd.read_parquet(episode_file(info, root, ep_idx, kind="parquet"), columns=["task_index"])
            ti = int(df_head["task_index"].iloc[0])
            prompt = tasks_idx.get(ti, f"episode{ep_idx:04d}")
        out_path = out_dir / f"{slugify(prompt)}_ep{ep_idx:04d}.mp4"
        try:
            n_frames, n_target = render_episode(info, root, ep_idx, prompt, out_path)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] ep{ep_idx} ({prompt}): {e}", file=sys.stderr)
            continue
        n_done += 1
        print(f"[ok]   ep{ep_idx:03d} {prompt[:30]:30s} -> {out_path.name}  ({n_frames} frames, {n_target} with OBB)")

    print(f"\nWrote {n_done}/{len(episodes)} mp4s under {out_dir}")
    return 0 if n_done else 1


if __name__ == "__main__":
    sys.exit(main())

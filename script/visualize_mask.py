"""Visualize object masks overlaid on RGB frames from a demo episode."""
import argparse
import io
import h5py
import numpy as np
import cv2
from PIL import Image

CAMERAS = ["head_camera", "front_camera", "left_camera", "right_camera"]
MASK_COLOR = np.array([0, 255, 0], dtype=np.uint8)  # green overlay


def decode_rgb(raw_bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(bytes(raw_bytes))))


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, alpha=0.5) -> np.ndarray:
    out = rgb.copy()
    out[mask > 0] = (alpha * MASK_COLOR + (1 - alpha) * rgb[mask > 0]).astype(np.uint8)
    return out


def make_grid(frames: list[np.ndarray]) -> np.ndarray:
    """Stack frames in a 2x2 grid (or fewer)."""
    n = len(frames)
    h, w = frames[0].shape[:2]
    if n == 1:
        return frames[0]
    if n == 2:
        return np.concatenate(frames, axis=1)
    if n <= 4:
        row1 = np.concatenate(frames[:2], axis=1)
        row2 = np.concatenate((frames[2:4] + [np.zeros_like(frames[0])] * (4 - n)), axis=1)
        return np.concatenate([row1, row2], axis=0)
    return np.concatenate(frames[:4], axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf5", help="Path to episode HDF5 file")
    parser.add_argument("--out", default="mask_vis.mp4", help="Output MP4 path")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--cameras", nargs="+", default=CAMERAS)
    parser.add_argument("--object", default="bottle", help="Object name for mask")
    args = parser.parse_args()

    with h5py.File(args.hdf5, "r") as f:
        # filter to cameras that actually exist
        cams = [c for c in args.cameras if f"observation/{c}/rgb" in f]
        n_frames = f[f"observation/{cams[0]}/rgb"].shape[0]

        all_rgb = {c: f[f"observation/{c}/rgb"][:] for c in cams}
        all_masks = {}
        for c in cams:
            key = f"observation/{c}/object_mask/{args.object}"
            all_masks[c] = f[key][:] if key in f else np.zeros((n_frames, *f[f"observation/{c}/rgb"].shape[1:2]), dtype=np.uint8)

    # determine output frame size from grid
    sample = decode_rgb(all_rgb[cams[0]][0])
    h, w = sample.shape[:2]
    n_cams = len(cams)
    grid_w = w * min(n_cams, 2)
    grid_h = h * ((n_cams + 1) // 2)

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (grid_w, grid_h))

    for i in range(n_frames):
        frames = []
        for c in cams:
            rgb = decode_rgb(all_rgb[c][i])
            mask = all_masks[c][i]
            vis = overlay_mask(rgb, mask)
            # add camera label
            cv2.putText(vis, c, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            frames.append(vis)

        grid = make_grid(frames)
        writer.write(cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"Saved {n_frames} frames → {args.out}")


if __name__ == "__main__":
    main()

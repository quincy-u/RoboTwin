"""Run save_obb2d on every collected task's data — generic batch runner.

For each subdirectory of `data/`, locate `<task>/demo_clean/data/episode*.hdf5`
and fill in `observation/{camera}/proj_3d_obb/{obj}` for every tracked object.

How an object's asset directory is resolved (in order):
  1. ``object_info/{obj}`` dataset in the HDF5 (auto-written by
     ``Base_Task.save_obb2d`` after the data-collection refactor; format
     ``"modelname/baseN"`` e.g. ``"001_bottle/base16"``).
  2. ``scene_info.json`` info field ``{A}``/``{B}``/... matched positionally
     against the sorted ``object_pose/*`` keys in the HDF5 (legacy data
     without ``object_info``).
  3. Manual --override mapping ``--override task:obj=001_bottle/base16`` (and
     ``--object_info`` keys passed through to ``save_obb2d.py``).

Usage:
    python script/save_obb2d_all.py                       # all tasks under data/
    python script/save_obb2d_all.py adjust_bottle ...     # named tasks
    python script/save_obb2d_all.py --demos demo_clean demo_randomized
    python script/save_obb2d_all.py --dry-run             # print actions
"""
import argparse
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from script.save_obb2d import (
    load_model_data,
    process_episode,
    CAMERAS,
)


def parse_label(label: str) -> tuple[str, int]:
    """'001_bottle/base16' -> ('001_bottle', 16)."""
    m = re.match(r"(.+?)/base(\d+)$", label)
    if m is None:
        raise ValueError(f"Cannot parse asset label: {label}")
    return m.group(1), int(m.group(2))


def discover_objects_from_hdf5(hdf5_path: Path) -> list[str]:
    with h5py.File(hdf5_path, "r") as f:
        if "object_pose" not in f:
            return []
        return sorted(list(f["object_pose"].keys()))


def read_object_info_from_hdf5(hdf5_path: Path, obj: str) -> str | None:
    with h5py.File(hdf5_path, "r") as f:
        key = f"object_info/{obj}"
        if key not in f:
            return None
        v = f[key][()]
        if isinstance(v, bytes):
            return v.decode()
        return str(v)


def per_episode_model_dirs(
    task_dir: Path,
    objects: list[str],
    overrides: dict[str, str],
) -> dict[str, tuple[Path, int, dict[int, int]]]:
    """Return {obj_name: (model_dir, fallback_model_id, ep_idx_to_model_id)}.

    For each object, build a per-episode map of model_id (so save_obb2d can
    pick the right model_data when episodes use varying variants), plus a
    fallback in case the map doesn't cover an episode.
    """
    data_dir = task_dir / "data"
    scene_info_path = task_dir / "scene_info.json"
    scene_info = {}
    if scene_info_path.exists():
        try:
            scene_info = json.loads(scene_info_path.read_text())
        except Exception:
            scene_info = {}

    episodes = sorted(data_dir.glob("episode*.hdf5"),
                      key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
    if not episodes:
        return {}

    # Step 1: try object_info from any episode (they should be identical
    # for modelname, but model_id can differ episode-by-episode).
    obj_resolution = {}  # obj -> (modelname, per-episode dict[ep_idx]->model_id)
    for obj in objects:
        if obj in overrides:
            modelname, model_id = parse_label(overrides[obj])
            obj_resolution[obj] = (modelname, {}, model_id)
            continue
        modelname = None
        ep_mid: dict[int, int] = {}
        for ep in episodes:
            label = read_object_info_from_hdf5(ep, obj)
            if label is None:
                continue
            mn, mid = parse_label(label)
            if modelname is None:
                modelname = mn
            elif modelname != mn:
                # different modelname per episode is unusual; skip
                pass
            ep_idx = int(re.search(r"(\d+)", ep.stem).group(1))
            ep_mid[ep_idx] = mid
        if modelname is not None:
            obj_resolution[obj] = (modelname, ep_mid, None)

    # Step 2: legacy fallback via scene_info {A}/{B}/...
    placeholder_keys = ["{A}", "{B}", "{C}", "{D}", "{E}", "{F}", "{G}", "{H}"]
    unresolved = [o for o in objects if o not in obj_resolution]
    for i, obj in enumerate(unresolved):
        # match positionally: i-th object → i-th placeholder
        if i >= len(placeholder_keys):
            continue
        ph = placeholder_keys[i]
        modelname = None
        ep_mid = {}
        for ep_key, info in scene_info.items():
            inf = info.get("info") or {}
            label = inf.get(ph)
            if not isinstance(label, str) or "/base" not in label:
                continue
            try:
                mn, mid = parse_label(label)
            except ValueError:
                continue
            if modelname is None:
                modelname = mn
            elif modelname != mn:
                pass
            try:
                ep_idx = int(ep_key.split("_")[-1])
            except ValueError:
                continue
            ep_mid[ep_idx] = mid
        if modelname is not None:
            obj_resolution[obj] = (modelname, ep_mid, None)

    # Resolve to (model_dir, fallback_id, per-ep map)
    out = {}
    for obj, (modelname, ep_mid, override_mid) in obj_resolution.items():
        model_dir = REPO / "assets" / "objects" / modelname
        if not model_dir.exists():
            print(f"  [warn] {obj}: asset dir {model_dir} does not exist; skipping")
            continue
        fallback = override_mid
        if fallback is None and ep_mid:
            fallback = next(iter(ep_mid.values()))
        if fallback is None:
            fallback = 0
        out[obj] = (model_dir, fallback, ep_mid)
    return out


def process_task(task_dir: Path, overrides: dict[str, str], cameras: list[str], dry_run: bool):
    data_dir = task_dir / "data"
    if not data_dir.is_dir():
        print(f"  [skip] no data/ dir in {task_dir}")
        return
    episodes = sorted(data_dir.glob("episode*.hdf5"),
                      key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
    if not episodes:
        print(f"  [skip] no episodes in {data_dir}")
        return

    objects = discover_objects_from_hdf5(episodes[0])
    if not objects:
        print(f"  [skip] no object_pose in {episodes[0].name} — task does not track objects")
        return

    resolution = per_episode_model_dirs(task_dir, objects, overrides)
    if not resolution:
        print(f"  [skip] could not resolve any asset for objects: {objects}")
        return

    print(f"  Objects: {objects}")
    for obj, (mdir, fb, ep_mid) in resolution.items():
        ep_count = len(ep_mid)
        print(f"    {obj} → {mdir.name} (per-episode={ep_count} variants, fallback id={fb})")

    if dry_run:
        return

    # For each object, iterate episodes and load model_data with the matching id.
    for obj, (mdir, fb, ep_mid) in resolution.items():
        for ep in episodes:
            ep_idx = int(re.search(r"(\d+)", ep.stem).group(1))
            mid = ep_mid.get(ep_idx, fb)
            try:
                model_data = load_model_data(mdir, mid)
            except Exception as e:
                print(f"    [err] {obj} ep{ep_idx}: load model_data({mid}): {e}")
                continue
            try:
                process_episode(ep, model_data, cameras, obj_name=obj)
            except Exception as e:
                print(f"    [err] {obj} ep{ep_idx}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks", nargs="*", help="Task names; default = all subdirs of data/")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--demos", nargs="+", default=["demo_clean"],
                        help="Which demo dirs to process per task (e.g. demo_clean demo_randomized)")
    parser.add_argument("--cameras", nargs="+", default=CAMERAS)
    parser.add_argument("--override", action="append", default=[],
                        help="Per-task object override: 'task:obj=modelname/baseN' "
                             "(e.g. --override pick_dual_bottles:bottle1=001_bottle/base13)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Parse overrides
    overrides_by_task: dict[str, dict[str, str]] = {}
    for ov in args.override:
        try:
            task_obj, label = ov.split("=", 1)
            task, obj = task_obj.split(":", 1)
        except ValueError:
            sys.exit(f"Bad --override (need 'task:obj=modelname/baseN'): {ov}")
        overrides_by_task.setdefault(task, {})[obj] = label

    data_root = Path(args.data_root)
    if args.tasks:
        task_names = args.tasks
    else:
        task_names = sorted([d.name for d in data_root.iterdir()
                             if d.is_dir() and not d.name.startswith(".")])
    print(f"Tasks: {task_names}")

    for task in task_names:
        for demo in args.demos:
            task_demo = data_root / task / demo
            if not task_demo.is_dir():
                continue
            print(f"\n=== {task}/{demo} ===")
            process_task(task_demo, overrides_by_task.get(task, {}), args.cameras, args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()

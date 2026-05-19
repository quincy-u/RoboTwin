"""Retro-fill object_target/{left,right} labels into already-collected HDF5s.

For every episode under data/<task>/<config>/data/episode*.hdf5, run
script/object_target_labeler.py and write/overwrite ``object_target/left`` and
``object_target/right`` (variable-length string, shape (T,)).

Usage:
    python script/label_object_targets_all.py                    # all tasks, default demo_obb_vis
    python script/label_object_targets_all.py adjust_bottle ...  # named tasks
    python script/label_object_targets_all.py --config demo_clean
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from script.object_target_labeler import compute_object_targets, write_object_targets  # noqa: E402


def process_episode(ep_path: Path) -> dict:
    with h5py.File(ep_path, "r+") as f:
        targets = compute_object_targets(f)
        write_object_targets(f, targets)
    L_set = sorted({s for s in targets["left"] if s})
    R_set = sorted({s for s in targets["right"] if s})
    L_active = int(sum(1 for s in targets["left"] if s))
    R_active = int(sum(1 for s in targets["right"] if s))
    T = len(targets["left"])
    return {
        "T": T, "left_objs": L_set, "right_objs": R_set,
        "L_active": L_active, "R_active": R_active,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tasks", nargs="*", help="task names; default = all under data/")
    p.add_argument("--data-root", default="data")
    p.add_argument("--config", default="demo_obb_vis")
    args = p.parse_args()

    data_root = Path(args.data_root)
    task_names = args.tasks if args.tasks else sorted(
        d.name for d in data_root.iterdir() if d.is_dir() and not d.name.startswith(".")
    )

    total_eps, total_ok, total_err = 0, 0, 0
    for task in task_names:
        ep_dir = data_root / task / args.config / "data"
        if not ep_dir.is_dir():
            continue
        episodes = sorted(ep_dir.glob("episode*.hdf5"))
        if not episodes:
            continue
        per_task_summary = []
        for ep in episodes:
            total_eps += 1
            try:
                rep = process_episode(ep)
                per_task_summary.append(
                    f"T={rep['T']} L={rep['L_active']}f({','.join(rep['left_objs']) or '-'}) "
                    f"R={rep['R_active']}f({','.join(rep['right_objs']) or '-'})"
                )
                total_ok += 1
            except Exception as e:
                per_task_summary.append(f"ERR: {e}")
                total_err += 1
        print(f"{task:35s}  {len(episodes):3d} eps  | {per_task_summary[0]}")

    print(f"\nDone. processed={total_eps}  ok={total_ok}  err={total_err}")


if __name__ == "__main__":
    main()

"""Aggregate DP policy eval results: mean ± std success rate per (task, eval_config).

Reads `eval_result/<task>/DP/<eval_config>/<ckpt_setting>/<timestamp>_seed<seed>/_result.txt`.
Each file's success rate is the LAST non-blank line (the float `suc/test_num`
that eval_policy.py writes). When multiple timestamped dirs exist for the same
seed, takes the LATEST (lexicographic sort works because timestamps are ISO).

Outputs a markdown summary table and prints it to stdout.
"""
import argparse
from pathlib import Path
import re
import statistics
import sys


def read_result(result_txt: Path) -> float | None:
    try:
        lines = [l.strip() for l in result_txt.read_text().splitlines() if l.strip()]
    except Exception:
        return None
    if not lines:
        return None
    last = lines[-1]
    try:
        return float(last)
    except ValueError:
        return None


def latest_seed_result(task: str, eval_config: str, ckpt_setting: str, seed: int) -> tuple[float | None, Path | None]:
    base = Path(f"eval_result/{task}/DP/{eval_config}/{ckpt_setting}")
    if not base.is_dir():
        return None, None
    pat = re.compile(rf"_seed{seed}$")
    candidates = sorted(
        [d for d in base.iterdir() if d.is_dir() and pat.search(d.name)],
        key=lambda p: p.name,
        reverse=True,
    )
    for d in candidates:
        rf = d / "_result.txt"
        if rf.exists():
            v = read_result(rf)
            if v is not None:
                return v, rf
    return None, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", required=True, help='space-separated, e.g. "a b c"')
    p.add_argument("--seeds", required=True, help='space-separated, e.g. "1 2 3"')
    p.add_argument("--eval-configs", required=True, help='space-separated')
    p.add_argument("--ckpt-setting", required=True)
    p.add_argument("--out", default=None, help="optional markdown output path")
    args = p.parse_args()

    tasks = args.tasks.split()
    seeds = [int(s) for s in args.seeds.split()]
    eval_configs = args.eval_configs.split()

    rows = []
    missing = []
    for task in tasks:
        for ec in eval_configs:
            seed_vals = {}
            for s in seeds:
                v, path = latest_seed_result(task, ec, args.ckpt_setting, s)
                if v is None:
                    missing.append((task, ec, s))
                else:
                    seed_vals[s] = (v, path)
            vals = [v for (v, _) in seed_vals.values()]
            if vals:
                mean = statistics.mean(vals)
                std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            else:
                mean = std = float("nan")
            rows.append((task, ec, seed_vals, mean, std))

    # Markdown table
    header = "| task | eval_config | " + " | ".join(f"seed {s}" for s in seeds) + " | mean | std |"
    sep    = "|" + "|".join(["---"] * (3 + len(seeds))) + "|"
    body = [header, sep]
    for task, ec, sv, mean, std in rows:
        cells = []
        for s in seeds:
            cells.append(f"{sv[s][0]:.3f}" if s in sv else "—")
        body.append(f"| {task} | {ec} | " + " | ".join(cells) + f" | {mean:.3f} | {std:.3f} |")

    out_text = "# DP eval summary\n\n" + "\n".join(body) + "\n"
    if missing:
        out_text += "\n## Missing results\n"
        for t, ec, s in missing:
            out_text += f"- {t} / {ec} / seed={s}\n"
    print(out_text)
    if args.out:
        Path(args.out).write_text(out_text)
        print(f"\nWritten to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

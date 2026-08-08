# Heuristic Baseline Integration Plan

Status: approved 2026-08-07

## Scope

Integrate `~/projects/simple-grasp` as RoboTwin's heuristic grasp policy using
simulator depth, GT instance labels, and GT 6D object poses. Keep
`deploy_policy.py` limited to evaluator entrypoints and keep the runtime/model
adapters in separate modules.

## Current baseline

- Branch: `heuristic` at `cc41a681c5445d4f68636773a385701c2c079481`.
- The current endpoint-based adapter completed two `shake_bottle` smoke runs at
  `1/1`, including the packaged `eval.sh` launcher.
- That task's success predicate verifies grasp/lift, not a shake motion.
- `docs/install.md` and `script/_install_uv.sh` contain pre-existing user changes
  and will remain untouched and excluded from commits.

## Plan

1. Tighten policy boundaries and target association.
   - Honor configured `simple_grasp_root` before importing the package.
   - Select only M2T2's GT-matched query, require a small contact-mask IoU, and
     require exact sampled target-contact membership.
   - Convert only expected no-target/no-grasp/no-IK outcomes into an episode
     failure; unexpected dependency or shape errors still abort visibly.
   - Verification: focused adapter tests plus all upstream `simple-grasp` tests.

2. Preserve planner trajectories.
   - Chain pregrasp, grasp, and retreat plans from the preceding planner result.
   - Replay bounded/subsampled CuRobo waypoints instead of only terminal qpos.
   - Preserve raw M2T2 confidence for thresholding and use a separate geometric
     ranking score.
   - Verification: synthetic planner/controller tests confirm seed chaining,
     candidate reset, waypoint order, qpos dimensions, and action-count bounds.

3. Harden reproducibility and packaging.
   - Reset backend sampling per episode, pin install dependencies, and verify the
     checkpoint checksum through an atomic download.
   - Add concise `SUMMARY.md`, `PROGRESS.md`, and `DEBUG.md` records.
   - Commit only heuristic-policy and workflow-document paths in atomic commits.
   - Verification: shell syntax, Python compile/import, YAML parsing, clean diff
     review, and confirmation that unrelated staged files remain unchanged.

4. Run one final simulator smoke test.
   - Host: `shenlong-gpu-01`; allowed devices: GPU 6 or 7 only; task: `shake_bottle`;
     target: `bottle`; one accepted seed; checkpoint: `m2t2.pth`.
   - Record host, commit, worktree state, Python/Torch/CUDA versions, GPU memory,
     command, checkpoint identity, and output directory beside the result.
   - Verification: evaluator exits cleanly and reports `1/1`; a planning failure
     is also confirmed to fail one episode without leaking simulator resources.

## Approval needed

Approved by the user. Before the run, inspect GPU 6 and 7 memory/processes,
select a non-conflicting device, and never stop another user process.

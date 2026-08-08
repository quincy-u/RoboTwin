# Summary

## Heuristic baseline

The [`policy/heuristic_baseline`](policy/heuristic_baseline) package connects
RoboTwin simulator observations to the adjacent `simple-grasp` policy.

- [`deploy_policy.py`](policy/heuristic_baseline/deploy_policy.py) contains only
  evaluator entrypoints and typed per-episode failure containment.
- [`observation.py`](policy/heuristic_baseline/observation.py) converts SAPIEN
  head-camera RGB-D into a world-frame cloud with GT instance labels and GT 6D
  object poses.
- [`model.py`](policy/heuristic_baseline/model.py) configures the requested
  simple-grasp checkout before imports and lazily owns the heavy runtime.
- [`m2t2_backend.py`](policy/heuristic_baseline/m2t2_backend.py) loads M2T2 once,
  selects only the GT-matched query, verifies target contacts, and isolates its
  NumPy/Torch randomness.
- [`runtime.py`](policy/heuristic_baseline/runtime.py) selects the target/arm,
  chains full-qpos CuRobo plans, and replays bounded collision-planned waypoints.
- [`eval.sh`](policy/heuristic_baseline/eval.sh) and
  [`install.sh`](policy/heuristic_baseline/install.sh) provide the smoke launcher
  and pinned/checksummed setup path.

CPU verification covers 18 integration tests, all six upstream simple-grasp
tests, Python compilation, YAML parsing, and shell syntax.

The post-review GPU attempt reached GT-matched M2T2 inference with 97 exact
target candidates and exposed a CuRobo seed-dtype mismatch before motion. That
bug is fixed in `e328493`; the final trajectory rerun is pending because both
user-approved GPUs are occupied by another training job.

The current smoke target is `shake_bottle`, whose success predicate validates a
grasp and short lift. It does not validate a shaking trajectory.

The current coordinator supports one selected object and one arm per rollout;
dual-object and coordinated dual-arm tasks remain out of scope.

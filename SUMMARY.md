# Summary

## Heuristic baseline

The `policy/heuristic_baseline` package connects RoboTwin simulator observations
to the adjacent `simple-grasp` policy. `deploy_policy.py` contains only evaluator
entrypoints; observation conversion, model lifetime, M2T2 inference, and RoboTwin
planning/control live in separate modules.

The current smoke target is `shake_bottle`, whose success predicate validates a
grasp and short lift. It does not validate a shaking trajectory.

Implementation links will be finalized after the approved correctness pass.

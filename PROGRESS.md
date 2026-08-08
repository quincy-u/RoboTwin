# Progress

## 2026-08-07

- Plan approved on branch `heuristic`.
- GPU work is restricted to devices 6 or 7 after checking active processes;
  no existing process may be stopped.
- Simulator RGB-D, GT instance masks, and GT 6D poses are encoded into the
  simple-grasp scene contract.
- The initial endpoint controller passed two one-episode grasp/lift smoke runs.
- Completed: matched-query filtering, full-qpos planner chaining and bounded path
  replay, typed episode failures, deterministic sampling, installer hardening,
  and CPU tests.
- Verification passed: 18 heuristic tests, six upstream simple-grasp tests,
  Python compilation, YAML parsing, and shell syntax.
- Next: atomic feature commit, then the approved one-episode smoke run on a
  non-conflicting GPU 6 or 7.
- Feature commit: `4b1d2a9`; CuRobo seed-dtype fix: `e328493`.
- First GPU-7 attempt reached GT-matched M2T2 inference (97 candidates, purity
  `1.0`) and exposed a float64 planner seed before motion. The fix now preserves
  RoboTwin float32 qpos and passes all CPU tests.
- Final rerun is externally blocked: after an extended wait, another user job
  still occupies about 88 GiB and 100% compute on both allowed GPUs. The
  read-only wait loop was stopped; the training job remains untouched.
- Resume command once GPU 6 or 7 is safely available:
  `policy/heuristic_baseline/eval.sh shake_bottle heuristic_smoke 0 7 bottle 1`.
- Blockers: none.

# Progress

## 2026-08-07

- Plan approved on branch `heuristic`.
- GPU work is restricted to devices 6 or 7 after checking active processes;
  no existing process may be stopped.
- Simulator RGB-D, GT instance masks, and GT 6D poses are encoded into the
  simple-grasp scene contract.
- The initial endpoint controller passed two one-episode grasp/lift smoke runs.
- In progress: matched-query filtering, planner-path replay, typed episode
  failures, deterministic sampling, installer hardening, and CPU tests.
- Blockers: none.

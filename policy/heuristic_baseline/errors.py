"""Expected, per-episode failures for the heuristic grasp policy."""
from __future__ import annotations


class HeuristicEpisodeFailure(RuntimeError):
    """A grasp outcome that should fail this rollout without aborting evaluation."""


class TargetSelectionFailure(HeuristicEpisodeFailure):
    """The configured target could not be selected from the scene."""


class NoVisibleTargetFailure(HeuristicEpisodeFailure):
    """The selected target has no usable visible points."""


class NoObjectQueryFailure(HeuristicEpisodeFailure):
    """M2T2 produced no object query associated with the selected target."""


class NoFeasiblePlanFailure(HeuristicEpisodeFailure):
    """No candidate yielded a collision-free executable grasp plan."""

"""Small compatibility shims required by the pinned RoboTwin stack."""


def patch_warp_for_curobo() -> None:
    """Support CuRobo 0.7.8 with Warp versions that moved torch helpers."""
    try:
        import warp as wp
    except ImportError:
        return
    if not hasattr(wp, "torch") and hasattr(wp, "device_from_torch"):
        wp.torch = wp

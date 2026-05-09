"""pi0.5 + joint OBB denoising runtime.

Mirrors ``policy/pi05/pi_model.py`` but unpacks the additional ``future_obb`` channel
returned by ``policy.infer`` when the underlying model is configured with
``predict_obb=True`` (see ``pi05_aloha_full_base`` in ``training/config.py``).

Reuses the *vendored* openpi tree at ``policy/pi05/src/openpi/`` (the OBB transforms
have been synced into it). We import openpi from there so the same .venv that runs
``policy/pi05/`` also runs this policy.
"""
from __future__ import annotations

import os
import sys

import numpy as np

# Prefer the vendored openpi tree alongside `pi05/` so this policy works inside the
# existing `policy/pi05/.venv` environment without needing a parallel install.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PI05_SRC = os.path.normpath(os.path.join(_THIS_DIR, "..", "pi05", "src"))
if os.path.isdir(_PI05_SRC) and _PI05_SRC not in sys.path:
    sys.path.insert(0, _PI05_SRC)

from openpi.policies import policy_config as _policy_config  # noqa: E402
from openpi.training import config as _config  # noqa: E402


class PI0OBB:
    """Thin wrapper around an ``openpi`` policy that returns predicted OBBs alongside actions."""

    def __init__(self, train_config_name: str, model_name: str, checkpoint_id: str | int, pi0_step: int):
        self.train_config_name = train_config_name
        self.model_name = model_name
        self.checkpoint_id = str(checkpoint_id)

        ckpt_root = (
            f"policy/pi05_obb/checkpoints/{self.train_config_name}/{self.model_name}/{self.checkpoint_id}"
        )
        assets_dir = os.path.join(ckpt_root, "assets")
        if not os.path.isdir(assets_dir):
            raise FileNotFoundError(
                f"Checkpoint assets/ not found at {assets_dir}. Symlink your trained "
                "checkpoint there (e.g. "
                "`ln -s /home/.../openpi/checkpoints/<name>/<exp>/<step> "
                "policy/pi05_obb/checkpoints/<name>/<exp>/<step>`)."
            )
        assets_id = os.listdir(assets_dir)[0]

        config = _config.get_config(self.train_config_name)
        self.policy = _policy_config.create_trained_policy(
            config,
            ckpt_root,
            robotwin_repo_id=assets_id,
        )
        print("loading model success!")
        self.img_size = (224, 224)
        self.observation_window: dict | None = None
        self.pi0_step = int(pi0_step)
        # Most recent predicted future OBB chunk, ``[H, max_objects, 8, 2]`` in [0, 1] image fractions.
        self.last_future_obb: np.ndarray | None = None

    def set_img_size(self, img_size):
        self.img_size = img_size

    def set_language(self, instruction: str):
        self.instruction = instruction
        print(f"successfully set instruction:{instruction}")

    def update_observation_window(self, img_arr, state):
        img_front, img_right, img_left = img_arr[0], img_arr[1], img_arr[2]
        # (H, W, 3) -> (3, H, W)
        img_front = np.transpose(img_front, (2, 0, 1))
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))

        self.observation_window = {
            "state": state,
            "images": {
                "cam_high": img_front,
                "cam_left_wrist": img_left,
                "cam_right_wrist": img_right,
            },
            "prompt": self.instruction,
        }

    def get_action(self):
        """Run a single forward pass; returns ``actions`` and caches the predicted OBB chunk."""
        assert self.observation_window is not None, "update observation_window first!"
        out = self.policy.infer(self.observation_window)
        # ``future_obb`` only present when model trained with ``predict_obb=True``.
        future_obb = out.get("future_obb_pixels", None)
        if future_obb is None:
            future_obb = out.get("future_obb", None)
        self.last_future_obb = None if future_obb is None else np.asarray(future_obb)
        return out["actions"]

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        self.last_future_obb = None
        print("successfully unset obs and language intruction")

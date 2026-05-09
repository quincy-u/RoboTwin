#!/usr/bin/env python3
"""Instantiate RoboTwin DP policy from Hydra config and print parameter counts."""
from __future__ import annotations

import argparse
import pathlib
import sys

import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent


def get_camera_config(camera_type: str) -> dict:
    path = _REPO_ROOT / "task_config" / "_camera_config.yml"
    with open(path, encoding="utf-8") as f:
        args = yaml.safe_load(f)
    return args[camera_type]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-dim", type=int, default=14, choices=(14, 16))
    parser.add_argument("--camera", type=str, default="D435", help="key in task_config/_camera_config.yml")
    args = parser.parse_args()

    OmegaConf.register_new_resolver("eval", eval, replace=True)

    config_dir = str(_SCRIPT_DIR / "diffusion_policy" / "config")
    cfg_name = f"robot_dp_{args.action_dim}"

    sys.path.insert(0, str(_SCRIPT_DIR))
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name=cfg_name,
            overrides=[
                "task.name=param_count",
                f"head_camera_type={args.camera}",
            ],
        )

    cam = get_camera_config(args.camera)
    h, w = cam["h"], cam["w"]
    cfg.task.image_shape = [3, h, w]
    cfg.task.shape_meta.obs.head_cam.shape = [3, h, w]
    OmegaConf.resolve(cfg)
    cfg.task.image_shape = [3, h, w]
    cfg.task.shape_meta.obs.head_cam.shape = [3, h, w]

    import torch
    from hydra.utils import instantiate

    policy = instantiate(cfg.policy)
    policy.eval()

    def count_params(module: torch.nn.Module) -> int:
        return sum(p.numel() for p in module.parameters())

    def count_trainable(module: torch.nn.Module) -> int:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    obs_n = count_params(policy.obs_encoder)
    unet_n = count_params(policy.model)
    total = count_params(policy)
    trainable = count_trainable(policy)

    print(f"Config: {cfg_name}.yaml  camera={args.camera}  image {h}x{w}")
    print(f"  obs_encoder (ResNet18 + low_dim): {obs_n:,}")
    print(f"  conditional_unet1d:               {unet_n:,}")
    print(f"  policy total:                     {total:,}")
    print(f"  trainable:                        {trainable:,}")


if __name__ == "__main__":
    main()

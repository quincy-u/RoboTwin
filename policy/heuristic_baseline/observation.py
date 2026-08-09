"""Convert RoboTwin simulator state into simple-grasp observations."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ._simple_grasp import ensure_simple_grasp_importable

if TYPE_CHECKING:
    from simple_grasp.types import ObjectState, SceneObservation


def _instance_id(actor: Any, fallback: int) -> int:
    """Return one label ID, including for multi-link articulated actors."""
    entity = getattr(actor, "actor", actor)
    instance_id = getattr(entity, "per_scene_id", getattr(entity, "id", None))
    return fallback if instance_id is None else int(instance_id)


def _actor_dataset_label(actor: Any) -> str | None:
    modelname = getattr(actor, "modelname", None)
    model_id = getattr(actor, "model_id", None)
    if not modelname:
        return None
    suffix = 0 if model_id is None or model_id == "" else model_id
    return f"{modelname}/base{suffix}"


def _resolve_target_name(
    task_env: Any, tracked: dict[str, Any], requested: str
) -> str | None:
    if requested != "auto":
        return requested if requested in tracked else None
    task_plan = getattr(task_env, "heuristic_task_plan", None)
    if task_plan is not None:
        primary_name = task_plan.primary_target
        if primary_name in tracked:
            return primary_name
        primary_label = None
    else:
        task_info = getattr(task_env, "heuristic_task_info", {}) or {}
        primary_label = task_info.get("{A}")
    if primary_label:
        matches = [
            name for name, actor in tracked.items()
            if _actor_dataset_label(actor) == primary_label
        ]
        if len(matches) == 1:
            return matches[0]
    candidates = [name for name in tracked if name != "wall"]
    return candidates[0] if len(candidates) == 1 else None


def _object_states(
    tracked: dict[str, Any], instance_ids: dict[str, int], object_state_type: type
) -> dict[str, "ObjectState"]:
    return {
        name: object_state_type(
            name=name,
            world_pose=np.asarray(
                actor.get_pose().to_transformation_matrix(), dtype=np.float64
            ),
            instance_id=instance_ids[name],
        )
        for name, actor in tracked.items()
        if actor is not None
    }


def _head_depth_m(task_env: Any, camera_obs: dict) -> np.ndarray:
    depth_mm = camera_obs.get("depth")
    if depth_mm is None:
        depth_mm = task_env.cameras.get_depth()["head_camera"]["depth"]

    depth_m = np.asarray(depth_mm, dtype=np.float32) / 1000.0
    if depth_m.ndim != 2:
        raise ValueError(f"head-camera depth must be HxW, got {depth_m.shape}")
    return depth_m


def _world_pointcloud(
    depth_m: np.ndarray,
    intrinsic: np.ndarray,
    world_to_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Back-project valid CV-camera depth pixels into world coordinates."""
    height, width = depth_m.shape
    v, u = np.indices((height, width), dtype=np.float64)
    z = depth_m.reshape(-1)
    valid = np.isfinite(z) & (z > 0.0)

    pixels = np.stack(
        (u.reshape(-1), v.reshape(-1), np.ones(height * width)), axis=0
    )
    rays = np.linalg.inv(np.asarray(intrinsic, dtype=np.float64)) @ pixels
    camera_xyz = rays[:, valid] * z[valid][None, :]
    camera_xyz_h = np.concatenate(
        (camera_xyz, np.ones((1, camera_xyz.shape[1]), dtype=np.float64)), axis=0
    )
    world_to_camera = np.asarray(world_to_camera, dtype=np.float64)
    if world_to_camera.shape == (3, 4):
        world_to_camera = np.vstack(
            (world_to_camera, [0.0, 0.0, 0.0, 1.0])
        )
    if world_to_camera.shape != (4, 4):
        raise ValueError(
            f"head-camera extrinsic must be 3x4 or 4x4, got "
            f"{world_to_camera.shape}"
        )
    camera_to_world = np.linalg.inv(world_to_camera)
    world_xyz = (camera_to_world @ camera_xyz_h)[:3].T.astype(np.float32)
    return world_xyz, valid, camera_to_world


def _head_world_pointcloud(
    task_env: Any,
    camera_obs: dict,
    depth_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use SAPIEN's Position buffer exactly as RoboTwin's native get_pcd."""
    cameras = task_env.cameras
    try:
        index = cameras.static_camera_name.index("head_camera")
    except ValueError as exc:
        raise RuntimeError("RoboTwin has no configured head_camera") from exc
    camera = cameras.static_camera_list[index]
    position = np.asarray(camera.get_picture("Position"), dtype=np.float32)
    if position.shape[:2] != depth_m.shape or position.shape[-1] < 4:
        raise ValueError(
            f"head Position/depth shapes do not align: "
            f"{position.shape} vs {depth_m.shape}"
        )

    valid = (
        np.isfinite(depth_m.reshape(-1))
        & (depth_m.reshape(-1) > 0.0)
        & (position[..., 3].reshape(-1) < 1.0)
    )
    camera_xyz = position[..., :3].reshape(-1, 3)[valid]
    camera_to_world = np.asarray(camera_obs["cam2world_gl"], dtype=np.float64)
    if camera_to_world.shape != (4, 4):
        raise ValueError(
            f"head cam2world_gl must be 4x4, got {camera_to_world.shape}"
        )
    world_xyz = (
        camera_xyz @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
    ).astype(np.float32)
    return world_xyz, valid, camera_to_world


def encode_obs(
    task_env: Any,
    observation: dict,
    *,
    simple_grasp_root: str | Path | None = None,
    target_name: str = "auto",
) -> "SceneObservation":
    """Build an RGB-D point cloud, GT segmentation, and tracked 6D object poses."""
    ensure_simple_grasp_importable(simple_grasp_root)
    from simple_grasp.types import ObjectState, SceneObservation

    camera_obs = observation["observation"]["head_camera"]
    rgb_image = np.asarray(camera_obs["rgb"], dtype=np.float32) / 255.0
    depth_m = _head_depth_m(task_env, camera_obs)
    if rgb_image.shape != (*depth_m.shape, 3):
        raise ValueError(
            f"head RGB/depth shapes do not align: "
            f"{rgb_image.shape} vs {depth_m.shape}"
        )

    xyz, valid, camera_to_world = _head_world_pointcloud(
        task_env,
        camera_obs,
        depth_m,
    )

    tracked = {
        name: actor
        for name, actor in (task_env.get_tracked_objects() or {}).items()
        if actor is not None
    }
    instance_ids = {
        name: _instance_id(actor, fallback=-(index + 2))
        for index, (name, actor) in enumerate(tracked.items())
    }
    selected_target = _resolve_target_name(task_env, tracked, target_name)

    labels = np.full(depth_m.shape, -1, dtype=np.int64)
    if selected_target is not None:
        target_only = {selected_target: tracked[selected_target]}
        target_mask = task_env.cameras.get_object_masks(target_only)["head_camera"]
        labels[np.asarray(target_mask[selected_target], dtype=bool)] = (
            instance_ids[selected_target]
        )

    return SceneObservation(
        xyz=xyz,
        rgb=rgb_image.reshape(-1, 3)[valid],
        instance_labels=labels.reshape(-1)[valid],
        camera_pose=camera_to_world,
        objects=_object_states(
            ({selected_target: tracked[selected_target]} if selected_target else tracked),
            instance_ids,
            ObjectState,
        ),
    )

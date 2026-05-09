"""Transforms for the joint OBB-trajectory channel used by pi0.5 with ``predict_obb=True``.

Reads a per-frame, per-object projected 3-D OBB trajectory in *native camera pixel
coordinates*, normalizes it to roughly ``[-1, 1]`` (matching the dino-wam recipe in
``vjepa2/app/dino_wam/train.py``), and emits the ``future_obb_gt`` and ``valid_obj_mask``
fields that the model consumes.

OBB layout (per frame):
    flat[0:16]   = object 0's 8 corners as [x0, y0, x1, y1, ..., x7, y7]
    flat[16:32]  = object 1's 8 corners (zeros for single-object tasks)

Normalization is per-axis because RoboTwin head-camera frames are **non-square**
(default 480×640): u (x) is divided by image_width and v (y) by image_height,
following the dino-wam ``_normalize_obb`` recipe (``clip(x01, -m, 1+m); 2x-1``).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from openpi import transforms


@dataclasses.dataclass(frozen=True)
class RobotwinObbInputs(transforms.DataTransformFn):
    """Convert raw-pixel OBB trajectories to normalized OBB inputs + a per-object validity mask.

    Reads ``data[src_key]`` of shape ``[H, max_objects * per_object_dim]`` and writes:
      ``future_obb_gt`` : float32 ``[H, max_objects * per_object_dim]`` in ~``[-1, 1]``.
      ``valid_obj_mask``: float32 ``[H, max_objects]`` (1 = real object, 0 = zero-padding).

    The mask is derived from the *raw* (un-normalized) values, since zero-pad slots are
    exactly ``0`` in pixel space and would map to ``-1`` after normalization.
    """

    # Native camera pixel dimensions used during OBB projection (RoboTwin head_camera default).
    image_width: float = 640.0
    image_height: float = 480.0
    # Off-image clamp tolerance from the dino-wam recipe.
    clamp_margin: float = 0.05

    # Layout of the flat OBB tensor.
    max_objects: int = 2
    per_object_dim: int = 16  # 8 corners × (u, v)

    # Where to read the raw-pixel OBB trajectory from.
    src_key: str = "future_obb_pixels"
    # Where to read the per-frame "is padding" boolean (LeRobot clamps OOB delta_timestamps to
    # the last valid frame; without this, end-of-episode windows train on stale targets).
    pad_key: str = "future_obb_is_pad"
    # Where to write the normalized OBB and the validity mask.
    dst_obb_key: str = "future_obb_gt"
    dst_mask_key: str = "valid_obj_mask"

    def __call__(self, data: dict) -> dict:
        if self.src_key not in data:
            return data
        obb_px = np.asarray(data.pop(self.src_key), dtype=np.float32)
        if obb_px.ndim != 2:
            raise ValueError(f"{self.src_key} must be 2-D [H, D]; got shape {obb_px.shape}")
        h, total = obb_px.shape
        per_obj = self.per_object_dim
        if total != self.max_objects * per_obj:
            raise ValueError(
                f"{self.src_key} last dim {total} != max_objects ({self.max_objects}) * "
                f"per_object_dim ({per_obj})"
            )

        # Validity mask from the raw pixel slot (computed BEFORE normalization).
        per_slot_max_abs = np.abs(obb_px.reshape(h, self.max_objects, per_obj)).max(axis=-1)
        valid_obj_mask = (per_slot_max_abs > 1e-6).astype(np.float32)

        # Zero out frames that LeRobot padded (end-of-episode windows where the future
        # trajectory walks off the end and same-value clamping would otherwise look "valid").
        if self.pad_key in data:
            is_pad = np.asarray(data.pop(self.pad_key), dtype=bool)  # [H]
            valid_obj_mask = valid_obj_mask * (~is_pad)[:, None].astype(np.float32)

        # Per-axis pixel→[-1, 1] normalization (head camera is non-square 480×640).
        obb_xy = obb_px.reshape(h, self.max_objects, per_obj // 2, 2)  # [H, M, 8, 2]
        x = obb_xy[..., 0] / float(self.image_width)
        y = obb_xy[..., 1] / float(self.image_height)
        x = np.clip(x, -self.clamp_margin, 1.0 + self.clamp_margin) * 2.0 - 1.0
        y = np.clip(y, -self.clamp_margin, 1.0 + self.clamp_margin) * 2.0 - 1.0
        obb_norm = np.stack([x, y], axis=-1).reshape(h, total).astype(np.float32)

        # Slots that are zero-padded should also be zero in the normalized tensor (not -1),
        # so the model never trains on a fake "centred-at-image-corner" target. The loss
        # mask handles correctness, but zeroing keeps things tidy.
        obb_norm = obb_norm * np.repeat(valid_obj_mask, per_obj, axis=-1)

        return {**data, self.dst_obb_key: obb_norm, self.dst_mask_key: valid_obj_mask}


@dataclasses.dataclass(frozen=True)
class RobotwinObbOutputs(transforms.DataTransformFn):
    """Inverse of :class:`RobotwinObbInputs` for inference.

    Reads the model's predicted OBB trajectory at ``data[src_key]`` (shape
    ``[H, max_objects, per_object_dim]`` in roughly ``[-1, 1]``, as returned by
    ``pi0.sample_actions`` when ``predict_obb=True``) and writes image-fractional
    corners at ``data[dst_key]`` (shape ``[H, max_objects, 8, 2]``) so drawing code
    can simply multiply by the runtime frame's ``(W, H)``.

    Train/native pixel-space mismatch
    ---------------------------------
    ``RobotwinObbInputs`` divides x by ``image_width`` and y by ``image_height``,
    so the model's output (after undoing the ``2x - 1`` shift) is exactly
    ``pixel / (image_width, image_height)`` in TRAIN-norm space.

    If the OBB pixel coords were collected against a smaller native image (e.g. the
    LeRobot dataset's ``observation.obb_head`` was projected with a D435 head camera
    at ``320×240`` while training divided by ``(640, 480)``), the predicted x01
    only spans ``[0, native_w / image_width]`` — i.e. only the upper-left fraction
    of the runtime frame. To recover image-fractional coords on the runtime frame,
    scale by ``(image_width / native_image_width, image_height / native_image_height)``.
    Set ``scale_x`` / ``scale_y`` accordingly (default ``1.0`` = no rescale, which is
    correct when train-norm matches the native projection resolution).
    """

    max_objects: int = 2
    per_object_dim: int = 16

    # Per-axis post-output scale; compensates for any train-norm vs native-pixel mismatch.
    # Use ``image_width_train / native_image_width`` (similarly for y).
    scale_x: float = 1.0
    scale_y: float = 1.0

    src_key: str = "future_obb"
    dst_key: str = "future_obb_pixels"

    def __call__(self, data: dict) -> dict:
        if self.src_key not in data:
            return data
        obb = np.asarray(data[self.src_key], dtype=np.float32)
        # Accept both flat ``[H, max_objects * per_object_dim]`` and split
        # ``[H, max_objects, per_object_dim]`` shapes.
        if obb.ndim == 2:
            h, total = obb.shape
            if total != self.max_objects * self.per_object_dim:
                raise ValueError(
                    f"{self.src_key} last dim {total} != max_objects ({self.max_objects}) * "
                    f"per_object_dim ({self.per_object_dim})"
                )
            obb = obb.reshape(h, self.max_objects, self.per_object_dim)
        elif obb.ndim != 3:
            raise ValueError(f"{self.src_key} must be 2-D or 3-D; got shape {obb.shape}")

        h, mo, per_obj = obb.shape
        # Undo ``2x - 1`` to get fractional per axis, then rescale to compensate for any
        # train-norm vs native-pixel-space mismatch.
        corners = obb.reshape(h, mo, per_obj // 2, 2)
        corners = (corners + 1.0) * 0.5
        if self.scale_x != 1.0 or self.scale_y != 1.0:
            corners = corners * np.array([self.scale_x, self.scale_y], dtype=np.float32)
        return {**data, self.dst_key: corners.astype(np.float32)}

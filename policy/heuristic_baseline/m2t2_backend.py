"""Concrete M2T2 backend for the deployment-neutral simple-grasp package."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from .errors import NoObjectQueryFailure, NoVisibleTargetFailure


class RoboTwinM2T2Backend:
    """Load M2T2 and retain grasps whose contacts lie on the GT target cloud."""

    _RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    @staticmethod
    def _empty_trace() -> dict[str, np.ndarray]:
        """Return an empty, shape-stable trace for grasp visualization."""
        return {
            "poses": np.empty((0, 4, 4), dtype=np.float64),
            "scores": np.empty(0, dtype=np.float64),
            "contacts": np.empty((0, 3), dtype=np.float64),
            "target_contacts": np.empty(0, dtype=bool),
            "query_ids": np.empty((0, 2), dtype=np.int64),
        }

    def __init__(
        self,
        *,
        simple_grasp_root: str | Path,
        checkpoint: str | Path,
        config: str | Path,
        device: str = "cuda:0",
        num_points: int = 16_384,
        num_object_points: int = 1_024,
        num_runs: int = 1,
        mask_threshold: float = 0.4,
        object_threshold: float = 0.4,
        max_predictions: int | None = 512,
        workspace_bounds: tuple[float, float, float, float, float, float] | None = None,
        contact_match_distance_m: float = 1e-5,
        seed: int = 0,
    ) -> None:
        self.root = Path(simple_grasp_root).expanduser().resolve()
        self.checkpoint = self._resolve(checkpoint)
        self.config_path = self._resolve(config)
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"M2T2 checkpoint not found: {self.checkpoint}")
        if not self.config_path.is_file():
            raise FileNotFoundError(f"M2T2 config not found: {self.config_path}")
        if num_points <= 0 or num_object_points <= 0 or num_runs <= 0:
            raise ValueError("M2T2 point counts and num_runs must be positive")

        m2t2_root = self.root / "third_party" / "M2T2"
        if str(m2t2_root) not in sys.path:
            sys.path.insert(0, str(m2t2_root))
        try:
            import torch
            from m2t2.m2t2 import M2T2
            from omegaconf import OmegaConf
        except ImportError as exc:
            raise RuntimeError(
                "M2T2 is not installed in the active RoboTwin environment; "
                "run policy/heuristic_baseline/install.sh"
            ) from exc

        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"requested {device}, but CUDA is unavailable")

        cfg = OmegaConf.load(self.config_path)
        cfg.eval.world_coord = True
        cfg.eval.mask_thresh = float(mask_threshold)
        cfg.eval.object_thresh = float(object_threshold)
        if max_predictions is not None:
            cfg.m2t2.action_decoder.max_num_pred = int(max_predictions)
        self.cfg = cfg

        model = M2T2.from_config(cfg.m2t2)
        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(payload.get("model", payload))
        self.model = model.to(self.device).eval()

        self.num_points = int(num_points)
        self.num_object_points = int(num_object_points)
        self.num_runs = int(num_runs)
        self.contact_match_distance_m = float(contact_match_distance_m)
        if self.contact_match_distance_m <= 0.0:
            raise ValueError("contact_match_distance_m must be positive")
        self.workspace_bounds = (
            None
            if workspace_bounds is None
            else np.asarray(workspace_bounds, dtype=np.float32)
        )
        if self.workspace_bounds is not None and self.workspace_bounds.shape != (6,):
            raise ValueError("workspace_bounds must contain [xmin,ymin,zmin,xmax,ymax,zmax]")
        self._seed = int(seed)
        self.reset()

    def _resolve(self, path: str | Path) -> Path:
        path = Path(path).expanduser()
        return path if path.is_absolute() else self.root / path

    def reset(self, seed: int | None = None) -> None:
        """Reset private sampling state without changing NumPy's global RNG."""
        if seed is not None:
            self._seed = int(seed)
        self.rng = np.random.default_rng(self._seed)
        self.last_trace = self._empty_trace()

    def _sample_indices(self, target_membership: np.ndarray) -> np.ndarray:
        """Sample exactly half target points and half non-target context."""
        target_membership = np.asarray(target_membership, dtype=bool)
        total = target_membership.shape[0]
        if total == 0:
            raise ValueError("cannot run M2T2 on an empty point cloud")
        target = np.flatnonzero(target_membership)
        if target.size == 0:
            raise NoVisibleTargetFailure("segmented target has no visible depth points")

        target_count = (self.num_points + 1) // 2
        target_idx = self.rng.choice(
            target, target_count, replace=target.size < target_count
        )
        context_count = self.num_points - target_count
        context = np.flatnonzero(~target_membership)
        if context.size == 0:
            context = target
        context_idx = self.rng.choice(
            context, context_count, replace=context.size < context_count
        )
        indices = np.concatenate((target_idx, context_idx))
        self.rng.shuffle(indices)
        return indices

    def _input_batch(
        self,
        xyz: np.ndarray,
        rgb: np.ndarray,
        indices: np.ndarray,
        sampled_target_membership: np.ndarray,
    ) -> dict[str, Any]:
        torch = self.torch
        points = np.asarray(xyz[indices], dtype=np.float32)
        colors = np.asarray(rgb[indices], dtype=np.float32)
        sampled_target_membership = np.asarray(
            sampled_target_membership, dtype=bool
        )
        if sampled_target_membership.shape != (len(points),):
            raise ValueError("sampled target mask must align with M2T2 points")
        colors = (colors - self._RGB_MEAN) / self._RGB_STD
        inputs = np.concatenate((points - points.mean(axis=0), colors), axis=1)

        def tensor(value: np.ndarray) -> Any:
            return torch.from_numpy(value).unsqueeze(0).to(self.device)

        return {
            "inputs": tensor(inputs),
            "points": tensor(points),
            "grasp_target_mask": tensor(sampled_target_membership),
            "object_inputs": torch.zeros(
                (1, self.num_object_points, 6), device=self.device
            ),
            "bottom_center": torch.zeros((1, 3), device=self.device),
            "cam_pose": torch.eye(4, device=self.device).unsqueeze(0),
            "ee_pose": torch.eye(4, device=self.device).unsqueeze(0),
            "task_is_pick": torch.ones((1,), dtype=torch.bool, device=self.device),
            "task_is_place": torch.zeros((1,), dtype=torch.bool, device=self.device),
        }

    @staticmethod
    def _target_conditioned_confidence(
        grasp_logits: Any,
        target_mask: Any,
        mask_threshold: float,
    ) -> Any:
        """Pool anonymous queries and hard-gate contacts to the GT target."""
        if getattr(grasp_logits, "ndim", 0) != 3:
            raise ValueError(
                "M2T2 grasp logits must have shape [batch, queries, points]"
            )
        if getattr(target_mask, "ndim", 0) != 2:
            raise ValueError("M2T2 target mask must have shape [batch, points]")
        if (
            grasp_logits.shape[0] != target_mask.shape[0]
            or grasp_logits.shape[2] != target_mask.shape[1]
        ):
            raise ValueError("M2T2 target mask must align with grasp logits")
        if grasp_logits.shape[1] == 0:
            raise ValueError("M2T2 returned no anonymous grasp queries")
        target_mask = target_mask.bool()
        if not bool(target_mask.any(dim=1).all()):
            raise NoVisibleTargetFailure(
                "sampled M2T2 target mask contains no target points"
            )

        pooled = grasp_logits.sigmoid().amax(dim=1, keepdim=True)
        confidence_floor = pooled.new_tensor(float(mask_threshold) + 1e-4)
        return pooled.maximum(confidence_floor).where(
            target_mask.unsqueeze(1), pooled.new_zeros(())
        )

    def _target_conditioned_infer(
        self, batch: dict[str, Any]
    ) -> dict[str, Any]:
        """Run M2T2 once while allowing grasp contacts only on the GT target."""
        model = self.model
        scene_features = model.backbone(batch["inputs"])
        object_features = (
            {}
            if model.object_encoder is None
            else model.object_encoder(batch["object_inputs"])
        )
        if "task_is_place" in batch:
            for key in object_features.get("features", {}):
                object_features["features"][key] = (
                    object_features["features"][key]
                    * batch["task_is_place"].view(
                        batch["task_is_place"].shape[0], 1, 1
                    )
                )
        embedding, decoder_outputs = model.transformer(
            scene_features, object_features, batch.get("lang_tokens")
        )
        outputs = decoder_outputs[-1]
        if "grasp" not in embedding or embedding["grasp"].shape[1] == 0:
            raise NoObjectQueryFailure("M2T2 returned no grasp embeddings")
        if model.grasp_mlp.use_embed:
            raise RuntimeError(
                "target-conditioned M2T2 requires action_decoder.use_embed=false"
            )

        confidence = self._target_conditioned_confidence(
            outputs["grasping_masks"],
            batch["grasp_target_mask"],
            float(self.cfg.eval.mask_thresh),
        )
        mask_features = scene_features["features"][
            model.transformer.mask_feature
        ]
        grasp_outputs = model.grasp_mlp(
            batch["points"],
            mask_features,
            list(confidence),
            float(self.cfg.eval.mask_thresh),
            [item[:1] for item in embedding["grasp"]],
        )
        outputs.update(grasp_outputs)
        outputs["grasping_masks"] = [
            mask.unsqueeze(0) for mask in batch["grasp_target_mask"].bool()
        ]
        return outputs

    @staticmethod
    def _validate_query_alignment(
        masks: Any,
        query_poses: Any,
        query_scores: Any,
        query_contacts: Any,
    ) -> int:
        """Validate M2T2's query-aligned output contract before indexing it."""
        if getattr(masks, "ndim", 0) != 2:
            raise ValueError("M2T2 grasping_masks must have shape [queries, points]")
        query_count = int(masks.shape[0])
        collections = {
            "grasps": query_poses,
            "grasp_confidence": query_scores,
            "grasp_contacts": query_contacts,
        }
        for name, collection in collections.items():
            try:
                collection_len = len(collection)
            except TypeError as exc:
                raise ValueError(f"M2T2 {name} must be query-indexable") from exc
            if collection_len != query_count:
                raise ValueError(
                    f"M2T2 {name} has {collection_len} queries; expected {query_count}"
                )
        for query_idx in range(query_count):
            lengths = {
                "poses": len(query_poses[query_idx]),
                "scores": len(query_scores[query_idx]),
                "contacts": len(query_contacts[query_idx]),
            }
            if len(set(lengths.values())) != 1:
                raise ValueError(
                    "M2T2 per-query pose/score/contact lengths differ for "
                    f"query {query_idx}: {lengths}"
                )
        return query_count

    @staticmethod
    def _target_contact_membership(
        contacts: np.ndarray,
        target_points: np.ndarray,
        tolerance_m: float,
    ) -> np.ndarray:
        """Keep contacts that match a ground-truth target point within tolerance."""
        contacts = np.asarray(contacts, dtype=np.float64)
        target_points = np.asarray(target_points, dtype=np.float64)
        if contacts.ndim != 2 or contacts.shape[1:] != (3,):
            raise ValueError("M2T2 grasp contacts must have shape [grasps, 3]")
        if target_points.ndim != 2 or target_points.shape[1:] != (3,):
            raise ValueError("target_points must have shape [points, 3]")
        if len(target_points) == 0:
            raise NoVisibleTargetFailure("segmented target contains no visible depth points")

        keep = np.zeros(len(contacts), dtype=bool)
        tolerance_sq = float(tolerance_m) ** 2
        for start in range(0, len(contacts), 256):
            chunk = contacts[start : start + 256]
            distance_sq = np.sum(
                (chunk[:, None, :] - target_points[None, :, :]) ** 2, axis=2
            )
            keep[start : start + len(chunk)] = np.any(distance_sq <= tolerance_sq, axis=1)
        return keep

    @staticmethod
    def _rigid_grasp_pose(
        pose: np.ndarray, *, max_projection_error: float = 1e-3
    ) -> np.ndarray | None:
        """Project small float drift to SO(3), rejecting malformed frames."""
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            return None
        if not np.allclose(
            pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8, rtol=0.0
        ):
            return None

        rotation = pose[:3, :3]
        try:
            u, _, vh = np.linalg.svd(rotation)
        except np.linalg.LinAlgError:
            return None
        handedness = np.eye(3)
        handedness[-1, -1] = np.sign(np.linalg.det(u @ vh))
        projected = u @ handedness @ vh
        if (
            not np.all(np.isfinite(projected))
            or np.linalg.norm(rotation - projected, ord="fro")
            > max_projection_error
        ):
            return None

        result = pose.copy()
        result[:3, :3] = projected
        result[3] = [0.0, 0.0, 0.0, 1.0]
        return result

    def predict(
        self,
        xyz: np.ndarray,
        rgb: np.ndarray,
        target_pose: np.ndarray,
        target_points: np.ndarray,
        target_membership: np.ndarray,
    ) -> tuple[list[np.ndarray], list[float]]:
        poses: list[np.ndarray] = []
        scores: list[float] = []
        trace_poses: list[np.ndarray] = []
        trace_scores: list[float] = []
        trace_contacts: list[np.ndarray] = []
        trace_target_contacts: list[bool] = []
        trace_query_ids: list[tuple[int, int]] = []
        self.last_trace = self._empty_trace()
        rejected_rotations = 0
        queries_with_target_contacts = 0
        xyz = np.asarray(xyz, dtype=np.float32)
        rgb = np.asarray(rgb, dtype=np.float32)
        target_pose = np.asarray(target_pose, dtype=np.float64)
        target_points = np.asarray(target_points, dtype=np.float32)
        target_membership = np.asarray(target_membership, dtype=bool)
        if target_pose.shape != (4, 4):
            raise ValueError("target_pose must have shape (4, 4)")
        if target_membership.shape != (len(xyz),):
            raise ValueError("target_membership must align with xyz")
        if target_points.ndim != 2 or target_points.shape[1:] != (3,):
            raise ValueError("target_points must have shape (N, 3)")
        if self.workspace_bounds is not None:
            lower, upper = self.workspace_bounds[:3], self.workspace_bounds[3:]
            within = np.all((xyz >= lower) & (xyz <= upper), axis=1)
            xyz, rgb, target_membership = (
                xyz[within], rgb[within], target_membership[within]
            )
        if not np.any(target_membership):
            raise NoVisibleTargetFailure("segmented target has no visible depth points")

        for run_index in range(self.num_runs):
            indices = self._sample_indices(target_membership)
            batch = self._input_batch(
                xyz, rgb, indices, target_membership[indices]
            )
            torch_seed = (self._seed + run_index) % (2**63 - 1)
            cuda_devices: list[int] = []
            if self.device.type == "cuda":
                device_index = self.device.index
                if device_index is None:
                    device_index = self.torch.cuda.current_device()
                cuda_devices.append(device_index)
            with self.torch.random.fork_rng(devices=cuda_devices):
                self.torch.random.default_generator.manual_seed(torch_seed)
                if self.device.type == "cuda":
                    with self.torch.cuda.device(self.device):
                        self.torch.cuda.manual_seed(torch_seed)
                with self.torch.inference_mode():
                    output = self._target_conditioned_infer(batch)

            masks = output["grasping_masks"][0]
            grasps = output["grasps"][0]
            confidences = output["grasp_confidence"][0]
            contacts = output["grasp_contacts"][0]
            query_count = self._validate_query_alignment(
                masks, grasps, confidences, contacts
            )
            # Target membership is imposed before the action decoder. This
            # geometric check is now an invariant, not target selection.
            for query_idx in range(query_count):
                query_poses = grasps[query_idx].detach().cpu().numpy()
                query_scores = confidences[query_idx].detach().cpu().numpy()
                query_contacts = contacts[query_idx].detach().cpu().numpy()
                if len(query_contacts) == 0:
                    continue
                keep = self._target_contact_membership(
                    query_contacts,
                    target_points,
                    self.contact_match_distance_m,
                )
                if not np.all(keep):
                    raise RuntimeError(
                        "target-conditioned M2T2 emitted a non-target contact"
                    )
                if np.any(keep):
                    queries_with_target_contacts += 1
                for pose, contact, score, is_target_contact in zip(
                    query_poses, query_contacts, query_scores, keep
                ):
                    pose = self._rigid_grasp_pose(pose)
                    if pose is None:
                        if is_target_contact:
                            rejected_rotations += 1
                        continue

                    # build_6d_grasp places the wrist at contact - depth * approach
                    # plus a learned half-width along the finger-closing axis.
                    # Repair rotation before clamping that learned lateral offset.
                    contact = np.asarray(contact, dtype=np.float64)
                    anchor = contact - 0.1034 * pose[:3, 2]
                    lateral = pose[:3, 3] - anchor
                    lateral_norm = np.linalg.norm(lateral)
                    max_half_width = 0.045
                    scale = min(
                        1.0, max_half_width / max(lateral_norm, 1e-12)
                    )
                    pose[:3, 3] = anchor + lateral * scale
                    trace_poses.append(pose.copy())
                    trace_scores.append(float(score))
                    trace_contacts.append(contact.copy())
                    trace_target_contacts.append(bool(is_target_contact))
                    trace_query_ids.append((run_index, query_idx))
                    if is_target_contact:
                        poses.append(pose)
                        scores.append(float(score))

        self.last_trace = {
            "poses": np.asarray(trace_poses, dtype=np.float64).reshape(-1, 4, 4),
            "scores": np.asarray(trace_scores, dtype=np.float64),
            "contacts": np.asarray(trace_contacts, dtype=np.float64).reshape(-1, 3),
            "target_contacts": np.asarray(trace_target_contacts, dtype=bool),
            "query_ids": np.asarray(trace_query_ids, dtype=np.int64).reshape(-1, 2),
        }

        if not poses:
            raise NoObjectQueryFailure(
                f"M2T2 target-conditioned decoding produced no valid grasp "
                f"poses after {self.num_runs} runs "
                f"(rejected_rotations={rejected_rotations})"
            )

        score_range = f"{min(scores):.3f}..{max(scores):.3f}"
        visible_target_points = xyz[target_membership]
        target_center = visible_target_points.mean(axis=0)
        distances = [np.linalg.norm(pose[:3, 3] - target_center) for pose in poses]
        distance_range = f"{min(distances):.3f}..{max(distances):.3f}m"
        print(
            f"[heuristic] M2T2 segmented-target candidates={len(poses)} "
            f"target_queries={queries_with_target_contacts} "
            f"confidence={score_range} target_distance={distance_range} "
            f"rejected_rotations={rejected_rotations} "
            f"target_center={np.array2string(target_center, precision=3)}"
        )
        return poses, scores

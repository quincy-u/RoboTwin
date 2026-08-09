"""Concrete M2T2 backend for the deployment-neutral simple-grasp package."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import NoObjectQueryFailure, NoVisibleTargetFailure


@dataclass(frozen=True)
class QueryMatch:
    """Association metrics for one M2T2 object query."""

    query_idx: int
    iou: float
    intersection: int
    purity: float

class RoboTwinM2T2Backend:
    """Load M2T2 once and associate object queries with a target rigid-frame crop."""

    _RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

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
        min_query_iou: float = 0.01,
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
        self.min_query_iou = float(min_query_iou)
        if not 0.0 <= self.min_query_iou <= 1.0:
            raise ValueError("min_query_iou must be in [0, 1]")
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



    def _sample_indices(self, target_membership: np.ndarray) -> np.ndarray:
        """Sample a scene while reserving segmented target points."""
        target_membership = np.asarray(target_membership, dtype=bool)
        total = target_membership.shape[0]
        if total == 0:
            raise ValueError("cannot run M2T2 on an empty point cloud")
        target = np.flatnonzero(target_membership)
        if target.size == 0:
            raise NoVisibleTargetFailure("segmented target has no visible depth points")

        target_count = min(target.size, max(1, self.num_points // 4))
        target_idx = self.rng.choice(target, target_count, replace=False)
        scene_count = self.num_points - target_count
        scene_idx = self.rng.choice(total, scene_count, replace=total < scene_count)
        indices = np.concatenate((target_idx, scene_idx))
        self.rng.shuffle(indices)
        return indices

    def _input_batch(
        self,
        xyz: np.ndarray,
        rgb: np.ndarray,
        indices: np.ndarray,
    ) -> dict[str, Any]:
        torch = self.torch
        points = np.asarray(xyz[indices], dtype=np.float32)
        colors = np.asarray(rgb[indices], dtype=np.float32)
        colors = (colors - self._RGB_MEAN) / self._RGB_STD
        inputs = np.concatenate((points - points.mean(axis=0), colors), axis=1)

        def tensor(value: np.ndarray) -> Any:
            return torch.from_numpy(value).unsqueeze(0).to(self.device)

        return {
            "inputs": tensor(inputs),
            "points": tensor(points),
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
    def _matching_query(masks: Any, target_mask: Any) -> QueryMatch:
        """Return the best query and diagnostics; thresholding is caller policy."""
        if getattr(masks, "ndim", 0) != 2:
            raise ValueError("M2T2 grasping_masks must have shape [queries, points]")
        if masks.shape[0] == 0:
            raise NoObjectQueryFailure("M2T2 found no object queries")
        if target_mask.ndim != 1 or target_mask.shape[0] != masks.shape[1]:
            raise ValueError("target mask must align with M2T2 mask points")
        target_mask = target_mask.to(device=masks.device, dtype=masks.dtype)
        predicted = masks.bool()
        intersection = (predicted & target_mask.unsqueeze(0)).sum(dim=1).float()
        union = (predicted | target_mask.unsqueeze(0)).sum(dim=1).clamp_min(1)
        iou = intersection / union
        query = int(iou.argmax().item())
        purity = intersection / predicted.sum(dim=1).clamp_min(1)
        return QueryMatch(
            query_idx=query,
            iou=float(iou[query].item()),
            intersection=int(intersection[query].item()),
            purity=float(purity[query].item()),
        )

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
    def _sampled_target_contact_membership(
        contacts: np.ndarray,
        sampled_target_points: np.ndarray,
        tolerance_m: float,
    ) -> np.ndarray:
        """Keep contacts that match an exact sampled target point within tolerance."""
        contacts = np.asarray(contacts, dtype=np.float64)
        target_points = np.asarray(sampled_target_points, dtype=np.float64)
        if contacts.ndim != 2 or contacts.shape[1:] != (3,):
            raise ValueError("M2T2 grasp contacts must have shape [grasps, 3]")
        if target_points.ndim != 2 or target_points.shape[1:] != (3,):
            raise ValueError("sampled target points must have shape [points, 3]")
        if len(target_points) == 0:
            raise NoVisibleTargetFailure("sampled target contains no visible depth points")

        keep = np.zeros(len(contacts), dtype=bool)
        tolerance_sq = float(tolerance_m) ** 2
        for start in range(0, len(contacts), 256):
            chunk = contacts[start : start + 256]
            distance_sq = np.sum(
                (chunk[:, None, :] - target_points[None, :, :]) ** 2, axis=2
            )
            keep[start : start + len(chunk)] = np.any(distance_sq <= tolerance_sq, axis=1)
        return keep

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
        query_matches: list[QueryMatch] = []
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
            batch = self._input_batch(xyz, rgb, indices)
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
                    output = self.model.infer(batch, self.cfg.eval)

            masks = output["grasping_masks"][0]
            grasps = output["grasps"][0]
            confidences = output["grasp_confidence"][0]
            contacts = output["grasp_contacts"][0]
            self._validate_query_alignment(masks, grasps, confidences, contacts)
            target_mask = self.torch.from_numpy(target_membership[indices])
            try:
                match = self._matching_query(masks, target_mask)
            except NoObjectQueryFailure:
                continue
            query_matches.append(match)
            if match.iou < self.min_query_iou:
                continue

            # M2T2 outputs are query-aligned: never pool or fall back to another query.
            query_poses = grasps[match.query_idx].detach().cpu().numpy()
            query_scores = confidences[match.query_idx].detach().cpu().numpy()
            query_contacts = contacts[match.query_idx].detach().cpu().numpy()
            if len(query_contacts) == 0:
                continue
            keep = self._sampled_target_contact_membership(
                query_contacts,
                target_points,
                self.contact_match_distance_m,
            )
            selected_poses = np.asarray(query_poses[keep], dtype=np.float64).copy()
            selected_contacts = np.asarray(query_contacts[keep], dtype=np.float64)
            # build_6d_grasp places the wrist at contact - depth * approach
            # plus a learned half-width along the finger-closing axis.  Clamp
            # that learned offset to the physical gripper half-width so an
            # out-of-distribution prediction cannot move a valid contact far
            # outside the robot workspace.
            anchor = selected_contacts - 0.1034 * selected_poses[:, :3, 2]
            lateral = selected_poses[:, :3, 3] - anchor
            lateral_norm = np.linalg.norm(lateral, axis=1)
            max_half_width = 0.045
            scale = np.minimum(1.0, max_half_width / np.maximum(lateral_norm, 1e-12))
            selected_poses[:, :3, 3] = anchor + lateral * scale[:, None]
            poses.extend(selected_poses)
            scores.extend(np.asarray(query_scores[keep], dtype=np.float64).tolist())

        if not poses:
            best_iou = max((match.iou for match in query_matches), default=0.0)
            raise NoObjectQueryFailure(
                f"M2T2 found no target candidates for the target-pose crop after "
                f"{self.num_runs} runs (best query IoU={best_iou:.3f}, "
                f"minimum={self.min_query_iou:.3f})"
            )

        score_range = f"{min(scores):.3f}..{max(scores):.3f}"
        visible_target_points = xyz[target_membership]
        target_center = visible_target_points.mean(axis=0)
        distances = [np.linalg.norm(pose[:3, 3] - target_center) for pose in poses]
        distance_range = f"{min(distances):.3f}..{max(distances):.3f}m"
        print(
            f"[heuristic] M2T2 segmented-target candidates={len(poses)} "
            f"query_iou={max((m.iou for m in query_matches), default=0.0):.3f} "
            f"intersection={max((m.intersection for m in query_matches), default=0)} "
            f"purity={max((m.purity for m in query_matches), default=0.0):.3f} "
            f"confidence={score_range} target_distance={distance_range} "
            f"target_center={np.array2string(target_center, precision=3)}"
        )
        return poses, scores

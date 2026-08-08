# Debug Notes

- Run the policy from the RoboTwin root environment; M2T2 needs its CUDA-enabled
  Torch plus the PointNet++ extension.
- `SIMPLE_GRASP_ROOT` must point to a checkout containing `src/simple_grasp` and
  `third_party/M2T2`.
- RoboTwin depth is millimeters. Convert it to meters before building the point
  cloud.
- Prefer the SAPIEN `Position` buffer with `cam2world_gl`; manually mixing CV and
  GL camera conventions can reflect the cloud.
- CuRobo 0.7.8 with Warp 1.16 may need the local `wp.torch` compatibility shim.
- CuRobo returns arm-only path columns, but `last_qpos` must be a full
  articulation vector. Inject path endpoints using active joint names.
- Preserve the native float32 articulation qpos when chaining planner seeds;
  converting it to float64 makes CuRobo fail with a scalar-type mismatch.
- M2T2 grasping masks are sparse contact masks, not full object segmentation;
  a small GT-mask IoU such as `0.01` is intentional.
- With `CUDA_VISIBLE_DEVICES=6`, the process still addresses that GPU as
  `cuda:0`.
- A CPU-only SAPIEN physics scene must be created explicitly with
  `sapien.Scene([sapien.physx.PhysxCpuSystem()])`; the default scene adds a
  render system. Local llvmpipe cannot satisfy SAPIEN rendering extensions, so
  it cannot produce the RGB-D observation required by this policy.

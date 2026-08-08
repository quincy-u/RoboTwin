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
- With `CUDA_VISIBLE_DEVICES=6`, the process still addresses that GPU as
  `cuda:0`.

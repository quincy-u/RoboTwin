from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from policy.heuristic_baseline._simple_grasp import ensure_simple_grasp_importable

ensure_simple_grasp_importable()

from simple_grasp.policy import PolicyConfig
from simple_grasp.types import GraspCandidate, ObjectState, SceneObservation

from policy.heuristic_baseline.errors import NoFeasiblePlanFailure, TargetSelectionFailure
from policy.heuristic_baseline.runtime import (
    M2T2_TO_ROBOTWIN,
    QposActionBuffer,
    ReachabilityRankedGrasps,
    RoboTwinHeuristicRuntime,
    RoboTwinMinkIK,
)

I = np.eye(4)


class Joint:
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name


class Entity:
    def __init__(self, qpos, joint_names):
        self.qpos = np.asarray(qpos, dtype=np.float32)
        self.joints = [Joint(name) for name in joint_names]

    def get_qpos(self):
        return self.qpos.copy()

    def get_active_joints(self):
        return self.joints


class Robot:
    def __init__(self, results):
        self.results = list(results)
        self.seeds = []
        self.right_arm_joints_name = ["right_0", "right_1", "right_2"]
        self.left_arm_joints_name = ["left_0", "left_1", "left_2"]
        self.right_entity = Entity(
            [90.0, 91.0, 92.0, 93.0],
            ["right_0", "right_gripper", "right_1", "right_2"],
        )
        self.left_entity = Entity(
            [80.0, 81.0, 82.0, 83.0],
            ["left_0", "left_gripper", "left_1", "left_2"],
        )
        self.right_entity_origion_pose = SimpleNamespace(p=np.array([0.0, 0.0, 0.0]))
        self.left_entity_origion_pose = SimpleNamespace(p=np.array([0.0, 0.0, 0.0]))

    def get_right_arm_jointState(self):
        return [4.0, 5.0, 6.0, 1.0]

    def get_left_arm_jointState(self):
        return [1.0, 2.0, 3.0, 1.0]

    def right_plan_path(self, pose, *, last_qpos=None, constraint_pose=None):
        self.seeds.append(np.asarray(last_qpos).copy())
        return self.results.pop(0)


class Env:
    def __init__(self, results):
        self.robot = Robot(results)
        self.episode_seed = 17


def success_path(start: float, length: int = 2):
    path = np.stack(
        [np.arange(3, dtype=np.float64) + start + index * 10 for index in range(length)]
    )
    return {"status": "Success", "position": path}


class Grasps:
    def __init__(self, candidates):
        self.candidates = candidates
        self.backend = SimpleNamespace(reset=lambda seed: None)

    def propose(self, observation, target):
        return self.candidates


def scene(*, objects):
    return SceneObservation(
        np.zeros((1, 3)), np.zeros((1, 3)), np.array([7]), I, objects
    )


@unittest.skip("superseded CuRobo planner contract")
class PlannerPathTest(unittest.TestCase):
    def test_chains_full_articulation_seed_and_resets_for_next_candidate(self):
        first = {"status": "Failure"}
        second, third, fourth = success_path(10), success_path(30), success_path(50)
        env = Env([first, second, third, fourth])
        ik = RoboTwinPlannerIK(env, I, relax_orientation_on_failure=False)

        self.assertIsNone(ik.solve("right", I))
        self.assertIsNone(ik.solve("right", I))
        self.assertIsNone(ik.solve("right", I))
        pregrasp = ik.solve("right", I)
        grasp = ik.solve("right", I)
        retreat = ik.solve("right", I)

        np.testing.assert_allclose(env.robot.seeds[0], [90, 91, 92, 93])
        np.testing.assert_allclose(env.robot.seeds[1], [90, 91, 92, 93])
        np.testing.assert_allclose(env.robot.seeds[2], [20, 91, 21, 22])
        np.testing.assert_allclose(env.robot.seeds[3], [40, 91, 41, 42])
        self.assertEqual(len(env.robot.seeds), 4)
        self.assertTrue(all(seed.dtype == np.float32 for seed in env.robot.seeds))
        np.testing.assert_allclose(pregrasp, second["position"][-1, :3])
        np.testing.assert_allclose(grasp, third["position"][-1, :3])
        np.testing.assert_allclose(retreat, fourth["position"][-1, :3])
        np.testing.assert_allclose(ik.consume_path("right", pregrasp), second["position"])
        np.testing.assert_allclose(ik.consume_path("right", grasp), third["position"])
        np.testing.assert_allclose(ik.consume_path("right", retreat), fourth["position"])

    def test_retries_failed_exact_pose_with_relaxed_orientation(self):
        relaxed = success_path(10)
        env = Env([{"status": "Fail"}, relaxed])
        poses = []
        constraints = []
        original = env.robot.right_plan_path

        def record_constraint(pose, *, last_qpos=None, constraint_pose=None):
            poses.append(np.asarray(pose).copy())
            constraints.append(constraint_pose)
            return original(
                pose,
                last_qpos=last_qpos,
                constraint_pose=constraint_pose,
            )

        env.robot.right_plan_path = record_constraint
        ik = RoboTwinPlannerIK(env, I)

        result = ik.solve("right", I)

        np.testing.assert_allclose(result, relaxed["position"][-1])
        self.assertEqual(constraints, [None, None])
        np.testing.assert_allclose(poses[1][3:], [-0.353523, 0.61239, -0.353524, -0.61239])
        self.assertEqual(ik.calls, 1)
        self.assertEqual(ik.planner_attempts, 2)
        self.assertEqual(ik.relaxed_successes, 1)

    def test_orientation_relaxation_can_be_disabled(self):
        env = Env([{"status": "Fail"}])
        ik = RoboTwinPlannerIK(env, I, relax_orientation_on_failure=False)

        self.assertIsNone(ik.solve("right", I))
        self.assertEqual(ik.planner_attempts, 1)

    def test_m2t2_wrist_pose_uses_only_robotwin_frame_calibration(self):
        pose = I.copy()
        converted = pose @ M2T2_TO_ROBOTWIN

        # M2T2 build_6d_grasp has already applied its 0.1034 m gripper depth;
        # applying it again here makes otherwise reachable poses miss by 12 cm.
        np.testing.assert_allclose(converted[:3, 3], [0.0, 0.0, -0.0166])

    def test_accepts_step_limit_fail_only_with_a_trajectory(self):
        over_limit = success_path(10)
        over_limit["status"] = "Fail"
        ik = RoboTwinPlannerIK(
            Env([over_limit]), I, relax_orientation_on_failure=False
        )

        result = ik.solve("right", I)

        np.testing.assert_allclose(result, over_limit["position"][-1])
        self.assertEqual(ik.over_limit_successes, 1)

    def test_replays_ordered_bounded_full_qpos_waypoints(self):
        paths = [success_path(10, 12), success_path(30, 12), success_path(50, 12)]
        env = Env(paths)
        ik = RoboTwinPlannerIK(env, I)
        buffer = QposActionBuffer(env, ik, max_waypoints_per_segment=3)
        buffer.reset()
        joints = [ik.solve("right", I) for _ in range(3)]

        buffer.open_gripper("right")
        buffer.move_joints("right", joints[0])
        buffer.move_joints("right", joints[1])
        buffer.close_gripper("right")
        buffer.move_joints("right", joints[2])

        self.assertLessEqual(len(buffer.actions), 2 + 3 * 3)
        self.assertEqual(len(buffer.actions), 2 + 3 * 3)
        self.assertTrue(all(action.shape == (8,) for action in buffer.actions))
        right_offsets = [4, 5, 6]
        expected = []
        for path in paths:
            expected.extend(path["position"][[0, 5, 11], :3])
        replay_actions = (
            buffer.actions[1:4] + buffer.actions[4:7] + buffer.actions[8:11]
        )
        actual = [action[right_offsets] for action in replay_actions]
        np.testing.assert_allclose(actual, expected)
        self.assertTrue(all(np.allclose(action[:3], [1, 2, 3]) for action in buffer.actions))

    def test_controller_fails_closed_without_completed_path(self):
        env = Env([])
        ik = RoboTwinPlannerIK(env, I)
        buffer = QposActionBuffer(env, ik)
        buffer.reset()

        with self.assertRaisesRegex(RuntimeError, "path cache is missing"):
            buffer.move_joints("right", np.array([7.0, 8.0, 9.0]))

    def test_planner_rejects_non_finite_waypoints(self):
        result = success_path(10)
        result["position"][0, 0] = np.nan
        ik = RoboTwinPlannerIK(Env([result]), I)

        with self.assertRaisesRegex(ValueError, "non-finite"):
            ik.solve("right", I)

    def test_raw_confidence_filter_precedes_geometric_ranking(self):
        target_pose = I.copy()
        target_pose[0, 3] = 1.0
        target = ObjectState("cube", target_pose, 7)
        low = GraspCandidate(I, 0.39, "cube")
        accepted = GraspCandidate(I, 0.4, "cube")
        ranked = ReachabilityRankedGrasps(
            Grasps([low, accepted]), Env([]), "right", min_confidence=0.4
        ).propose(scene(objects={"cube": target}), target)

        self.assertEqual(len(ranked), 1)
        self.assertGreaterEqual(ranked[0].confidence, 0.4)

    def test_runtime_raises_typed_failures_for_target_and_complete_ik(self):
        target = ObjectState("cube", I, 7)
        candidate = GraspCandidate(I, 0.9, "cube")
        runtime = RoboTwinHeuristicRuntime(
            task_env=Env([{"status": "Failure"}]),
            grasps=Grasps([candidate]),
            config=PolicyConfig(object_name="cube"),
            automatic_target=False,
            automatic_arm=False,
            grasp_to_robotwin=I,
            relax_orientation_on_failure=False,
        )
        with self.assertRaises(NoFeasiblePlanFailure):
            runtime.get_action(scene=scene(objects={"cube": target}))

        ambiguous = RoboTwinHeuristicRuntime(
            task_env=Env([]),
            grasps=Grasps([]),
            config=PolicyConfig(object_name="auto"),
            automatic_target=True,
            automatic_arm=False,
            grasp_to_robotwin=I,
        )
        with self.assertRaises(TargetSelectionFailure):
            ambiguous.get_action(scene=scene(objects={"a": target, "b": target}))


if __name__ == "__main__":
    unittest.main()

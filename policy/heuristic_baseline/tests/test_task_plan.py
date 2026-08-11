from __future__ import annotations

import unittest

import numpy as np

from policy.heuristic_baseline.task_plan import (
    Handoff,
    Pick,
    Place,
    ProceduralTaskRecorder,
    task_plan_from_trace,
)


def pose(x=0.0, y=0.0, z=0.8):
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = [x, y, z]
    return result


def local_pose(x=0.0, y=0.0, z=0.0):
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = [x, y, z]
    return result


class Pose:
    def __init__(self, matrix):
        self.matrix = np.asarray(matrix, dtype=np.float64)

    def to_transformation_matrix(self):
        return self.matrix.copy()


class Actor:
    POINTS = {
        "contact": "contact_points_pose",
        "functional": "functional_matrix",
    }

    def __init__(self, world_pose, *, functional=(), contacts=()):
        self.world_pose = np.asarray(world_pose, dtype=np.float64)
        self.config = {
            "scale": [1.0, 1.0, 1.0],
            "functional_matrix": [np.asarray(item).tolist() for item in functional],
            "contact_points_pose": [np.asarray(item).tolist() for item in contacts],
        }

    def get_pose(self):
        return Pose(self.world_pose)

    def get_functional_point(self, index, representation="matrix"):
        matrix = self.world_pose @ np.asarray(
            self.config["functional_matrix"][index], dtype=np.float64
        )
        return matrix if representation == "matrix" else Pose(matrix)

    def get_contact_point(self, index, representation="matrix"):
        matrix = self.world_pose @ np.asarray(
            self.config["contact_points_pose"][index], dtype=np.float64
        )
        return matrix if representation == "matrix" else Pose(matrix)


class Env:
    def __init__(self, tracked):
        self.tracked = tracked
        self.moves = []

    def get_tracked_objects(self):
        return self.tracked

    def grasp_actor(
        self,
        actor,
        arm_tag,
        pre_grasp_dis=0.1,
        grasp_dis=0,
        gripper_pos=0.0,
        contact_point_id=None,
    ):
        return arm_tag, ("grasp", actor)

    def place_actor(
        self,
        actor,
        arm_tag,
        target_pose,
        functional_point_id=None,
        pre_dis=0.1,
        dis=0.02,
        is_open=True,
        **args,
    ):
        return arm_tag, ("place", actor, target_pose, args)

    def move_by_displacement(
        self,
        arm_tag,
        x=0.0,
        y=0.0,
        z=0.0,
        quat=None,
        move_axis="world",
    ):
        return arm_tag, ("displacement", x, y, z, quat, move_axis)

    def open_gripper(self, arm_tag, pos=1.0):
        return arm_tag, ("open", pos)

    def move(self, *actions):
        self.moves.append(actions)


def local_point(x, y, z):
    return local_pose(x, y, z)


class TaskPlanTest(unittest.TestCase):
    def test_single_place_is_recorded_without_task_metadata(self):
        shoe = Actor(pose(0.2, 0.03, 0.74))
        target_fp = local_pose()
        target = Actor(pose(0.0, -0.08, 0.74), functional=(target_fp,))
        env = Env({"shoe": shoe, "platform": target})

        with ProceduralTaskRecorder(env) as recorder:
            env.move(env.grasp_actor(shoe, "right", pre_grasp_dis=0.10))
            env.move(env.move_by_displacement("right", z=0.07))
            env.move(
                env.place_actor(
                    shoe,
                    "right",
                    target.get_functional_point(0),
                    functional_point_id=0,
                    pre_dis=0.12,
                    dis=0.02,
                    constrain="align",
                )
            )
            env.move(env.open_gripper("right"))

        plan = task_plan_from_trace(env, "arbitrary_name", recorder.trace)

        self.assertEqual(plan.family, "pick_place")
        self.assertEqual(plan.manipulation_targets, ("shoe",))
        self.assertEqual(plan.pose_objects, ("shoe", "platform"))
        pick, place_stage = plan.stages
        self.assertEqual(
            pick,
            Pick(
                "shoe",
                "right",
                0.10,
                (0.0, 0.0, 0.07),
                group_id=0,
            ),
        )
        self.assertEqual(place_stage.destination, "platform")
        self.assertEqual(place_stage.destination_functional_point_id, 0)
        self.assertEqual(place_stage.object_functional_point_id, 0)
        self.assertEqual(place_stage.preplace_offset_m, 0.12)
        self.assertEqual(place_stage.place_offset_m, 0.02)
        self.assertEqual(place_stage.constrain, "align")
        self.assertEqual(place_stage.group_id, 2)
        self.assertIsNotNone(place_stage.target_pose)
        self.assertTrue(place_stage.release)

    def test_arbitrary_target_pose_does_not_invent_destination_actor(self):
        source = Actor(pose(-0.2, 0.1, 0.75))
        context = Actor(pose(0.0, 0.0, 0.75))
        target_pose = pose(-0.18, 0.0, 0.741)
        env = Env({"source": source, "context": context})

        with ProceduralTaskRecorder(env) as recorder:
            env.move(env.grasp_actor(source, "left", pre_grasp_dis=0.05))
            env.move(
                env.move_by_displacement("left", y=-0.10, z=0.10)
            )
            env.move(
                env.place_actor(
                    source,
                    "left",
                    target_pose,
                    pre_dis=0.05,
                    dis=0.0,
                )
            )

        pick, place_stage = task_plan_from_trace(
            env, "another_name", recorder.trace
        ).stages
        self.assertEqual(pick.target, "source")
        self.assertEqual(pick.postgrasp_displacement, (0.0, -0.10, 0.10))
        self.assertIsNone(place_stage.destination)
        self.assertIsNone(place_stage.destination_functional_point_id)
        np.testing.assert_allclose(place_stage.target_pose, target_pose)

    def test_actor_alias_is_not_misclassified_as_destination(self):
        source = Actor(pose(-0.2, 0.1, 0.75))
        env = Env({"source": source, "source_alias": source})

        with ProceduralTaskRecorder(env) as recorder:
            env.move(env.grasp_actor(source, "left"))
            env.move(
                env.place_actor(
                    source,
                    "left",
                    source.get_pose(),
                )
            )

        pick, place_stage = task_plan_from_trace(
            env, "alias_generic", recorder.trace
        ).stages
        self.assertEqual(pick.target, "source")
        self.assertIsNone(place_stage.destination)

    def test_opposite_arm_regrasp_reduces_to_handoff(self):
        contacts = tuple(
            local_point(x, 0.0, z)
            for z in (0.08, 0.08, -0.08, -0.08)
            for x in (0.01,)
        )
        source_fp = local_pose(0.0, 0.0, -0.10)
        box = Actor(
            pose(-0.15, 0.2, 0.84),
            functional=(source_fp,),
            contacts=contacts,
        )
        destination = Actor(
            pose(0.18, 0.16, 0.84),
            functional=(local_pose(), local_pose(0.0, 0.0, 0.01)),
        )
        middle = pose(0.0, 0.0, 0.9)
        env = Env({"object": box, "goal": destination})

        with ProceduralTaskRecorder(env) as recorder:
            env.move(
                env.grasp_actor(
                    box, "left", pre_grasp_dis=0.07,
                    gripper_pos=0.25,
                    contact_point_id=[0, 1],
                )
            )
            env.move(env.move_by_displacement("left", z=0.10))
            env.move(
                env.place_actor(
                    box,
                    "left",
                    middle,
                    functional_point_id=0,
                    pre_dis=0.0,
                    dis=0.0,
                    is_open=False,
                    constrain="free",
                )
            )
            env.move(
                env.grasp_actor(
                    box, "right", pre_grasp_dis=0.07,
                    gripper_pos=0.35,
                    contact_point_id=[2, 3],
                )
            )
            env.move(env.open_gripper("left"))
            env.move(
                env.place_actor(
                    box,
                    "right",
                    destination.get_functional_point(1),
                    functional_point_id=0,
                    pre_dis=0.05,
                    dis=0.0,
                    constrain="align",
                    pre_dis_axis="fp",
                )
            )

        plan = task_plan_from_trace(env, "generic_transfer", recorder.trace)

        self.assertEqual(plan.family, "handoff")
        self.assertEqual(plan.manipulation_targets, ("object",))
        self.assertEqual(plan.pose_objects, ("object", "goal"))
        pick, handoff, place_stage = plan.stages
        self.assertEqual(pick.arm, "left")
        self.assertEqual(pick.gripper_target, 0.25)
        self.assertEqual(pick.postgrasp_displacement, (0.0, 0.0, 0.10))
        self.assertIsInstance(handoff, Handoff)
        self.assertEqual((handoff.from_arm, handoff.to_arm), ("left", "right"))
        self.assertEqual(handoff.object_functional_point_id, 0)
        self.assertEqual(handoff.group_id, 2)
        self.assertEqual(handoff.gripper_target, 0.35)
        self.assertIsNotNone(handoff.rendezvous_pose)
        np.testing.assert_allclose(
            handoff.allowed_contact_points_local,
            ((0.01, 0.0, -0.08), (0.01, 0.0, -0.08)),
        )
        self.assertEqual(place_stage.destination, "goal")
        self.assertEqual(place_stage.destination_functional_point_id, 1)
        self.assertEqual(place_stage.preplace_axis, "fp")

    def test_grouped_dual_pick_and_nonrelease_places_are_preserved(self):
        first = Actor(pose(-0.15, 0.1, 0.8), functional=(local_pose(),))
        second = Actor(pose(0.15, 0.1, 0.8), functional=(local_pose(),))
        env = Env({"first": first, "second": second})

        with ProceduralTaskRecorder(env) as recorder:
            env.move(
                env.grasp_actor(first, "left", pre_grasp_dis=0.08),
                env.grasp_actor(second, "right", pre_grasp_dis=0.08),
            )
            env.move(
                env.move_by_displacement("left", z=0.10),
                env.move_by_displacement("right", z=0.10),
            )
            env.move(
                env.place_actor(
                    first,
                    "left",
                    pose(-0.06, -0.105, 1.0),
                    functional_point_id=0,
                    pre_dis=0.0,
                    dis=0.0,
                    is_open=False,
                ),
                env.place_actor(
                    second,
                    "right",
                    pose(0.06, -0.105, 1.0),
                    functional_point_id=0,
                    pre_dis=0.0,
                    dis=0.0,
                    is_open=False,
                ),
            )

        plan = task_plan_from_trace(env, "dual_generic", recorder.trace)

        self.assertEqual(plan.family, "pick_place")
        self.assertEqual(plan.manipulation_targets, ("first", "second"))
        self.assertEqual(len(plan.stages), 4)
        first_pick, second_pick, first_place, second_place = plan.stages
        self.assertEqual((first_pick.group_id, second_pick.group_id), (0, 0))
        self.assertEqual(
            (first_pick.postgrasp_displacement, second_pick.postgrasp_displacement),
            ((0.0, 0.0, 0.10), (0.0, 0.0, 0.10)),
        )
        self.assertIsInstance(first_place, Place)
        self.assertIsInstance(second_place, Place)
        self.assertEqual((first_place.group_id, second_place.group_id), (2, 2))
        self.assertFalse(first_place.release)
        self.assertFalse(second_place.release)
        self.assertIsNone(first_place.destination)
        self.assertIsNone(second_place.destination)

    def test_grouped_dual_pick_of_one_object_preserves_both_arm_intents(self):
        contacts = (
            local_point(-0.11, 0.01, 0.02),
            local_point(0.12, -0.01, 0.03),
        )
        roller = Actor(
            pose(0.07, -0.18, 0.76),
            contacts=contacts,
        )
        env = Env({"roller": roller})

        with ProceduralTaskRecorder(env) as recorder:
            env.move(
                env.grasp_actor(
                    roller,
                    "left",
                    pre_grasp_dis=0.06,
                    grasp_dis=0.01,
                    gripper_pos=0.25,
                    contact_point_id=0,
                ),
                env.grasp_actor(
                    roller,
                    "right",
                    pre_grasp_dis=0.09,
                    grasp_dis=0.02,
                    gripper_pos=0.35,
                    contact_point_id=1,
                ),
            )
            env.move(
                env.move_by_displacement("left", x=0.01, z=0.11),
                env.move_by_displacement("right", y=-0.02, z=0.12),
            )

        self.assertEqual(
            [(call.kind, call.arm, call.group_id) for call in recorder.trace],
            [
                ("grasp", "left", 0),
                ("grasp", "right", 0),
                ("displacement", "left", 1),
                ("displacement", "right", 1),
            ],
        )

        plan = task_plan_from_trace(env, "generic_shared_pick", recorder.trace)

        self.assertEqual(plan.family, "pick")
        self.assertEqual(plan.manipulation_targets, ("roller",))
        self.assertEqual(len(plan.stages), 2)
        left, right = plan.stages
        self.assertEqual(
            (
                left.target,
                left.arm,
                left.pregrasp_offset_m,
                left.postgrasp_displacement,
                left.grasp_offset_m,
                left.gripper_target,
                left.group_id,
            ),
            ("roller", "left", 0.06, (0.01, 0.0, 0.11), 0.01, 0.25, 0),
        )
        self.assertEqual(
            (
                right.target,
                right.arm,
                right.pregrasp_offset_m,
                right.postgrasp_displacement,
                right.grasp_offset_m,
                right.gripper_target,
                right.group_id,
            ),
            ("roller", "right", 0.09, (0.0, -0.02, 0.12), 0.02, 0.35, 0),
        )
        np.testing.assert_allclose(
            left.allowed_contact_points_local,
            ((-0.11, 0.01, 0.02),),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            right.allowed_contact_points_local,
            ((0.12, -0.01, 0.03),),
            atol=1e-12,
        )

    def test_terminal_nonrelease_place_plus_open_becomes_release(self):
        item = Actor(pose(), functional=(local_pose(),))
        env = Env({"item": item})
        with ProceduralTaskRecorder(env) as recorder:
            env.move(env.grasp_actor(item, "left"))
            env.move(
                env.place_actor(
                    item,
                    "left",
                    pose(0.1, 0.0, 0.8),
                    is_open=False,
                )
            )
            env.move(env.open_gripper("left"))

        plan = task_plan_from_trace(env, "generic", recorder.trace)
        self.assertEqual(len(plan.stages), 2)
        self.assertTrue(plan.stages[1].release)

    def test_no_manipulation_calls_produces_other_plan(self):
        env = Env({"item": Actor(pose())})
        with ProceduralTaskRecorder(env) as recorder:
            pass
        plan = task_plan_from_trace(env, "non_manipulation", recorder.trace)
        self.assertEqual(plan.family, "other")
        self.assertEqual(plan.stages, ())

    def test_trace_follows_move_order_not_action_construction_order(self):
        item = Actor(pose())
        env = Env({"item": item})
        with ProceduralTaskRecorder(env) as recorder:
            lift_action = env.move_by_displacement("left", z=0.06)
            grasp_action = env.grasp_actor(item, "left")
            env.move(grasp_action)
            env.move(lift_action)

        self.assertEqual(
            [(call.kind, call.group_id) for call in recorder.trace],
            [("grasp", 0), ("displacement", 1)],
        )
        pick, = task_plan_from_trace(
            env, "delayed_actions", recorder.trace
        ).stages
        self.assertEqual(pick.postgrasp_displacement, (0.0, 0.0, 0.06))

    def test_trace_ignores_constructed_but_unexecuted_action(self):
        item = Actor(pose())
        env = Env({"item": item})
        with ProceduralTaskRecorder(env) as recorder:
            env.grasp_actor(item, "left")

        self.assertEqual(recorder.trace, ())

    def test_recorder_restores_original_bound_methods(self):
        env = Env({"item": Actor(pose())})
        original = env.grasp_actor
        with ProceduralTaskRecorder(env):
            self.assertNotEqual(env.grasp_actor, original)
        self.assertEqual(env.grasp_actor, original)


if __name__ == "__main__":
    unittest.main()

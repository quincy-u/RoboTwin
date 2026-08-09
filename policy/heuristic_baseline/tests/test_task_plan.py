from __future__ import annotations

import unittest

from policy.heuristic_baseline.task_plan import (
    Pick,
    Place,
    task_family,
    task_plan_from_task,
    task_primitives,
)


class Actor:
    def __init__(self, modelname, model_id):
        self.modelname = modelname
        self.model_id = model_id


class Env:
    def __init__(self, tracked):
        self.tracked = tracked

    def get_tracked_objects(self):
        return self.tracked


class TaskPlanTest(unittest.TestCase):
    def test_pick_plan_uses_live_actor_name(self):
        env = Env({"bottle": Actor("001_bottle", 3)})
        plan = task_plan_from_task(
            env,
            "shake_bottle",
            {"{A}": "001_bottle/base3", "{a}": "right"},
        )
        self.assertEqual(plan.family, "pick")
        self.assertEqual(plan.stages, (Pick("bottle", "right"),))
        self.assertEqual(plan.primary_target, "bottle")

    def test_place_plan_uses_object_and_destination_names(self):
        env = Env({
            "cup": Actor("021_cup", 0),
            "coaster": Actor("019_coaster", 0),
        })
        plan = task_plan_from_task(
            env,
            "place_empty_cup",
            {"{A}": "021_cup/base0", "{B}": "019_coaster/base0", "{a}": "left"},
        )
        self.assertEqual(plan.family, "pick_place")
        self.assertEqual(
            plan.stages,
            (Pick("cup", "left"), Place("cup", "coaster", "left")),
        )

    def test_family_is_derived_from_task_source(self):
        self.assertEqual(task_family("shake_bottle"), "pick")
        self.assertEqual(task_family("place_empty_cup"), "pick_place")
        self.assertEqual(task_family("click_alarmclock"), "other")
        self.assertIn("grasp_actor", task_primitives("shake_bottle"))
        self.assertIn("place_actor", task_primitives("place_empty_cup"))

    def test_other_task_has_no_invented_pick_or_place(self):
        plan = task_plan_from_task(
            Env({"object": Actor("object", 0)}),
            "click_alarmclock",
            {"{A}": "alarmclock/base0"},
        )
        self.assertEqual(plan.stages, ())
        self.assertIsNone(plan.primary_target)


if __name__ == "__main__":
    unittest.main()

import tempfile

import torch
from mpmath import workdps

from src.tests.base import *
from src.util import SequentialWorkflow


class Workflow1(SequentialWorkflow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_step = 0
        self.num_executed = {1: 0, 2: 0, 3: 0}

    def step_001(self):
        if self.max_step < 1:
            raise KeyboardInterrupt

        self.num_executed[1] += 1
        return {"arg1": "a", "arg2": "b"}

    def step_002(self, arg1: str, arg2: str):
        if self.max_step < 2:
            raise KeyboardInterrupt

        self.num_executed[2] += 1
        return {"arg": arg1 + arg2}

    def step_003(self, arg: str):
        if self.max_step < 3:
            raise KeyboardInterrupt

        self.num_executed[3] += 1
        return arg * 3



class TestSequentialWorkflow(TestBase):

    def test_100_sequential(self):
        with tempfile.TemporaryDirectory("nn-test") as root:
            workflow = Workflow1(root_path=root)

            with self.assertRaises(KeyboardInterrupt):
                workflow.run()

            self.assertFalse(workflow.result_exists("step_001"))
            self.assertFalse(workflow.result_exists("step_002"))
            self.assertFalse(workflow.result_exists("step_003"))
            self.assertEqual({1: 0, 2: 0, 3: 0}, workflow.num_executed)

            workflow.max_step = 1
            with self.assertRaises(KeyboardInterrupt):
                workflow.run()

            self.assertTrue(workflow.result_exists("step_001"))
            self.assertFalse(workflow.result_exists("step_002"))
            self.assertFalse(workflow.result_exists("step_003"))
            self.assertEqual({1: 1, 2: 0, 3: 0}, workflow.num_executed)

            workflow.max_step = 2
            with self.assertRaises(KeyboardInterrupt):
                workflow.run()

            self.assertTrue(workflow.result_exists("step_001"))
            self.assertTrue(workflow.result_exists("step_002"))
            self.assertFalse(workflow.result_exists("step_003"))
            self.assertEqual({1: 1, 2: 1, 3: 0}, workflow.num_executed)

            workflow.max_step = 3
            for i in range(3):
                self.assertEqual(
                    "ababab",
                    workflow.run(),
                )

                self.assertTrue(workflow.result_exists("step_001"))
                self.assertTrue(workflow.result_exists("step_002"))
                self.assertTrue(workflow.result_exists("step_003"))
                self.assertEqual({1: 1, 2: 1, 3: 1}, workflow.num_executed)

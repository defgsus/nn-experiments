import unittest

from src.tests.base import TestBase
from src.datasets import make_compact_whitespace


class TestText(TestBase):

    def test_100_compact_whitespace(self):
        self.assertEqual(
            " hello world, this is compact whitespace ",
            make_compact_whitespace("""
            hello
            world,\t\tthis
            is\n\ncompact\r\r\rwhitespace
            """)
        )
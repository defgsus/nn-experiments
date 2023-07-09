import unittest

from src import console
from src.console import CC


class TestConsole(unittest.TestCase):

    def test_100_colors(self):
        print(" ".join(
            f"{CC.palette_signed(v)}{v}{CC.Off}"
            for v in [-1., -.7, -.3, -.1, 0., .1, .3, .7, 1.]
        ))

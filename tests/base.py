from pathlib import Path
import unittest


class TestBase(unittest.TestCase):

    DATA_PATH = Path(__file__).resolve().parent / "data"

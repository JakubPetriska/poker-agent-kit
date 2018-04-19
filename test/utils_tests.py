import unittest

from tools.utils import flatten


class UtilsTests(unittest.TestCase):
    def test_flatten(self):
        data = [[1, 2], [3], [4, 5]]
        flattened = flatten(*data)
        self.assertEqual(flattened, [1, 2, 3, 4, 5])

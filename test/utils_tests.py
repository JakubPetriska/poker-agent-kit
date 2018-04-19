import unittest

from tools.utils import flatten, intersection, is_unique


class UtilsTests(unittest.TestCase):
    def test_flatten(self):
        self.assertEqual(
            flatten([1, 2], [3], [4, 5]),
            [1, 2, 3, 4, 5])

    def test_intersection_empty(self):
        self.assertEqual(
            intersection([1, 2], [3, 4]),
            set([]))

    def test_intersection_non_empty(self):
        self.assertEqual(
            intersection([1, 2, 3, 4], [2, 4]),
            set([2, 4]))

    def test_is_unique_true(self):
        self.assertEqual(
            is_unique((1, 2), [3], [4, 5]),
            True)

    def test_is_unique_false(self):
        self.assertEqual(
            is_unique([1, 2], [3], [3, 4, 5]),
            False)

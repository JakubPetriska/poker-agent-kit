import unittest

from tools.hand_evaluation import get_winners, get_utility

#        Card values
#
#           Suit
#        c  d  h  s
#       ------------
#    2 | 0  1  2  3
#    3 | 4  5  6  7
#    4 | 8  9  10 11
#    5 | 12 13 14 15
# R  6 | 16 17 18 19
# a  7 | 20 21 22 23
# n  8 | 24 25 26 27
# k  9 | 28 29 30 31
#    T | 32 33 34 35
#    J | 36 37 38 39
#    Q | 40 41 42 43
#    K | 44 45 46 47
#    A | 48 49 50 51


class HandEvaluationTests(unittest.TestCase):
    def test_get_winners_folded_player(self):
        hands = [(51, 47, 43, 39, 35), None]
        winners = get_winners(hands)
        self.assertEqual(len(winners), 1)
        self.assertEqual(winners[0], 0)

    def test_get_winners_leduc_higher_card(self):
        hands = [(43, 22), (51, 23)]
        winners = get_winners(hands)
        self.assertEqual(len(winners), 1)
        self.assertEqual(winners[0], 1)

    def test_get_winners_leduc_pair(self):
        hands = [(22, 23), (51, 23)]
        winners = get_winners(hands)
        self.assertEqual(len(winners), 1)
        self.assertEqual(winners[0], 0)

    def test_get_winners_leduc_equal_cards(self):
        hands = [(50, 23), (51, 23)]
        winners = get_winners(hands)
        self.assertEqual(len(winners), 2)
        self.assertTrue(0 in winners)
        self.assertTrue(1 in winners)

    def test_get_utility_folded_player(self):
        self.assertEqual(
            get_utility(
                [(51,), (47,)],
                [],
                [True, False],
                [1, 1]).tolist(),
            [-1, 1])

    def test_get_utility_uneven_pot(self):
        self.assertEqual(
            get_utility(
                [(51,), (47,)],
                [],
                [False, False],
                [5, 1]).tolist(),
            [1, -1])

    def test_get_utility_board_cards(self):
        self.assertEqual(
            get_utility(
                [(51,), (47,)],
                [46],
                [False, False],
                [1, 1]).tolist(),
            [-1, 1])

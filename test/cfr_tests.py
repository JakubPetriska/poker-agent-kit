import unittest

import acpc_python_client as acpc

from cfr.main import Cfr

KUHN_POKER_GAME_FILE_PATH = 'games/kuhn.limit.2p.game'
KUHN_BIG_DECK_POKER_GAME_FILE_PATH = 'games/kuhn.bigdeck.limit.2p.game'
KUHN_BIG_DECK_2ROUND_POKER_GAME_FILE_PATH = 'games/kuhn.bigdeck.2round.limit.2p.game'
LEDUC_POKER_GAME_FILE_PATH = 'games/leduc.limit.2p.game'


class CfrTests(unittest.TestCase):
    def test_kuhn_cfr_works(self):
        game = acpc.read_game_file(KUHN_POKER_GAME_FILE_PATH)
        cfr = Cfr(game, show_progress=False)
        cfr.train(100)

    def test_kuhn_bigdeck_cfr_works(self):
        game = acpc.read_game_file(KUHN_BIG_DECK_POKER_GAME_FILE_PATH)
        cfr = Cfr(game, show_progress=False)
        cfr.train(100)

    def test_kuhn_bigdeck_2round_cfr_works(self):
        game = acpc.read_game_file(KUHN_BIG_DECK_2ROUND_POKER_GAME_FILE_PATH)
        cfr = Cfr(game, show_progress=False)
        cfr.train(100)

    def test_leduc_cfr_works(self):
        game = acpc.read_game_file(LEDUC_POKER_GAME_FILE_PATH)
        cfr = Cfr(game, show_progress=False)
        cfr.train(100)

    def test_leduc_cfr_checkpointing(self):
        game = acpc.read_game_file(LEDUC_POKER_GAME_FILE_PATH)
        cfr = Cfr(game, show_progress=False)

        checkpoints_count = 0
        def checkpoint_callback(game_tree, checkpoint_index, iterations):
            nonlocal checkpoints_count
            self.assertTrue(game_tree is not None)
            self.assertEqual(checkpoint_index, checkpoints_count)
            checkpoints_count += 1

        cfr.train(90, 15, checkpoint_callback)

        self.assertEqual(checkpoints_count, 6)

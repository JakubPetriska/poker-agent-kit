import unittest

import acpc_python_client as acpc

from cfr.main import Cfr

KUHN_POKER_GAME_FILE_PATH = 'games/kuhn.limit.2p.game'
LEDUC_POKER_GAME_FILE_PATH = 'games/leduc.limit.2p.game'


class CfrTests(unittest.TestCase):
    def test_kuhn_cfr_works(self):
        game = acpc.read_game_file(KUHN_POKER_GAME_FILE_PATH)
        cfr = Cfr(game, show_progress=False)
        cfr.train(100)

    def test_leduc_cfr_works(self):
        game = acpc.read_game_file(LEDUC_POKER_GAME_FILE_PATH)
        cfr = Cfr(game, show_progress=False)
        cfr.train(100)

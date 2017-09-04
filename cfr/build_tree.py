import copy
import itertools

import acpc_python_client as acpc

from cfr.game_tree import HoleCardsNode, ActionNode, TerminalNode, BoardCardsNode


class GameTreeBuilder:
    class GameState:
        def __init__(self, game, deck):
            # Game properties
            self.players_folded = [False] * game.get_num_players()
            self.pot_commitment = [game.get_blind(p) for p in range(game.get_num_players())]
            self.deck = deck

            # Round properties
            self.rounds_left = game.get_num_rounds()
            self.round_raise_count = 0
            self.players_acted = 0
            self.current_player = game.get_first_player(0)

        def next_round_state(self):
            res = copy.deepcopy(self)
            res.rounds_left -= 1
            res.round_raise_count = 0
            res.players_acted = 0
            return res

        def next_move_state(self):
            res = copy.deepcopy(self)
            res.players_acted += 1
            return res

    def __init__(self, game):
        self.game = game
        if game.get_betting_type() != acpc.BettingType.LIMIT:
            raise AttributeError('No limit betting games not supported')

    def build_tree(self):
        deck = acpc.game_utils.generate_deck(self.game)

        root = HoleCardsNode(None, self.game.get_num_hole_cards())
        num_hole_cards = self.game.get_num_hole_cards()
        hole_card_combinations = itertools.combinations(range(len(deck)), num_hole_cards)
        for hole_cards_indexes in hole_card_combinations:
            hole_cards = tuple(map(lambda i: deck[i], hole_cards_indexes))
            next_deck = list(deck)
            for hole_card_index in hole_cards_indexes:
                del next_deck[hole_card_index]
            game_state = GameTreeBuilder.GameState(self.game, next_deck)
            self._generate_board_cards_node(root, hole_cards, game_state)
        return root

    def _generate_board_cards_node(self, parent, child_key, game_state):
        rounds_left = game_state.rounds_left
        round_index = self.game.get_num_rounds() - rounds_left
        num_board_cards = self.game.get_num_board_cards(round_index)
        if num_board_cards <= 0:
            self._generate_action_node(parent, child_key, game_state)
        else:
            new_node = BoardCardsNode(parent, num_board_cards)
            parent.children[child_key] = new_node

            deck = game_state.deck
            board_card_combinations = itertools.combinations(range(len(deck)), num_board_cards)

            for board_cards_idxs in board_card_combinations:
                next_game_state = copy.deepcopy(game_state)
                for board_card_index in board_cards_idxs:
                    del next_game_state.deck[board_card_index]
                board_cards = tuple(map(lambda i: deck[i], board_cards_idxs))
                self._generate_action_node(new_node, board_cards, next_game_state)

    @staticmethod
    def _bets_settled(bets, players_folded):
        non_folded_bets = filter(lambda bet: not players_folded[bet[0]], enumerate(bets))
        non_folded_bets = list(map(lambda bet_enum: bet_enum[1], non_folded_bets))
        return non_folded_bets.count(non_folded_bets[0]) == len(non_folded_bets)

    def _generate_action_node(self, parent, child_key, game_state):
        players_folded = game_state.players_folded
        pot_commitment = game_state.pot_commitment
        current_player = game_state.current_player
        rounds_left = game_state.rounds_left

        bets_settled = GameTreeBuilder._bets_settled(pot_commitment, players_folded)
        all_acted = game_state.players_acted >= (self.game.get_num_players() - sum(players_folded))
        if bets_settled and all_acted:
            if rounds_left > 1:
                next_game_state = game_state.next_round_state()
                next_game_state.current_player = \
                    self.game.get_first_player(self.game.get_num_rounds() - rounds_left + 1)

                self._generate_board_cards_node(parent, child_key, next_game_state)
            else:
                new_node = TerminalNode(parent, pot_commitment)
                parent.children[child_key] = new_node
            return

        new_node = ActionNode(parent, current_player)
        parent.children[child_key] = new_node

        round_index = self.game.get_num_rounds() - rounds_left
        next_player = (current_player + 1) % self.game.get_num_players()
        max_pot_commitment = max(pot_commitment)
        valid_actions = [1]
        if not bets_settled:
            valid_actions.append(0)
        if game_state.round_raise_count < self.game.get_max_raises(round_index):
            valid_actions.append(2)
        for a in valid_actions:
            next_game_state = game_state.next_move_state()
            next_game_state.current_player = next_player

            if a == 0:
                next_game_state.players_folded[current_player] = True
            elif a == 1:
                next_game_state.pot_commitment[current_player] = max_pot_commitment
            elif a == 2:
                next_game_state.round_raise_count += 2
                next_game_state.pot_commitment[current_player] = \
                    max_pot_commitment + self.game.get_raise_size(round_index)

            self._generate_action_node(new_node, a, next_game_state)

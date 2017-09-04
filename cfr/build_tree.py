import acpc_python_client as acpc

from cfr.game_tree import HoleCardNode, ActionNode, TerminalNode


class GameTreeBuilder:
    def __init__(self, game):
        self.game = game
        if game.get_betting_type() != acpc.BettingType.LIMIT:
            raise AttributeError('No limit betting games not supported')
        if game.get_total_num_board_cards(game.get_num_rounds() - 1) > 0:
            raise AttributeError('Games with board cards not supported yet')

    def build_tree(self):
        deck = acpc.game_utils.generate_deck(self.game)
        return self._generate_hole_card_node(None, None,
                                             self.game.get_num_hole_cards(), deck)

    def _generate_hole_card_node(self, parent, child_key, hole_cards_left, deck):
        if hole_cards_left == 0:
            return self._generate_action_nodes(parent, child_key)
        new_node = HoleCardNode(parent, self.game.get_num_hole_cards() - hole_cards_left)
        if parent and child_key:
            parent.children[child_key] = new_node
        deck_offset = -hole_cards_left + 1
        if deck_offset == 0:
            deck_offset = len(deck)
        for i, hole_card in enumerate(deck[:deck_offset]):
            self._generate_hole_card_node(new_node, hole_card, hole_cards_left - 1, deck[i + 1:])
        return new_node

    def _generate_action_nodes(self, parent, child_key):
        blinds = [self.game.get_blind(p) for p in range(self.game.get_num_players())]
        self._generate_action_node(
            parent, child_key,
            self.game.get_num_rounds(), 0,
            0, [False] * self.game.get_num_players(),
            self.game.get_first_player(0), blinds)

    @staticmethod
    def _bets_settled(bets, players_folded):
        non_folded_bets = filter(lambda bet: not players_folded[bet[0]], enumerate(bets))
        non_folded_bets = list(map(lambda bet_enum: bet_enum[1], non_folded_bets))
        return non_folded_bets.count(non_folded_bets[0]) == len(non_folded_bets)

    def _generate_action_node(self, parent, child_key,
                              rounds_left, round_raise_count,
                              players_acted, players_folded,
                              current_player, pot_commitment):
        bets_settled = GameTreeBuilder._bets_settled(pot_commitment, players_folded)
        all_acted = players_acted >= (self.game.get_num_players() - sum(players_folded))
        if bets_settled and all_acted:
            if rounds_left > 1:
                next_round_first_player = \
                    self.game.get_first_player(self.game.get_num_rounds() - rounds_left + 1)
                self._generate_action_node(
                    parent, child_key, rounds_left - 1, 0,
                    0, players_folded,
                    next_round_first_player, pot_commitment)
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
        if round_raise_count < self.game.get_max_raises(round_index):
            valid_actions.append(2)
        for a in valid_actions:
            next_round_raise_count = round_raise_count
            next_players_folded = players_folded
            next_pot_commitment = pot_commitment

            if a == 0:
                next_players_folded = list(players_folded)
                next_players_folded[current_player] = True
            elif a == 1:
                next_pot_commitment = list(pot_commitment)
                next_pot_commitment[current_player] = max_pot_commitment
            elif a == 2:
                next_round_raise_count += 2
                next_pot_commitment = list(pot_commitment)
                next_pot_commitment[current_player] = \
                    max_pot_commitment + self.game.get_raise_size(round_index)

            self._generate_action_node(
                new_node, a, rounds_left, next_round_raise_count,
                players_acted + 1, next_players_folded, next_player,
                next_pot_commitment)

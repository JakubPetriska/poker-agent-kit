import scipy.misc
import scipy.special


def get_num_hole_card_combinations(game):
    num_players = game.get_num_players()
    num_hole_cards = game.get_num_hole_cards()
    num_cards = game.get_num_suits() * game.get_num_ranks()
    num_total_hole_cards = num_players * num_hole_cards
    return scipy.misc.comb(num_cards, num_total_hole_cards, exact=True) \
        * scipy.special.perm(num_total_hole_cards, num_total_hole_cards, exact=True)

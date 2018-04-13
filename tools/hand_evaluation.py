from functools import reduce

import acpc_python_client as acpc


def get_winners(hands):
    """Evaluate hands of players and determine winners.

    !!! This function is currently only capable of evaluating hands that contain up to 5 cards. !!!

    Args:
        hands (list(list(int))): List which contains player's hands. Each player's hand is a list of integers
                                 that represent player's cards. Board cards must be included in each player's hand.

    Returns:
        list(int): Indexes of winners. The pot should be split evenly between all winners.
    """
    scores = [(i, _score(hand) if hand else ((0,), (0,)))
              for i, hand in enumerate(hands)]
    sorted_scores = sorted(scores, key=lambda x: x[1])
    winning_score = sorted_scores[-1][1]
    winner_count = 1
    for i in range(len(hands) - 2, 0 - 1, -1):
        if sorted_scores[-i][1] == winning_score:
            winner_count += 1
        else:
            break
    return [score[0] for score in sorted_scores[len(hands) - winner_count:]]


def _parse_hand(hand):
    return map(lambda card: (acpc.game_utils.card_rank(card), acpc.game_utils.card_suit(card)), hand)


def _score(hand):
    if len(hand) <= 5:
        return _score_hand_combination(_parse_hand(hand))
    else:
        # TODO create multiple 5 card combinations from longer hand to allow Texas Hold'em hand evaluation
        return ((0,), (0,))


def _score_hand_combination(hand):
    rank_counts = {r: reduce(lambda count, card: count + (card[0] == r), hand, 0)
                   for r, _ in hand}.items()
    score, ranks = zip(*sorted((cnt, rank) for rank, cnt in rank_counts)[::-1])
    if len(score) == 5:
        if ranks[0:2] == (12, 3):  # adjust if 5 high straight
            ranks = (3, 2, 1, 0, -1)
        straight = ranks[0] - ranks[4] == 4
        flush = len({suit for _, suit in hand}) == 1
        score = ([(1,), (3, 1, 1, 1)], [(3, 1, 1, 2), (5,)])[flush][straight]
    return score, ranks

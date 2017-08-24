import random

NUM_ACTIONS = 2


class Node:
    def __init__(self, info_set):
        super().__init__()
        self.info_set = info_set
        self.regret_sum = [0] * NUM_ACTIONS
        self.strategy = [0] * NUM_ACTIONS
        self.strategy_sum = [0] * NUM_ACTIONS

    def get_strategy(self, realization_weight):
        normalizing_sum = 0
        for a in range(NUM_ACTIONS):
            self.strategy[a] = self.regret_sum[a] if self.regret_sum[a] > 0 else 0
            normalizing_sum += self.strategy[a]

        for a in range(NUM_ACTIONS):
            if normalizing_sum > 0:
                self.strategy[a] /= normalizing_sum
            else:
                self.strategy[a] = 1.0 / NUM_ACTIONS
                self.strategy_sum[a] += realization_weight * self.strategy[a]
        return self.strategy

    def get_average_strategy(self):
        avg_strategy = [0] * NUM_ACTIONS
        normalizing_sum = 0
        for a in range(NUM_ACTIONS):
            normalizing_sum += self.strategy_sum[a]

        for a in range(NUM_ACTIONS):
            if normalizing_sum > 0:
                avg_strategy[a] = self.strategy_sum[a] / normalizing_sum
            else:
                avg_strategy[a] = 1.0 / NUM_ACTIONS
        return avg_strategy

    def __str__(self):
        return '%s: %s' % (self.info_set, self.get_average_strategy())


nodeMap = {}


def cfr(cards, history, p0, p1):
    plays = len(history)
    player = plays % 2
    opponent = 1 - player

    if plays > 1:
        terminal_pass = history[plays - 1] == 'p'
        double_bet = history[plays - 2:plays] == 'bb'
        is_player_card_higher = cards[player] > cards[opponent]
        if terminal_pass:
            if history == 'pp':
                return 1 if is_player_card_higher else -1
            else:
                return 1

        elif double_bet:
            return 2 if is_player_card_higher else -2

    info_set = str(cards[player]) + history
    if info_set in nodeMap:
        node = nodeMap[info_set]
    else:
        node = Node(info_set)
        node.infoSet = info_set
        nodeMap[info_set] = node

    strategy = node.get_strategy(p0 if player == 0 else p1)
    util = [0] * NUM_ACTIONS
    node_util = 0
    for a in range(NUM_ACTIONS):
        next_history = history + ('p' if a == 0 else 'b')
        if player == 0:
            util[a] = -cfr(cards, next_history, p0 * strategy[a], p1)
        else:
            util[a] = -cfr(cards, next_history, p0, p1 * strategy[a])
        node_util += strategy[a] * util[a]

    for a in range(NUM_ACTIONS):
        regret = util[a] - node_util
        node.regret_sum[a] += (p1 if player == 0 else p0) * regret
    return node_util


def train(iterations):
    cards = [1, 2, 3]
    util = 0
    for i in range(iterations):
        for card_index_1 in range(len(cards) - 1, -1, -1):
            card_index_2 = random.randint(0, card_index_1)
            tmp = cards[card_index_1]
            cards[card_index_1] = cards[card_index_2]
            cards[card_index_2] = tmp
        util += cfr(cards, '', 1, 1)

    print('Average game value: %s' % (util / iterations))
    for key, node in nodeMap.items():
        print(str(node))


if __name__ == "__main__":
    iterations = 1000
    train(iterations)

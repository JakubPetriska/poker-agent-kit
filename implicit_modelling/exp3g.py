import numpy as np


class Exp3G:
    def __init__(self, gamma, eta, experts_count):
        self.gamma = gamma
        self.eta = eta
        self.experts_count = experts_count
        self.weights = np.ones(experts_count)

    def get_current_expert_probabilities(self):
        weights_sum = np.sum(self.weights)
        return ((1 - self.gamma) * (self.weights / weights_sum)) + (self.gamma / self.experts_count)

    def update_weights(self, expert_payoffs):
        self.weights *= np.exp(expert_payoffs * self.eta)

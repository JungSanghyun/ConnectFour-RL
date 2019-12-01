# -*- coding: utf-8 -*-
import random

from agents.agent import Agent


class RandomAgent(Agent):
    def __init__(self, ):
        super(RandomAgent, self).__init__(trainable=False)

    def get_action(self, state, valid_actions):
        actions = []
        for i in range(7):
            if valid_actions[i] == 1:
                actions.append(i)
        return random.choice(actions)

# -*- coding: utf-8 -*-
from agents.agent import Agent


class HumanAgent(Agent):
    def __init__(self, ):
        super(HumanAgent, self).__init__(trainable=False)

    def get_action(self, state, valid_actions):
        action = int(input('Select Action: ')) - 1
        return action

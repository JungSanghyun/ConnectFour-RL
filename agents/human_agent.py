# -*- coding: utf-8 -*-
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path: sys.path.append(PROJECT_ROOT)
from agents.agent import Agent


class HumanAgent(Agent):
    def __init__(self, ):
        super(HumanAgent, self).__init__(trainable=False)

    def get_action(self, state, valid_actions):
        action = int(input('Select Action: ')) - 1
        return action

# -*- coding: utf-8 -*-


class Agent(object):
    def __init__(self, trainable=False):
        self.trainable = trainable

    def get_action(self, state, valid_actions):
        raise NotImplementedError

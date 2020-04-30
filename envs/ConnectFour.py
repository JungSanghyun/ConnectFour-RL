# -*- coding: utf-8 -*-
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path: sys.path.append(PROJECT_ROOT)
from envs.rendering import ConnectFourViewer


class ConnectFour(object):
    def __init__(self):
        self.w = 7
        self.h = 6
        self.board, self.turn = self._initialize()
        self.viewer = None
        self.winner = None
        self.action_space = [i for i in range(self.w)]

    def _initialize(self):
        self.winner = None
        return [[0 for _ in range(self.w)] for _ in range(self.h)], 1

    def _check_done(self, turn):
        done, winner = False, None

        for i in range(self.w - 3):
            for j in range(self.h):
                if self.board[j][i] == turn and self.board[j][i + 1] == turn and self.board[j][i + 2] == turn and self.board[j][i + 3] == turn:
                    return True, turn

        for i in range(self.w):
            for j in range(self.h - 3):
                if self.board[j][i] == turn and self.board[j + 1][i] == turn and self.board[j + 2][i] == turn and self.board[j + 3][i] == turn:
                    return True, turn

        for i in range(self.w - 3):
            for j in range(self.h - 3):
                if self.board[j][i] == turn and self.board[j + 1][i + 1] == turn and self.board[j + 2][i + 2] == turn and self.board[j + 3][i + 3] == turn:
                    return True, turn

        for i in range(self.w - 3):
            for j in range(3, self.h):
                if self.board[j][i] == turn and self.board[j - 1][i + 1] == turn and self.board[j - 2][i + 2] == turn and self.board[j - 3][i + 3] == turn:
                    return True, turn

        tie = True
        for i in range(self.w):
            tie = tie and (self.board[self.h - 1][i] != 0)
        if tie:
            done, winner = True, 0
        return done, winner

    def reset(self):
        self.board, self.turn = self._initialize()
        observation = self.board
        if self.viewer is not None:
            self.viewer.reset()
        return observation

    def step(self, action):
        info, pos = dict(), 0
        for i in range(self.h):
            if self.board[i][action] != 0:
                pos = i + 1
        self.board[pos][action] = self.turn

        done, winner = self._check_done(self.turn)
        info['winner'] = winner

        reward = 0
        if winner is None: reward = 0
        if winner == 0: reward = 0
        elif winner == self.turn: reward = 1
        else: reward = -1

        self.turn = 3 - self.turn  # 1 to 2, 2 to 1
        observation = self.board
        return observation, reward, done, info

    def render(self):
        if self.viewer is None:
            self.viewer = ConnectFourViewer(self.w, self.h)
        self.viewer.render(self.board)

    def valid_actions(self):
        valid_actions = [1 for _ in range(self.w)]
        for i in range(self.w):
            if self.board[self.h - 1][i] != 0:
                valid_actions[i] = 0
        return valid_actions
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

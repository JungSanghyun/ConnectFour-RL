# -*- coding: utf-8 -*-
import numpy as np
import torch


def observation_to_tensor(observation):
    x = np.zeros((1, 2, 6, 7))
    for i in range(6):
        for j in range(7):
            if observation[i][j] == 1:
                x[0][0][i][j] = 1
            elif observation[i][j] == 2:
                x[0][1][i][j] = 1
    return torch.tensor(x, dtype=torch.float)

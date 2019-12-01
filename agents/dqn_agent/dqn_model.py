# -*- coding: utf-8 -*-
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=4, stride=1, padding=3),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.3, inplace=True),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=9*10*64, out_features=64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Linear(in_features=64, out_features=7),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

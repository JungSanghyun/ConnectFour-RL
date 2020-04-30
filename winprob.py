# -*- coding: utf-8 -*-
import os
import sys
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path: sys.path.append(PROJECT_ROOT)
from envs.ConnectFour import ConnectFour

from agents.random_agent import RandomAgent
from agents.dqn_agent.dqn_agent import DQNAgent


def winprob(model_prefix, player, start, end, step, num_episodes=500):
    """
    :param model_prefix: prefix of model name
    :param player: player turn of models (1 or 2)
    :param start: start number of models
    :param end: end number of models
    :param step: number interval of models
    :param num_episodes: max number of episodes iteration for each model
    """
    x, y = [], []
    agents = [None, None]
    agents[2 - player] = RandomAgent()
    env = ConnectFour()
    for model_num in tqdm(range(start, end+1, step)):
        agents[player - 1] = DQNAgent(trainable=False, filename=f'{model_prefix}_{model_num}.pt')
        total_rewards = 0
        for i_episode in range(1, num_episodes + 1):
            observation = env.reset()
            for t in range(50):
                agent = agents[env.turn - 1]
                action = agent.get_action(observation, env.valid_actions())
                observation, reward, done, info = env.step(action)
                if done:
                    total_rewards += [[0, 1, -1], [0, -1, 1]][player-1][info['winner']]
                    break
        x.append(model_num)
        y.append(total_rewards / num_episodes)

    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='prefix of model file')
    parser.add_argument('-p', '--player', type=int, help='player turn of model (1 or 2)')
    parser.add_argument('--start', type=int, help='start number of model')
    parser.add_argument('--end', type=int, default=None, help='end number of model')
    parser.add_argument('--step', type=int, default=None, help='step of number of model')
    parser.add_argument('-e', '--episodes', type=int, default=500, help='max number of episodes iteration')

    args = parser.parse_args()
    winprob(model_prefix=args.model,
            player=args.player,
            start=args.start,
            end=args.end,
            step=args.step,
            num_episodes=args.episodes)

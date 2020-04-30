# -*- coding: utf-8 -*-
import os
import sys
import torch
from argparse import ArgumentParser
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path: sys.path.append(PROJECT_ROOT)
from envs.ConnectFour import ConnectFour
from utils.utils import observation_to_tensor

from agents.random_agent import RandomAgent
from agents.dqn_agent.dqn_agent import DQNAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(agent1, agent2, num_episodes, train_freq, target_freq, save_freq):
    """
    :param agent1: first turn agent
    :param agent2: second turn agent
    :param num_episodes: max number of episodes iteration
    :param train_freq: train agent frequency (episodes per 1 train)
    :param target_freq: target update frequency (episodes per 1 target update)
    :param save_freq: model save frequency (episodes per 1 model save)
    """
    agents = [agent1, agent2]
    env = ConnectFour()
    reward_table = [[0., 1., -1.], [0., -1., 1.]]
    for i_episode in tqdm(range(1, num_episodes+1)):
        observation = env.reset()
        state_memory = [[], []]
        action_memory = [[], []]
        reward_memory = [None, None]
        for t in range(50):
            agent = agents[env.turn-1]
            if agent.trainable:
                state_memory[env.turn-1].append(observation_to_tensor(observation))
                action = agent.get_action_for_train(observation, env.valid_actions())
                action_memory[env.turn-1].append(action)
            else:
                action = agent.get_action(observation, env.valid_actions())
            observation, reward, done, info = env.step(action)

            if done:
                for player in range(2):
                    agent = agents[player]
                    if agent.trainable:
                        agent = agents[player]
                        memory_len = (t + 2 - player) // 2
                        state_memory[player].append(None)
                        reward_memory[player] = [0. for _ in range(memory_len)]
                        reward_memory[player][-1] = reward_table[player][info['winner']]
                        for i in range(memory_len):
                            action = torch.tensor([[action_memory[player][i]]]).to(device)
                            reward = torch.tensor([reward_memory[player][i]]).to(device)
                            agent.memory.push(state_memory[player][i], action, state_memory[player][i+1], reward)
                        agent.episodes_done += 1
                break

        for player in range(2):
            agent = agents[player]
            if agent.trainable:
                if agent.episodes_done % target_freq == 0:
                    agent.target_update()
                if agent.episodes_done % train_freq == 0:
                    agent.train()
                if agent.episodes_done % save_freq == 0:
                    agent.save_model(player=player+1)
    env.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-1p', '--agent1', type=str, help='choose agent1 ("random", "dqn_train", "dqn_eval")')
    parser.add_argument('-1m', '--model1', type=str, default=None, help='filename of model of agent1')
    parser.add_argument('-2p', '--agent2', type=str, help='choose agent2 ("random", "dqn_train", "dqn_eval")')
    parser.add_argument('-2m', '--model2', type=str, default=None, help='filename of model of agent2')
    parser.add_argument('-e', '--episodes', type=int, default=100000, help='max number of episodes iteration')
    parser.add_argument('--train', type=int, default=5, help='train agent frequency (episodes per 1 train)')
    parser.add_argument('--target', type=int, default=50, help='target update frequency (episodes per 1 target update)')
    parser.add_argument('--save', type=int, default=1000, help='model save frequency (episodes per 1 model save)')

    def get_agent(agent_name, model):
        if agent_name == 'random': agent = RandomAgent()
        elif agent_name == 'dqn_train': agent = DQNAgent(trainable=True, filename=model)
        elif agent_name == 'dqn_eval': agent = DQNAgent(trainable=False, filename=model)
        else: raise NameError(f'There is no agent name: {agent_name}')
        return agent

    args = parser.parse_args()
    train(agent1=get_agent(args.agent1, args.model1),
          agent2=get_agent(args.agent2, args.model2),
          num_episodes=args.episodes,
          train_freq=args.train,
          target_freq=args.target,
          save_freq=args.save)

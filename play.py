# -*- coding: utf-8 -*-
import os
import sys
import time
from argparse import ArgumentParser

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path: sys.path.append(PROJECT_ROOT)
from agents.random_agent import RandomAgent
from agents.dqn_agent.dqn_agent import DQNAgent
from agents.human_agent import HumanAgent
from envs.ConnectFour import ConnectFour


def play(agent1, agent2, num_episodes=10, render=False, verbose=False, delay=0.5):
    """
    :param agent1: first turn agent
    :param agent2: second turn agent
    :param num_episodes: max number of episodes iteration
    :param render: render environment
    :param verbose: print additional information
    :param delay: delay between actions (seconds)
    """
    agents = [agent1, agent2]
    env = ConnectFour()
    result = [0, 0, 0]
    for i_episode in range(1, num_episodes+1):
        observation = env.reset()
        if render: env.render()
        for t in range(50):
            agent = agents[env.turn-1]
            action = agent.get_action(observation, env.valid_actions())
            observation, reward, done, info = env.step(action)
            if render:
                env.render()
                time.sleep(delay)
            if done:
                if verbose: print(f'Game Result: {["Draw", "Player1 Wins", "Player2 Wins"][info["winner"]]}')
                result[info['winner']] += 1
                break
    print()
    print(f'Player1 Wins: {result[1]}')
    print(f'Player2 Wins: {result[2]}')
    print(f'Draw: {result[0]}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-1p', '--agent1', type=str, help='choose agent1 ("random", "dqn", "human")')
    parser.add_argument('-1m', '--model1', type=str, default='model1_100000.pt', help='filename of model of agent1')
    parser.add_argument('-2p', '--agent2', type=str, help='choose agent2 ("random", "dqn", "human")')
    parser.add_argument('-2m', '--model2', type=str, default='model2_100000.pt', help='filename of model of agent2')
    parser.add_argument('-e', '--episodes', type=int, default=10, help='max number of episodes iteration')
    parser.add_argument("-r", "--render", action="store_true", help="render the game state with GUI")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose the game winner for each game")
    parser.add_argument('-d', '--delay', type=float, default=0.5, help='delay between actions (seconds)')

    def get_agent(agent_name, model):
        if agent_name == 'random': agent = RandomAgent()
        elif agent_name == 'dqn': agent = DQNAgent(trainable=False, filename=model)
        elif agent_name == 'human': agent = HumanAgent()
        else: raise NameError(f'There is no agent name: {agent_name}')
        return agent

    args = parser.parse_args()
    play(agent1=get_agent(args.agent1, args.model1),
         agent2=get_agent(args.agent2, args.model2),
         num_episodes=args.episodes,
         render=args.render,
         verbose=args.verbose,
         delay=args.delay)

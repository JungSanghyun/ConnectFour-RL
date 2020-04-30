# -*- coding: utf-8 -*-
import os
import sys
import random
import math
import torch
import torch.optim as optim
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path: sys.path.append(PROJECT_ROOT)
from agents.agent import Agent
from agents.dqn_agent.dqn_model import DQN
from agents.dqn_agent.replay_memory import ReplayMemory, Transition
from utils.utils import observation_to_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = os.path.join(PROJECT_ROOT, 'agents/dqn_agent/models/')


class DQNAgent(Agent):
    def __init__(self, trainable=False, filename=None):
        super(DQNAgent, self).__init__(trainable=trainable)
        self.dqn = DQN().to(device)
        self.episodes_done = 0
        self.steps_done = 0

        # hyperparams
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 2000

        self.GAMMA = 0.99
        self.BATCH_SIZE = 1024

        if filename is not None:
            model_path = os.path.join(model_dir, filename)
            if os.path.exists(model_path):
                model = torch.load(model_path)
                try:
                    self.dqn.load_state_dict(model['dqn'])
                    self.episodes_done = model['episodes_done']
                    self.steps_done = model['steps_done']
                except RuntimeError as e:
                    print('Wrong saved file.')
                else:
                    print(f'Model: {filename} loaded.')
            else:
                print(f'Model: {filename} does not exist.')

        if self.trainable:
            self.optimizer = optim.Adam(self.dqn.parameters())
            self.memory = ReplayMemory(10000)

            self.target_dqn = DQN().to(device)
            self.target_dqn.load_state_dict(self.dqn.state_dict())
            self.target_dqn.eval()

    def get_action(self, state, valid_actions):
        self.dqn.eval()
        with torch.no_grad():
            x = observation_to_tensor(state).to(device)
            q_value = self.dqn(x).detach()[0].cpu()
            action = torch.argmax(F.softmax(q_value, dim=0) * torch.tensor(valid_actions, dtype=torch.float)).item()
        return action

    def get_action_for_train(self, state, valid_actions):
        self.dqn.eval()
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        if sample > eps_threshold:
            with torch.no_grad():
                x = observation_to_tensor(state).to(device)
                q_value = self.dqn(x).detach()[0].cpu()
                action = torch.argmax(F.softmax(q_value, dim=0) * torch.tensor(valid_actions, dtype=torch.float)).item()
        else:
            valid_action = []
            for i in range(7):
                if valid_actions[i] == 1:
                    valid_action.append(i)
            action = random.choice(valid_action)
        self.steps_done += 1
        return action

    def train(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        self.dqn.train()
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.dqn(state_batch.to(device)).gather(1, action_batch)
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_dqn(non_final_next_states.to(device)).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.dqn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def save_model(self, player):
        save_dict = {'dqn': self.dqn.state_dict(), 'episodes_done': self.episodes_done, 'steps_done': self.steps_done}
        torch.save(save_dict, os.path.join(model_dir, f'model{player}_{self.episodes_done}.pt'))

    def target_update(self):
        self.target_dqn.load_state_dict(self.dqn.state_dict())

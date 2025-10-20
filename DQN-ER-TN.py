import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt

MEMORY_SIZE=10000
MAX_EPISODES=500
MAX_STEPS=1000
MAX_EXPLORATION = 1.0
DECAY_EXPLORATION = 0.995
MIN_EXPLORATION = 0.001
GAMMA = 0.95
FC1_SIZE=512
FC2_SIZE=64
LEARNING_RATE = 1e-3

ENV = gym.make("CartPole-v1")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STATE_SHAPE = ENV.observation_space.shape[0]
ACTION_SHAPE = ENV.action_space.n

class DQN_Network(nn.Module):
    def __init__(self, state_shape=STATE_SHAPE, action_shape=ACTION_SHAPE):
        super(DQN_Network, self).__init__()
        self.fc1 = nn.Linear(state_shape, FC1_SIZE)
        self.fc2 = nn.Linear(FC1_SIZE, FC2_SIZE)
        self.fc3 = nn.Linear(FC2_SIZE, action_shape)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Agent():
    def __init__(self, device=DEVICE, state_shape=STATE_SHAPE, action_shape=ACTION_SHAPE, gamma=GAMMA, lr=LEARNING_RATE,action_value_noise_std=0.1):
        self.device = device
        self.state_dim = state_shape
        self.action_dim = action_shape
        self.gamma = gamma
        self.policy_net = DQN_Network(state_shape, action_shape).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.steps = 0
        self.writer = SummaryWriter()
        self.epsilon = MAX_EXPLORATION  
        self.temp = 1.0  
        self.action_value_noise_std = action_value_noise_std

    @staticmethod
    def softmax(x,temp=1.0):
        x = x / temp  
        z = x - max(x)  
        return np.exp(z)/np.sum(np.exp(z))
    
    def select_action(self,state,policy,epsilon,temp):

        if policy == 'egreedy': 
            if epsilon is None:
                raise KeyError("Provide an epsilon")
            if random.random() < epsilon:
                return random.randint(0, self.action_dim - 1)
            else:
                state = torch.FloatTensor(state).to(self.device)
                with torch.no_grad():
                    action = self.policy_net(state).argmax().item()

        if policy == 'Boltzmann':
            if temp is None:
                raise KeyError("Provide a temperature")
            state = torch.FloatTensor(state).to(self.device)
            with torch.no_grad():
                action_values = self.policy_net(state)
                action_probabilities = self.softmax(action_values.cpu().numpy(), temp)
                action = np.random.choice(np.arange(self.action_dim), p=action_probabilities)

        if policy == 'Thompsonsampling':
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action_values = self.policy_net(state)
                noise = torch.randn(action_values.size()).to(self.device) * self.action_value_noise_std
                sampled_values = action_values + noise
                action = torch.argmax(sampled_values, dim=1).item()

        return action  

    def train(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        action = torch.LongTensor([action]).unsqueeze(0).to(self.device)
        reward = torch.FloatTensor([reward]).unsqueeze(0).to(self.device)
        done = torch.FloatTensor([done]).unsqueeze(0).to(self.device)

        q_values = self.policy_net(state).gather(1, action)
        next_q_values = self.policy_net(next_state).max(1)[0]
        expected_q_values = reward + self.gamma * next_q_values * (1 - done)

        loss = self.loss_fn(q_values.unsqueeze(-1), expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        self.writer.add_scalar("Loss", loss.item(), self.steps)


def experiment(env, agent, strategy, max_episodes=MAX_EPISODES, max_steps=MAX_STEPS):
    rewards_per_episode = []
    for episode in tqdm(range(max_episodes), desc=f"Training with {strategy}"):
        state = ENV.reset()
        total_reward = 0
        for step in range(max_steps):
            if strategy == 'egreedy':
                action = agent.select_action(state, policy='egreedy', epsilon=agent.epsilon, temp=1.0)
                agent.epsilon = max(agent.epsilon * DECAY_EXPLORATION, MIN_EXPLORATION)
            elif strategy == 'Boltzmann':
                temp = max(1.0 - episode / max_episodes, 0.1)  
                action = agent.select_action(state, policy='Boltzmann', epsilon=agent.epsilon, temp=temp)
            elif strategy == 'Thompsonsampling':
                action = agent.select_action(state, policy='Thompsonsampling', epsilon=agent.epsilon, temp=1.0)
            next_state, reward, done, _ = ENV.step(action)
            agent.train(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
        rewards_per_episode.append(total_reward)
    return rewards_per_episode

if __name__ == "__main__":
    strategies = ['egreedy', 'Boltzmann', 'Thompsonsampling']
    strategy_rewards = {}

    for strategy in strategies:
        agent = Agent()  
        rewards = experiment(env=ENV, agent=agent, strategy=strategy)
        strategy_rewards[strategy] = rewards

    plt.figure()
    for strategy, rewards in strategy_rewards.items():
        average_rewards = np.cumsum(rewards) / (np.arange(MAX_EPISODES) + 1)
        plt.plot(average_rewards, label=f"{strategy} Average Reward")
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Ablation experiment results: DQN-ER-TN')
    plt.legend()
    plt.savefig('DQN-ER-TN.png')
    plt.show()
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
import os

MEMORY_SIZE = 10000
MAX_EPISODES = 500
MAX_STEPS = 1000
MAX_EXPLORATION = 1.0
DECAY_EXPLORATION = 0.995
MIN_EXPLORATION = 0.001
BATCH_SIZE = 64
GAMMA = 0.95
FC1_SIZE = 256
FC2_SIZE = 128

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

class Replay_Buffer():
    def __init__(self, memory_size=MEMORY_SIZE):
        self.capacity = deque(maxlen=memory_size)

    def push(self, state, action, reward, next_state, done):
        self.capacity.append((state, action, reward, next_state, done))

    def sample(self):
        results = random.sample(self.capacity, BATCH_SIZE)
        return results
        
    def __len__(self):
        return len(self.capacity)

class Agent():
    def __init__(self, device=DEVICE, state_shape=STATE_SHAPE, action_shape=ACTION_SHAPE, memory_size=MEMORY_SIZE, batch_size=BATCH_SIZE, gamma=GAMMA, lr=None, action_value_noise_std=0.1):
        self.device = device
        self.state_dim = state_shape
        self.action_dim = action_shape  
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.policy_net = DQN_Network(state_shape, action_shape).to(self.device)
        self.target_net = DQN_Network(state_shape, action_shape).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()
        self.steps = 0
        self.writer = SummaryWriter()
        self.epsilon = MAX_EXPLORATION  
        self.temp = 1.0  
        self.memory = Replay_Buffer(memory_size)
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

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample()
        batch = list(zip(*transitions))
        state_batch = torch.FloatTensor(batch[0]).to(self.device)
        action_batch = torch.LongTensor(batch[1]).to(self.device)
        reward_batch = torch.FloatTensor(batch[2]).to(self.device)
        next_state_batch = torch.FloatTensor(batch[3]).to(self.device)
        done_batch = torch.FloatTensor(batch[4]).to(self.device)
       
        next_actions = self.policy_net(next_state_batch).max(1)[1].view(-1, 1)      
        next_q_values = self.target_net(next_state_batch).gather(1, next_actions).squeeze(1) 
        expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)
        q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(-1)).squeeze(-1)
        loss = self.loss_fn(q_values, expected_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        self.writer.add_scalar("Loss", loss.item(), self.steps)

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, strategy):
        path = os.path.join('model', '_'.join(['DOUBLEDQN', strategy, "model.pth"]))   
        if not os.path.exists(os.path.dirname(path)):   
            os.makedirs(os.path.dirname(path))
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.update_target()


def Experiment(env, agent, strategies, max_episodes=MAX_EPISODES, max_steps=MAX_STEPS):
    strategy_rewards = {}  

    for strategy in strategies:
        total_rewards = []
        epsilon = MAX_EXPLORATION
        
        for episode in tqdm(range(max_episodes), desc=f"Training with {strategy}"):
            state = env.reset()
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
                
                next_state, reward, done, _ = env.step(action)
                agent.memory.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                agent.train()
                if done:
                    break
            agent.update_target()
            total_rewards.append(total_reward)         
            strategy_rewards[strategy] = total_rewards
            agent.save_model(strategy=strategy)
    
    plt.figure()
    for strategy, rewards in strategy_rewards.items():
        average_rewards = np.cumsum(rewards) / (np.arange(max_episodes) + 1)
        plt.plot(average_rewards, label=f"{strategy} Average Reward")
    
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Average Reward by different Strategies about Double DQN')
    plt.legend()
    plt.savefig('DOUBLEDQN.png')
    plt.show()

def test_agent(env, agent, strategies, episodes=10):
    for strategy in strategies:
        model_path = os.path.join('model', '_'.join(['DOUBLEDQN', strategy, "model.pth"]))   
        print(f"Testing with {strategy} strategy...")
        agent.load_model(path=model_path)
        total_rewards = 0   
        for episode in range(episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            while not done:
                if strategy == 'egreedy':
                    action = agent.select_action(state, policy='egreedy', epsilon=0, temp=0.2)
                elif strategy == 'Boltzmann':
                    action = agent.select_action(state, policy='Boltzmann', epsilon=0, temp=0.2)
                elif strategy == 'Thompsonsampling':
                    action = agent.select_action(state, policy='Thompsonsampling', epsilon=0, temp=0.2) 
                next_state, reward, done, _ = env.step(action)
                #colab does not have a graphical interface, so comment it out
                #env.render()
                episode_reward += reward
                state = next_state
            total_rewards+= episode_reward
        average_reward = total_rewards / episodes  
        print(f"Strategy: {strategy}, Average Reward: {average_reward:.2f}")

if __name__ == "__main__":
    agent = Agent(lr=1e-3)
    strategy_types = ['egreedy', 'Boltzmann', 'Thompsonsampling']
    #strategy_types = ['egreedy']
    Experiment(env=ENV, agent=agent, strategies=strategy_types)
    test_agent(env=ENV, agent=agent, strategies=strategy_types, episodes=10)
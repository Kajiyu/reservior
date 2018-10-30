import numpy as np
from skimage.transform import resize
import gym
import gym.spaces
import gym_oculoenv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from torch.distributions import Categorical
import torch.optim as optim

from esncell import ESNCell
import utils

import matplotlib.pyplot as plt

from itertools import count
from collections import namedtuple


# Dataset params
sample_length = 1000
n_samples = 40
batch_size = 1

n_obs = 50*50
n_action = 9
oh_mat = np.eye(n_action)

# ESN properties
input_dim = n_obs + n_action
n_hidden = 500
w_sparsity=0.5
leaky_rate = 0.2
input_scaling=0.01
spectral_radius = 0.9
n_iterations = 50

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class ActorCritic(nn.Module):
    def __init__(self, n_action, input_dim, hidden_dim=500):
        super(ActorCritic, self).__init__()
        self.affine_a = nn.Linear(input_dim, hidden_dim) # for actor
        self.affine_v = nn.Linear(input_dim, hidden_dim) # for critic
        self.action_head = nn.Linear(hidden_dim, n_action)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x1 = F.relu(self.affine_a(x))
        x2 = F.relu(self.affine_v(x))
        action_scores = self.action_head(x1)
        state_values = self.value_head(x2)
        return F.softmax(action_scores, dim=-1), state_values


# resnet = models.resnet50(pretrained=True)
# resnet = nn.Sequential(*list(resnet.children())[:-1])
resevior = ESNCell(input_dim, n_hidden, batch_size, spectral_radius=spectral_radius, input_scaling=input_scaling, w_sparsity=w_sparsity, leaky_rate=leaky_rate)
ac_model = ActorCritic(n_action, n_hidden)

optimizer = optim.Adam(ac_model.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float()
    state.requires_grad_(requires_grad=True)
    probs, state_value = ac_model(state)
#     print(probs)
    m = Categorical(probs)
    action = m.sample()
    ac_model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    return action.item()


def finish_episode(gamma=0.99):
    R = 0
    saved_actions = ac_model.saved_actions
    policy_losses = []
    value_losses = []
    rewards = []
    for r in ac_model.rewards[::-1]:
        R = r + gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for (log_prob, value), r in zip(saved_actions, rewards):
        reward = r - value.item()
        policy_losses.append(-log_prob * reward)
        value_losses.append(F.smooth_l1_loss(value.view(1), torch.tensor([r])))
    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    # for p in ac_model.parameters():
    #     print(p.grad)
    loss.backward()
    # for p in ac_model.parameters():
    #     print(p.grad)
#     retain_graph=True
    optimizer.step()
    print(loss.item(), np.sum(ac_model.rewards))
    del ac_model.rewards[:]
    del ac_model.saved_actions[:]


env = gym.make("RedCursor-v0")

running_reward = 10
hidden_x = resevior.init_hidden(batch_size)
for i_episode in range(10000):
    obs = env.reset()
    action_arr = np.array([0.] * n_action).reshape(1, -1)
    for t in range(50):  # Don't infinite loop while learning
        obs = obs/255.
        obs = resize(obs, (50, 50))
        obs = obs.transpose((2, 0, 1)).astype("float32")
        obs = (obs[0].astype("float32") + obs[1].astype("float32") + obs[2].astype("float32")) / 3.0
        # print(obs.shape)
        # obs = torch.tensor(np.expand_dims(obs.transpose((2, 0, 1)), axis=0).astype("float32"))
        obs = torch.tensor(obs).view(1, -1)
        obs = torch.cat([obs, torch.tensor(action_arr.astype("float32")).requires_grad_(requires_grad=False)], dim=-1)
        hidden_x = resevior(obs)
#         print(hidden_x)
        state = hidden_x.detach().numpy().reshape(1, -1)
        action = select_action(state)
        action_arr = np.expand_dims(oh_mat[action], axis=0)
        time_step = env.step(action)
        obs, reward, done, _ = env.step(action)
        env.render()
        ac_model.rewards.append(reward)
        if done or (t+1)%20 == 0:
            # print(hidden_x)
            running_reward = running_reward * 0.99 + t * 0.01
            finish_episode()
            if done:
                print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(i_episode, t, running_reward))
                break
        if t+1 == 50:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(i_episode, t, running_reward))
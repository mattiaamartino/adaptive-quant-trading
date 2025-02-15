import numpy as np

import torch
import torch.nn as nn

from tqdm import trange

from Env import POMDPTEnv, dt_policy 

# EPISODES 

class Episode:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.expert_actions = []  # 'prophetic' (intraday actions)
        self.dones = [] # 'done' flags
        self.is_demo = False # if the episode is a demonstration
        self.priority = 1.0 # priority for PER

class PERBuffer:
    def __init__(self, max_episodes=1000, alpha=0.6, beta=0.4, device="cuda"):
        self.max_episodes = max_episodes
        self.alpha = alpha # priority exponent
        self.beta = beta # importance sampling exponent
        self.device = device
        self.episodes = []
        self.priorities = [] # for PER
    
    def add_episode(self, ep):
        # Remove the oldest episode if buffer is full
        if len(self.episodes) >= self.max_episodes:
            self.episodes.pop(0)
            self.priorities.pop(0)

        ep.obs = torch.tensor(np.array(ep.obs, dtype=np.float32), device=self.device)
        ep.actions = torch.tensor(np.array(ep.actions, dtype=np.int64), device=self.device)
        ep.rewards = torch.tensor(np.array(ep.rewards, dtype=np.float32), device=self.device)
        ep.expert_actions = torch.tensor(np.array(ep.expert_actions, dtype=np.int64), device=self.device)
        ep.dones = torch.tensor(np.array(ep.done_flags, dtype=bool), device=self.device)
        
        self.episodes.append(ep)
        self.priorities.append(ep.priority ** self.alpha)

    def update_priorities(self, indices, new_priorities):
        # Eq (10)
        for i, p in zip(indices, new_priorities):
            self.priorities[i] = p ** self.alpha

    def sample(self, batch_size):
        probs = np.array(self.priorities) / np.sum(self.priorities)
        indices = np.random.choice(len(self.episodes), batch_size, p=probs)
        batch = [self.episodes[i] for i in indices]
        
        # Eq (10)
        weights = (len(self.episodes) * probs[indices]) ** (-self.beta)
        weights = weights / np.max(weights)
        
        episodes = [self.episodes[i] for i in indices]

        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        return batch, episodes, indices, weights
    
    def __len__(self):
        return len(self.episodes)
    
# AGENT

class iRDPGAgent(nn.Module):
    def __init__(self, obs_dim, action_dim=3, hidden_dim=64, device="cuda"):
        super().__init__()

        self.actor_gru = nn.GRU(obs_dim, hidden_dim, batch_first=True) # [batch, seq_len, obs_dim]
        self.actor_fc = nn.Linear(hidden_dim, action_dim) # [batch, seq_len, action_dim]

        self.critic_gru = nn.GRU(obs_dim + action_dim, hidden_dim, batch_first=True) # [batch, seq_len, obs_dim + action_dim]
        self.critic_fc = nn.Linear(hidden_dim, action_dim) # [batch, seq_len, action_dim]

        self.device = device
        self.action_dim = action_dim

    def forward(self, obs, actions, h_actor=None, h_critic=None):

        obs = obs.to(self.device)
        
        # Actor
        z_actor, h_actor_next = self.actor_gru(obs, h_actor)
        logits = self.actor_fc(z_actor)
        # Critic
        actions_onehot = torch.nn.functional.one_hot(actions, num_classes=self.action_dim).float() # Ensure correct structure

        z_critic, h_critic_next = self.critic_gru(torch.concat([obs, actions_onehot], dim=-1), h_critic)
        q_value = self.critic_fc(z_critic[:, -1])

        return logits, q_value, h_actor_next, h_critic_next
    
    def act(self, obs, h_actor=None):

        dummy_actions = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).view(1,1,-1)  
            logits, _, h_actor, _ = self.forward(obs_t, dummy_actions, h_actor)
            probs = torch.softmax(logits[0,0,:], dim=-1)  
            action = probs.argmax().item()
            
        return action, h_actor
    

# UTILS

def collect_episode(env, agent, noise, device="cuda"):

    ep = Episode()
    obs = env.reset()
    done = False

    dummy_action = torch.zeros((1,1), dtype=torch.int64, device=device)
    
    while not done:

        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float, device=device).view(1,1,-1)
            logits, Q_vals, _, _ = agent(obs_t, dummy_action)
            # pick a policy action
            probs = torch.softmax(logits[0,0,:], dim=-1)
            if noise:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()
            else:
                action = probs.argmax().item()
        
        next_obs, reward, done, _ = env.step(action)
        
        ep.obs.append(obs)
        ep.actions.append(action)
        ep.rewards.append(reward)
        ep.done_flags.append(done)
        
        obs = next_obs
    
    return ep

def collect_demonstrations(df, window_size=60, n_episodes=50):
    demos = []
    env = POMDPTEnv(df, window_size=window_size)
    for _ in trange(n_episodes, desc="Collecting Demonstrations"):
        ep = Episode()
        obs = env.reset()
        done = False
        while not done:
            a = dt_policy(env)
            next_obs, rew, done, _ = env.step(a)
            ep.obs.append(obs)
            ep.actions.append(a)
            ep.rewards.append(rew)
            ep.done_flags.append(done)
            obs = next_obs
        demos.append(ep)
    return demos
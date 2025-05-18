import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import os # For BATTLESNAKE_RENDER
import wandb # Import wandb
import time
import glob
import argparse
from pathlib import Path
from collections import defaultdict

from Gym.snake_gym import BattlesnakeGym
from Gym.snake import Snake # For action definitions
# --- Actor-Critic Network ---
class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions): # input_dims is (channels, height, width) which is (3, 11, 11) for the snake representation
        """
        Actor-Critic Network for PPO
        Args:
            input_dims (tuple): Dimensions of the input state (channels, height, width)
            n_actions (int): Number of possible actions
        """
        # Initialize the parent class
        super().__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_dims[0], 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Calculate flattened size after conv layers
        dummy_input = torch.zeros(1, *input_dims)
        conv_out = self.conv2(F.relu(self.conv1(dummy_input)))
        self.flattened_size = int(np.prod(conv_out.size()[1:]))
        
        # Fully connected layers
        self.fc_shared = nn.Linear(self.flattened_size, 128)
        
        # Actor head
        self.actor_head = nn.Linear(128, n_actions)
        # Critic head
        self.critic_head = nn.Linear(128, 1)

    def forward(self, state): # state shape (batch, channels, height, width)
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # Flatten
        
        x_shared = F.relu(self.fc_shared(x))
        
        action_logits = self.actor_head(x_shared)
        value = self.critic_head(x_shared)
        
        return action_logits, value

# --- PPO Memory Buffer ---
class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.vals = []
        
        self.batch_size = batch_size # Not strictly used for PPO batching in this version, but good to have, little bit of a remainder from the DQN code

    def store_memory(self, state, action, reward, done, log_prob, val):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.vals.append(val)

    def sample_all(self):
        # Convert lists to tensors for processing
        # States are numpy arrays, need to stack and convert
        states_tensor = torch.tensor(np.array(self.states), dtype=torch.float32)
        actions_tensor = torch.tensor(self.actions, dtype=torch.long)
        rewards_tensor = torch.tensor(self.rewards, dtype=torch.float32)
        dones_tensor = torch.tensor(self.dones, dtype=torch.bool)
        old_log_probs_tensor = torch.tensor(self.log_probs, dtype=torch.float32)
        vals_tensor = torch.tensor(self.vals, dtype=torch.float32)
        
        return states_tensor, actions_tensor, rewards_tensor, dones_tensor, old_log_probs_tensor, vals_tensor

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.vals = []
    
    def __len__(self):
        return len(self.states)

# --- PPO Agent ---
class PPOAgent:
    def __init__(self, input_dims, n_actions, lr=0.0003, gamma=0.99, gae_lambda=0.95, ppo_clip=0.2, n_epochs=10, batch_size_ppo=64, device='cpu'):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_clip = ppo_clip
        self.n_epochs = n_epochs
        self.batch_size_ppo = batch_size_ppo
        self.device = device
        self.input_dims = input_dims
        self.actor_critic = ActorCritic(input_dims, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)

    def choose_action(self, observation_tensor, action_mask=None): # observation_tensor shape (1, C, H, W), action_mask boolean list/tensor
        self.actor_critic.eval() # Set to evaluation mode for action selection
        with torch.no_grad():
            logits, value = self.actor_critic(observation_tensor.to(self.device)) # logits shape (1, n_actions)

        
        if action_mask is not None:
            # Apply the mask: set logits of invalid actions to a very small number
            # Ensure action_mask is a boolean tensor on the same device
            mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=self.device) # mask_tensor shape (n_actions,)
            
            # Apply mask to the action dimension of the logits
            # logits has shape [1, n_actions], mask_tensor has shape [n_actions]
            logits[0, mask_tensor] = -float('inf') 
            # Handle case where all actions are masked
            # Check if all elements in the action dimension of logits are -inf
            if torch.all(logits[0] == -float('inf')):
                # If all actions are masked, unmask all by re-fetching original logits.
                # This allows the policy to choose, even if it leads to a collision.
                # A more sophisticated handling might pick a default "least bad" action.
                original_logits, _ = self.actor_critic(observation_tensor.to(self.device))
                logits = original_logits



        probs = F.softmax(logits, dim=-1)
        # Handle NaN probabilities if all logits were -inf and not caught by the above check (should be rare)
        if torch.isnan(probs).any():
            # If probs are NaN (e.g. all logits were -inf and softmax resulted in NaN)
            # Fallback to uniform distribution over all actions or re-fetch original logits
            original_logits, _ = self.actor_critic(observation_tensor.to(self.device))
            probs = F.softmax(original_logits, dim=-1)

        distribution = Categorical(probs)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()

    def learn(self, memory, last_val_of_final_state, global_step_counter=None): # Added global_step_counter for logging
        self.actor_critic.train() # Set to training mode

        states, actions, rewards, dones, old_log_probs, vals = memory.sample_all()
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device) # dones is a bool tensor
        old_log_probs = old_log_probs.to(self.device)
        vals = vals.to(self.device)

        # Calculate GAE (Generalized Advantage Estimation)
        advantages = torch.zeros_like(rewards).to(self.device)
        gae = 0
        # The last_val_of_final_state is V(S_N) where N is the number of steps collected
        # If S_N was terminal, this should be 0.
        # The dones array indicates if S_{t+1} is terminal.
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1: # Last step in the collected trajectory
                next_non_terminal = 1.0 - dones[t].float() # dones[t] is True if S_{t+1} (i.e., S_N) is terminal
                next_value = last_val_of_final_state # This is V(S_N)
            else:
                next_non_terminal = 1.0 - dones[t].float() # dones[t] is True if S_{t+1} is terminal
                next_value = vals[t+1]             # This is V(S_{t+1})
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - vals[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
        
        returns = advantages + vals
        
        # Normalize advantages (optional, but often helpful)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO Update
        num_samples = len(states)
        indices = np.arange(num_samples)

        # For logging average losses over epochs
        avg_actor_loss = 0
        avg_critic_loss = 0
        avg_total_loss = 0
        num_updates = 0

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, num_samples, self.batch_size_ppo):
                end = start + self.batch_size_ppo
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                logits, critic_values = self.actor_critic(batch_states)
                critic_values = critic_values.squeeze()

                new_probs = F.softmax(logits, dim=-1)
                new_distribution = Categorical(new_probs)
                new_log_probs = new_distribution.log_prob(batch_actions)
                
                # PPO Loss
                prob_ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = prob_ratio * batch_advantages
                surr2 = torch.clamp(prob_ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                critic_loss = F.mse_loss(critic_values, batch_returns)
                
                entropy_loss = -new_distribution.entropy().mean() # Optional: for exploration

                total_loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss # Adjust entropy coefficient

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5) # Grad clipping
                self.optimizer.step()

                avg_actor_loss += actor_loss.item()
                avg_critic_loss += critic_loss.item()
                avg_total_loss += total_loss.item()
                num_updates += 1

            # Log losses to wandb which we dont do anymore
            '''if num_updates > 0 and global_step_counter is not None:
            wandb.log({
                "actor_loss": avg_actor_loss / num_updates,
                "critic_loss": avg_critic_loss / num_updates,
                "total_ppo_loss": avg_total_loss / num_updates,
                "global_step": global_step_counter
            })'''

    def save_model(self, filepath):
        """Save model state dictionary to a file and export ONNX"""
        # Save PyTorch checkpoint
        torch.save({
            'model_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, filepath)
        print(f"Model saved to {filepath}")
        
        # Export ONNX
        onnx_path = filepath.replace(".pt", ".onnx")
        self.actor_critic.eval()
        # Dummy inputs (adjust shapes to your model)
        dummy_obs = torch.randn(1, *self.input_dims).to(self.device)


        torch.onnx.export(
            self.actor_critic,
            (dummy_obs),
            onnx_path,
            input_names=["input"],
            output_names=["logits", "value"],
            opset_version=11,
            do_constant_folding=True
        )

        print(f"ONNX model exported to {onnx_path}")
    
    def load_model(self, filepath):
        """Load model state dictionary from a file"""
        if not os.path.exists(filepath):
            print(f"Warning: Model file {filepath} not found. Starting with a fresh model.")
            return False
        
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {filepath}")
        return True
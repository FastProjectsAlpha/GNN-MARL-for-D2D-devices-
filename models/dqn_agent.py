import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from typing import List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DQNNetwork(nn.Module):
    """Deep Q-Network architecture"""
    
    def __init__(self, state_dim: int, action_dim: int):
        """
        Initialize the DQN network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
        """
        super(DQNNetwork, self).__init__()
        
        try:
            self.fc1 = nn.Linear(state_dim, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3 = nn.Linear(128, 64)
            self.fc4 = nn.Linear(64, action_dim)
            
            logger.debug(f"Initialized DQN with state_dim={state_dim}, action_dim={action_dim}")
            
        except Exception as e:
            logger.error(f"Failed to initialize DQN network: {str(e)}")
            raise
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class DQNAgent:
    """Deep Q-Network Agent for spectrum allocation"""
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int, 
                 lr: float = 0.001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 batch_size: int = 32,
                 memory_size: int = 10000):
        """
        Initialize the DQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            lr: Learning rate
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Exploration rate decay
            batch_size: Batch size for experience replay
            memory_size: Size of replay memory
        """
        try:
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.lr = lr
            self.gamma = gamma
            self.epsilon = epsilon_start
            self.epsilon_min = epsilon_end
            self.epsilon_decay = epsilon_decay
            self.batch_size = batch_size
            
            # Experience replay memory
            self.memory = deque(maxlen=memory_size)
            
            # Neural networks
            self.q_network = DQNNetwork(state_dim, action_dim)
            self.target_network = DQNNetwork(state_dim, action_dim)
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
            
            # Update target network
            self.update_target_network()
            
            logger.info("DQN Agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize DQN agent: {str(e)}")
            raise
    
    def update_target_network(self):
        """Update target network with Q-network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        logger.debug("Target network updated")
    
    def remember(self, 
                 state: np.ndarray, 
                 action: int, 
                 reward: float, 
                 next_state: np.ndarray, 
                 done: bool):
        """
        Store experience in replay memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            
        Returns:
            Selected action index
        """
        try:
            if np.random.random() <= self.epsilon:
                return random.randrange(self.action_dim)
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return np.argmax(q_values.cpu().data.numpy())
            
        except Exception as e:
            logger.error(f"Error in action selection: {str(e)}")
            return random.randrange(self.action_dim)  # Fallback to random action
    
    def replay(self) -> Optional[float]:
        """
        Train on a batch of experiences from replay memory.
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.memory) < self.batch_size:
            logger.debug("Not enough samples in memory for training")
            return None
        
        try:
            # Sample batch from memory
            batch = random.sample(self.memory, self.batch_size)
            states = torch.FloatTensor([e[0] for e in batch])
            actions = torch.LongTensor([e[1] for e in batch])
            rewards = torch.FloatTensor([e[2] for e in batch])
            next_states = torch.FloatTensor([e[3] for e in batch])
            dones = torch.BoolTensor([e[4] for e in batch])
            
            # Current Q values for chosen actions
            current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
            
            # Target Q values
            with torch.no_grad():
                next_q = self.target_network(next_states).max(1)[0]
                target_q = rewards + (self.gamma * next_q * ~dones)
            
            # Compute loss and update
            loss = F.mse_loss(current_q.squeeze(), target_q)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Decay exploration rate
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            logger.debug(f"Training step - loss: {loss.item():.4f}, epsilon: {self.epsilon:.4f}")
            return loss.item()
            
        except Exception as e:
            logger.error(f"Error in experience replay: {str(e)}")
            return None
import torch
import numpy as np
from typing import Dict, List, Tuple
import yaml
import logging
from collections import defaultdict
import time
import os
from tqdm import tqdm

from environment import D2DEnvironment
from models.gnn import GraphNeuralNetwork
from models.dqn_agent import DQNAgent
from utils.logger import setup_logging
from utils.visualization import plot_training_results

class GNNMARLSystem:
    """Complete GNN-MARL System for D2D Spectrum Allocation"""
    
    def __init__(self, config_path: str = 'configs/default_config.yaml'):
        """
        Initialize the GNN-MARL system.
        
        Args:
            config_path: Path to configuration file
        """
        try:
            # Load configuration
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            # Initialize logging
            setup_logging(self.config['logging']['log_dir'])
            self.logger = logging.getLogger(__name__)
            
            # Initialize environment
            self.env = D2DEnvironment(config_path)
            self.num_channels = self.env.num_channels
            self.num_power_levels = 4  # [10, 15, 20, 23] dBm
            
            # Initialize GNN
            self.gnn = GraphNeuralNetwork(
                input_dim=self.config['gnn']['input_dim'],
                hidden_dim=self.config['gnn']['hidden_dim'],
                output_dim=self.config['gnn']['output_dim'],
                dropout=self.config['gnn']['dropout']
            )
            
            # Create agents for each D2D pair
            self.agents = []
            action_dim = self.num_channels * self.num_power_levels  # Combined action space
            
            for i in range(self.env.num_d2d_pairs):
                agent = DQNAgent(
                    state_dim=38,  # Adjusted state dimension
                    action_dim=action_dim,
                    lr=self.config['training']['lr'],
                    gamma=self.config['training']['gamma'],
                    epsilon_start=self.config['training']['epsilon_start'],
                    epsilon_end=self.config['training']['epsilon_end'],
                    epsilon_decay=self.config['training']['epsilon_decay'],
                    batch_size=self.config['training']['batch_size']
                )
                self.agents.append(agent)
            
            # Training history
            self.training_history = defaultdict(list)
            
            self.logger.info("GNN-MARL System initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize GNN-MARL system: {str(e)}")
            raise
    
    def create_graph_data(self, state: Dict) -> Dict:
        """
        Convert network state to graph representation.
        
        Args:
            state: Current network state
            
        Returns:
            Dictionary containing:
            - x: Node features [num_nodes, feature_dim]
            - edge_index: Edge indices [2, num_edges]
            - edge_weight: Edge weights [num_edges]
        """
        try:
            positions = state['positions']
            num_nodes = len(positions)
            
            # Node features: [x, y, tx_power, channel, sinr, is_d2d]
            node_features = []
            
            # D2D nodes (transmitters and receivers)
            for i in range(self.env.num_d2d_pairs):
                tx_pos, rx_pos = positions[i]
                
                # Transmitter node features
                tx_features = [
                    tx_pos[0] / self.env.area_size,  # Normalized x
                    tx_pos[1] / self.env.area_size,  # Normalized y
                    state['power_levels'][i] / 23,   # Normalized power
                    state['spectrum_allocation'][i] / self.num_channels,  # Normalized channel
                    (state['sinr_values'][i] + 20) / 40,  # Normalized SINR
                    1.0  # Is D2D transmitter
                ]
                node_features.append(tx_features)
                
                # Receiver node features
                rx_features = [
                    rx_pos[0] / self.env.area_size,
                    rx_pos[1] / self.env.area_size,
                    0,  # Receiver doesn't transmit
                    state['spectrum_allocation'][i] / self.num_channels,
                    (state['sinr_values'][i] + 20) / 40,
                    0.5  # Is D2D receiver
                ]
                node_features.append(rx_features)
            
            # Cellular user nodes
            for i in range(self.env.num_cellular_users):
                pos = positions[self.env.num_d2d_pairs + i][0]  # Same for tx and rx in cellular
                features = [
                    pos[0] / self.env.area_size,
                    pos[1] / self.env.area_size,
                    1.0,  # Full power for cellular
                    0,    # Different channel allocation for cellular
                    0,    # No SINR calculation for cellular
                    0.0   # Is cellular user
                ]
                node_features.append(features)
            
            # Create edges based on interference relationships
            edge_indices = []
            edge_weights = []
            
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    # Get positions based on node type
                    if i < 2 * self.env.num_d2d_pairs:
                        pos_i = positions[i // 2][i % 2]
                    else:
                        pos_i = positions[self.env.num_d2d_pairs + (i - 2 * self.env.num_d2d_pairs)][0]
                    
                    if j < 2 * self.env.num_d2d_pairs:
                        pos_j = positions[j // 2][j % 2]
                    else:
                        pos_j = positions[self.env.num_d2d_pairs + (j - 2 * self.env.num_d2d_pairs)][0]
                    
                    distance = self.env.calculate_distance(pos_i, pos_j)
                    if distance < 200:  # Only connect nearby nodes
                        weight = 1 / (1 + distance / 100)  # Inverse distance weighting
                        edge_indices.extend([[i, j], [j, i]])  # Undirected graph
                        edge_weights.extend([weight, weight])
            
            # Convert to tensors
            x = torch.FloatTensor(node_features)
            edge_index = torch.LongTensor(edge_indices).t().contiguous() if edge_indices else torch.empty((2, 0), dtype=torch.long)
            edge_weight = torch.FloatTensor(edge_weights) if edge_weights else torch.empty(0)
            
            return {
                'x': x,
                'edge_index': edge_index,
                'edge_weight': edge_weight
            }
            
        except Exception as e:
            self.logger.error(f"Error creating graph data: {str(e)}")
            raise
    
    def get_agent_state(self, agent_idx: int, gnn_embeddings: torch.Tensor, state: Dict) -> np.ndarray:
        """
        Get state representation for a specific agent.
        
        Args:
            agent_idx: Index of the agent/D2D pair
            gnn_embeddings: GNN embeddings for all nodes
            state: Current network state
            
        Returns:
            State representation vector for the agent
        """
        try:
            # Agent's own embedding (transmitter node)
            own_embedding = gnn_embeddings[agent_idx * 2].detach().numpy()
            
            # Local network information
            local_info = [
                state['spectrum_allocation'][agent_idx] / self.num_channels,
                state['power_levels'][agent_idx] / 23,
                (state['sinr_values'][agent_idx] + 20) / 40,
            ]
            
            # Neighbor information (simplified)
            neighbor_info = []
            for i in range(min(3, self.env.num_d2d_pairs)):  # Consider up to 3 neighbors
                if i != agent_idx and i < len(state['sinr_values']):
                    neighbor_info.extend([
                        state['spectrum_allocation'][i] / self.num_channels,
                        (state['sinr_values'][i] + 20) / 40
                    ])
                else:
                    neighbor_info.extend([0, 0])  # Padding
            
            # Combine all information
            full_state = np.concatenate([own_embedding, local_info, neighbor_info])
            
            # Ensure consistent state size (pad or truncate to 38 dimensions)
            if len(full_state) < 38:
                full_state = np.pad(full_state, (0, 38 - len(full_state)))
            elif len(full_state) > 38:
                full_state = full_state[:38]
                
            return full_state
            
        except Exception as e:
            self.logger.error(f"Error getting agent state for agent {agent_idx}: {str(e)}")
            return np.zeros(38)  # Return zero state in case of error
    
    def train_episode(self) -> Tuple[float, int, float, float]:
        """
        Train for one episode.
        
        Returns:
            Tuple of (total_reward, steps, sum_rate, avg_sinr)
        """
        state = self.env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(100):  # Max steps per episode
            # Get GNN embeddings
            graph_data = self.create_graph_data(state)
            
            if graph_data['edge_index'].size(1) > 0:
                gnn_embeddings = self.gnn(
                    graph_data['x'], 
                    graph_data['edge_index'],
                    graph_data['edge_weight']
                )
            else:
                gnn_embeddings = torch.zeros(len(graph_data['x']), self.config['gnn']['output_dim'])
            
            # Get actions from all agents
            actions = []
            agent_states = []
            
            for i, agent in enumerate(self.agents):
                agent_state = self.get_agent_state(i, gnn_embeddings, state)
                agent_states.append(agent_state)
                action_idx = agent.act(agent_state)
                
                # Convert action index to (channel, power_level)
                channel = action_idx // self.num_power_levels
                power_level = action_idx % self.num_power_levels
                actions.append((channel, power_level))
            
            # Execute actions
            next_state, rewards = self.env.step(actions)
            total_reward += sum(rewards)
            
            # Get next GNN embeddings
            next_graph_data = self.create_graph_data(next_state)
            if next_graph_data['edge_index'].size(1) > 0:
                next_gnn_embeddings = self.gnn(
                    next_graph_data['x'], 
                    next_graph_data['edge_index'],
                    next_graph_data['edge_weight']
                )
            else:
                next_gnn_embeddings = torch.zeros(len(next_graph_data['x']), self.config['gnn']['output_dim'])
            
            # Store experiences and train agents
            for i, agent in enumerate(self.agents):
                next_agent_state = self.get_agent_state(i, next_gnn_embeddings, next_state)
                action_idx = actions[i][0] * self.num_power_levels + actions[i][1]
                
                agent.remember(agent_states[i], action_idx, rewards[i], next_agent_state, False)
                agent.replay()
            
            state = next_state
            steps += 1
            
            # Early stopping if converged
            if steps > 50 and abs(total_reward) < 0.1:
                break
        
        # Calculate final metrics
        sum_rate = self.env.calculate_sum_rate()
        avg_sinr = np.mean(state['sinr_values'])
        
        return total_reward, steps, sum_rate, avg_sinr
    
    def train(self, episodes: int =  1000) -> Dict:
        """
        Train the complete system.
        
        Args:
            episodes: Number of training episodes (overrides config if provided)
            
        Returns:
            Training history dictionary
        """
        try:
            if episodes is None:
                episodes = self.config['training']['episodes']
            
            self.logger.info(f"Starting GNN-MARL Training for {episodes} episodes")
            
            start_time = time.time()
            
            for episode in tqdm(range(episodes), desc="Training Progress"):
                episode_reward, steps, sum_rate, avg_sinr = self.train_episode()
                
                # Store training history
                self.training_history['episode'].append(episode)
                self.training_history['reward'].append(episode_reward)
                self.training_history['steps'].append(steps)
                self.training_history['sum_rate'].append(sum_rate)
                self.training_history['avg_sinr'].append(avg_sinr)
                self.training_history['epsilon'].append(self.agents[0].epsilon)
                
                print("Training History stored successfully")
                
                # Update target networks periodically
                if episode % self.config['training']['target_update_freq'] == 0:
                    for agent in self.agents:
                        agent.update_target_network()
                
                # Log progress
                if episode % 100 == 0:
                    self.logger.info(
                        f"Episode {episode:4d} | Sum Rate: {sum_rate:6.2f} Mbps | "
                        f"Avg SINR: {avg_sinr:5.1f} dB | Steps: {steps:3d} | "
                        f"Epsilon: {self.agents[0].epsilon:.3f}"
                    )
            
            training_time = time.time() - start_time
            self.logger.info(f"\nTraining Completed in {training_time:.1f} seconds!")
            
            # Save models if configured
            if self.config['logging']['save_model']:
                self.save_models()
            
            print("Training history: ", self.training_history)
            return self.training_history
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise
    
    def save_models(self):
        """Save trained models to disk"""
        try:
            model_dir = self.config['logging']['model_dir']
            os.makedirs(model_dir, exist_ok=True)
            
            # Save GNN
            torch.save(self.gnn.state_dict(), os.path.join(model_dir, 'gnn_model.pth'))
            
            # Save DQN agents
            for i, agent in enumerate(self.agents):
                torch.save(agent.q_network.state_dict(), os.path.join(model_dir, f'agent_{i}_q_network.pth'))
                torch.save(agent.target_network.state_dict(), os.path.join(model_dir, f'agent_{i}_target_network.pth'))
            
            self.logger.info(f"Models saved to {model_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}")
            raise
        
        
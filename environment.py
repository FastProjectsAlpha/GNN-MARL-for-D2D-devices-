import numpy as np
import yaml
from typing import Dict, List, Tuple
import logging
from dataclasses import dataclass
from scipy.spatial.distance import cdist
from sklearn.cluster import SpectralClustering
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.pyplot as plt


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class D2DPair:
    tx_position: Tuple[float, float]
    rx_position: Tuple[float, float]
    channel: int
    power_level: float

@dataclass
class CellularUser:
    position: Tuple[float, float]

class D2DEnvironment:
    """D2D Network Environment Simulator"""
    
    def __init__(self, config_path: str = 'configs/default_config.yaml'):
        """
        Initialize the D2D environment with configuration.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)['environment']
            
            self.area_size = self.config['area_size']
            self.num_d2d_pairs = self.config['num_d2d_pairs']
            self.num_cellular_users = self.config['num_cellular_users']
            self.num_channels = self.config['num_channels']
            self.max_power = self.config['max_power']
            self.noise_power = self.config['noise_power']
            self.bandwidth = self.config['bandwidth']
            
            # Initialize network
            self.d2d_pairs: List[D2DPair] = []
            self.cellular_users: List[CellularUser] = []
            self.reset()
            
            logger.info("D2D Environment initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize environment: {str(e)}")
            raise
    
    
    def get_state(self) -> Dict:
        """
        Get current network state.
        
        Returns:
            Dictionary containing:
            - positions: List of all node positions
            - spectrum_allocation: Current channel allocation
            - power_levels: Current power levels
            - sinr_values: Current SINR values for D2D pairs
        """
        positions = [(pair.tx_position, pair.rx_position) for pair in self.d2d_pairs] + \
                   [(user.position, user.position) for user in self.cellular_users]
        
        spectrum_allocation = [pair.channel for pair in self.d2d_pairs]
        power_levels = [pair.power_level for pair in self.d2d_pairs]
        sinr_values = [self.calculate_sinr(i) for i in range(self.num_d2d_pairs)]
        
        return {
            'positions': positions,
            'spectrum_allocation': spectrum_allocation,
            'power_levels': power_levels,
            'sinr_values': sinr_values
        }
        
    def reset(self) -> Dict:
        """Reset environment and generate new network topology"""
        try:
            # Clear existing network
            self.d2d_pairs = []
            self.cellular_users = []
            
            # Generate D2D pairs positions
            for _ in range(self.num_d2d_pairs):
                # Each D2D pair: transmitter and receiver within 50m
                tx_x, tx_y = np.random.uniform(0, self.area_size, 2)
                distance = np.random.uniform(10, 50)
                angle = np.random.uniform(0, 2*np.pi)
                rx_x = tx_x + distance * np.cos(angle)
                rx_y = tx_y + distance * np.sin(angle)
                
                # Keep receiver within area
                rx_x = np.clip(rx_x, 0, self.area_size)
                rx_y = np.clip(rx_y, 0, self.area_size)
                
                # Initialize with random channel and max power
                channel = np.random.randint(0, self.num_channels)
                power_level = self.max_power
                
                self.d2d_pairs.append(
                    D2DPair(
                        tx_position=(tx_x, tx_y),
                        rx_position=(rx_x, rx_y),
                        channel=channel,
                        power_level=power_level
                    )
                )
            
            # Generate cellular users positions
            for _ in range(self.num_cellular_users):
                x, y = np.random.uniform(0, self.area_size, 2)
                self.cellular_users.append(CellularUser(position=(x, y)))
            
            logger.debug("Environment reset with new network topology")
            return self.get_state()
            
        except Exception as e:
            logger.error(f"Error resetting environment: {str(e)}")
            raise

    def calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two positions"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def calculate_path_loss(self, distance: float) -> float:
        """
        Calculate path loss using urban macro model.
        
        Args:
            distance: Distance between transmitter and receiver in meters
            
        Returns:
            Path loss in dB
        """
        if distance < 1:
            distance = 1
        # Urban macro path loss model (3GPP)
        pl_db = 128.1 + 37.6 * np.log10(distance / 1000)  # distance in km
        return pl_db

    def calculate_sinr(self, d2d_idx: int):
        """
        Calculate SINR for a specific D2D pair.
        
        Args:
            d2d_idx: Index of the D2D pair
            
        Returns:
            SINR in dB
        """
        try:
            d2d_pair = self.d2d_pairs[d2d_idx]
            tx_pos = d2d_pair.tx_position
            rx_pos = d2d_pair.rx_position
            channel = d2d_pair.channel
            power_dbm = d2d_pair.power_level
            
            # Signal power
            distance = self.calculate_distance(tx_pos, rx_pos)
            path_loss = self.calculate_path_loss(distance)
            signal_power_dbm = power_dbm - path_loss
            
            # Interference calculation
            interference_power_dbm = -float('inf')
            
            # Interference from other D2D pairs on same channel
            for i, other_pair in enumerate(self.d2d_pairs):
                if i != d2d_idx and other_pair.channel == channel:
                    other_tx_pos = other_pair.tx_position
                    interference_distance = self.calculate_distance(other_tx_pos, rx_pos)
                    interference_path_loss = self.calculate_path_loss(interference_distance)
                    interference_power = other_pair.power_level - interference_path_loss
                    
                    if interference_power_dbm == -float('inf'):
                        interference_power_dbm = interference_power
                    else:
                        # Add interference powers (convert to linear, add, convert back)
                        interference_power_dbm = 10 * np.log10(
                            10**(interference_power_dbm/10) + 10**(interference_power/10)
                        )
            
            # Total interference + noise
            noise_power_dbm = self.noise_power + 10 * np.log10(self.bandwidth)
            
            if interference_power_dbm == -float('inf'):
                total_interference_dbm = noise_power_dbm
            else:
                total_interference_dbm = 10 * np.log10(
                    10**(interference_power_dbm/10) + 10**(noise_power_dbm/10))
            
            # SINR calculation
            sinr_db = signal_power_dbm - total_interference_dbm
            return max(sinr_db, -20)  # Minimum SINR threshold    
        except Exception as e:
            logger.error(f"Error calculating SINR for D2D pair {d2d_idx}: {str(e)}")
            return -20  # Return minimum SINR in case of error

    def calculate_sum_rate(self) -> float:
        """Calculate total sum rate of the network in Mbps"""
        sum_rate = 0
        for i in range(self.num_d2d_pairs):
            sinr_db = self.calculate_sinr(i)
            sinr_linear = 10**(sinr_db/10)
            rate = self.bandwidth * np.log2(1 + sinr_linear)
            sum_rate += rate
        return sum_rate / 1e6  # Convert to Mbps

     
    def step(self, actions: List[Tuple[int, int]]) -> Tuple[Dict, List[float]]:
        """
        Execute actions and return new state, rewards.
        
        Args:
            actions: List of (channel, power_level_idx) for each D2D pair
            
        Returns:
            Tuple of (new_state, rewards)
        """
        try:
            rewards = []
            old_sum_rate = self.calculate_sum_rate()
            
            # Available power levels (dBm)
            power_levels = [10, 15, 20, 23]
            
            for i, (channel, power_idx) in enumerate(actions):
                # Validate action
                if not (0 <= channel < self.num_channels):
                    raise ValueError(f"Invalid channel {channel}. Must be 0-{self.num_channels-1}")
                if not (0 <= power_idx < len(power_levels)):
                    raise ValueError(f"Invalid power index {power_idx}. Must be 0-{len(power_levels)-1}")
                
                # Update D2D pair configuration
                self.d2d_pairs[i].channel = channel
                self.d2d_pairs[i].power_level = power_levels[power_idx]
            
            new_sum_rate = self.calculate_sum_rate()
            
            # Calculate individual rewards
            for i in range(self.num_d2d_pairs):
                sinr = self.calculate_sinr(i)
                sinr_linear = 10**(sinr/10)
                rate = self.bandwidth * np.log2(1 + sinr_linear) / 1e6
                
                # Reward components
                rate_reward = rate / 10  # Normalize
                interference_penalty = -0.1 if sinr < 0 else 0
                
                reward = rate_reward + interference_penalty + (new_sum_rate - old_sum_rate) * 0.1
                rewards.append(reward)
            
            return self.get_state(), rewards
            
        except Exception as e:
            logger.error(f"Error in environment step: {str(e)}")
            raise

    def perform_interference_based_clustering(self, num_clusters: int = 3):
        """
        Cluster D2D pairs based on interference levels at their receivers.
        
        Args:
            num_clusters: Number of desired clusters
        """
        try:
            # Collect interference levels at each receiver
            interference_levels = []
            
            for i, pair in enumerate(self.d2d_pairs):
                total_interference_dbm = -float('inf')
                
                for j, other_pair in enumerate(self.d2d_pairs):
                    if i != j and other_pair.channel == pair.channel:
                        interference_distance = self.calculate_distance(other_pair.tx_position, pair.rx_position)
                        interference_path_loss = self.calculate_path_loss(interference_distance)
                        interference_power = other_pair.power_level - interference_path_loss
                        
                        if total_interference_dbm == -float('inf'):
                            total_interference_dbm = interference_power
                        else:
                            total_interference_dbm = 10 * np.log10(10**(total_interference_dbm/10) + 10**(interference_power/10))
                
                # Add noise if no interference
                noise_power_dbm = self.noise_power + 10 * np.log10(self.bandwidth)
                if total_interference_dbm == -float('inf'):
                    total_interference_dbm = noise_power_dbm
                else:
                    total_interference_dbm = 10 * np.log10(10**(total_interference_dbm/10) + 10**(noise_power_dbm/10))
                
                interference_levels.append([total_interference_dbm])
            
            interference_array = np.array(interference_levels)

            if len(interference_array) < num_clusters:
                raise ValueError("Number of clusters cannot exceed number of D2D pairs.")
            
            # Spectral Clustering on interference patterns
            clustering = SpectralClustering(
                n_clusters=num_clusters,
                affinity='nearest_neighbors',
                assign_labels='kmeans',
                random_state=42
            )
            
            cluster_labels = clustering.fit_predict(interference_array)

            # Assign cluster labels to D2D pairs
            for pair, label in zip(self.d2d_pairs, cluster_labels):
                pair.cluster = label
            
            print(f"Interference-based clustering completed. {num_clusters} clusters formed.")
        
        except Exception as e:
            logger.error(f"Interference-based clustering failed: {str(e)}")
            raise

    def visualize_clusters_with_circles(self):
        """
        Visualize D2D pairs colored by cluster with circles showing group boundaries.
        """
        
        # Unique clusters
        clusters = set(getattr(pair, 'cluster', 0) for pair in self.d2d_pairs)
        colors = cm.rainbow(np.linspace(0, 1, len(clusters)))

        for cluster_id, color in zip(clusters, colors):
            cluster_pairs = [pair for pair in self.d2d_pairs if getattr(pair, 'cluster', 0) == cluster_id]
            
            # Collect transmitter positions
            tx_positions = np.array([pair.tx_position for pair in cluster_pairs])
            
            # Plot pairs with color
            for pair in cluster_pairs:
                tx_x, tx_y = pair.tx_position
                rx_x, rx_y = pair.rx_position
                plt.scatter(tx_x, tx_y, color=color, marker='o', label=f'Cluster {cluster_id}' if f'Cluster {cluster_id}' not in plt.gca().get_legend_handles_labels()[1] else "")
                plt.scatter(rx_x, rx_y, color=color, marker='x')
                plt.plot([tx_x, rx_x], [tx_y, rx_y], color=color, linestyle='dotted')
            
            # Draw circle around cluster
            if len(tx_positions) > 0:
                center_x, center_y = np.mean(tx_positions, axis=0)
                max_distance = np.max(np.linalg.norm(tx_positions - [center_x, center_y], axis=1))
                circle = patches.Circle((center_x, center_y), max_distance + 10, edgecolor=color, facecolor='none', linestyle='dashed', linewidth=2)
                plt.gca().add_patch(circle)
        
        # Cellular users
        for user in self.cellular_users:
            x, y = user.position
            plt.scatter(x, y, c='black', marker='s', label='Cellular User' if 'Cellular User' not in plt.gca().get_legend_handles_labels()[1] else "")
        
        plt.legend()
        plt.grid(True)
        plt.show() 

if __name__ == "__main__":
    env = D2DEnvironment()
    env.perform_interference_based_clustering(num_clusters=3)
    env.visualize_clusters_with_circles()
   
        
        
def main():
    d2denvironment = D2DEnvironment()
   
    
if __name__ == "__main__":
    main()
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List
import logging
from matplotlib import cm, patches

logger = logging.getLogger(__name__)

def plot_training_results(training_history: Dict, save_dir: str):
    """
    Plot training results and save to file.
    
    Args:
        training_history: Dictionary containing training metrics
        save_dir: Directory to save plots
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(15, 10))
        
        # Plot sum rate
        plt.subplot(2, 2, 1)
        plt.plot(training_history['episode'], training_history['sum_rate'])
        plt.title('Training Sum Rate')
        plt.xlabel('Episode')
        plt.ylabel('Sum Rate (Mbps)')
        plt.grid(True)
        
        # Plot average SINR
        plt.subplot(2, 2, 2)
        plt.plot(training_history['episode'], training_history['avg_sinr'])
        plt.title('Training Average SINR')
        plt.xlabel('Episode')
        plt.ylabel('SINR (dB)')
        plt.grid(True)
        
        # Plot episode reward
        plt.subplot(2, 2, 3)
        plt.plot(training_history['episode'], training_history['reward'])
        plt.title('Episode Reward')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)
        
        # Plot exploration rate
        plt.subplot(2, 2, 4)
        plt.plot(training_history['episode'], training_history['epsilon'])
        plt.title('Exploration Rate')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_results.png'))
        plt.close()
        
        logger.info("Saved training plots")
    
    except Exception as e:
        logger.error(f"Error plotting training results: {str(e)}")

def plot_evaluation_results(eval_results: Dict, save_dir: str):
    """
    Plot evaluation results and save to file.
    
    Args:
        eval_results: Dictionary containing evaluation metrics
        save_dir: Directory to save plots
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(15, 10))
        
        # Sum rate distribution
        plt.subplot(2, 2, 1)
        plt.hist(eval_results['sum_rates'], bins=20, alpha=0.7)
        plt.title('Sum Rate Distribution')
        plt.xlabel('Sum Rate (Mbps)')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        # SINR distribution
        plt.subplot(2, 2, 2)
        plt.hist(eval_results['sinr_values'], bins=20, alpha=0.7)
        plt.title('SINR Distribution')
        plt.xlabel('SINR (dB)')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        # Spectral efficiency
        plt.subplot(2, 2, 3)
        plt.hist(eval_results['spectral_efficiency'], bins=20, alpha=0.7)
        plt.title('Spectral Efficiency Distribution')
        plt.xlabel('Mbps/MHz')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        # Convergence steps
        plt.subplot(2, 2, 4)
        plt.hist(eval_results['convergence_steps'], bins=20, alpha=0.7)
        plt.title('Convergence Steps Distribution')
        plt.xlabel('Steps to Converge')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'evaluation_results.png'))
        plt.close()
        
        logger.info("Saved evaluation plots")
    
    except Exception as e:
        logger.error(f"Error plotting evaluation results: {str(e)}")

def plot_sum_rate_vs_devices(device_counts: List[int], sum_rates: List[float], save_dir: str, label: str = 'Proposed', color: str = 'b'):
    """
    Plot Sum Rate against Number of D2D User Pairs and save to file, styled for clarity and publication.
    
    Args:
        device_counts: List of device counts (e.g., number of D2D pairs)
        sum_rates: Corresponding sum rates for each device count
        save_dir: Directory to save the plot
        label: Label for the curve (default 'Proposed')
        color: Color for the curve (default 'b')
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        plt.figure(figsize=(8, 6))
        plt.plot(device_counts, sum_rates, marker='o', markersize=8, linestyle='-', linewidth=2.5, color=color, label=label)
        plt.title('Sum Rate vs Number of D2D User Pairs', fontsize=16)
        plt.xlabel('Number of D2D User Pairs', fontsize=14)
        plt.ylabel('Sum Rate (Mbps)', fontsize=14)
        plt.xticks(np.arange(min(device_counts), max(device_counts)+1, 10), fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)
        plt.legend(fontsize=13, loc='best', frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'sum_rate_vs_devices.png'), dpi=300)
        plt.close()
        logger.info("Saved Sum Rate vs Number of D2D User Pairs plot")
    except Exception as e:
        logger.error(f"Error plotting Sum Rate vs Number of D2D User Pairs: {str(e)}")

def plot_clusters_with_circles(d2d_pairs, cellular_users, save_dir: str):
    """
    Visualize D2D pairs colored by cluster with circles showing group boundaries and save to file.
    
    Args:
        d2d_pairs: List of D2DPair objects (must have .tx_position, .rx_position, .cluster)
        cellular_users: List of CellularUser objects (must have .position)
        save_dir: Directory to save the plot
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        plt.figure(figsize=(10, 10))
        clusters = sorted(set(getattr(pair, 'cluster', 0) for pair in d2d_pairs))
        cmap = plt.cm.get_cmap('rainbow', len(clusters))
        colors = [cmap(i) for i in range(len(clusters))]
        for cluster_id, color in zip(clusters, colors):
            cluster_pairs = [pair for pair in d2d_pairs if getattr(pair, 'cluster', 0) == cluster_id]
            tx_positions = np.array([pair.tx_position for pair in cluster_pairs])
            for pair in cluster_pairs:
                tx_x, tx_y = pair.tx_position
                rx_x, rx_y = pair.rx_position
                plt.scatter(tx_x, tx_y, color=color, marker='o', label=f'Cluster {cluster_id}' if f'Cluster {cluster_id}' not in plt.gca().get_legend_handles_labels()[1] else "")
                plt.scatter(rx_x, rx_y, color=color, marker='x')
                plt.plot([tx_x, rx_x], [tx_y, rx_y], color=color, linestyle='dotted')
            if len(tx_positions) > 0:
                center_x, center_y = np.mean(tx_positions, axis=0)
                max_distance = np.max(np.linalg.norm(tx_positions - [center_x, center_y], axis=1))
                circle = patches.Circle((center_x, center_y), max_distance + 10, edgecolor=color, facecolor='none', linestyle='dashed', linewidth=2)
                plt.gca().add_patch(circle)
        for user in cellular_users:
            x, y = user.position
            plt.scatter(x, y, c='black', marker='s', label='Cellular User' if 'Cellular User' not in plt.gca().get_legend_handles_labels()[1] else "")
        plt.legend()
        plt.grid(True)
        plt.title('D2D Clusters with Circles')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'clusters.png'))
        plt.close()
        logger.info("Saved clusters plot as clusters.png")
    except Exception as e:
        logger.error(f"Error plotting clusters: {str(e)}")

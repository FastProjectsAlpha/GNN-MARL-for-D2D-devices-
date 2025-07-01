import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List
import logging

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
        




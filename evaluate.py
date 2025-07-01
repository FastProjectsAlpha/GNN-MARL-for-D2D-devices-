import numpy as np
from typing import Dict, List
import logging
from tqdm import tqdm
import torch
from utils.visualization import plot_evaluation_results

def evaluate_system(system, num_episodes: int = 100) -> Dict:
    """
    Evaluate the trained GNN-MARL system.
    
    Args:
        system: Trained GNNMARLSystem instance
        num_episodes: Number of evaluation episodes
        
    Returns:
        Dictionary containing evaluation results
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Evaluating system for {num_episodes} episodes...")
    
    results = {
        'sum_rates': [],
        'sinr_values': [],
        'spectral_efficiency': [],
        'convergence_steps': []
    }
    
    try:
        # Store original exploration rates
        original_epsilons = [agent.epsilon for agent in system.agents]
        
        # Set agents to evaluation mode (no exploration)
        for agent in system.agents:
            agent.epsilon = 0.0
        
        for _ in tqdm(range(num_episodes), desc="Evaluation Progress"):
            state = system.env.reset()
            steps = 0
            
            for _ in range(50):  # Max evaluation steps
                # Get GNN embeddings
                graph_data = system.create_graph_data(state)
                
                if graph_data['edge_index'].size(1) > 0:
                    gnn_embeddings = system.gnn(
                        graph_data['x'], 
                        graph_data['edge_index'],
                        graph_data['edge_weight']
                    )
                else:
                    gnn_embeddings = torch.zeros(len(graph_data['x']), system.config['gnn']['output_dim'])
                
                # Get actions from all agents
                actions = []
                for i, agent in enumerate(system.agents):
                    agent_state = system.get_agent_state(i, gnn_embeddings, state)
                    action_idx = agent.act(agent_state)
                    channel = action_idx // system.num_power_levels
                    power_level = action_idx % system.num_power_levels
                    actions.append((channel, power_level))
                
                # Execute actions
                state, _ = system.env.step(actions)
                steps += 1
            
            # Calculate final metrics
            sum_rate = system.env.calculate_sum_rate()
            sinr_values = [system.env.calculate_sinr(i) for i in range(system.env.num_d2d_pairs)]
            spectral_eff = sum_rate / (system.env.num_channels * 0.18)  # Mbps/MHz
            
            results['sum_rates'].append(sum_rate)
            results['sinr_values'].extend(sinr_values)
            results['spectral_efficiency'].append(spectral_eff)
            results['convergence_steps'].append(steps)
        
        # Restore exploration rates
        for i, agent in enumerate(system.agents):
            agent.epsilon = original_epsilons[i]
        
        logger.info("Evaluation completed successfully")
        return results
    
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise


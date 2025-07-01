import numpy as np
import torch
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)

def normalize_features(features: np.ndarray, 
                      feature_ranges: Dict[str, Tuple[float, float]]) -> np.ndarray:
    """
    Normalize features to [0, 1] range based on provided ranges.
    
    Args:
        features: Input features to normalize
        feature_ranges: Dictionary mapping feature names to (min, max) ranges
        
    Returns:
        Normalized features
    """
    try:
        normalized = np.zeros_like(features)
        for i, (name, (min_val, max_val)) in enumerate(feature_ranges.items()):
            if max_val == min_val:
                normalized[:, i] = 0.5  # Handle constant features
            else:
                normalized[:, i] = (features[:, i] - min_val) / (max_val - min_val)
        return normalized
    except Exception as e:
        logger.error(f"Error normalizing features: {str(e)}")
        return features

def create_mini_batches(data, batch_size: int) -> List[List[Dict]]:
    """
    Create mini-batches from training data.
    
    Args:
        data: List of training samples
        batch_size: Size of each mini-batch
        
    Returns:
        List of mini-batches
    """
    try:
        np.random.shuffle(data)
        batches = []
        for i in range(0, len(data), batch_size):
            batches.append(data[i:i + batch_size])
        return batches
    except Exception as e:
        logger.error(f"Error creating mini-batches: {str(e)}")
        return [data]

def calculate_edge_weights(positions: List[Tuple[float, float]], 
                          max_distance: float = 200.0) -> np.ndarray:
    """
    Calculate edge weights based on inverse distance.
    
    Args:
        positions: List of node positions
        max_distance: Maximum distance to consider for edges
        
    Returns:
        Array of edge weights
    """
    try:
        n_nodes = len(positions)
        weights = np.zeros((n_nodes, n_nodes))
        
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                distance = np.sqrt((positions[i][0] - positions[j][0])**2 + 
                           (positions[i][1] - positions[j][1])**2)
                if distance < max_distance:
                    weight = 1 / (1 + distance / 100)  # Inverse distance weighting
                    weights[i, j] = weight
                    weights[j, i] = weight
        
        return weights
    except Exception as e:
        logger.error(f"Error calculating edge weights: {str(e)}")
        return np.zeros((len(positions), len(positions))) 
    
    if __name__ -- "__main__":
        print("Data processing module loaded successfully!")
        print("Available functions:", [name for name in globals() if callable(globals()[name]) and not name.startswith('_')])
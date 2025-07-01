import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from typing import Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphNeuralNetwork(nn.Module):
    """Graph Neural Network for D2D interference modeling"""
    
    def __init__(self, 
                 input_dim: int = 6, 
                 hidden_dim: int = 64, 
                 output_dim: int = 32,
                 dropout: float = 0.2):
        """
        Initialize the GNN model.
        
        Args:
            input_dim: Dimension of input node features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output embeddings
            dropout: Dropout rate
        """
        super(GraphNeuralNetwork, self).__init__()
        
        try:
            # Graph convolutional layers
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            
            # Graph attention layer
            self.attention = GATConv(hidden_dim, output_dim, heads=4, concat=False)
            
            # Dropout layer
            self.dropout = nn.Dropout(dropout)
            
            logger.info("GNN model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize GNN model: {str(e)}")
            raise
    
    def forward(self, 
                x: torch.Tensor, 
                edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the GNN.
        
        Args:
            x: Node feature matrix [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_weight: Optional edge weights [num_edges]
            
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        try:
            # First GCN layer
            x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
            x = self.dropout(x)
            
            # Second GCN layer
            x = F.relu(self.conv2(x, edge_index, edge_weight=edge_weight))
            x = self.dropout(x)
            
            # Attention layer
            x = self.attention(x, edge_index, edge_attr=edge_weight.unsqueeze(1) if edge_weight is not None else edge_weight)
            
            return x
            
        except Exception as e:
            logger.error(f"Error in GNN forward pass: {str(e)}")
            raise
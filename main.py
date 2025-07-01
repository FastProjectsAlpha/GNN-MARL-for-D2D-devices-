import argparse
import yaml
import logging
from train import GNNMARLSystem
from evaluate import evaluate_system
from utils.visualization import plot_training_results, plot_evaluation_results
from utils.logger import setup_logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='GNN-MARL for D2D Spectrum Allocation')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--train', action='store_true',
                       help='Run training')
    parser.add_argument('--eval', action='store_true',
                       help='Run evaluation')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of training episodes (overrides config)')
    parser.add_argument('--eval-episodes', type=int, default=100,
                       help='Number of evaluation episodes')
    
    return parser.parse_args()

def main():
    """Main application entry point"""
    args = parse_args()
    
    try:
        # Load configuration
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
  
        logger.info("Starting GNN-MARL D2D Spectrum Allocation System")
        
        # Initialize system
        system = GNNMARLSystem(args.config)
        
        if args.train:
            # Training mode
            logger.info("Starting training process...")
            training_history = system.train(episodes=args.episodes)

            
            # Plot training results
            plot_training_results(training_history, config['logging']['log_dir'])
        
        if args.eval:
            # Evaluation mode
            logger.info("Starting evaluation process...")
            eval_results = evaluate_system(system, args.eval_episodes)
            
            # Plot evaluation results
            plot_evaluation_results(eval_results, config['logging']['log_dir'])
            
            # Print summary
            print("\nEvaluation Summary:")
            print(f"Average Sum Rate: {np.mean(eval_results['sum_rates']):.2f} Mbps")
            print(f"Average SINR: {np.mean(eval_results['sinr_values']):.1f} dB")
        
        logger.info("Application completed successfully")
    
    except Exception as e:
        logger.error(f"")
        logger.error(f"Application error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
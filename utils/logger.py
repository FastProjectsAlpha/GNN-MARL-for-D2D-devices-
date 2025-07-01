import logging
import os
from typing import Optional
import sys
from datetime import datetime

def setup_logging(log_dir: Optional[str] = None, 
                 log_level: str = "INFO",
                 console: bool = True):
    """
    Configure logging system.
    
    Args:
        log_dir: Directory to save log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console: Whether to log to console
    """
    # Create log directory if specified
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    else:
        log_file = None
    
    # Configure logging
    log_level = getattr(logging, log_level.upper())
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[]
    )
    
    # Add file handler if log directory specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)
    
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(console_handler)
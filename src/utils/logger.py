import logging
import sys
from src.config import LOG_FILE_PATH, LOGS_DIR

def setup_logger(name="movie_rec"):
    """
    Sets up a logger that logs to both the console (INFO level) and a log file (DEBUG level).
    
    Args:
        name (str): The name of the logger. Default is 'movie_rec'.
        
    Returns:
        logging.Logger: The configured Logger instance.
    """
    # Ensure logs directory exists
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Prevent duplicate handlers
    if not logger.handlers:
        # Create formatter for console (simple and clean)
        console_formatter = logging.Formatter(
            "[%(levelname)s] %(asctime)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Create formatter for log file (detailed)
        file_formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File Handler
        try:
            file_handler = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            # Fallback if log file cannot be opened/created
            print(f"Warning: Failed to set up file logging: {e}", file=sys.stderr)
            
    return logger

def get_logger(name="movie_rec"):
    """
    Returns the configured logger. If it has not been configured, sets it up.
    
    Args:
        name (str): The name of the logger.
        
    Returns:
        logging.Logger: The configured Logger instance.
    """
    return setup_logger(name)

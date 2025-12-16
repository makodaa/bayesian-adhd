import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

# Create logs directory if it doesn't exist
LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Log file paths
APP_LOG_FILE = LOG_DIR / "app.log"
DB_LOG_FILE = LOG_DIR / "database.log"
ML_LOG_FILE = LOG_DIR / "ml.log"

# Logging format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(name: str, log_file: Path = None, level=logging.INFO) -> logging.Logger:
    """
    Setup a logger with both file and console handlers.
    
    Args:
        name: Logger name (typically __name__ of the module)
        log_file: Path to log file (optional)
        level: Logging level (default: INFO)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    
    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation (if log_file provided)
    if log_file:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_app_logger(name: str) -> logging.Logger:
    """Get logger for general application code."""
    return setup_logger(name, APP_LOG_FILE)


def get_db_logger(name: str) -> logging.Logger:
    """Get logger for database operations."""
    return setup_logger(name, DB_LOG_FILE)


def get_ml_logger(name: str) -> logging.Logger:
    """Get logger for machine learning operations."""
    return setup_logger(name, ML_LOG_FILE)

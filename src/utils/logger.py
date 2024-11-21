import logging
import os
from datetime import datetime
from ..utils.config import Config

def setup_logger(name, log_file=None, level=logging.INFO):
    """
    args:
        name: name of the logger
        log_file: path to log file
        level: logging level
    """

    os.makedirs(Config.LOG_PATH, exist_ok=True)
    
    #creating logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    if log_file is None:
        log_file = os.path.join(
            Config.LOG_PATH, 
            f'{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    
    #log formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

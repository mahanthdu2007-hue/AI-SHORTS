import logging
import sys


class CheckmarkFormatter(logging.Formatter):
    """Custom formatter to display clean, checkmark-style logs like `✓ Message`."""
    
    def format(self, record: logging.LogRecord) -> str:
        if record.levelno >= logging.ERROR:
            prefix = "✗"
        elif record.levelno >= logging.WARNING:
            prefix = "!"
        else:
            prefix = "✓"
            
        return f"{prefix} {record.getMessage()}"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(CheckmarkFormatter())
        # Make sure to handle unicode on windows if needed, though main.py already reconfigures stdout.
        logger.addHandler(handler)
        logger.propagate = False
    return logger


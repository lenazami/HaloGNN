# utils / logger.py
'''Implements customized formatting for logs.'''
import sys
from enum import Enum

class LogLevel(Enum):
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    
class Logger:
    '''logger that behaves like print with debug levels.'''
    def __init__(self, name: str = "", level: LogLevel = LogLevel.INFO):
        self.name = name
        self.level = level
        self.colors = {
            LogLevel.DEBUG: "\033[90m",    # Gray
            LogLevel.INFO: "\033[0m",      # Default
            LogLevel.WARNING: "\033[93m",  # Yellow
            LogLevel.ERROR: "\033[91m"     # Red
        }
        self.reset = "\033[0m"
    
    def _log(self, level: LogLevel, *args, **kwargs):
        if level.value >= self.level.value:
            prefix = f"[{self.name}] " if self.name else ""
            color = self.colors.get(level, "")
            print(f"{color}{prefix}", *args, f"{self.reset}", **kwargs)
    
    def debug(self, *args, **kwargs):
        self._log(LogLevel.DEBUG, *args, **kwargs)
    
    def info(self, *args, **kwargs):
        self._log(LogLevel.INFO, *args, **kwargs)
    
    def warning(self, *args, **kwargs):
        self._log(LogLevel.WARNING, *args, **kwargs)
    
    def error(self, *args, **kwargs):
        self._log(LogLevel.ERROR, *args, **kwargs)
        
# global logger instance
_logger = Logger(level=LogLevel.INFO)

# qol functions
def set_verbosity(verbose: bool):
    """Set global verbosity level."""
    _logger.level = LogLevel.DEBUG if verbose else LogLevel.INFO

def debug(*args, **kwargs):
    _logger.debug(*args, **kwargs)

def info(*args, **kwargs):
    _logger.info(*args, **kwargs)

def warning(*args, **kwargs):
    _logger.warning(*args, **kwargs)

def error(*args, **kwargs):
    _logger.error(*args, **kwargs)
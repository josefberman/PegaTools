"""
Pega Tools - A collection of tools for working with Pega systems.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .core import PegaClient
from .utils import PegaException

__all__ = ["PegaClient", "PegaException"] 
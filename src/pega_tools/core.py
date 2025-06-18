"""
Core functionality for Pega Tools.
"""

import requests
from typing import Optional, Dict, Any
from .utils import PegaException


class PegaClient:
    """
    Main client for interacting with Pega systems.
    """
    
    def __init__(self, base_url: str, username: Optional[str] = None, 
                 password: Optional[str] = None, timeout: int = 30):
        """
        Initialize the Pega client.
        
        Args:
            base_url: Base URL of the Pega instance
            username: Username for authentication
            password: Password for authentication
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.username = username
        self.password = password
        self.timeout = timeout
        self.session = requests.Session()
        
        if username and password:
            self.session.auth = (username, password)
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get the health status of the Pega instance.
        
        Returns:
            Dict containing health status information
            
        Raises:
            PegaException: If the request fails
        """
        try:
            response = self.session.get(
                f"{self.base_url}/api/health",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise PegaException(f"Failed to get health status: {str(e)}")
    
    def close(self):
        """Close the client session."""
        self.session.close() 
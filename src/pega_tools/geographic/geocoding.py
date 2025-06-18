"""
Geocoding functionality for address to coordinate conversion.
"""

from typing import Optional, Dict, Any
import requests
from .coordinates import Coordinate
from ..utils import PegaException


class Geocoder:
    """Geocoding service for converting addresses to coordinates."""
    
    def __init__(self, api_key: Optional[str] = None, service: str = "nominatim"):
        """
        Initialize geocoder.
        
        Args:
            api_key: API key for geocoding service
            service: Geocoding service to use ('nominatim', 'google', etc.)
        """
        self.api_key = api_key
        self.service = service.lower()
        self.session = requests.Session()
        
        # Set user agent for Nominatim (required)
        if self.service == "nominatim":
            self.session.headers.update({
                'User-Agent': 'pega-tools/1.0 (https://github.com/yourname/pega-tools)'
            })
    
    def geocode(self, address: str) -> Optional[Coordinate]:
        """
        Convert address to coordinates.
        
        Args:
            address: Address to geocode
            
        Returns:
            Coordinate object or None if not found
            
        Raises:
            PegaException: If geocoding fails
        """
        try:
            if self.service == "nominatim":
                return self._geocode_nominatim(address)
            else:
                raise PegaException(f"Unsupported geocoding service: {self.service}")
        except Exception as e:
            raise PegaException(f"Geocoding failed: {str(e)}")
    
    def reverse_geocode(self, coordinate: Coordinate) -> Optional[str]:
        """
        Convert coordinates to address.
        
        Args:
            coordinate: Coordinate to reverse geocode
            
        Returns:
            Address string or None if not found
        """
        try:
            if self.service == "nominatim":
                return self._reverse_geocode_nominatim(coordinate)
            else:
                raise PegaException(f"Unsupported geocoding service: {self.service}")
        except Exception as e:
            raise PegaException(f"Reverse geocoding failed: {str(e)}")
    
    def _geocode_nominatim(self, address: str) -> Optional[Coordinate]:
        """Geocode using Nominatim service."""
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            'q': address,
            'format': 'json',
            'limit': 1
        }
        
        response = self.session.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if data:
            result = data[0]
            return Coordinate(float(result['lat']), float(result['lon']))
        return None
    
    def _reverse_geocode_nominatim(self, coordinate: Coordinate) -> Optional[str]:
        """Reverse geocode using Nominatim service."""
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {
            'lat': coordinate.latitude,
            'lon': coordinate.longitude,
            'format': 'json'
        }
        
        response = self.session.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        return data.get('display_name')
    
    def close(self):
        """Close the session."""
        self.session.close() 
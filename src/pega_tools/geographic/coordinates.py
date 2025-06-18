"""
Coordinate handling and distance calculations.
"""

import math
from typing import Tuple, Union
from dataclasses import dataclass


@dataclass
class Coordinate:
    """Represents a geographic coordinate with latitude and longitude."""
    
    latitude: float
    longitude: float
    
    def __post_init__(self):
        """Validate coordinate values."""
        if not -90 <= self.latitude <= 90:
            raise ValueError("Latitude must be between -90 and 90 degrees")
        if not -180 <= self.longitude <= 180:
            raise ValueError("Longitude must be between -180 and 180 degrees")
    
    def to_tuple(self) -> Tuple[float, float]:
        """Convert to (latitude, longitude) tuple."""
        return (self.latitude, self.longitude)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {"latitude": self.latitude, "longitude": self.longitude}
    
    def __str__(self) -> str:
        return f"({self.latitude}, {self.longitude})"


def distance_between(coord1: Union[Coordinate, Tuple[float, float]], 
                    coord2: Union[Coordinate, Tuple[float, float]], 
                    unit: str = "km") -> float:
    """
    Calculate the great circle distance between two points on Earth.
    
    Args:
        coord1: First coordinate
        coord2: Second coordinate  
        unit: Distance unit ('km', 'miles', 'nautical_miles')
        
    Returns:
        Distance between coordinates
    """
    # Convert to Coordinate objects if needed
    if isinstance(coord1, tuple):
        coord1 = Coordinate(coord1[0], coord1[1])
    if isinstance(coord2, tuple):
        coord2 = Coordinate(coord2[0], coord2[1])
    
    # Convert to radians
    lat1, lon1 = math.radians(coord1.latitude), math.radians(coord1.longitude)
    lat2, lon2 = math.radians(coord2.latitude), math.radians(coord2.longitude)
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Earth radius in kilometers
    earth_radius_km = 6371.0
    distance_km = earth_radius_km * c
    
    # Convert to requested unit
    if unit == "km":
        return distance_km
    elif unit == "miles":
        return distance_km * 0.621371
    elif unit == "nautical_miles":
        return distance_km * 0.539957
    else:
        raise ValueError("Unit must be 'km', 'miles', or 'nautical_miles'") 
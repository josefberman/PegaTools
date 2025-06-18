"""
Geographic utility functions.
"""

from typing import Union, Tuple
from .coordinates import Coordinate


def validate_coordinates(lat: float, lon: float) -> bool:
    """
    Validate latitude and longitude values.
    
    Args:
        lat: Latitude value
        lon: Longitude value
        
    Returns:
        True if coordinates are valid
    """
    return -90 <= lat <= 90 and -180 <= lon <= 180


def format_coordinates(coord: Union[Coordinate, Tuple[float, float]], 
                      format_type: str = "decimal") -> str:
    """
    Format coordinates in different formats.
    
    Args:
        coord: Coordinate to format
        format_type: Format type ('decimal', 'dms', 'dm')
        
    Returns:
        Formatted coordinate string
    """
    if isinstance(coord, tuple):
        coord = Coordinate(coord[0], coord[1])
    
    if format_type == "decimal":
        return f"{coord.latitude:.6f}, {coord.longitude:.6f}"
    elif format_type == "dms":
        return f"{_to_dms(coord.latitude, 'lat')}, {_to_dms(coord.longitude, 'lon')}"
    elif format_type == "dm":
        return f"{_to_dm(coord.latitude, 'lat')}, {_to_dm(coord.longitude, 'lon')}"
    else:
        raise ValueError("format_type must be 'decimal', 'dms', or 'dm'")


def _to_dms(decimal_degrees: float, coord_type: str) -> str:
    """Convert decimal degrees to degrees, minutes, seconds format."""
    abs_dd = abs(decimal_degrees)
    degrees = int(abs_dd)
    minutes = int((abs_dd - degrees) * 60)
    seconds = ((abs_dd - degrees) * 60 - minutes) * 60
    
    if coord_type == 'lat':
        direction = 'N' if decimal_degrees >= 0 else 'S'
    else:
        direction = 'E' if decimal_degrees >= 0 else 'W'
    
    return f"{degrees}°{minutes}'{seconds:.2f}\"{direction}"


def _to_dm(decimal_degrees: float, coord_type: str) -> str:
    """Convert decimal degrees to degrees, decimal minutes format."""
    abs_dd = abs(decimal_degrees)
    degrees = int(abs_dd)
    minutes = (abs_dd - degrees) * 60
    
    if coord_type == 'lat':
        direction = 'N' if decimal_degrees >= 0 else 'S'
    else:
        direction = 'E' if decimal_degrees >= 0 else 'W'
    
    return f"{degrees}°{minutes:.4f}'{direction}" 
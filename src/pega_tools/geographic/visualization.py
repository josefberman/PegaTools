"""
Geographic visualization utilities.

This module provides functions for visualizing geographic data including
scatter plots, 2D histograms, and gradient visualizations.
"""

import numpy as np
from typing import List, Union, Optional, Dict, Any, Tuple
from datetime import datetime
import warnings

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.colors import LinearSegmentedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("matplotlib is required for geographic visualization. Install with: pip install pega-tools[viz]")

from .coordinates import Coordinate


def _check_matplotlib():
    """Check if matplotlib is available."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib is required for geographic visualization. "
            "Install with: pip install pega-tools[viz]"
        )


def _extract_coordinates(coordinates: List[Union[Coordinate, Tuple[float, float]]]) -> Tuple[List[float], List[float]]:
    """Extract latitude and longitude lists from coordinate objects."""
    lats, lons = [], []
    
    for coord in coordinates:
        if isinstance(coord, Coordinate):
            lats.append(coord.latitude)
            lons.append(coord.longitude)
        elif isinstance(coord, (tuple, list)) and len(coord) >= 2:
            lats.append(float(coord[0]))
            lons.append(float(coord[1]))
        else:
            raise ValueError(f"Invalid coordinate format: {coord}")
    
    return lats, lons


def plot_geographic_scatter(
    coordinates: List[Union[Coordinate, Tuple[float, float]]],
    values: Optional[List[float]] = None,
    title: str = "Geographic Scatter Plot",
    figsize: Tuple[int, int] = (12, 8),
    marker_size: Union[int, List[int]] = 50,
    color_map: str = "viridis",
    alpha: float = 0.7,
    show_colorbar: bool = True,
    colorbar_label: str = "Value",
    background_map: bool = False,
    **kwargs
) -> plt.Figure:
    """
    Create a scatter plot of geographic coordinates.
    
    Args:
        coordinates: List of Coordinate objects or (lat, lon) tuples
        values: Optional list of values for color mapping
        title: Plot title
        figsize: Figure size (width, height)
        marker_size: Size of scatter points (single value or list)
        color_map: Matplotlib colormap name
        alpha: Transparency of points
        show_colorbar: Whether to show colorbar
        colorbar_label: Label for colorbar
        background_map: Whether to add a simple background map (basic implementation)
        **kwargs: Additional matplotlib scatter plot parameters
        
    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()
    
    lats, lons = _extract_coordinates(coordinates)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up scatter plot parameters
    scatter_kwargs = {
        'alpha': alpha,
        'cmap': color_map,
        **kwargs
    }
    
    if values is not None:
        scatter = ax.scatter(lons, lats, c=values, s=marker_size, **scatter_kwargs)
        if show_colorbar:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(colorbar_label)
    else:
        ax.scatter(lons, lats, s=marker_size, **scatter_kwargs)
    
    # Basic background map (simple grid)
    if background_map:
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')
    
    return fig


def plot_geographic_histogram_2d(
    coordinates: List[Union[Coordinate, Tuple[float, float]]],
    bins: Union[int, Tuple[int, int]] = 50,
    title: str = "Geographic 2D Histogram",
    figsize: Tuple[int, int] = (12, 8),
    color_map: str = "viridis",
    log_scale: bool = False,
    show_colorbar: bool = True,
    colorbar_label: str = "Count",
    background_map: bool = False,
    **kwargs
) -> plt.Figure:
    """
    Create a 2D histogram of geographic coordinates.
    
    Args:
        coordinates: List of Coordinate objects or (lat, lon) tuples
        bins: Number of bins (single value or (lat_bins, lon_bins))
        title: Plot title
        figsize: Figure size (width, height)
        color_map: Matplotlib colormap name
        log_scale: Whether to use logarithmic color scale
        show_colorbar: Whether to show colorbar
        colorbar_label: Label for colorbar
        background_map: Whether to add a simple background map
        **kwargs: Additional matplotlib hist2d parameters
        
    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()
    
    lats, lons = _extract_coordinates(coordinates)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up histogram parameters
    hist_kwargs = {
        'cmap': color_map,
        'norm': mcolors.LogNorm() if log_scale else None,
        **kwargs
    }
    
    # Create 2D histogram
    hist, xedges, yedges, im = ax.hist2d(
        lons, lats, bins=bins, **hist_kwargs
    )
    
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(colorbar_label)
    
    # Basic background map
    if background_map:
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')
    
    return fig


def plot_temporal_gradient_histogram(
    coordinates: List[Union[Coordinate, Tuple[float, float]]],
    timestamps: List[Union[datetime, str]],
    bins: Union[int, Tuple[int, int]] = 50,
    time_bins: int = 10,
    title: str = "Temporal Gradient 2D Histogram",
    figsize: Tuple[int, int] = (15, 10),
    color_map: str = "plasma",
    log_scale: bool = False,
    show_colorbar: bool = True,
    colorbar_label: str = "Temporal Density",
    background_map: bool = False,
    **kwargs
) -> plt.Figure:
    """
    Create a 2D histogram showing temporal gradient of geographic points.
    
    Args:
        coordinates: List of Coordinate objects or (lat, lon) tuples
        timestamps: List of datetime objects or timestamp strings
        bins: Number of spatial bins (single value or (lat_bins, lon_bins))
        time_bins: Number of time bins for temporal analysis
        title: Plot title
        figsize: Figure size (width, height)
        color_map: Matplotlib colormap name
        log_scale: Whether to use logarithmic color scale
        show_colorbar: Whether to show colorbar
        colorbar_label: Label for colorbar
        background_map: Whether to add a simple background map
        **kwargs: Additional matplotlib parameters
        
    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()
    
    if len(coordinates) != len(timestamps):
        raise ValueError("Coordinates and timestamps must have the same length")
    
    lats, lons = _extract_coordinates(coordinates)
    
    # Convert timestamps to datetime objects if needed
    dt_timestamps = []
    for ts in timestamps:
        if isinstance(ts, str):
            dt_timestamps.append(datetime.fromisoformat(ts.replace('Z', '+00:00')))
        elif isinstance(ts, datetime):
            dt_timestamps.append(ts)
        else:
            raise ValueError(f"Invalid timestamp format: {ts}")
    
    # Convert to numerical values for analysis
    time_values = np.array([ts.timestamp() for ts in dt_timestamps])
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Spatial distribution with temporal coloring
    scatter_kwargs = {
        'cmap': color_map,
        'norm': mcolors.LogNorm() if log_scale else None,
        'alpha': 0.7,
        **kwargs
    }
    
    scatter = ax1.scatter(lons, lats, c=time_values, s=30, **scatter_kwargs)
    if show_colorbar:
        cbar1 = plt.colorbar(scatter, ax=ax1)
        cbar1.set_label("Timestamp")
    
    ax1.set_title("Spatial Distribution (Colored by Time)")
    if background_map:
        ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.set_aspect('equal', adjustable='box')
    
    # Plot 2: Temporal gradient histogram
    # Create time-based weights for histogram
    time_weights = np.ones_like(time_values)
    
    hist_kwargs = {
        'cmap': color_map,
        'norm': mcolors.LogNorm() if log_scale else None,
        **kwargs
    }
    
    hist, xedges, yedges, im = ax2.hist2d(
        lons, lats, bins=bins, weights=time_weights, **hist_kwargs
    )
    
    if show_colorbar:
        cbar2 = plt.colorbar(im, ax=ax2)
        cbar2.set_label(colorbar_label)
    
    ax2.set_title("Temporal Gradient Density")
    if background_map:
        ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.set_aspect('equal', adjustable='box')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    return fig


def plot_spatial_gradient_histogram(
    coordinates: List[Union[Coordinate, Tuple[float, float]]],
    values: List[float],
    bins: Union[int, Tuple[int, int]] = 50,
    title: str = "Spatial Gradient 2D Histogram",
    figsize: Tuple[int, int] = (15, 10),
    color_map: str = "coolwarm",
    log_scale: bool = False,
    show_colorbar: bool = True,
    colorbar_label: str = "Value",
    background_map: bool = False,
    gradient_method: str = "density",
    **kwargs
) -> plt.Figure:
    """
    Create a 2D histogram showing spatial gradient of geographic points.
    
    Args:
        coordinates: List of Coordinate objects or (lat, lon) tuples
        values: List of values associated with each coordinate
        bins: Number of bins (single value or (lat_bins, lon_bins))
        title: Plot title
        figsize: Figure size (width, height)
        color_map: Matplotlib colormap name
        log_scale: Whether to use logarithmic color scale
        show_colorbar: Whether to show colorbar
        colorbar_label: Label for colorbar
        background_map: Whether to add a simple background map
        gradient_method: Method for gradient calculation ('density' or 'interpolation')
        **kwargs: Additional matplotlib parameters
        
    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()
    
    if len(coordinates) != len(values):
        raise ValueError("Coordinates and values must have the same length")
    
    lats, lons = _extract_coordinates(coordinates)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Scatter plot with value coloring
    scatter_kwargs = {
        'cmap': color_map,
        'norm': mcolors.LogNorm() if log_scale else None,
        'alpha': 0.7,
        **kwargs
    }
    
    scatter = ax1.scatter(lons, lats, c=values, s=50, **scatter_kwargs)
    if show_colorbar:
        cbar1 = plt.colorbar(scatter, ax=ax1)
        cbar1.set_label(colorbar_label)
    
    ax1.set_title("Spatial Distribution (Colored by Value)")
    if background_map:
        ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.set_aspect('equal', adjustable='box')
    
    # Plot 2: Spatial gradient histogram
    if gradient_method == "density":
        # Use weighted histogram based on values
        hist_kwargs = {
            'cmap': color_map,
            'norm': mcolors.LogNorm() if log_scale else None,
            **kwargs
        }
        
        hist, xedges, yedges, im = ax2.hist2d(
            lons, lats, bins=bins, weights=values, **hist_kwargs
        )
        
    elif gradient_method == "interpolation":
        # Simple interpolation-based approach
        from scipy.interpolate import griddata
        
        # Create regular grid
        lon_min, lon_max = min(lons), max(lons)
        lat_min, lat_max = min(lats), max(lats)
        
        if isinstance(bins, int):
            lon_bins = lat_bins = bins
        else:
            lon_bins, lat_bins = bins
        
        lon_grid = np.linspace(lon_min, lon_max, lon_bins)
        lat_grid = np.linspace(lat_min, lat_max, lat_bins)
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        
        # Interpolate values
        points = np.column_stack((lons, lats))
        grid_values = griddata(points, values, (lon_mesh, lat_mesh), method='linear')
        
        # Plot interpolated surface
        im = ax2.pcolormesh(lon_mesh, lat_mesh, grid_values, 
                           cmap=color_map, shading='auto', **kwargs)
        
    else:
        raise ValueError("gradient_method must be 'density' or 'interpolation'")
    
    if show_colorbar:
        cbar2 = plt.colorbar(im, ax=ax2)
        cbar2.set_label(colorbar_label)
    
    ax2.set_title(f"Spatial Gradient ({gradient_method.title()})")
    if background_map:
        ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.set_aspect('equal', adjustable='box')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    return fig


def create_geographic_animation(
    coordinates: List[Union[Coordinate, Tuple[float, float]]],
    timestamps: List[Union[datetime, str]],
    values: Optional[List[float]] = None,
    output_file: str = "geographic_animation.gif",
    duration: int = 10,
    figsize: Tuple[int, int] = (12, 8),
    **kwargs
) -> str:
    """
    Create an animated visualization of geographic data over time.
    
    Args:
        coordinates: List of Coordinate objects or (lat, lon) tuples
        timestamps: List of datetime objects or timestamp strings
        values: Optional list of values for color mapping
        output_file: Output file path for the animation
        duration: Animation duration in seconds
        figsize: Figure size (width, height)
        **kwargs: Additional animation parameters
        
    Returns:
        Path to the created animation file
    """
    _check_matplotlib()
    
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError:
        raise ImportError("matplotlib.animation is required for animations")
    
    if len(coordinates) != len(timestamps):
        raise ValueError("Coordinates and timestamps must have the same length")
    
    lats, lons = _extract_coordinates(coordinates)
    
    # Convert timestamps
    dt_timestamps = []
    for ts in timestamps:
        if isinstance(ts, str):
            dt_timestamps.append(datetime.fromisoformat(ts.replace('Z', '+00:00')))
        elif isinstance(ts, datetime):
            dt_timestamps.append(ts)
        else:
            raise ValueError(f"Invalid timestamp format: {ts}")
    
    # Sort by timestamp
    sorted_data = sorted(zip(lats, lons, dt_timestamps, values or [0]*len(lats)), 
                        key=lambda x: x[2])
    lats, lons, dt_timestamps, values = zip(*sorted_data)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Initial plot
    if values:
        scatter = ax.scatter([], [], c=[], cmap='viridis', alpha=0.7, **kwargs)
    else:
        scatter = ax.scatter([], [], alpha=0.7, **kwargs)
    
    ax.set_xlim(min(lons) - 0.01, max(lons) + 0.01)
    ax.set_ylim(min(lats) - 0.01, max(lats) + 0.01)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Geographic Animation")
    ax.grid(True, alpha=0.3)
    
    def animate(frame):
        # Show data up to current frame
        idx = int(frame * len(lats) / 100)
        current_lats = lats[:idx+1]
        current_lons = lons[:idx+1]
        current_values = values[:idx+1] if values else None
        
        if current_values:
            scatter.set_offsets(np.column_stack((current_lons, current_lats)))
            scatter.set_array(current_values)
        else:
            scatter.set_offsets(np.column_stack((current_lons, current_lats)))
        
        ax.set_title(f"Geographic Animation - {dt_timestamps[idx] if idx < len(dt_timestamps) else dt_timestamps[-1]}")
        return scatter,
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=100, interval=duration*10, blit=True)
    
    # Save animation
    writer = PillowWriter(fps=10)
    anim.save(output_file, writer=writer)
    
    plt.close(fig)
    
    return output_file 
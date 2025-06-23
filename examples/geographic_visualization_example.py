"""
Geographic Visualization Examples

This example demonstrates the new geographic visualization functions
including scatter plots, 2D histograms, and gradient visualizations.
"""

import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from pega_tools.geographic import (
    Coordinate,
    plot_geographic_scatter,
    plot_geographic_histogram_2d,
    plot_temporal_gradient_histogram,
    plot_spatial_gradient_histogram,
    create_geographic_animation
)


def generate_sample_data(n_points=100):
    """Generate sample geographic data for demonstration."""
    np.random.seed(42)
    
    # Generate coordinates around New York City
    base_lat, base_lon = 40.7128, -74.0060
    
    # Create clustered data
    coordinates = []
    timestamps = []
    values = []
    
    for i in range(n_points):
        # Add some clustering
        if i < n_points // 3:
            # Cluster 1: Manhattan
            lat = base_lat + np.random.normal(0, 0.01)
            lon = base_lon + np.random.normal(0, 0.01)
            value = np.random.exponential(10)
        elif i < 2 * n_points // 3:
            # Cluster 2: Brooklyn
            lat = base_lat - 0.02 + np.random.normal(0, 0.01)
            lon = base_lon + 0.01 + np.random.normal(0, 0.01)
            value = np.random.exponential(15)
        else:
            # Cluster 3: Queens
            lat = base_lat + 0.01 + np.random.normal(0, 0.01)
            lon = base_lon + 0.02 + np.random.normal(0, 0.01)
            value = np.random.exponential(8)
        
        coordinates.append(Coordinate(lat, lon))
        
        # Generate timestamps over the last week
        timestamp = datetime.now() - timedelta(
            days=np.random.uniform(0, 7),
            hours=np.random.uniform(0, 24)
        )
        timestamps.append(timestamp)
        values.append(value)
    
    return coordinates, timestamps, values


def example_scatter_plot():
    """Demonstrate geographic scatter plot with different parameters."""
    print("Creating geographic scatter plot...")
    
    coordinates, timestamps, values = generate_sample_data(50)
    
    # Basic scatter plot
    fig1 = plot_geographic_scatter(
        coordinates=coordinates,
        title="Basic Geographic Scatter Plot",
        figsize=(10, 8),
        marker_size=100,
        alpha=0.8
    )
    plt.savefig("geographic_scatter_basic.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # Scatter plot with values
    fig2 = plot_geographic_scatter(
        coordinates=coordinates,
        values=values,
        title="Geographic Scatter Plot with Values",
        figsize=(10, 8),
        marker_size=values,  # Size based on values
        color_map="plasma",
        colorbar_label="Activity Level",
        background_map=True
    )
    plt.savefig("geographic_scatter_values.png", dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    print("✓ Scatter plots saved as 'geographic_scatter_basic.png' and 'geographic_scatter_values.png'")


def example_2d_histogram():
    """Demonstrate 2D histogram with different parameters."""
    print("Creating 2D histogram...")
    
    coordinates, timestamps, values = generate_sample_data(200)
    
    # Basic 2D histogram
    fig1 = plot_geographic_histogram_2d(
        coordinates=coordinates,
        title="Basic Geographic 2D Histogram",
        figsize=(10, 8),
        bins=30,
        color_map="viridis",
        show_colorbar=True,
        background_map=True
    )
    plt.savefig("geographic_histogram_basic.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # 2D histogram with log scale
    fig2 = plot_geographic_histogram_2d(
        coordinates=coordinates,
        title="Geographic 2D Histogram (Log Scale)",
        figsize=(10, 8),
        bins=(40, 40),
        color_map="hot",
        log_scale=True,
        colorbar_label="Log Count"
    )
    plt.savefig("geographic_histogram_log.png", dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    print("✓ 2D histograms saved as 'geographic_histogram_basic.png' and 'geographic_histogram_log.png'")


def example_temporal_gradient():
    """Demonstrate temporal gradient visualization."""
    print("Creating temporal gradient histogram...")
    
    coordinates, timestamps, values = generate_sample_data(150)
    
    fig = plot_temporal_gradient_histogram(
        coordinates=coordinates,
        timestamps=timestamps,
        title="Temporal Gradient Analysis",
        figsize=(15, 6),
        bins=25,
        color_map="plasma",
        log_scale=False,
        background_map=True
    )
    plt.savefig("temporal_gradient.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("✓ Temporal gradient saved as 'temporal_gradient.png'")


def example_spatial_gradient():
    """Demonstrate spatial gradient visualization."""
    print("Creating spatial gradient histogram...")
    
    coordinates, timestamps, values = generate_sample_data(200)
    
    # Density-based gradient
    fig1 = plot_spatial_gradient_histogram(
        coordinates=coordinates,
        values=values,
        title="Spatial Gradient (Density Method)",
        figsize=(15, 6),
        bins=30,
        color_map="coolwarm",
        gradient_method="density",
        background_map=True
    )
    plt.savefig("spatial_gradient_density.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # Interpolation-based gradient
    try:
        fig2 = plot_spatial_gradient_histogram(
            coordinates=coordinates,
            values=values,
            title="Spatial Gradient (Interpolation Method)",
            figsize=(15, 6),
            bins=40,
            color_map="RdYlBu",
            gradient_method="interpolation",
            background_map=True
        )
        plt.savefig("spatial_gradient_interpolation.png", dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print("✓ Spatial gradients saved as 'spatial_gradient_density.png' and 'spatial_gradient_interpolation.png'")
    except ImportError:
        print("⚠️ Interpolation method requires scipy. Skipping interpolation example.")
        print("✓ Spatial gradient saved as 'spatial_gradient_density.png'")


def example_animation():
    """Demonstrate geographic animation."""
    print("Creating geographic animation...")
    
    coordinates, timestamps, values = generate_sample_data(100)
    
    try:
        output_file = create_geographic_animation(
            coordinates=coordinates,
            timestamps=timestamps,
            values=values,
            output_file="geographic_animation.gif",
            duration=5,
            figsize=(10, 8)
        )
        print(f"✓ Animation saved as '{output_file}'")
    except ImportError as e:
        print(f"⚠️ Animation creation failed: {e}")
        print("   This requires matplotlib.animation support.")


def example_custom_parameters():
    """Demonstrate custom parameter configurations."""
    print("Creating custom parameter examples...")
    
    coordinates, timestamps, values = generate_sample_data(80)
    
    # Custom scatter plot with specific parameters
    fig = plot_geographic_scatter(
        coordinates=coordinates,
        values=values,
        title="Custom Scatter Plot",
        figsize=(12, 10),
        marker_size=[v/2 for v in values],  # Variable marker sizes
        color_map="Spectral",
        alpha=0.6,
        show_colorbar=True,
        colorbar_label="Custom Values",
        background_map=True,
        edgecolors='black',  # Additional matplotlib parameter
        linewidth=0.5
    )
    plt.savefig("custom_scatter.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Custom histogram with specific parameters
    fig = plot_geographic_histogram_2d(
        coordinates=coordinates,
        title="Custom 2D Histogram",
        figsize=(12, 10),
        bins=(50, 50),
        color_map="jet",
        log_scale=True,
        show_colorbar=True,
        colorbar_label="Custom Count",
        background_map=True,
        alpha=0.8  # Additional matplotlib parameter
    )
    plt.savefig("custom_histogram.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("✓ Custom examples saved as 'custom_scatter.png' and 'custom_histogram.png'")


def main():
    """Run all visualization examples."""
    print("🌍 Geographic Visualization Examples")
    print("=" * 50)
    
    try:
        example_scatter_plot()
        example_2d_histogram()
        example_temporal_gradient()
        example_spatial_gradient()
        example_animation()
        example_custom_parameters()
        
        print("\n✅ All examples completed successfully!")
        print("\nGenerated files:")
        print("- geographic_scatter_basic.png")
        print("- geographic_scatter_values.png")
        print("- geographic_histogram_basic.png")
        print("- geographic_histogram_log.png")
        print("- temporal_gradient.png")
        print("- spatial_gradient_density.png")
        print("- custom_scatter.png")
        print("- custom_histogram.png")
        print("- geographic_animation.gif (if animation supported)")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("Make sure you have installed the visualization dependencies:")
        print("pip install pega-tools[viz]")


if __name__ == "__main__":
    main() 
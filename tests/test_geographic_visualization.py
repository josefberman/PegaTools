"""
Tests for geographic visualization functions.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from pega_tools.geographic import (
    Coordinate,
    plot_geographic_scatter,
    plot_geographic_histogram_2d,
    plot_temporal_gradient_histogram,
    plot_spatial_gradient_histogram,
    create_geographic_animation
)


class TestGeographicVisualization:
    """Test geographic visualization functions."""
    
    def setup_method(self):
        """Set up test data."""
        self.coordinates = [
            Coordinate(40.7128, -74.0060),  # New York
            Coordinate(34.0522, -118.2437),  # Los Angeles
            Coordinate(41.8781, -87.6298),  # Chicago
        ]
        
        self.values = [100, 85, 92]
        
        self.timestamps = [
            datetime(2023, 1, 1, 10, 0),
            datetime(2023, 1, 1, 11, 0),
            datetime(2023, 1, 1, 12, 0),
        ]
    
    @patch('pega_tools.geographic.visualization.MATPLOTLIB_AVAILABLE', True)
    @patch('pega_tools.geographic.visualization.plt')
    def test_plot_geographic_scatter_basic(self, mock_plt):
        """Test basic scatter plot creation."""
        # Mock matplotlib components
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        # Test basic scatter plot
        fig = plot_geographic_scatter(
            coordinates=self.coordinates,
            title="Test Scatter Plot"
        )
        
        # Verify function calls
        mock_plt.subplots.assert_called_once()
        mock_ax.scatter.assert_called_once()
        mock_ax.set_title.assert_called_with("Test Scatter Plot")
        assert fig == mock_fig
    
    @patch('pega_tools.geographic.visualization.MATPLOTLIB_AVAILABLE', True)
    @patch('pega_tools.geographic.visualization.plt')
    def test_plot_geographic_scatter_with_values(self, mock_plt):
        """Test scatter plot with values."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        # Mock colorbar
        mock_scatter = MagicMock()
        mock_ax.scatter.return_value = mock_scatter
        mock_plt.colorbar.return_value = MagicMock()
        
        fig = plot_geographic_scatter(
            coordinates=self.coordinates,
            values=self.values,
            title="Test Scatter with Values",
            show_colorbar=True
        )
        
        # Verify scatter was called with values
        mock_ax.scatter.assert_called_once()
        call_args = mock_ax.scatter.call_args
        assert 'c' in call_args[1]  # Check that color parameter was passed
        mock_plt.colorbar.assert_called_once()
    
    @patch('pega_tools.geographic.visualization.MATPLOTLIB_AVAILABLE', True)
    @patch('pega_tools.geographic.visualization.plt')
    def test_plot_geographic_histogram_2d(self, mock_plt):
        """Test 2D histogram creation."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        # Mock hist2d return values
        mock_ax.hist2d.return_value = (None, None, None, MagicMock())
        mock_plt.colorbar.return_value = MagicMock()
        
        fig = plot_geographic_histogram_2d(
            coordinates=self.coordinates,
            title="Test 2D Histogram",
            bins=20
        )
        
        # Verify hist2d was called
        mock_ax.hist2d.assert_called_once()
        mock_ax.set_title.assert_called_with("Test 2D Histogram")
    
    @patch('pega_tools.geographic.visualization.MATPLOTLIB_AVAILABLE', True)
    @patch('pega_tools.geographic.visualization.plt')
    def test_plot_temporal_gradient_histogram(self, mock_plt):
        """Test temporal gradient histogram."""
        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
        
        # Mock scatter and hist2d
        mock_scatter = MagicMock()
        mock_ax1.scatter.return_value = mock_scatter
        mock_ax2.hist2d.return_value = (None, None, None, MagicMock())
        mock_plt.colorbar.return_value = MagicMock()
        
        fig = plot_temporal_gradient_histogram(
            coordinates=self.coordinates,
            timestamps=self.timestamps,
            title="Test Temporal Gradient"
        )
        
        # Verify both subplots were created
        mock_ax1.scatter.assert_called_once()
        mock_ax2.hist2d.assert_called_once()
        mock_plt.suptitle.assert_called_with("Test Temporal Gradient")
    
    @patch('pega_tools.geographic.visualization.MATPLOTLIB_AVAILABLE', True)
    @patch('pega_tools.geographic.visualization.plt')
    def test_plot_spatial_gradient_histogram_density(self, mock_plt):
        """Test spatial gradient histogram with density method."""
        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
        
        # Mock scatter and hist2d
        mock_scatter = MagicMock()
        mock_ax1.scatter.return_value = mock_scatter
        mock_ax2.hist2d.return_value = (None, None, None, MagicMock())
        mock_plt.colorbar.return_value = MagicMock()
        
        fig = plot_spatial_gradient_histogram(
            coordinates=self.coordinates,
            values=self.values,
            title="Test Spatial Gradient",
            gradient_method="density"
        )
        
        # Verify both subplots were created
        mock_ax1.scatter.assert_called_once()
        mock_ax2.hist2d.assert_called_once()
    
    @patch('pega_tools.geographic.visualization.MATPLOTLIB_AVAILABLE', True)
    @patch('pega_tools.geographic.visualization.plt')
    def test_plot_spatial_gradient_histogram_interpolation(self, mock_plt):
        """Test spatial gradient histogram with interpolation method."""
        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
        
        # Mock scatter and pcolormesh
        mock_scatter = MagicMock()
        mock_ax1.scatter.return_value = mock_scatter
        mock_ax2.pcolormesh.return_value = MagicMock()
        mock_plt.colorbar.return_value = MagicMock()
        
        # Mock scipy.interpolate.griddata
        with patch('pega_tools.geographic.visualization.griddata') as mock_griddata:
            mock_griddata.return_value = np.array([[1, 2], [3, 4]])
            
            fig = plot_spatial_gradient_histogram(
                coordinates=self.coordinates,
                values=self.values,
                title="Test Spatial Gradient",
                gradient_method="interpolation"
            )
            
            # Verify interpolation was used
            mock_griddata.assert_called_once()
            mock_ax2.pcolormesh.assert_called_once()
    
    @patch('pega_tools.geographic.visualization.MATPLOTLIB_AVAILABLE', True)
    @patch('pega_tools.geographic.visualization.plt')
    def test_create_geographic_animation(self, mock_plt):
        """Test geographic animation creation."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        # Mock animation components
        with patch('pega_tools.geographic.visualization.FuncAnimation') as mock_anim:
            with patch('pega_tools.geographic.visualization.PillowWriter') as mock_writer:
                mock_anim_instance = MagicMock()
                mock_anim.return_value = mock_anim_instance
                mock_writer_instance = MagicMock()
                mock_writer.return_value = mock_writer_instance
                
                output_file = create_geographic_animation(
                    coordinates=self.coordinates,
                    timestamps=self.timestamps,
                    values=self.values,
                    output_file="test_animation.gif"
                )
                
                # Verify animation was created
                mock_anim.assert_called_once()
                mock_anim_instance.save.assert_called_once()
                assert output_file == "test_animation.gif"
    
    def test_extract_coordinates(self):
        """Test coordinate extraction from different formats."""
        from pega_tools.geographic.visualization import _extract_coordinates
        
        # Test with Coordinate objects
        coords = [Coordinate(40.7128, -74.0060), Coordinate(34.0522, -118.2437)]
        lats, lons = _extract_coordinates(coords)
        
        assert lats == [40.7128, 34.0522]
        assert lons == [-74.0060, -118.2437]
        
        # Test with tuples
        coord_tuples = [(40.7128, -74.0060), (34.0522, -118.2437)]
        lats, lons = _extract_coordinates(coord_tuples)
        
        assert lats == [40.7128, 34.0522]
        assert lons == [-74.0060, -118.2437]
        
        # Test with invalid format
        with pytest.raises(ValueError):
            _extract_coordinates([(40.7128,)])  # Missing longitude
    
    @patch('pega_tools.geographic.visualization.MATPLOTLIB_AVAILABLE', False)
    def test_matplotlib_not_available(self):
        """Test that functions raise ImportError when matplotlib is not available."""
        with pytest.raises(ImportError, match="matplotlib is required"):
            plot_geographic_scatter(coordinates=self.coordinates)
        
        with pytest.raises(ImportError, match="matplotlib is required"):
            plot_geographic_histogram_2d(coordinates=self.coordinates)
        
        with pytest.raises(ImportError, match="matplotlib is required"):
            plot_temporal_gradient_histogram(
                coordinates=self.coordinates,
                timestamps=self.timestamps
            )
        
        with pytest.raises(ImportError, match="matplotlib is required"):
            plot_spatial_gradient_histogram(
                coordinates=self.coordinates,
                values=self.values
            )
    
    def test_invalid_gradient_method(self):
        """Test that invalid gradient method raises ValueError."""
        with patch('pega_tools.geographic.visualization.MATPLOTLIB_AVAILABLE', True):
            with patch('pega_tools.geographic.visualization.plt'):
                with pytest.raises(ValueError, match="gradient_method must be"):
                    plot_spatial_gradient_histogram(
                        coordinates=self.coordinates,
                        values=self.values,
                        gradient_method="invalid_method"
                    )
    
    def test_mismatched_lengths(self):
        """Test that mismatched coordinate and value lengths raise ValueError."""
        with patch('pega_tools.geographic.visualization.MATPLOTLIB_AVAILABLE', True):
            with patch('pega_tools.geographic.visualization.plt'):
                # Test temporal gradient with mismatched lengths
                with pytest.raises(ValueError, match="must have the same length"):
                    plot_temporal_gradient_histogram(
                        coordinates=self.coordinates,
                        timestamps=self.timestamps[:2]  # Different length
                    )
                
                # Test spatial gradient with mismatched lengths
                with pytest.raises(ValueError, match="must have the same length"):
                    plot_spatial_gradient_histogram(
                        coordinates=self.coordinates,
                        values=self.values[:2]  # Different length
                    ) 
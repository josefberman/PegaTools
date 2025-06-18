"""
Time series data handling for forecasting.
"""

from typing import List, Optional, Union
from datetime import datetime, timedelta
import numpy as np


class TimeSeriesData:
    """Container for time series data."""
    
    def __init__(self, values: List[float], 
                 timestamps: Optional[List[datetime]] = None,
                 frequency: Optional[str] = None):
        """
        Initialize time series data.
        
        Args:
            values: Time series values
            timestamps: Optional timestamps for each value
            frequency: Data frequency ('daily', 'weekly', 'monthly', etc.)
        """
        self.values = np.array(values)
        self.frequency = frequency
        
        if timestamps is None:
            # Generate default timestamps
            self.timestamps = [datetime.now() + timedelta(days=i) 
                             for i in range(len(values))]
        else:
            if len(timestamps) != len(values):
                raise ValueError("Timestamps and values must have same length")
            self.timestamps = timestamps
    
    def __len__(self) -> int:
        """Return the length of the time series."""
        return len(self.values)
    
    def __getitem__(self, index: int) -> float:
        """Get value at index."""
        return self.values[index]
    
    def add_point(self, value: float, timestamp: Optional[datetime] = None) -> None:
        """
        Add a new data point to the time series.
        
        Args:
            value: Value to add
            timestamp: Optional timestamp, defaults to next in sequence
        """
        self.values = np.append(self.values, value)
        
        if timestamp is None:
            # Generate next timestamp based on last one
            if self.timestamps:
                last_ts = self.timestamps[-1]
                if self.frequency == 'daily':
                    next_ts = last_ts + timedelta(days=1)
                elif self.frequency == 'weekly':
                    next_ts = last_ts + timedelta(weeks=1)
                elif self.frequency == 'monthly':
                    next_ts = last_ts + timedelta(days=30)  # Approximate
                else:
                    next_ts = last_ts + timedelta(days=1)
            else:
                next_ts = datetime.now()
        else:
            next_ts = timestamp
        
        self.timestamps.append(next_ts)
    
    def get_statistics(self) -> dict:
        """
        Get basic statistics of the time series.
        
        Returns:
            Dictionary with statistical measures
        """
        return {
            'count': len(self.values),
            'mean': np.mean(self.values),
            'std': np.std(self.values),
            'min': np.min(self.values),
            'max': np.max(self.values),
            'median': np.median(self.values)
        }
    
    def get_trend(self) -> str:
        """
        Determine the overall trend of the time series.
        
        Returns:
            Trend description ('increasing', 'decreasing', 'stable')
        """
        if len(self.values) < 2:
            return 'insufficient_data'
        
        # Simple trend calculation using first and last values
        start_avg = np.mean(self.values[:len(self.values)//3])
        end_avg = np.mean(self.values[-len(self.values)//3:])
        
        diff_ratio = (end_avg - start_avg) / start_avg if start_avg != 0 else 0
        
        if diff_ratio > 0.05:  # 5% increase
            return 'increasing'
        elif diff_ratio < -0.05:  # 5% decrease
            return 'decreasing'
        else:
            return 'stable'
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            'values': self.values.tolist(),
            'timestamps': [ts.isoformat() for ts in self.timestamps],
            'frequency': self.frequency,
            'statistics': self.get_statistics(),
            'trend': self.get_trend()
        } 
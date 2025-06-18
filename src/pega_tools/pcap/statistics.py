"""
Network statistics and metrics for PCAP analysis.
"""

import math
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from .parser import Packet


class NetworkStatistics:
    """Calculate network statistics and metrics from packet data."""
    
    def __init__(self, packets: List[Packet] = None):
        """
        Initialize network statistics calculator.
        
        Args:
            packets: List of packets to analyze
        """
        self.packets = packets or []
    
    def calculate_basic_stats(self) -> Dict[str, Any]:
        """
        Calculate basic packet statistics.
        
        Returns:
            Dictionary with basic statistics
        """
        if not self.packets:
            return {}
        
        packet_sizes = [len(p.data) for p in self.packets]
        total_packets = len(self.packets)
        total_bytes = sum(packet_sizes)
        
        # Time statistics
        start_time = self.packets[0].header.timestamp
        end_time = self.packets[-1].header.timestamp
        duration = (end_time - start_time).total_seconds()
        
        return {
            'total_packets': total_packets,
            'total_bytes': total_bytes,
            'duration_seconds': duration,
            'average_packet_size': total_bytes / total_packets,
            'min_packet_size': min(packet_sizes),
            'max_packet_size': max(packet_sizes),
            'packets_per_second': total_packets / duration if duration > 0 else 0,
            'bytes_per_second': total_bytes / duration if duration > 0 else 0,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat()
        }
    
    def calculate_throughput_stats(self, interval_seconds: int = 1) -> Dict[str, Any]:
        """
        Calculate throughput statistics over time intervals.
        
        Args:
            interval_seconds: Time interval for calculations
            
        Returns:
            Dictionary with throughput statistics
        """
        if not self.packets:
            return {}
        
        # Group packets by time intervals
        interval_data = defaultdict(lambda: {'packets': 0, 'bytes': 0})
        start_time = self.packets[0].header.timestamp
        
        for packet in self.packets:
            elapsed = (packet.header.timestamp - start_time).total_seconds()
            interval_index = int(elapsed // interval_seconds)
            interval_key = interval_index
            
            interval_data[interval_key]['packets'] += 1
            interval_data[interval_key]['bytes'] += len(packet.data)
        
        # Calculate statistics
        packet_rates = [data['packets'] / interval_seconds for data in interval_data.values()]
        byte_rates = [data['bytes'] / interval_seconds for data in interval_data.values()]
        
        return {
            'avg_packet_rate': sum(packet_rates) / len(packet_rates) if packet_rates else 0,
            'max_packet_rate': max(packet_rates) if packet_rates else 0,
            'min_packet_rate': min(packet_rates) if packet_rates else 0,
            'avg_throughput_bps': sum(byte_rates) / len(byte_rates) if byte_rates else 0,
            'max_throughput_bps': max(byte_rates) if byte_rates else 0,
            'min_throughput_bps': min(byte_rates) if byte_rates else 0,
            'interval_seconds': interval_seconds,
            'total_intervals': len(interval_data)
        }
    
    def calculate_protocol_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Calculate statistics per protocol.
        
        Returns:
            Dictionary with protocol-specific statistics
        """
        protocol_stats = defaultdict(lambda: {
            'packet_count': 0,
            'total_bytes': 0,
            'packet_sizes': []
        })
        
        for packet in self.packets:
            ip_info = packet.get_ip_info()
            protocol = ip_info.get('protocol_name', 'Unknown') if ip_info else 'Non-IP'
            packet_size = len(packet.data)
            
            protocol_stats[protocol]['packet_count'] += 1
            protocol_stats[protocol]['total_bytes'] += packet_size
            protocol_stats[protocol]['packet_sizes'].append(packet_size)
        
        # Calculate additional metrics for each protocol
        result = {}
        for protocol, stats in protocol_stats.items():
            sizes = stats['packet_sizes']
            result[protocol] = {
                'packet_count': stats['packet_count'],
                'total_bytes': stats['total_bytes'],
                'avg_packet_size': stats['total_bytes'] / stats['packet_count'],
                'min_packet_size': min(sizes) if sizes else 0,
                'max_packet_size': max(sizes) if sizes else 0,
                'percentage_of_total': (stats['packet_count'] / len(self.packets)) * 100
            }
        
        return result
    
    def calculate_host_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Calculate statistics per host (IP address).
        
        Returns:
            Dictionary with host-specific statistics
        """
        host_stats = defaultdict(lambda: {
            'packets_sent': 0,
            'packets_received': 0,
            'bytes_sent': 0,
            'bytes_received': 0,
            'protocols_used': set()
        })
        
        for packet in self.packets:
            ip_info = packet.get_ip_info()
            if not ip_info:
                continue
            
            src_ip = ip_info['source_ip']
            dest_ip = ip_info['destination_ip']
            protocol = ip_info.get('protocol_name', 'Unknown')
            packet_size = len(packet.data)
            
            # Source host statistics
            host_stats[src_ip]['packets_sent'] += 1
            host_stats[src_ip]['bytes_sent'] += packet_size
            host_stats[src_ip]['protocols_used'].add(protocol)
            
            # Destination host statistics
            host_stats[dest_ip]['packets_received'] += 1
            host_stats[dest_ip]['bytes_received'] += packet_size
            host_stats[dest_ip]['protocols_used'].add(protocol)
        
        # Convert to final format
        result = {}
        for host, stats in host_stats.items():
            total_packets = stats['packets_sent'] + stats['packets_received']
            total_bytes = stats['bytes_sent'] + stats['bytes_received']
            
            result[host] = {
                'total_packets': total_packets,
                'total_bytes': total_bytes,
                'packets_sent': stats['packets_sent'],
                'packets_received': stats['packets_received'],
                'bytes_sent': stats['bytes_sent'],
                'bytes_received': stats['bytes_received'],
                'protocols_used': list(stats['protocols_used']),
                'avg_packet_size': total_bytes / total_packets if total_packets > 0 else 0
            }
        
        return result
    
    def calculate_packet_size_distribution(self, bins: int = 10) -> Dict[str, Any]:
        """
        Calculate packet size distribution.
        
        Args:
            bins: Number of bins for histogram
            
        Returns:
            Dictionary with size distribution data
        """
        if not self.packets:
            return {}
        
        packet_sizes = [len(p.data) for p in self.packets]
        min_size = min(packet_sizes)
        max_size = max(packet_sizes)
        
        # Calculate bin edges
        bin_width = (max_size - min_size) / bins
        bin_edges = [min_size + i * bin_width for i in range(bins + 1)]
        bin_counts = [0] * bins
        
        # Count packets in each bin
        for size in packet_sizes:
            bin_index = min(int((size - min_size) / bin_width), bins - 1)
            bin_counts[bin_index] += 1
        
        return {
            'bin_edges': bin_edges,
            'bin_counts': bin_counts,
            'bin_width': bin_width,
            'total_packets': len(packet_sizes),
            'size_stats': {
                'min': min_size,
                'max': max_size,
                'avg': sum(packet_sizes) / len(packet_sizes),
                'median': sorted(packet_sizes)[len(packet_sizes) // 2]
            }
        }
    
    def calculate_inter_arrival_times(self) -> Dict[str, Any]:
        """
        Calculate packet inter-arrival time statistics.
        
        Returns:
            Dictionary with inter-arrival time statistics
        """
        if len(self.packets) < 2:
            return {}
        
        inter_arrival_times = []
        for i in range(1, len(self.packets)):
            prev_time = self.packets[i-1].header.timestamp
            curr_time = self.packets[i].header.timestamp
            inter_arrival = (curr_time - prev_time).total_seconds()
            inter_arrival_times.append(inter_arrival)
        
        if not inter_arrival_times:
            return {}
        
        avg_inter_arrival = sum(inter_arrival_times) / len(inter_arrival_times)
        min_inter_arrival = min(inter_arrival_times)
        max_inter_arrival = max(inter_arrival_times)
        
        # Calculate standard deviation
        variance = sum((t - avg_inter_arrival) ** 2 for t in inter_arrival_times) / len(inter_arrival_times)
        std_dev = math.sqrt(variance)
        
        return {
            'average_inter_arrival_seconds': avg_inter_arrival,
            'min_inter_arrival_seconds': min_inter_arrival,
            'max_inter_arrival_seconds': max_inter_arrival,
            'std_dev_seconds': std_dev,
            'total_intervals': len(inter_arrival_times)
        }
    
    def calculate_network_efficiency(self) -> Dict[str, Any]:
        """
        Calculate network efficiency metrics.
        
        Returns:
            Dictionary with efficiency metrics
        """
        if not self.packets:
            return {}
        
        total_packets = len(self.packets)
        total_bytes = sum(len(p.data) for p in self.packets)
        
        # Count different packet types
        broadcast_count = 0
        multicast_count = 0
        unicast_count = 0
        
        for packet in self.packets:
            eth_info = packet.get_ethernet_info()
            if eth_info:
                dest_mac = eth_info.get('destination_mac', '')
                if dest_mac.lower() == 'ff:ff:ff:ff:ff:ff':
                    broadcast_count += 1
                elif dest_mac and int(dest_mac.split(':')[0], 16) & 0x01:
                    multicast_count += 1
                else:
                    unicast_count += 1
        
        return {
            'total_packets': total_packets,
            'unicast_packets': unicast_count,
            'broadcast_packets': broadcast_count,
            'multicast_packets': multicast_count,
            'unicast_percentage': (unicast_count / total_packets) * 100,
            'broadcast_percentage': (broadcast_count / total_packets) * 100,
            'multicast_percentage': (multicast_count / total_packets) * 100,
            'average_packet_size': total_bytes / total_packets,
            'bandwidth_utilization_estimate': self._estimate_bandwidth_utilization()
        }
    
    def _estimate_bandwidth_utilization(self) -> float:
        """
        Estimate bandwidth utilization (simplified calculation).
        
        Returns:
            Estimated bandwidth utilization percentage
        """
        if not self.packets:
            return 0.0
        
        # Simple estimation based on packet rate and size
        # This is a very basic estimation and would need more sophisticated
        # calculation for real network analysis
        
        duration = (self.packets[-1].header.timestamp - 
                   self.packets[0].header.timestamp).total_seconds()
        
        if duration <= 0:
            return 0.0
        
        total_bits = sum(len(p.data) * 8 for p in self.packets)
        bits_per_second = total_bits / duration
        
        # Assume 100 Mbps link (this would normally be configured)
        assumed_link_capacity = 100 * 1000 * 1000  # 100 Mbps
        
        utilization = (bits_per_second / assumed_link_capacity) * 100
        return min(utilization, 100.0)  # Cap at 100%
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive statistics report.
        
        Returns:
            Dictionary with all calculated statistics
        """
        return {
            'basic_stats': self.calculate_basic_stats(),
            'throughput_stats': self.calculate_throughput_stats(),
            'protocol_stats': self.calculate_protocol_stats(),
            'host_stats': self.calculate_host_stats(),
            'packet_size_distribution': self.calculate_packet_size_distribution(),
            'inter_arrival_times': self.calculate_inter_arrival_times(),
            'network_efficiency': self.calculate_network_efficiency()
        } 
"""
Network packet analysis functionality.
"""

from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from .parser import Packet, PCAPParser


class PacketAnalyzer:
    """Analyze network packets for patterns and insights."""
    
    def __init__(self, packets: List[Packet] = None):
        """
        Initialize packet analyzer.
        
        Args:
            packets: List of packets to analyze
        """
        self.packets = packets or []
        self.analysis_cache = {}
    
    def load_from_parser(self, parser: PCAPParser) -> None:
        """
        Load packets from PCAP parser.
        
        Args:
            parser: PCAPParser instance with parsed packets
        """
        self.packets = parser.packets
        self.analysis_cache = {}
    
    def get_traffic_volume_over_time(self, interval_seconds: int = 60) -> Dict[str, List]:
        """
        Analyze traffic volume over time intervals.
        
        Args:
            interval_seconds: Time interval for grouping packets
            
        Returns:
            Dictionary with timestamps and packet counts
        """
        if not self.packets:
            return {'timestamps': [], 'packet_counts': [], 'byte_counts': []}
        
        # Group packets by time intervals
        interval_data = defaultdict(lambda: {'packets': 0, 'bytes': 0})
        
        start_time = self.packets[0].header.timestamp
        
        for packet in self.packets:
            # Calculate which interval this packet falls into
            elapsed = (packet.header.timestamp - start_time).total_seconds()
            interval_index = int(elapsed // interval_seconds)
            interval_key = start_time + timedelta(seconds=interval_index * interval_seconds)
            
            interval_data[interval_key]['packets'] += 1
            interval_data[interval_key]['bytes'] += len(packet.data)
        
        # Sort by timestamp and extract data
        sorted_intervals = sorted(interval_data.items())
        
        return {
            'timestamps': [ts.isoformat() for ts, _ in sorted_intervals],
            'packet_counts': [data['packets'] for _, data in sorted_intervals],
            'byte_counts': [data['bytes'] for _, data in sorted_intervals]
        }
    
    def get_protocol_distribution(self) -> Dict[str, int]:
        """
        Get distribution of protocols in the packet capture.
        
        Returns:
            Dictionary mapping protocol names to packet counts
        """
        if 'protocol_distribution' in self.analysis_cache:
            return self.analysis_cache['protocol_distribution']
        
        protocols = Counter()
        
        for packet in self.packets:
            ip_info = packet.get_ip_info()
            if ip_info:
                protocol = ip_info.get('protocol_name', 'Unknown')
                protocols[protocol] += 1
            else:
                protocols['Non-IP'] += 1
        
        result = dict(protocols)
        self.analysis_cache['protocol_distribution'] = result
        return result
    
    def get_top_talkers(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get top talking hosts by packet count.
        
        Args:
            limit: Maximum number of hosts to return
            
        Returns:
            List of top talking hosts with their statistics
        """
        host_stats = defaultdict(lambda: {
            'packets_sent': 0,
            'packets_received': 0,
            'bytes_sent': 0,
            'bytes_received': 0
        })
        
        for packet in self.packets:
            ip_info = packet.get_ip_info()
            if not ip_info:
                continue
            
            src_ip = ip_info['source_ip']
            dest_ip = ip_info['destination_ip']
            packet_size = len(packet.data)
            
            host_stats[src_ip]['packets_sent'] += 1
            host_stats[src_ip]['bytes_sent'] += packet_size
            
            host_stats[dest_ip]['packets_received'] += 1
            host_stats[dest_ip]['bytes_received'] += packet_size
        
        # Calculate total activity for each host
        host_activity = []
        for host, stats in host_stats.items():
            total_packets = stats['packets_sent'] + stats['packets_received']
            total_bytes = stats['bytes_sent'] + stats['bytes_received']
            
            host_activity.append({
                'host': host,
                'total_packets': total_packets,
                'total_bytes': total_bytes,
                'packets_sent': stats['packets_sent'],
                'packets_received': stats['packets_received'],
                'bytes_sent': stats['bytes_sent'],
                'bytes_received': stats['bytes_received']
            })
        
        # Sort by total activity and return top talkers
        host_activity.sort(key=lambda x: x['total_packets'], reverse=True)
        return host_activity[:limit]
    
    def get_port_analysis(self) -> Dict[str, Any]:
        """
        Analyze port usage in the network traffic.
        
        Returns:
            Dictionary with port statistics
        """
        # This is a simplified version - would need more detailed parsing for TCP/UDP ports
        # For now, return protocol-based analysis
        return {
            'note': 'Port analysis requires deeper packet parsing',
            'protocols': self.get_protocol_distribution()
        }
    
    def detect_suspicious_activity(self) -> List[Dict[str, Any]]:
        """
        Detect potentially suspicious network activity.
        
        Returns:
            List of suspicious activity alerts
        """
        alerts = []
        
        # Check for high-volume hosts (potential DDoS sources)
        top_talkers = self.get_top_talkers(5)
        if top_talkers:
            highest_talker = top_talkers[0]
            if highest_talker['total_packets'] > len(self.packets) * 0.3:
                alerts.append({
                    'type': 'high_volume_host',
                    'severity': 'medium',
                    'description': f"Host {highest_talker['host']} generated {highest_talker['total_packets']} packets",
                    'host': highest_talker['host'],
                    'packet_count': highest_talker['total_packets']
                })
        
        # Check for unusual protocol distribution
        protocols = self.get_protocol_distribution()
        if 'Unknown' in protocols and protocols['Unknown'] > len(self.packets) * 0.1:
            alerts.append({
                'type': 'unknown_protocols',
                'severity': 'low',
                'description': f"High number of packets with unknown protocols: {protocols['Unknown']}",
                'count': protocols['Unknown']
            })
        
        # Check for very large packets
        large_packets = [p for p in self.packets if len(p.data) > 1500]
        if len(large_packets) > len(self.packets) * 0.05:
            alerts.append({
                'type': 'large_packets',
                'severity': 'low',
                'description': f"Unusually large packets detected: {len(large_packets)}",
                'count': len(large_packets)
            })
        
        return alerts
    
    def get_connection_patterns(self) -> Dict[str, Any]:
        """
        Analyze connection patterns between hosts.
        
        Returns:
            Dictionary with connection statistics
        """
        connections = defaultdict(int)
        unique_sources = set()
        unique_destinations = set()
        
        for packet in self.packets:
            ip_info = packet.get_ip_info()
            if not ip_info:
                continue
            
            src_ip = ip_info['source_ip']
            dest_ip = ip_info['destination_ip']
            
            connection_key = f"{src_ip} -> {dest_ip}"
            connections[connection_key] += 1
            
            unique_sources.add(src_ip)
            unique_destinations.add(dest_ip)
        
        # Get top connections
        top_connections = sorted(connections.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'total_connections': len(connections),
            'unique_sources': len(unique_sources),
            'unique_destinations': len(unique_destinations),
            'top_connections': [
                {'connection': conn, 'packet_count': count}
                for conn, count in top_connections
            ]
        }
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive analysis summary.
        
        Returns:
            Dictionary with complete analysis summary
        """
        if not self.packets:
            return {'error': 'No packets to analyze'}
        
        # Basic statistics
        total_packets = len(self.packets)
        total_bytes = sum(len(p.data) for p in self.packets)
        start_time = self.packets[0].header.timestamp
        end_time = self.packets[-1].header.timestamp
        duration = (end_time - start_time).total_seconds()
        
        return {
            'basic_stats': {
                'total_packets': total_packets,
                'total_bytes': total_bytes,
                'duration_seconds': duration,
                'average_packet_size': total_bytes / total_packets if total_packets > 0 else 0,
                'packets_per_second': total_packets / duration if duration > 0 else 0,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat()
            },
            'protocol_distribution': self.get_protocol_distribution(),
            'top_talkers': self.get_top_talkers(5),
            'connection_patterns': self.get_connection_patterns(),
            'suspicious_activity': self.detect_suspicious_activity(),
            'traffic_volume': self.get_traffic_volume_over_time(60)  # 1-minute intervals
        } 
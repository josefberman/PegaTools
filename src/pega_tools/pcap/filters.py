"""
Packet filtering functionality.
"""

from typing import List, Callable, Optional, Union
from datetime import datetime
from .parser import Packet


class PacketFilter:
    """Filter packets based on various criteria."""
    
    def __init__(self):
        """Initialize packet filter."""
        self.filters = []
    
    def add_filter(self, filter_func: Callable[[Packet], bool]) -> None:
        """
        Add a custom filter function.
        
        Args:
            filter_func: Function that takes a packet and returns True if it should be included
        """
        self.filters.append(filter_func)
    
    def apply_filters(self, packets: List[Packet]) -> List[Packet]:
        """
        Apply all filters to a list of packets.
        
        Args:
            packets: List of packets to filter
            
        Returns:
            Filtered list of packets
        """
        result = packets
        for filter_func in self.filters:
            result = [p for p in result if filter_func(p)]
        return result
    
    def clear_filters(self) -> None:
        """Clear all filters."""
        self.filters = []
    
    def filter_by_protocol(self, packets: List[Packet], protocol: str) -> List[Packet]:
        """
        Filter packets by IP protocol.
        
        Args:
            packets: List of packets to filter
            protocol: Protocol name (e.g., 'TCP', 'UDP', 'ICMP')
            
        Returns:
            Filtered packets
        """
        def protocol_filter(packet: Packet) -> bool:
            ip_info = packet.get_ip_info()
            if ip_info:
                return ip_info.get('protocol_name', '').upper() == protocol.upper()
            return False
        
        return [p for p in packets if protocol_filter(p)]
    
    def filter_by_ip_address(self, packets: List[Packet], 
                           ip_address: str, 
                           direction: str = 'both') -> List[Packet]:
        """
        Filter packets by IP address.
        
        Args:
            packets: List of packets to filter
            ip_address: IP address to filter by
            direction: Filter direction ('src', 'dest', 'both')
            
        Returns:
            Filtered packets
        """
        def ip_filter(packet: Packet) -> bool:
            ip_info = packet.get_ip_info()
            if not ip_info:
                return False
            
            src_ip = ip_info.get('source_ip', '')
            dest_ip = ip_info.get('destination_ip', '')
            
            if direction == 'src':
                return src_ip == ip_address
            elif direction == 'dest':
                return dest_ip == ip_address
            else:  # both
                return src_ip == ip_address or dest_ip == ip_address
        
        return [p for p in packets if ip_filter(p)]
    
    def filter_by_ip_range(self, packets: List[Packet], 
                          ip_range: str, 
                          direction: str = 'both') -> List[Packet]:
        """
        Filter packets by IP address range (CIDR notation).
        
        Args:
            packets: List of packets to filter
            ip_range: IP range in CIDR notation (e.g., '192.168.1.0/24')
            direction: Filter direction ('src', 'dest', 'both')
            
        Returns:
            Filtered packets
        """
        import ipaddress
        
        try:
            network = ipaddress.ip_network(ip_range, strict=False)
        except ValueError:
            return []
        
        def range_filter(packet: Packet) -> bool:
            ip_info = packet.get_ip_info()
            if not ip_info:
                return False
            
            src_ip = ip_info.get('source_ip', '')
            dest_ip = ip_info.get('destination_ip', '')
            
            try:
                if direction == 'src':
                    return ipaddress.ip_address(src_ip) in network
                elif direction == 'dest':
                    return ipaddress.ip_address(dest_ip) in network
                else:  # both
                    return (ipaddress.ip_address(src_ip) in network or 
                           ipaddress.ip_address(dest_ip) in network)
            except ValueError:
                return False
        
        return [p for p in packets if range_filter(p)]
    
    def filter_by_time_range(self, packets: List[Packet], 
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None) -> List[Packet]:
        """
        Filter packets by time range.
        
        Args:
            packets: List of packets to filter
            start_time: Start time (inclusive)
            end_time: End time (inclusive)
            
        Returns:
            Filtered packets
        """
        def time_filter(packet: Packet) -> bool:
            packet_time = packet.header.timestamp
            
            if start_time and packet_time < start_time:
                return False
            if end_time and packet_time > end_time:
                return False
            
            return True
        
        return [p for p in packets if time_filter(p)]
    
    def filter_by_packet_size(self, packets: List[Packet], 
                            min_size: Optional[int] = None,
                            max_size: Optional[int] = None) -> List[Packet]:
        """
        Filter packets by size.
        
        Args:
            packets: List of packets to filter
            min_size: Minimum packet size in bytes
            max_size: Maximum packet size in bytes
            
        Returns:
            Filtered packets
        """
        def size_filter(packet: Packet) -> bool:
            packet_size = len(packet.data)
            
            if min_size is not None and packet_size < min_size:
                return False
            if max_size is not None and packet_size > max_size:
                return False
            
            return True
        
        return [p for p in packets if size_filter(p)]
    
    def filter_by_mac_address(self, packets: List[Packet], 
                            mac_address: str, 
                            direction: str = 'both') -> List[Packet]:
        """
        Filter packets by MAC address.
        
        Args:
            packets: List of packets to filter
            mac_address: MAC address to filter by (format: 'aa:bb:cc:dd:ee:ff')
            direction: Filter direction ('src', 'dest', 'both')
            
        Returns:
            Filtered packets
        """
        def mac_filter(packet: Packet) -> bool:
            eth_info = packet.get_ethernet_info()
            if not eth_info:
                return False
            
            src_mac = eth_info.get('source_mac', '').lower()
            dest_mac = eth_info.get('destination_mac', '').lower()
            target_mac = mac_address.lower()
            
            if direction == 'src':
                return src_mac == target_mac
            elif direction == 'dest':
                return dest_mac == target_mac
            else:  # both
                return src_mac == target_mac or dest_mac == target_mac
        
        return [p for p in packets if mac_filter(p)]
    
    def filter_broadcast_packets(self, packets: List[Packet]) -> List[Packet]:
        """
        Filter to include only broadcast packets.
        
        Args:
            packets: List of packets to filter
            
        Returns:
            Broadcast packets only
        """
        def broadcast_filter(packet: Packet) -> bool:
            eth_info = packet.get_ethernet_info()
            if eth_info:
                dest_mac = eth_info.get('destination_mac', '')
                return dest_mac.lower() == 'ff:ff:ff:ff:ff:ff'
            return False
        
        return [p for p in packets if broadcast_filter(p)]
    
    def filter_multicast_packets(self, packets: List[Packet]) -> List[Packet]:
        """
        Filter to include only multicast packets.
        
        Args:
            packets: List of packets to filter
            
        Returns:
            Multicast packets only
        """
        def multicast_filter(packet: Packet) -> bool:
            eth_info = packet.get_ethernet_info()
            if eth_info:
                dest_mac = eth_info.get('destination_mac', '')
                if dest_mac:
                    # Check if first octet has multicast bit set
                    first_octet = int(dest_mac.split(':')[0], 16)
                    return (first_octet & 0x01) == 1
            return False
        
        return [p for p in packets if multicast_filter(p)]
    
    def filter_by_connection(self, packets: List[Packet], 
                           src_ip: str, dest_ip: str) -> List[Packet]:
        """
        Filter packets for a specific connection.
        
        Args:
            packets: List of packets to filter
            src_ip: Source IP address
            dest_ip: Destination IP address
            
        Returns:
            Packets belonging to the specific connection (bidirectional)
        """
        def connection_filter(packet: Packet) -> bool:
            ip_info = packet.get_ip_info()
            if not ip_info:
                return False
            
            packet_src = ip_info.get('source_ip', '')
            packet_dest = ip_info.get('destination_ip', '')
            
            # Check both directions of the connection
            return ((packet_src == src_ip and packet_dest == dest_ip) or
                   (packet_src == dest_ip and packet_dest == src_ip))
        
        return [p for p in packets if connection_filter(p)]
    
    def get_unique_ips(self, packets: List[Packet]) -> List[str]:
        """
        Get list of unique IP addresses from packets.
        
        Args:
            packets: List of packets to analyze
            
        Returns:
            List of unique IP addresses
        """
        ips = set()
        for packet in packets:
            ip_info = packet.get_ip_info()
            if ip_info:
                ips.add(ip_info.get('source_ip', ''))
                ips.add(ip_info.get('destination_ip', ''))
        
        return sorted(list(ips))
    
    def get_unique_protocols(self, packets: List[Packet]) -> List[str]:
        """
        Get list of unique protocols from packets.
        
        Args:
            packets: List of packets to analyze
            
        Returns:
            List of unique protocol names
        """
        protocols = set()
        for packet in packets:
            ip_info = packet.get_ip_info()
            if ip_info:
                protocols.add(ip_info.get('protocol_name', 'Unknown'))
        
        return sorted(list(protocols)) 
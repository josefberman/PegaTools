"""
PCAP file parsing functionality.
"""

import struct
from typing import List, Dict, Any, Optional, BinaryIO
from pathlib import Path
from datetime import datetime
from ..utils import PegaException


class PCAPHeader:
    """PCAP file header structure."""
    
    def __init__(self, magic: int, version_major: int, version_minor: int,
                 thiszone: int, sigfigs: int, snaplen: int, network: int):
        self.magic = magic
        self.version_major = version_major
        self.version_minor = version_minor
        self.thiszone = thiszone
        self.sigfigs = sigfigs
        self.snaplen = snaplen
        self.network = network
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'magic': hex(self.magic),
            'version': f"{self.version_major}.{self.version_minor}",
            'timezone': self.thiszone,
            'timestamp_accuracy': self.sigfigs,
            'max_packet_length': self.snaplen,
            'data_link_type': self.network
        }


class PacketHeader:
    """Packet header structure."""
    
    def __init__(self, ts_sec: int, ts_usec: int, incl_len: int, orig_len: int):
        self.ts_sec = ts_sec
        self.ts_usec = ts_usec
        self.incl_len = incl_len
        self.orig_len = orig_len
        self.timestamp = datetime.fromtimestamp(ts_sec + ts_usec / 1000000)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'timestamp_sec': self.ts_sec,
            'timestamp_usec': self.ts_usec,
            'captured_length': self.incl_len,
            'original_length': self.orig_len
        }


class Packet:
    """Network packet container."""
    
    def __init__(self, header: PacketHeader, data: bytes):
        self.header = header
        self.data = data
        self.parsed_data = {}
    
    def get_ethernet_info(self) -> Optional[Dict[str, Any]]:
        """Extract Ethernet header information."""
        if len(self.data) < 14:
            return None
        
        # Ethernet header: 6 bytes dest + 6 bytes src + 2 bytes type
        dest_mac = ':'.join(f'{b:02x}' for b in self.data[0:6])
        src_mac = ':'.join(f'{b:02x}' for b in self.data[6:12])
        eth_type = struct.unpack('!H', self.data[12:14])[0]
        
        return {
            'destination_mac': dest_mac,
            'source_mac': src_mac,
            'ethernet_type': hex(eth_type),
            'protocol': self._get_ethernet_protocol(eth_type)
        }
    
    def get_ip_info(self) -> Optional[Dict[str, Any]]:
        """Extract IP header information."""
        if len(self.data) < 34:  # Ethernet (14) + IP (20)
            return None
        
        ip_data = self.data[14:]  # Skip Ethernet header
        
        if len(ip_data) < 20:
            return None
        
        # IP header fields
        version_ihl = ip_data[0]
        version = (version_ihl >> 4) & 0xF
        ihl = version_ihl & 0xF
        
        if version != 4:  # Only IPv4 for now
            return None
        
        tos = ip_data[1]
        total_length = struct.unpack('!H', ip_data[2:4])[0]
        identification = struct.unpack('!H', ip_data[4:6])[0]
        flags_frag = struct.unpack('!H', ip_data[6:8])[0]
        ttl = ip_data[8]
        protocol = ip_data[9]
        checksum = struct.unpack('!H', ip_data[10:12])[0]
        src_ip = '.'.join(str(b) for b in ip_data[12:16])
        dest_ip = '.'.join(str(b) for b in ip_data[16:20])
        
        return {
            'version': version,
            'header_length': ihl * 4,
            'type_of_service': tos,
            'total_length': total_length,
            'identification': identification,
            'flags': (flags_frag >> 13) & 0x7,
            'fragment_offset': flags_frag & 0x1FFF,
            'ttl': ttl,
            'protocol': protocol,
            'protocol_name': self._get_ip_protocol(protocol),
            'checksum': hex(checksum),
            'source_ip': src_ip,
            'destination_ip': dest_ip
        }
    
    def _get_ethernet_protocol(self, eth_type: int) -> str:
        """Get protocol name from Ethernet type."""
        protocols = {
            0x0800: 'IPv4',
            0x0806: 'ARP',
            0x86DD: 'IPv6',
            0x8100: 'VLAN'
        }
        return protocols.get(eth_type, 'Unknown')
    
    def _get_ip_protocol(self, protocol: int) -> str:
        """Get protocol name from IP protocol number."""
        protocols = {
            1: 'ICMP',
            6: 'TCP',
            17: 'UDP',
            2: 'IGMP',
            89: 'OSPF'
        }
        return protocols.get(protocol, 'Unknown')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert packet to dictionary."""
        result = {
            'header': self.header.to_dict(),
            'size': len(self.data)
        }
        
        eth_info = self.get_ethernet_info()
        if eth_info:
            result['ethernet'] = eth_info
        
        ip_info = self.get_ip_info()
        if ip_info:
            result['ip'] = ip_info
        
        return result


class PCAPParser:
    """PCAP file parser."""
    
    def __init__(self):
        """Initialize PCAP parser."""
        self.header = None
        self.packets = []
    
    def parse_file(self, file_path: str) -> List[Packet]:
        """
        Parse PCAP file and return packets.
        
        Args:
            file_path: Path to PCAP file
            
        Returns:
            List of parsed packets
            
        Raises:
            PegaException: If file cannot be parsed
        """
        try:
            with open(file_path, 'rb') as f:
                self.header = self._parse_header(f)
                self.packets = self._parse_packets(f)
            return self.packets
        except FileNotFoundError:
            raise PegaException(f"PCAP file not found: {file_path}")
        except Exception as e:
            raise PegaException(f"Error parsing PCAP file: {str(e)}")
    
    def _parse_header(self, f: BinaryIO) -> PCAPHeader:
        """Parse PCAP file header."""
        header_data = f.read(24)
        if len(header_data) < 24:
            raise PegaException("Invalid PCAP file: header too short")
        
        # Unpack header fields
        magic, version_major, version_minor, thiszone, sigfigs, snaplen, network = \
            struct.unpack('<LHHLLLL', header_data)
        
        # Check magic number
        if magic not in (0xa1b2c3d4, 0xd4c3b2a1):
            raise PegaException("Invalid PCAP file: bad magic number")
        
        return PCAPHeader(magic, version_major, version_minor, 
                         thiszone, sigfigs, snaplen, network)
    
    def _parse_packets(self, f: BinaryIO) -> List[Packet]:
        """Parse packets from PCAP file."""
        packets = []
        
        while True:
            # Read packet header
            packet_header_data = f.read(16)
            if len(packet_header_data) < 16:
                break  # End of file
            
            ts_sec, ts_usec, incl_len, orig_len = \
                struct.unpack('<LLLL', packet_header_data)
            
            packet_header = PacketHeader(ts_sec, ts_usec, incl_len, orig_len)
            
            # Read packet data
            packet_data = f.read(incl_len)
            if len(packet_data) < incl_len:
                break  # Incomplete packet
            
            packet = Packet(packet_header, packet_data)
            packets.append(packet)
        
        return packets
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary information about parsed PCAP.
        
        Returns:
            Dictionary with summary information
        """
        if not self.header:
            return {}
        
        packet_count = len(self.packets)
        total_size = sum(len(p.data) for p in self.packets)
        
        # Protocol distribution
        protocols = {}
        for packet in self.packets:
            ip_info = packet.get_ip_info()
            if ip_info:
                protocol = ip_info.get('protocol_name', 'Unknown')
                protocols[protocol] = protocols.get(protocol, 0) + 1
        
        # Time range
        if self.packets:
            start_time = self.packets[0].header.timestamp
            end_time = self.packets[-1].header.timestamp
            duration = (end_time - start_time).total_seconds()
        else:
            start_time = end_time = None
            duration = 0
        
        return {
            'header': self.header.to_dict(),
            'packet_count': packet_count,
            'total_size_bytes': total_size,
            'protocol_distribution': protocols,
            'start_time': start_time.isoformat() if start_time else None,
            'end_time': end_time.isoformat() if end_time else None,
            'duration_seconds': duration
        } 
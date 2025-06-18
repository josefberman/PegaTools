"""
Tests for the utils module.
"""

import pytest
from pega_tools.utils import PegaException, validate_url


class TestPegaException:
    """Test cases for PegaException class."""
    
    def test_exception_with_message_only(self):
        """Test exception with message only."""
        exc = PegaException("Test error")
        assert str(exc) == "Test error"
        assert exc.message == "Test error"
        assert exc.error_code is None
    
    def test_exception_with_error_code(self):
        """Test exception with error code."""
        exc = PegaException("Test error", "E001")
        assert str(exc) == "[E001] Test error"
        assert exc.message == "Test error"
        assert exc.error_code == "E001"


class TestValidateUrl:
    """Test cases for validate_url function."""
    
    def test_valid_https_url(self):
        """Test valid HTTPS URL."""
        assert validate_url("https://example.com") is True
    
    def test_valid_http_url(self):
        """Test valid HTTP URL."""
        assert validate_url("http://example.com") is True
    
    def test_valid_url_with_port(self):
        """Test valid URL with port."""
        assert validate_url("https://example.com:8080") is True
    
    def test_valid_url_with_path(self):
        """Test valid URL with path."""
        assert validate_url("https://example.com/path") is True
    
    def test_localhost_url(self):
        """Test localhost URL."""
        assert validate_url("http://localhost:3000") is True
    
    def test_ip_address_url(self):
        """Test IP address URL."""
        assert validate_url("https://192.168.1.1:8080") is True
    
    def test_invalid_url_no_protocol(self):
        """Test invalid URL without protocol."""
        assert validate_url("example.com") is False
    
    def test_invalid_url_empty(self):
        """Test invalid empty URL."""
        assert validate_url("") is False
    
    def test_invalid_url_malformed(self):
        """Test invalid malformed URL."""
        assert validate_url("not-a-url") is False 
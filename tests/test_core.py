"""
Tests for the core module.
"""

import pytest
from unittest.mock import Mock, patch
from pega_tools.core import PegaClient
from pega_tools.utils import PegaException


class TestPegaClient:
    """Test cases for PegaClient class."""
    
    def test_init_basic(self):
        """Test basic initialization of PegaClient."""
        client = PegaClient("https://example.com")
        assert client.base_url == "https://example.com"
        assert client.username is None
        assert client.password is None
        assert client.timeout == 30
    
    def test_init_with_auth(self):
        """Test initialization with authentication."""
        client = PegaClient("https://example.com", "user", "pass")
        assert client.username == "user"
        assert client.password == "pass"
        assert client.session.auth == ("user", "pass")
    
    def test_init_trailing_slash_removal(self):
        """Test that trailing slash is removed from base_url."""
        client = PegaClient("https://example.com/")
        assert client.base_url == "https://example.com"
    
    @patch('pega_tools.core.requests.Session.get')
    def test_get_health_status_success(self, mock_get):
        """Test successful health status retrieval."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "ok", "version": "8.7"}
        mock_get.return_value = mock_response
        
        client = PegaClient("https://example.com")
        result = client.get_health_status()
        
        assert result == {"status": "ok", "version": "8.7"}
        mock_get.assert_called_once_with(
            "https://example.com/api/health",
            timeout=30
        )
    
    @patch('pega_tools.core.requests.Session.get')
    def test_get_health_status_failure(self, mock_get):
        """Test health status retrieval failure."""
        mock_get.side_effect = Exception("Connection failed")
        
        client = PegaClient("https://example.com")
        
        with pytest.raises(PegaException) as exc_info:
            client.get_health_status()
        
        assert "Failed to get health status" in str(exc_info.value)
    
    def test_close(self):
        """Test closing the client session."""
        client = PegaClient("https://example.com")
        with patch.object(client.session, 'close') as mock_close:
            client.close()
            mock_close.assert_called_once() 
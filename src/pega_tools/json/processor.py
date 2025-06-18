"""
JSON processing and manipulation functionality.
"""

import json
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from ..utils import PegaException


class JSONProcessor:
    """JSON processor for parsing, validation, and manipulation."""
    
    def __init__(self, encoding: str = 'utf-8'):
        """
        Initialize JSON processor.
        
        Args:
            encoding: File encoding for reading/writing JSON files
        """
        self.encoding = encoding
    
    def load_from_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load JSON data from file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Parsed JSON data
            
        Raises:
            PegaException: If file cannot be read or parsed
        """
        try:
            with open(file_path, 'r', encoding=self.encoding) as f:
                return json.load(f)
        except FileNotFoundError:
            raise PegaException(f"JSON file not found: {file_path}")
        except json.JSONDecodeError as e:
            raise PegaException(f"Invalid JSON in file {file_path}: {str(e)}")
        except Exception as e:
            raise PegaException(f"Error reading JSON file {file_path}: {str(e)}")
    
    def save_to_file(self, data: Dict[str, Any], file_path: Union[str, Path], 
                     indent: int = 2, sort_keys: bool = False) -> None:
        """
        Save JSON data to file.
        
        Args:
            data: Data to save
            file_path: Output file path
            indent: JSON indentation
            sort_keys: Whether to sort keys
            
        Raises:
            PegaException: If file cannot be written
        """
        try:
            with open(file_path, 'w', encoding=self.encoding) as f:
                json.dump(data, f, indent=indent, sort_keys=sort_keys, 
                         ensure_ascii=False)
        except Exception as e:
            raise PegaException(f"Error writing JSON file {file_path}: {str(e)}")
    
    def parse_string(self, json_string: str) -> Dict[str, Any]:
        """
        Parse JSON from string.
        
        Args:
            json_string: JSON string to parse
            
        Returns:
            Parsed JSON data
            
        Raises:
            PegaException: If string cannot be parsed
        """
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            raise PegaException(f"Invalid JSON string: {str(e)}")
    
    def to_string(self, data: Dict[str, Any], indent: Optional[int] = None, 
                  sort_keys: bool = False) -> str:
        """
        Convert data to JSON string.
        
        Args:
            data: Data to convert
            indent: JSON indentation
            sort_keys: Whether to sort keys
            
        Returns:
            JSON string
        """
        return json.dumps(data, indent=indent, sort_keys=sort_keys, 
                         ensure_ascii=False)
    
    def minify(self, data: Dict[str, Any]) -> str:
        """
        Convert data to minified JSON string.
        
        Args:
            data: Data to minify
            
        Returns:
            Minified JSON string
        """
        return json.dumps(data, separators=(',', ':'), ensure_ascii=False)
    
    def prettify(self, json_string: str, indent: int = 2) -> str:
        """
        Prettify a JSON string.
        
        Args:
            json_string: JSON string to prettify
            indent: Indentation level
            
        Returns:
            Prettified JSON string
        """
        data = self.parse_string(json_string)
        return self.to_string(data, indent=indent, sort_keys=True)
    
    def extract_values(self, data: Dict[str, Any], key: str) -> List[Any]:
        """
        Extract all values for a given key from nested JSON.
        
        Args:
            data: JSON data to search
            key: Key to search for
            
        Returns:
            List of values found for the key
        """
        values = []
        
        def _extract_recursive(obj, target_key):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k == target_key:
                        values.append(v)
                    elif isinstance(v, (dict, list)):
                        _extract_recursive(v, target_key)
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, (dict, list)):
                        _extract_recursive(item, target_key)
        
        _extract_recursive(data, key)
        return values
    
    def get_nested_value(self, data: Dict[str, Any], path: str, 
                        separator: str = '.') -> Any:
        """
        Get value from nested JSON using dot notation.
        
        Args:
            data: JSON data
            path: Path to value (e.g., 'user.profile.name')
            separator: Path separator
            
        Returns:
            Value at path or None if not found
        """
        keys = path.split(separator)
        current = data
        
        try:
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                elif isinstance(current, list) and key.isdigit():
                    current = current[int(key)]
                else:
                    return None
            return current
        except (KeyError, IndexError, TypeError):
            return None
    
    def set_nested_value(self, data: Dict[str, Any], path: str, value: Any,
                        separator: str = '.') -> None:
        """
        Set value in nested JSON using dot notation.
        
        Args:
            data: JSON data to modify
            path: Path to set value (e.g., 'user.profile.name')
            value: Value to set
            separator: Path separator
        """
        keys = path.split(separator)
        current = data
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value 
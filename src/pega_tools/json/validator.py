"""
JSON validation functionality.
"""

import json
from typing import Any, Dict, List, Optional, Union
from ..utils import PegaException


class JSONValidator:
    """JSON validation with schema support."""
    
    def __init__(self):
        """Initialize JSON validator."""
        self.errors = []
    
    def validate_syntax(self, json_string: str) -> bool:
        """
        Validate JSON syntax.
        
        Args:
            json_string: JSON string to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            json.loads(json_string)
            return True
        except json.JSONDecodeError:
            return False
    
    def validate_required_fields(self, data: Dict[str, Any], 
                                required_fields: List[str]) -> bool:
        """
        Validate that required fields are present.
        
        Args:
            data: JSON data to validate
            required_fields: List of required field names
            
        Returns:
            True if all required fields present
        """
        self.errors = []
        missing_fields = []
        
        for field in required_fields:
            if field not in data:
                missing_fields.append(field)
        
        if missing_fields:
            self.errors.append(f"Missing required fields: {', '.join(missing_fields)}")
            return False
        
        return True
    
    def validate_field_types(self, data: Dict[str, Any], 
                           field_types: Dict[str, type]) -> bool:
        """
        Validate field types.
        
        Args:
            data: JSON data to validate
            field_types: Dictionary mapping field names to expected types
            
        Returns:
            True if all field types are correct
        """
        self.errors = []
        type_errors = []
        
        for field, expected_type in field_types.items():
            if field in data:
                actual_value = data[field]
                if not isinstance(actual_value, expected_type):
                    type_errors.append(
                        f"Field '{field}' expected {expected_type.__name__}, "
                        f"got {type(actual_value).__name__}"
                    )
        
        if type_errors:
            self.errors.extend(type_errors)
            return False
        
        return True
    
    def validate_string_formats(self, data: Dict[str, Any], 
                              format_rules: Dict[str, str]) -> bool:
        """
        Validate string field formats using simple patterns.
        
        Args:
            data: JSON data to validate
            format_rules: Dictionary mapping field names to format patterns
            
        Returns:
            True if all formats are valid
        """
        import re
        
        self.errors = []
        format_errors = []
        
        format_patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'phone': r'^\+?1?-?\.?\s?\(?(\d{3})\)?[\s.-]?(\d{3})[\s.-]?(\d{4})$',
            'url': r'^https?://[^\s/$.?#].[^\s]*$',
            'uuid': r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        }
        
        for field, format_name in format_rules.items():
            if field in data and isinstance(data[field], str):
                if format_name in format_patterns:
                    pattern = format_patterns[format_name]
                    if not re.match(pattern, data[field], re.IGNORECASE):
                        format_errors.append(
                            f"Field '{field}' does not match {format_name} format"
                        )
                else:
                    format_errors.append(f"Unknown format: {format_name}")
        
        if format_errors:
            self.errors.extend(format_errors)
            return False
        
        return True
    
    def validate_ranges(self, data: Dict[str, Any], 
                       range_rules: Dict[str, Dict[str, Union[int, float]]]) -> bool:
        """
        Validate numeric field ranges.
        
        Args:
            data: JSON data to validate
            range_rules: Dictionary mapping field names to range constraints
            
        Returns:
            True if all ranges are valid
        """
        self.errors = []
        range_errors = []
        
        for field, constraints in range_rules.items():
            if field in data and isinstance(data[field], (int, float)):
                value = data[field]
                
                if 'min' in constraints and value < constraints['min']:
                    range_errors.append(
                        f"Field '{field}' value {value} is below minimum {constraints['min']}"
                    )
                
                if 'max' in constraints and value > constraints['max']:
                    range_errors.append(
                        f"Field '{field}' value {value} is above maximum {constraints['max']}"
                    )
        
        if range_errors:
            self.errors.extend(range_errors)
            return False
        
        return True
    
    def get_errors(self) -> List[str]:
        """
        Get validation errors from last validation.
        
        Returns:
            List of error messages
        """
        return self.errors.copy()


def validate_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """
    Validate JSON data against a simple schema.
    
    Args:
        data: JSON data to validate
        schema: Schema definition
        
    Returns:
        True if data matches schema
        
    Raises:
        PegaException: If validation fails
    """
    validator = JSONValidator()
    
    # Check required fields
    if 'required' in schema:
        if not validator.validate_required_fields(data, schema['required']):
            raise PegaException(f"Schema validation failed: {validator.get_errors()}")
    
    # Check field types
    if 'types' in schema:
        type_map = {
            'string': str,
            'integer': int,
            'number': (int, float),
            'boolean': bool,
            'array': list,
            'object': dict
        }
        
        field_types = {}
        for field, type_name in schema['types'].items():
            if type_name in type_map:
                field_types[field] = type_map[type_name]
        
        if not validator.validate_field_types(data, field_types):
            raise PegaException(f"Schema validation failed: {validator.get_errors()}")
    
    # Check string formats
    if 'formats' in schema:
        if not validator.validate_string_formats(data, schema['formats']):
            raise PegaException(f"Schema validation failed: {validator.get_errors()}")
    
    # Check ranges
    if 'ranges' in schema:
        if not validator.validate_ranges(data, schema['ranges']):
            raise PegaException(f"Schema validation failed: {validator.get_errors()}")
    
    return True 
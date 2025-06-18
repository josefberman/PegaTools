"""
JSON transformation functionality.
"""

from typing import Any, Dict, List, Callable, Optional


class JSONTransformer:
    """JSON data transformation utilities."""
    
    def __init__(self):
        """Initialize JSON transformer."""
        self.transformations = []
    
    def add_transformation(self, func: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
        """
        Add a transformation function.
        
        Args:
            func: Function that takes and returns a dictionary
        """
        self.transformations.append(func)
    
    def apply_transformations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply all registered transformations to data.
        
        Args:
            data: Data to transform
            
        Returns:
            Transformed data
        """
        result = data.copy()
        for transform in self.transformations:
            result = transform(result)
        return result
    
    def rename_fields(self, data: Dict[str, Any], field_mapping: Dict[str, str]) -> Dict[str, Any]:
        """
        Rename fields in JSON data.
        
        Args:
            data: Data to transform
            field_mapping: Dictionary mapping old names to new names
            
        Returns:
            Data with renamed fields
        """
        result = {}
        for key, value in data.items():
            new_key = field_mapping.get(key, key)
            result[new_key] = value
        return result
    
    def filter_fields(self, data: Dict[str, Any], include_fields: List[str]) -> Dict[str, Any]:
        """
        Filter to include only specified fields.
        
        Args:
            data: Data to filter
            include_fields: List of fields to include
            
        Returns:
            Filtered data
        """
        return {key: value for key, value in data.items() if key in include_fields}
    
    def exclude_fields(self, data: Dict[str, Any], exclude_fields: List[str]) -> Dict[str, Any]:
        """
        Filter to exclude specified fields.
        
        Args:
            data: Data to filter
            exclude_fields: List of fields to exclude
            
        Returns:
            Filtered data
        """
        return {key: value for key, value in data.items() if key not in exclude_fields}
    
    def apply_field_transformations(self, data: Dict[str, Any], 
                                   field_transforms: Dict[str, Callable]) -> Dict[str, Any]:
        """
        Apply transformations to specific fields.
        
        Args:
            data: Data to transform
            field_transforms: Dictionary mapping field names to transformation functions
            
        Returns:
            Data with transformed fields
        """
        result = data.copy()
        for field, transform_func in field_transforms.items():
            if field in result:
                result[field] = transform_func(result[field])
        return result
    
    def normalize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize data by converting all string values to lowercase and stripping whitespace.
        
        Args:
            data: Data to normalize
            
        Returns:
            Normalized data
        """
        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = value.strip().lower()
            elif isinstance(value, dict):
                result[key] = self.normalize_data(value)
            elif isinstance(value, list):
                result[key] = [
                    self.normalize_data(item) if isinstance(item, dict)
                    else item.strip().lower() if isinstance(item, str)
                    else item
                    for item in value
                ]
            else:
                result[key] = value
        return result
    
    def convert_types(self, data: Dict[str, Any], 
                     type_conversions: Dict[str, type]) -> Dict[str, Any]:
        """
        Convert field types.
        
        Args:
            data: Data to convert
            type_conversions: Dictionary mapping field names to target types
            
        Returns:
            Data with converted types
        """
        result = data.copy()
        for field, target_type in type_conversions.items():
            if field in result:
                try:
                    if target_type == bool:
                        # Handle boolean conversion specially
                        value = result[field]
                        if isinstance(value, str):
                            result[field] = value.lower() in ('true', '1', 'yes', 'on')
                        else:
                            result[field] = bool(value)
                    else:
                        result[field] = target_type(result[field])
                except (ValueError, TypeError):
                    # Skip conversion if it fails
                    pass
        return result
    
    def aggregate_fields(self, data: List[Dict[str, Any]], 
                        group_by: str, 
                        aggregations: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Aggregate data by grouping and applying aggregation functions.
        
        Args:
            data: List of data dictionaries
            group_by: Field to group by
            aggregations: Dictionary mapping field names to aggregation functions
            
        Returns:
            Aggregated data
        """
        # Group data
        groups = {}
        for item in data:
            key = item.get(group_by)
            if key not in groups:
                groups[key] = []
            groups[key].append(item)
        
        # Apply aggregations
        result = []
        for group_key, group_items in groups.items():
            aggregated = {group_by: group_key}
            
            for field, agg_func in aggregations.items():
                values = [item.get(field, 0) for item in group_items 
                         if field in item and isinstance(item[field], (int, float))]
                
                if values:
                    if agg_func == 'sum':
                        aggregated[field] = sum(values)
                    elif agg_func == 'avg':
                        aggregated[field] = sum(values) / len(values)
                    elif agg_func == 'min':
                        aggregated[field] = min(values)
                    elif agg_func == 'max':
                        aggregated[field] = max(values)
                    elif agg_func == 'count':
                        aggregated[field] = len(values)
                else:
                    aggregated[field] = 0
            
            result.append(aggregated)
        
        return result 
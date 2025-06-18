"""
JSON utility functions.
"""

from typing import Any, Dict, List, Union


def flatten_json(data: Dict[str, Any], separator: str = '.', parent_key: str = '') -> Dict[str, Any]:
    """
    Flatten nested JSON data.
    
    Args:
        data: JSON data to flatten
        separator: Separator for nested keys
        parent_key: Parent key prefix
        
    Returns:
        Flattened dictionary
    """
    items = []
    
    for key, value in data.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key
        
        if isinstance(value, dict):
            items.extend(flatten_json(value, separator, new_key).items())
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    items.extend(flatten_json(item, separator, f"{new_key}[{i}]").items())
                else:
                    items.append((f"{new_key}[{i}]", item))
        else:
            items.append((new_key, value))
    
    return dict(items)


def unflatten_json(data: Dict[str, Any], separator: str = '.') -> Dict[str, Any]:
    """
    Unflatten JSON data back to nested structure.
    
    Args:
        data: Flattened JSON data
        separator: Separator used in flattened keys
        
    Returns:
        Unflattened nested dictionary
    """
    result = {}
    
    for key, value in data.items():
        parts = key.split(separator)
        current = result
        
        for i, part in enumerate(parts[:-1]):
            # Handle array indices
            if '[' in part and ']' in part:
                base_key = part.split('[')[0]
                index = int(part.split('[')[1].split(']')[0])
                
                if base_key not in current:
                    current[base_key] = []
                
                # Extend list if necessary
                while len(current[base_key]) <= index:
                    current[base_key].append({})
                
                current = current[base_key][index]
            else:
                if part not in current:
                    current[part] = {}
                current = current[part]
        
        # Set the final value
        final_key = parts[-1]
        if '[' in final_key and ']' in final_key:
            base_key = final_key.split('[')[0]
            index = int(final_key.split('[')[1].split(']')[0])
            
            if base_key not in current:
                current[base_key] = []
            
            while len(current[base_key]) <= index:
                current[base_key].append(None)
            
            current[base_key][index] = value
        else:
            current[final_key] = value
    
    return result


def merge_json(dict1: Dict[str, Any], dict2: Dict[str, Any], 
               strategy: str = 'override') -> Dict[str, Any]:
    """
    Merge two JSON dictionaries.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary
        strategy: Merge strategy ('override', 'preserve_first', 'combine_lists')
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result:
            if strategy == 'override':
                result[key] = value
            elif strategy == 'preserve_first':
                # Keep original value, don't override
                pass
            elif strategy == 'combine_lists':
                if isinstance(result[key], list) and isinstance(value, list):
                    result[key] = result[key] + value
                elif isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_json(result[key], value, strategy)
                else:
                    result[key] = value
        else:
            result[key] = value
    
    return result


def deep_merge_json(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two JSON dictionaries, recursively merging nested objects.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary
        
    Returns:
        Deep merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result:
            if isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge_json(result[key], value)
            else:
                result[key] = value
        else:
            result[key] = value
    
    return result


def find_differences(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Find differences between two JSON dictionaries.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary
        
    Returns:
        Dictionary containing differences
    """
    differences = {
        'added': {},
        'removed': {},
        'modified': {},
        'unchanged': {}
    }
    
    all_keys = set(dict1.keys()) | set(dict2.keys())
    
    for key in all_keys:
        if key not in dict1:
            differences['added'][key] = dict2[key]
        elif key not in dict2:
            differences['removed'][key] = dict1[key]
        elif dict1[key] != dict2[key]:
            differences['modified'][key] = {
                'old': dict1[key],
                'new': dict2[key]
            }
        else:
            differences['unchanged'][key] = dict1[key]
    
    return differences


def extract_paths(data: Dict[str, Any], separator: str = '.') -> List[str]:
    """
    Extract all possible paths from nested JSON data.
    
    Args:
        data: JSON data
        separator: Path separator
        
    Returns:
        List of all paths in the data
    """
    paths = []
    
    def _extract_paths(obj, current_path=''):
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{current_path}{separator}{key}" if current_path else key
                paths.append(new_path)
                if isinstance(value, (dict, list)):
                    _extract_paths(value, new_path)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                new_path = f"{current_path}[{i}]"
                paths.append(new_path)
                if isinstance(item, (dict, list)):
                    _extract_paths(item, new_path)
    
    _extract_paths(data)
    return paths


def get_size_info(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get size information about JSON data.
    
    Args:
        data: JSON data to analyze
        
    Returns:
        Dictionary with size information
    """
    import json
    import sys
    
    def count_elements(obj):
        if isinstance(obj, dict):
            return 1 + sum(count_elements(v) for v in obj.values())
        elif isinstance(obj, list):
            return 1 + sum(count_elements(item) for item in obj)
        else:
            return 1
    
    json_string = json.dumps(data)
    
    return {
        'total_elements': count_elements(data),
        'json_size_bytes': len(json_string.encode('utf-8')),
        'json_size_chars': len(json_string),
        'memory_size_bytes': sys.getsizeof(data),
        'max_depth': _get_max_depth(data),
        'total_keys': len(flatten_json(data))
    }


def _get_max_depth(obj, current_depth=0):
    """Get maximum nesting depth of JSON object."""
    if isinstance(obj, dict):
        if not obj:
            return current_depth
        return max(_get_max_depth(v, current_depth + 1) for v in obj.values())
    elif isinstance(obj, list):
        if not obj:
            return current_depth
        return max(_get_max_depth(item, current_depth + 1) for item in obj)
    else:
        return current_depth 
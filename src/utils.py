# utils.py
import json
import logging
import networkx as nx
from typing import List, Dict, Any, Tuple
import re


def fix_trailing_comma(json_str):
    # Replace problematic trailing commas before closing brackets or braces
    fixed_str = re.sub(r',\s*}', '}', json_str)
    fixed_str = re.sub(r',\s*\]', ']', fixed_str)
    return fixed_str

def parse_json_response(s):
    # Extract content between code fences
    pattern = r'```(?:json)?\n([\s\S]*?)\n```'
    match = re.search(pattern, s)
    
    if match:
        json_str = match.group(1)
        # Fix trailing commas
        json_str = fix_trailing_comma(json_str)
        try:
            # Parse the JSON string
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return s
    return s

def save_json(data: List[Dict[str, Any]], filepath: str):
    """Saves data to a JSON file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logging.info(f"Successfully saved data to {filepath}")
    except IOError as e:
        logging.error(f"Error saving data to {filepath}: {e}")

def load_json(filepath: str) -> List[Dict[str, Any]]:
    """Loads data from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.info(f"Successfully loaded data from {filepath}")
        return data
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        return []
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {filepath}: {e}")
        return []
    except IOError as e:
        logging.error(f"Error loading data from {filepath}: {e}")
        return []

def save_jsonl(data: List[Dict[str, Any]], filepath: str):
    """Saves data to a JSON Lines file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logging.info(f"Successfully saved data to {filepath}")
    except IOError as e:
        logging.error(f"Error saving data to {filepath}: {e}")

def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Loads data from a JSON Lines file."""
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    logging.warning(f"Skipping invalid line in {filepath}: {line.strip()} - Error: {e}")
        logging.info(f"Successfully loaded {len(data)} items from {filepath}")
        return data
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        return []
    except IOError as e:
        logging.error(f"Error loading data from {filepath}: {e}")
        return []


def export_graph(graph: nx.DiGraph, output_path: str):
    """Export the graph to a file in specified format."""
    logging.info(f"Exporting graph to {output_path}...")
    # Before exporting the graph
    for node, data in graph.nodes(data=True):
        # Replace None values with strings in node attributes
        for key, value in data.items():
            if value is None:
                data[key] = "none"  # or ""

    for u, v, data in graph.edges(data=True):
        # Replace None values with strings in edge attributes
        for key, value in data.items():
            if value is None:
                data[key] = "none"  # or ""
    
    # Determine export format based on file extension
    if output_path.endswith('.gml'):
        nx.write_gml(graph, output_path)
    elif output_path.endswith('.graphml'):
        nx.write_graphml(graph, output_path)
    elif output_path.endswith('.json'):
        from networkx.readwrite import json_graph
        import json
        with open(output_path, 'w') as f:
            json.dump(json_graph.node_link_data(graph), f, indent=2)
    else:
        # Default to GML format
        nx.write_gml(graph, output_path)
    
    logging.info(f"Graph exported successfully to {output_path}")


def load_graph(filepath: str) -> nx.Graph:
    """Loads a NetworkX graph from a file."""
    try:
        graph = nx.read_gml(filepath)
        logging.info(f"Successfully loaded graph from {filepath}")
        # Ensure node attributes are correctly typed if needed after loading GML
        for node, data in graph.nodes(data=True):
             # Example: convert type back if it was stored as string
             if 'node_type' in data and isinstance(data['node_type'], str):
                 pass # Add type conversion if necessary
        return graph
    except FileNotFoundError:
        logging.error(f"Graph file not found: {filepath}")
        return nx.Graph()
    except IOError as e:
        logging.error(f"Error loading graph from {filepath}: {e}")
        return nx.Graph()
    except Exception as e:
         logging.error(f"An unexpected error occurred loading graph: {e}")
         return nx.Graph()

    
# scenario_generator.py
import logging
from typing import List
import networkx as nx
import matplotlib.pyplot as plt
import random

from src.logging_utils import setup_logging


# --- Sampling Strategies ---
SAMPLING_STRATEGY_ALL_AXIOMS = "all_axioms"
SAMPLING_STRATEGY_LEAF_CONCEPTS = "leaf_concepts"
SAMPLING_STRATEGY_AXIOMS_AND_LEAVES = "axioms_and_leaves"
SAMPLING_STRATEGY_ALL_NODES = "all_nodes"
SAMPLING_STRATEGY_ALL_CONCEPTS = "all_concepts"

def select_target_nodes(
    graph: nx.Graph,
    strategy: str = SAMPLING_STRATEGY_AXIOMS_AND_LEAVES, # Default strategy
    ) -> List[str]:
    """Selects nodes from the graph based on the chosen sampling strategy."""
    logging.info(f"Selecting target nodes using strategy: {strategy}")
    all_nodes = list(graph.nodes)
    if not all_nodes:
        logging.warning("Graph has no nodes to sample from.")
        return []

    targets = set() # Use a set to avoid duplicates if strategies overlap

    if strategy == SAMPLING_STRATEGY_ALL_AXIOMS:
        targets.update(n for n, data in graph.nodes(data=True) if data.get('node_type') == 'axiom')

    elif strategy == SAMPLING_STRATEGY_LEAF_CONCEPTS:
        # Get all concept nodes
        all_concepts = {n for n, data in graph.nodes(data=True) if data.get('node_type') == 'concept'}
        logging.info(f"Total concept nodes: {len(all_concepts)}")
        
        # A leaf concept is one that is not a parent in any 'is_example_of' relationship
        # First, identify all parent concepts (those that have children)
        parent_concepts = set()
        for u, v, data in graph.edges(data=True):
            if data.get('type') == 'is_example_of':
                # v is the parent concept in an is_example_of relationship
                parent_concepts.add(v)
        
        # Leaf concepts are those that aren't parents
        leaf_concepts = all_concepts - parent_concepts
        logging.info(f"Found {len(parent_concepts)} parent concepts and {len(leaf_concepts)} leaf concepts")
        targets.update(leaf_concepts)

    elif strategy == SAMPLING_STRATEGY_AXIOMS_AND_LEAVES:
        # Combine All Axioms and Leaf Concepts
        axiom_nodes = {n for n, data in graph.nodes(data=True) if data.get('node_type') == 'axiom'}
        targets.update(axiom_nodes)
        
        # Get all concept nodes
        all_concepts = {n for n, data in graph.nodes(data=True) if data.get('node_type') == 'concept'}
        
        # A leaf concept is one that is not a parent in any 'is_example_of' relationship
        parent_concepts = set()
        for u, v, data in graph.edges(data=True):
            if data.get('type') == 'is_example_of':
                # v is the parent concept in an is_example_of relationship
                parent_concepts.add(v)
        
        leaf_concepts = all_concepts - parent_concepts
        targets.update(leaf_concepts)
        
    elif strategy == SAMPLING_STRATEGY_ALL_NODES:
        targets.update(all_nodes)
        
    elif strategy == SAMPLING_STRATEGY_ALL_CONCEPTS:
        targets.update(n for n, data in graph.nodes(data=True) if data.get('node_type') == 'concept')

    else:
        logging.error(f"Unknown sampling strategy: {strategy}. Defaulting to all axioms and leaves.")
        axiom_nodes = {n for n, data in graph.nodes(data=True) if data.get('node_type') == 'axiom'}
        targets.update(axiom_nodes)
        
        all_concepts = {n for n, data in graph.nodes(data=True) if data.get('node_type') == 'concept'}
        parent_concepts = set()
        for u, v, data in graph.edges(data=True):
            if data.get('type') == 'is_example_of':
                parent_concepts.add(v)
        
        leaf_concepts = all_concepts - parent_concepts
        targets.update(leaf_concepts)

    target_list = list(targets)
    logging.info(f"Selected {len(target_list)} target nodes for query generation using strategy '{strategy}'.")
    return target_list

def get_node_context(node_id: str, graph: nx.DiGraph):
    """Find the root axiom node id for a given concept node id"""
    # Check if node exists in the graph
    if node_id not in graph.nodes:
        logging.warning(f"Node {node_id} not found in graph")
        return None
        
    # First check if the current node is already an axiom
    if graph.nodes[node_id].get('node_type') == 'axiom':
        return graph.nodes[node_id]
    
    # Use a breadth-first approach to find connected axioms
    visited = set([node_id])
    queue = [node_id]
    
    while queue:
        current = queue.pop(0)
        
        # Check both incoming and outgoing edges
        for edge_func in [graph.in_edges, graph.out_edges]:
            for u, v, data in edge_func(current, data=True):
                # Determine the neighbor node
                neighbor = u if v == current else v
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    
                    # Check if this neighbor is an axiom
                    if graph.nodes[neighbor].get('node_type') == 'axiom':
                        return graph.nodes[neighbor]
    
    # No axiom found in the connected component
    logging.warning(f"No axiom context found for node {node_id}")
    return None


def demo():
    setup_logging(task_name="graph_sampling")

    # --- Load the graph ---
    GRAPH_PATH = "expanded_graph.gml"
    graph = nx.read_gml(GRAPH_PATH)
    # --- Print summary statistics ---
    logging.info(f"Graph built with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    logging.info(f"Node types: {set(nx.get_node_attributes(graph, 'node_type').values())}")
    logging.info(f"Edge types: {set(nx.get_edge_attributes(graph, 'type').values())}")

    targets = select_target_nodes(graph, SAMPLING_STRATEGY_LEAF_CONCEPTS)
    logging.info(f"Selected {len(targets)} target nodes for query generation using strategy '{SAMPLING_STRATEGY_LEAF_CONCEPTS}'.")

    targets = select_target_nodes(graph, SAMPLING_STRATEGY_ALL_NODES)
    logging.info(f"Selected {len(targets)} target nodes for query generation using strategy '{SAMPLING_STRATEGY_ALL_NODES}'.")

    targets = select_target_nodes(graph, SAMPLING_STRATEGY_AXIOMS_AND_LEAVES)
    logging.info(f"Selected {len(targets)} target nodes for query generation using strategy '{SAMPLING_STRATEGY_AXIOMS_AND_LEAVES}'.")

    targets = select_target_nodes(graph, SAMPLING_STRATEGY_ALL_AXIOMS)
    logging.info(f"Selected {len(targets)} target nodes for query generation using strategy '{SAMPLING_STRATEGY_ALL_AXIOMS}'.")

    targets = select_target_nodes(graph, SAMPLING_STRATEGY_ALL_CONCEPTS)
    logging.info(f"Selected {len(targets)} target nodes for query generation using strategy '{SAMPLING_STRATEGY_ALL_CONCEPTS}'.")

    # --- Sample a target node ---
    selected_target = random.sample(targets, 1)[0]
    logging.info(f"Selected target: {selected_target}")
    
    context = get_node_context(selected_target, graph)
    logging.info(f"\nContext: {context}")

            

if __name__ == "__main__":
    demo()



    


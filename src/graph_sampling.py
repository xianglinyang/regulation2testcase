# scenario_generator.py
import logging
from typing import List
import networkx as nx


# --- Sampling Strategies ---
SAMPLING_STRATEGY_ALL_AXIOMS = "all_axioms"
SAMPLING_STRATEGY_LEAF_CONCEPTS = "leaf_concepts"
SAMPLING_STRATEGY_AXIOMS_AND_LEAVES = "axioms_and_leaves"
SAMPLING_STRATEGY_ALL_NODES = "all_nodes"

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
        # Leaf concepts = concept nodes with no outgoing 'is_a' edges (no sub-concepts)
        # Note: This definition assumes 'is_a' edges go from child to parent (sub -> super)
        # If your 'is_a' edges go parent->child, reverse the logic (in_degree == 0 for 'is_a')
        all_concepts = {n for n, data in graph.nodes(data=True) if data.get('node_type') == 'concept'}
        non_leaf_concepts = set()
        for u, v, data in graph.edges(data=True):
             # Assuming is_a edge goes FROM sub-concept TO parent-concept
             if data.get('type') == 'is_a' and u in all_concepts:
                  non_leaf_concepts.add(u) # u is a sub-concept, therefore not a leaf in this direction

        leaf_concepts = all_concepts - non_leaf_concepts
        targets.update(leaf_concepts)

    elif strategy == SAMPLING_STRATEGY_AXIOMS_AND_LEAVES:
        # Combine All Axioms and Leaf Concepts
        targets.update(n for n, data in graph.nodes(data=True) if data.get('node_type') == 'axiom')
        # --- Duplicating leaf finding logic for clarity ---
        all_concepts = {n for n, data in graph.nodes(data=True) if data.get('node_type') == 'concept'}
        non_leaf_concepts = set()
        for u, v, data in graph.edges(data=True):
             if data.get('type') == 'is_example_of' and u in all_concepts: # Assuming sub -> super
                  non_leaf_concepts.add(u)
        leaf_concepts = all_concepts - non_leaf_concepts
        # --- End Duplication ---
        targets.update(leaf_concepts)

    else:
        logging.error(f"Unknown sampling strategy: {strategy}. Defaulting to all axioms and leaves.")
        targets.update(n for n, data in graph.nodes(data=True) if data.get('node_type') == 'axiom')
        all_concepts = {n for n, data in graph.nodes(data=True) if data.get('node_type') == 'concept'}
        non_leaf_concepts = set()
        for u, v, data in graph.edges(data=True):
             if data.get('type') == 'is_example_of' and u in all_concepts: # Assuming sub -> super
                  non_leaf_concepts.add(u)
        leaf_concepts = all_concepts - non_leaf_concepts
        targets.update(leaf_concepts)


    target_list = list(targets)
    logging.info(f"Selected {len(target_list)} target nodes for query generation using strategy '{strategy}'.")
    return target_list

def get_node_context(node_id: str, graph: nx.DiGraph):
    """Find the root axiom node id for a given concept node id"""
    current_node = node_id
    visited = set()
    
    # First check if the current node is already an axiom
    if graph.nodes[current_node].get('node_type') == 'axiom':
        return graph.nodes[current_node]
        
    # Get all edges where current_node is the target
    for source, target, data in graph.in_edges(current_node, data=True):
        if source not in visited:
            visited.add(source)
            # Check all possible edge types that might connect to an axiom
            if (data.get('type') in ['related_axiom', 'relates_to', 'is_a'] and 
                graph.nodes[source].get('node_type') == 'axiom'):
                return graph.nodes[source]
            # If not a direct axiom connection, check the source node
            result = get_node_context(source, graph)
            if result is not None:
                return graph.nodes[result]
                
    return None

            

if __name__ == "__main__":
    import networkx as nx
    import matplotlib.pyplot as plt
    import random

    # --- Load the graph ---
    GRAPH_PATH = "expanded_graph.gml"
    graph = nx.read_gml(GRAPH_PATH)

    # --- Print summary statistics ---
    print(f"Graph built with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    print(f"Node types: {set(nx.get_node_attributes(graph, 'node_type').values())}")
    print(f"Edge types: {set(nx.get_edge_attributes(graph, 'type').values())}")

    targets = select_target_nodes(graph, SAMPLING_STRATEGY_AXIOMS_AND_LEAVES)

    selected_targets = random.sample(targets, 1)[0]
    print(f"Selected target: {selected_targets}")
    
    context = get_node_context(selected_targets, graph)
    print(f"\nContext: {context}")


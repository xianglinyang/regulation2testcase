'''Merge a policy axioms into a single graph'''

from src.utils import load_graph
from src.policy_extractor import Axiom
from src.llms import LLMClient
from src.graph_builder import extract_concepts
import logging
from typing import List
import networkx as nx


def is_similar_axiom(axiom: Axiom, current_axioms: List[Axiom]) -> bool:
    """Check if the axiom is similar to any of the current axioms"""
    pass


def merge_graph_nodes(axioms: List[Axiom], graph: nx.DiGraph, llm_client: LLMClient) -> nx.DiGraph:
    """Merge the axioms into the graph"""
    logging.info("Merging graph nodes...")

    current_axioms = [node for node in graph.nodes() if graph.nodes[node].get('node_type') == 'axiom']
    current_concepts = [node for node in graph.nodes() if graph.nodes[node].get('node_type') == 'concept']
    remaining_axioms = []
    # 1. merge the axioms
    for axiom in axioms:
        if not is_similar_axiom(axiom, current_axioms):
            remaining_axioms.append(axiom)

    # 2. add new axioms to the graph
    for i, axiom in enumerate(remaining_axioms, len(current_axioms)):
        # Handle both dict-like objects and Axiom dataclass instances
        if isinstance(axiom, dict):
            axiom_id = axiom.get("id", f"axiom_{i}")
            axiom_data = axiom
        else:
            axiom_id = axiom.id if hasattr(axiom, 'id') else f"axiom_{i}"
            # Convert dataclass to dictionary if needed
            axiom_data = axiom.__dict__ if hasattr(axiom, '__dict__') else {}
        
        # Add axiom node with all its attributes
        graph.add_node(axiom_id, node_type="axiom", **axiom_data)
        
        # Extract concepts from various axiom fields
        logging.info(f"Extracting concepts from axiom {axiom_id}...")
        concepts_to_add = extract_concepts(axiom_data, llm_client)

        # 3. Add concepts and connect to axiom, field as role of the concept
        for field, concepts in concepts_to_add.items():
            for concept in concepts:
                if concept not in current_concepts:
                    graph.add_node(concept, node_type="concept", label=concept, role=field)
                graph.add_edge(axiom_id, concept, type="relates_to")
                
    logging.info(f"Created graph with {graph.number_of_nodes()} nodes initially.")
    return graph

if __name__ == "__main__":
    from src.llms import OpenAILLMClient
    from src.utils import export_graph, load_graph
    from src.logging_utils import setup_logging
    from src.policy_extractor import policy_extraction, extract_axioms, pretty_print_axioms
    from src.utils import load_regulation_text

    setup_logging(task_name="graph_merger")

    new_policy_file = "/home/ljiahao/xianglin/git_space/regulation2testcase/docs/openai_new.txt"
    graph_file = "expanded_graph.gml"
    logging.info(f"Loading graph from {graph_file}...")
    
    graph = load_graph(graph_file)
    logging.info(f"Graph loaded with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")

    # --- Load new policy ---
    policy_llm_client = OpenAILLMClient(model_name="gpt-4o")
    regulation_text = load_regulation_text(new_policy_file)
    rules = policy_extraction(policy_llm_client, regulation_text)
    logging.info(f"Extracted {len(rules)} rules from the regulation text.")

    # --- Extract axioms ---
    axioms = extract_axioms(rules)
    logging.info(f"Extracted {len(axioms)} axioms from the rules.")
    # pretty print the axioms
    pretty_print_axioms(axioms)

    # --- Load old graph ---
    graph = load_graph(graph_file) 

    # --- Merge graph ---
    graph_llm_client = OpenAILLMClient(model_name="gpt-4o")
    graph = merge_graph_nodes(axioms, graph, graph_llm_client)

    # --- Export graph ---
    export_graph(graph, "merged_graph.gml")
    
    
    

    

'''
Graph Builder
Objective: To organize the extracted axioms A and their embedded concepts into a structured graph G representing the regulatory landscape.
Process:
    Node Creation: Create an AxiomNode in V for each a ∈ A. Create ConceptNodes for unique canonicalized concepts identified in the Subject, Object, and Keywords fields of the axioms. Canonicalization can use stemming, lemmatization, and potentially mapping to a controlled vocabulary or ontology (e.g., map "PII" and "Personal Identifiable Information" to the same node 'Personal Data').
    Edge Inference:
        - relates_to(AxiomNode, ConceptNode): Create edges connecting each axiom to the concept nodes representing its core subject, object, and keywords.
        - is_example_of(ConceptNode_sub, ConceptNode_super): Infer initial hierarchical links based on explicit definitions within the text ("Personal data includes name, address...") or simple subsumption identified during extraction/canonicalization. This forms the base hierarchy for expansion.
        - depends_on(AxiomNode_i, AxiomNode_j): Infer dependencies based on explicit cross-references ("See section 3.b"), shared specific conditions, or semantic similarity suggesting prerequisite relationships. Graph algorithms analyzing text references or embedding similarities might be used.

Axioms Example:
    ID: P014
    SourceLocation: Document line 14
    SourceText: Respect our safeguards—don’t circumvent safeguards or safety mitigations in our services unless supported by OpenAI (e.g., domain experts in our Red Teaming Network)⁠ or related to research conducted in accordance with our Sharing & Publication Policy⁠.
    Subject: User
    Action: circumvent
    Object: safeguards or safety mitigations
    Modality: MUST_NOT
    Method: None
    Domain: None
    Temporal: None
    Purpose: 'unless supported by OpenAI or related to research'
    Condition: {'type': 'unless', 'clause': 'supported by OpenAI or related to research conducted in accordance with Sharing & Publication Policy'}
    Keywords: ['circumvent', 'safeguards', 'safety', 'OpenAI', 'research']


Graph Building Process:
1. for each axiom, extract concepts from the axiom.
2. remove redundant and duplicate concepts.
3. create graph nodes for axioms and concepts.
4. infer additional edges like is_a, depends_on, contradicts.
5. save the graph to a file.
'''
# TODO:
# 1. merge redundant concepts
# 2. add nodes and edges to an existing graph

import logging
import networkx as nx
from typing import List, Dict, Any, Set
import re
from collections import defaultdict

from src.policy_extractor import Axiom
from src.utils import parse_json_response
from src.llms import LLMClient

# --- Concept Extraction and Canonicalization ---

def refine_concepts(concepts: Set[str], llm_client: LLMClient) -> Set[str]:
    """Refine concepts from the set."""
    prompt = f"""You are a helpful concept refinement model tasked with ensuring concepts in the corresponding set are clean, concrete, concise, and accurate. \
    Your task is to remove the redundant concepts from the set and refine the concepts to be more concise and accurate. You should NOT leave any concept out.
    
    #### Output Requirement:
    - Each concept should be a single word or phrase.
    - Each concept should be in lowercase.
    - Each concept should be unique.
    - You should remove the redundant concepts from the set and refine the concepts to be more concise and accurate.

    #### Output Format:
    ```json
    [
        "concept1",
        "concept2",
        "concept3"
    ]

    #### Concepts:
    {concepts}
    """
    response = llm_client.invoke(prompt)
    concepts = parse_json_response(response)
    logging.info(f"Refined concepts: {concepts}")
    return concepts

def extract_concepts(axiom_data: Dict[str, Any], llm_client: LLMClient) -> Set[str]:
    """Extract and canonicalize concepts from axiom data."""
    concepts = set()
    
    # Fields to extract concepts from
    fields_to_process = [
        "Subject",
        "Action",
        "Object",
        "Method",
        "Domain",
        "Purpose",
        "Condition"
    ]
    logging.info(f"Extracting concepts from axiom {axiom_data['ID']}...")
    for field in fields_to_process:
        if field in axiom_data:
            value = axiom_data[field]
            if value is None:
                continue
            if isinstance(value, list):
                # For keyword lists
                for item in value:
                    if item and isinstance(item, str):
                        concepts.add(item)
            elif value and isinstance(value, str):
                concepts.add(value)
            elif value and isinstance(value, dict):
                for k, v in value.items():
                    concepts.add(v)
            else:
                logging.warning(f"Unexpected value type: {type(value)} for field: {field}")
    
    # refine and merge concepts
    concepts = refine_concepts(concepts, llm_client)
    return concepts


def create_graph_nodes(axioms: List[Axiom], llm_client: LLMClient) -> nx.DiGraph:
    """Creates graph nodes for axioms and initial concepts."""
    logging.info("Creating graph nodes...")
    
    G = nx.DiGraph() # Using DiGraph for directional relationships
    concepts = set()

    for i, axiom in enumerate(axioms):
        # Handle both dict-like objects and Axiom dataclass instances
        if isinstance(axiom, dict):
            axiom_id = axiom.get("id", f"axiom_{i}")
            axiom_data = axiom
        else:
            axiom_id = axiom.id if hasattr(axiom, 'id') else f"axiom_{i}"
            # Convert dataclass to dictionary if needed
            axiom_data = axiom.__dict__ if hasattr(axiom, '__dict__') else {}
        
        # Add axiom node with all its attributes
        G.add_node(axiom_id, node_type="axiom", **axiom_data)
        
        # Extract concepts from various axiom fields
        logging.info(f"Extracting concepts from axiom {axiom_id}...")
        concepts_to_add = extract_concepts(axiom_data, llm_client)
        
        # Add concepts and connect to axiom
        for concept in concepts_to_add:
            concepts.add(concept)
            G.add_edge(axiom_id, concept, type="relates_to")

    # Ensure all concepts are added as nodes with proper attributes
    for concept in concepts:
        if concept not in G:
            G.add_node(concept, node_type="concept", label=concept)
        elif G.nodes[concept].get('node_type') is None:
            G.nodes[concept]['node_type'] = "concept"
            G.nodes[concept]['label'] = concept

    logging.info(f"Created graph with {G.number_of_nodes()} nodes initially.")
    return G


def infer_graph_edges(graph: nx.DiGraph):
    """Infers and adds edges like is_a, depends_on, contradicts.
    Steps:
    1. Find concepts that co-occur in the same axioms
    2. Add related_to edges between frequently co-occurring concepts
    3. Identify potential is_a hierarchical relationships
    4. Identify potential axiom dependencies
    """
    logging.info("Inferring additional graph edges...")
    
    # Dictionary to track concept co-occurrences
    concept_cooccurrence = defaultdict(set)
    
    # 1. Find concepts that co-occur in the same axioms
    for node in graph.nodes():
        if graph.nodes[node].get('node_type') == 'axiom':
            # Get all concepts connected to this axiom
            connected_concepts = [
                neighbor for neighbor in graph.neighbors(node) 
                if graph.nodes[neighbor].get('node_type') == 'concept'
            ]
            
            # Record co-occurrences
            for concept in connected_concepts:
                concept_cooccurrence[concept].update(
                    c for c in connected_concepts if c != concept
                )
    
    # 2. Add related_to edges between frequently co-occurring concepts
    for concept, related_concepts in concept_cooccurrence.items():
        for related in related_concepts:
            # Only connect if the relationship is strong enough (occurring in >1 axiom)
            if concept != related and len(graph.out_edges(concept)) > 1 and len(graph.out_edges(related)) > 1:
                if not graph.has_edge(concept, related) and not graph.has_edge(related, concept):
                    graph.add_edge(concept, related, type="related_to", weight=1)
    
    # 3. Identify potential hierarchical relationships
    identify_hierarchical_relationships(graph)
    
    # 4. Identify potential axiom dependencies
    identify_axiom_dependencies(graph)
    
    logging.info(f"Added additional edges. Graph now has {graph.number_of_edges()} edges.")


def identify_hierarchical_relationships(graph: nx.DiGraph):
    """Identify potential is_a hierarchical relationships between concepts."""
    concept_nodes = [n for n in graph.nodes() if graph.nodes[n].get('node_type') == 'concept']
    
    # Basic implementation looking for substring relationships
    for concept1 in concept_nodes:
        for concept2 in concept_nodes:
            if concept1 != concept2:
                # If one concept contains the other, it might be a broader category
                if concept1 in concept2 and len(concept1) < len(concept2):
                    graph.add_edge(concept2, concept1, type="is_example_of")
                elif concept2 in concept1 and len(concept2) < len(concept1):
                    graph.add_edge(concept1, concept2, type="is_example_of")


def identify_axiom_dependencies(graph: nx.DiGraph):
    """Identify dependencies between axioms based on shared concepts."""
    axiom_nodes = [n for n in graph.nodes() if graph.nodes[n].get('node_type') == 'axiom']
    
    for axiom1 in axiom_nodes:
        # Get concepts connected to this axiom
        axiom1_concepts = set(
            neighbor for neighbor in graph.neighbors(axiom1) 
            if graph.nodes[neighbor].get('node_type') == 'concept'
        )
        
        for axiom2 in axiom_nodes:
            if axiom1 != axiom2:
                # Get concepts connected to the second axiom
                axiom2_concepts = set(
                    neighbor for neighbor in graph.neighbors(axiom2) 
                    if graph.nodes[neighbor].get('node_type') == 'concept'
                )
                
                # If axioms share concepts, they might be related
                shared_concepts = axiom1_concepts.intersection(axiom2_concepts)
                if len(shared_concepts) >= 2:  # Threshold for related axioms
                    graph.add_edge(axiom1, axiom2, type="related_axiom", 
                                   shared_concepts=list(shared_concepts))


def build_akg(axioms: List[Axiom], llm_client: LLMClient) -> nx.DiGraph:
    """Builds the initial Axiomatic Knowledge Graph."""
    logging.info("Building Axiomatic Knowledge Graph (AKG)...")
    graph = create_graph_nodes(axioms, llm_client)
    infer_graph_edges(graph)
    logging.info(f"AKG built with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    return graph


def export_graph(graph: nx.DiGraph, output_path: str):
    """Export the graph to a file in specified format."""
    logging.info(f"Exporting graph to {output_path}...")
    # Before exporting the graph
    for node, data in graph.nodes(data=True):
        # Replace None values with strings in node attributes
        for key, value in data.items():
            if value is None:
                data[key] = "None"  # or ""

    for u, v, data in graph.edges(data=True):
        # Replace None values with strings in edge attributes
        for key, value in data.items():
            if value is None:
                data[key] = "None"  # or ""
    
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



if __name__ == "__main__":
    from src.policy_extractor import extract_axioms
    from src.llms import OpenAILLMClient
    from src.policy_loader import load_regulation_text
    from src.policy_extractor import policy_extraction, pretty_print_axioms

    POLICY_FILE_PATH = "/home/ljiahao/xianglin/git_space/regulation2testcase/docs/openai.txt"
    GRAPH_FILE = "graph.gml"
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Initialize LLM client
    policy_llm_client = OpenAILLMClient("gpt-4o")
    builder_llm_client = OpenAILLMClient("gpt-4o-mini")
    
    # Load policy text
    policy_text = load_regulation_text(POLICY_FILE_PATH)

    # extract axioms from policy text
    rules = policy_extraction(policy_llm_client, policy_text)
    axioms = extract_axioms(rules)

    # pretty print the axioms
    pretty_print_axioms(axioms)
    
    # Build knowledge graph
    graph = build_akg(axioms, builder_llm_client)
    
    # Export graph
    export_graph(graph, GRAPH_FILE)
    
    # Print summary statistics
    print(f"Graph built with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    print(f"Node types: {set(nx.get_node_attributes(graph, 'node_type').values())}")
    print(f"Edge types: {set(nx.get_edge_attributes(graph, 'type').values())}")
# graph_builder.py
import logging
import networkx as nx
from typing import List, Dict, Any, Set
from src.policy_extractor import Axiom
import re
import spacy
from collections import defaultdict

# Try to load spaCy model if available
try:
    nlp = spacy.load("en_core_web_sm")
except:
    logging.warning("spaCy model not found. Using simplified text processing.")
    nlp = None

def create_graph_nodes(axioms: List[Axiom]) -> nx.DiGraph:
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
        concepts_to_add = extract_concepts(axiom_data)
        
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

def extract_concepts(axiom_data: Dict[str, Any]) -> Set[str]:
    """Extract and canonicalize concepts from axiom data."""
    concepts = set()
    
    # Fields to extract concepts from
    fields_to_process = [
        "prohibited_action", 
        "object_context", 
        "condition", 
        "keywords"
    ]
    
    for field in fields_to_process:
        if field in axiom_data:
            value = axiom_data[field]
            if isinstance(value, list):
                # For keyword lists
                for item in value:
                    if item and isinstance(item, str):
                        canonicalized = canonicalize_concept(item)
                        if canonicalized:
                            concepts.add(canonicalized)
            elif value and isinstance(value, str) and value.lower() != "null":
                # For text fields, extract noun phrases if spaCy is available
                if nlp:
                    concepts.update(extract_noun_phrases(value))
                else:
                    # Fallback to simple tokenization
                    words = re.findall(r'\b[a-zA-Z]{3,}\b', value)
                    for word in words:
                        canonicalized = canonicalize_concept(word)
                        if canonicalized:
                            concepts.add(canonicalized)
    
    return concepts

def canonicalize_concept(concept: str) -> str:
    """Standardize concept format (lowercase, remove extra spaces)."""
    concept = concept.lower().strip()
    # Remove special characters and normalize spaces
    concept = re.sub(r'[^\w\s]', ' ', concept)
    concept = re.sub(r'\s+', ' ', concept)
    # Only return if concept is meaningful (3+ chars)
    return concept if len(concept) >= 3 else ""

def extract_noun_phrases(text: str) -> Set[str]:
    """Extract noun phrases from text using spaCy."""
    concepts = set()
    if nlp and text:
        doc = nlp(text)
        # Extract noun chunks and named entities
        for chunk in doc.noun_chunks:
            canon = canonicalize_concept(chunk.text)
            if canon:
                concepts.add(canon)
        for ent in doc.ents:
            canon = canonicalize_concept(ent.text)
            if canon:
                concepts.add(canon)
    return concepts

def infer_graph_edges(graph: nx.DiGraph, axioms: List[Axiom]):
    """Infers and adds edges like is_a, depends_on, contradicts."""
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
    identify_axiom_dependencies(graph, axioms)
    
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
                    graph.add_edge(concept2, concept1, type="is_a")
                elif concept2 in concept1 and len(concept2) < len(concept1):
                    graph.add_edge(concept1, concept2, type="is_a")

def identify_axiom_dependencies(graph: nx.DiGraph, axioms: List[Axiom]):
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

def build_akg(axioms: List[Axiom]) -> nx.DiGraph:
    """Builds the initial Axiomatic Knowledge Graph."""
    logging.info("Building Axiomatic Knowledge Graph (AKG)...")
    graph = create_graph_nodes(axioms)
    infer_graph_edges(graph, axioms)
    logging.info(f"AKG built with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    return graph

def export_graph(graph: nx.DiGraph, output_path: str):
    """Export the graph to a file in specified format."""
    logging.info(f"Exporting graph to {output_path}...")
    
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
    from src.config import POLICY_FILE_PATH, GRAPH_FILE
    from src.llms import OpenAILLMClient
    from src.policy_loader import load_policy_text
    from src.policy_extractor import policy_extraction
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Initialize LLM client
    llm_client = OpenAILLMClient("gpt-4o")
    
    # Load policy text
    policy_text = load_policy_text(POLICY_FILE_PATH)

    # extract rules from policy text
    rules = policy_extraction(llm_client, policy_text)
    
    # Extract axioms
    axioms = extract_axioms(rules)
    logging.info(f"Extracted {len(axioms)} axioms from policy.")
    
    # Build knowledge graph
    graph = build_akg(axioms)
    
    # Export graph
    export_graph(graph, GRAPH_FILE)
    
    # Print summary statistics
    print(f"Graph built with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    print(f"Node types: {set(nx.get_node_attributes(graph, 'node_type').values())}")
    print(f"Edge types: {set(nx.get_edge_attributes(graph, 'type').values())}")
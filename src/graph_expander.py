'''Graph Expander

Objective: To enrich the AKG G into G' by hierarchically decomposing abstract ConceptNodes into more concrete sub-concepts, increasing the granularity for scenario generation.
Techniques: LLM-driven Hierarchical Expansion:
    - Selection: Identify target ConceptNodes for expansion (e.g., those with high centrality, representing broad categories like 'Privacy', 'Harmful Content', 'Security Measure').
    - Contextual Prompting: Generate prompts for a capable LLM (e.g., GPT-4, Claude 3) providing the concept name, its definition (if available from an axiom), and its context within the AKG (parent concept, related axioms). Example prompt: "The regulation discusses '[Concept Name]', defined as '[Definition]'. List specific, distinct sub-categories or concrete examples of '[Concept Name]' relevant within this regulatory context (e.g., [Regulation Name]). Provide up to N distinct items."
    - Iterative Refinement: The process can be applied recursively to newly generated sub-concepts up to a defined depth or until concepts become sufficiently concrete.
    - Validation: Generated sub-concepts should be validated for relevance and accuracy. This can involve: (a) Checking if the term appears elsewhere in the RC, (b) Using embedding similarity to ensure semantic coherence with the parent, (c) Filtering based on confidence scores if the LLM provides them, (d) Limited human review for critical concepts.
'''
import logging
import networkx as nx
from typing import List

from src.llms import LLMClient, OpenAILLMClient
from src.utils import parse_json_response


EXPANSION_SYSTEM_PROMPT = """You are an advanced concept expansion model designed to expand the given concept into more concrete sub-concepts.\
Your task is to expand a given concept from a policy into more concrete and diverse sub-concepts."""

EXPANSION_USER_PROMPT = """As a concept expansion model, your tasks are as follows:
1. Carefully analyze the context of the concept.
2. Identify specific, distinct, diverse examples or sub-types of subconcepts relevant to the concept.

#### Identification Guidelines
- The sub-concepts should be specific, distinct, and diverse.
- The sub-concepts should be relevant to the concept.
- The sub-concepts should be concrete and actionable.

#### Output Format
Provide the output in the following JSON format:
```json
[
{
"concept": "The concept to expand.",
"sub-concepts": ["List of sub-concepts."],
},
...
]
```

#### Output Requirements
- Each sub-concept must be specific, distinct, and relevant to the concept and should be less than 5 words.
- Ensure each sub-concept is distinct from the concept and other sub-concepts.
- Do not combine unrelated statements into one policy block.

#### Input Context and Concept
"""
VALIDATION_SYSTEM_PROMPT = """
You are a helpful assistant that validates the sub-concepts for a given concept.
"""

VALIDATION_USER_PROMPT = """

""" 

# ---------- Helper functions ----------

def select_concepts_for_expansion(graph: nx.Graph, depth: int = 0) -> List[str]:
    """Selects concept nodes suitable for expansion."""
    # TODO: Implement selection strategy (e.g., based on node degree, abstraction level, current depth)
    candidates = [
        node for node, data in graph.nodes(data=True)
        if data.get('node_type') == 'concept' and data.get('expansion_depth', -1) < depth
    ]
    logging.info(f"Selected {len(candidates)} concept nodes for expansion at depth {depth}.")
    return candidates

# core function
def generate_subconcepts_llm(llm_client: LLMClient, concept_node_id: str, graph: nx.Graph) -> List[str]:
    """Uses LLM to generate sub-concepts for a given concept node."""
    node_data = graph.nodes[concept_node_id]
    concept_label = node_data.get('label', concept_node_id)

    context = f"Context: The policy relates to '{concept_label}'.\n"
    concept = f"Concept: {concept_label}\n"
    prompt = EXPANSION_USER_PROMPT + context + concept

    logging.debug(f"Generating sub-concepts for: {concept_label}")

    response = llm_client.invoke(
        prompt=prompt,
        system_prompt=EXPANSION_SYSTEM_PROMPT,
    )
    llm_output = parse_json_response(response)
    sub_concepts = llm_output[0]['sub-concepts']

    logging.debug(f"Generated sub-concepts for {concept_label}: {sub_concepts}")
    return sub_concepts


def add_expansion_to_graph(graph: nx.Graph, parent_node_id: str, sub_concepts: List[str], current_depth: int):
    """Adds expanded nodes and edges to the graph."""
    for sub_concept in sub_concepts:
        # TODO: Add canonicalization and validation for sub-concepts
        sub_concept_id = sub_concept.lower().strip()
        if sub_concept_id and sub_concept_id != parent_node_id:
            if sub_concept_id not in graph:
                graph.add_node(sub_concept_id, node_type="concept", label=sub_concept, expansion_depth=current_depth + 1)
            graph.add_edge(sub_concept_id, parent_node_id, type="is_a") # Child -> Parent for is_a


def expand_graph(llm_client: LLMClient, graph: nx.Graph, max_depth: int = 1) -> nx.Graph:
    """Performs semantic expansion on the graph."""
    logging.info(f"Starting graph expansion up to depth {max_depth}...")

    current_graph = graph.copy() # Work on a copy

    for depth in range(max_depth):
        logging.info(f"--- Expansion Level {depth + 1} ---")
        concepts_to_expand = select_concepts_for_expansion(current_graph, depth)
        
        if not concepts_to_expand:
            logging.info("No more concepts to expand at this level.")
            break

        for concept_id in concepts_to_expand:
            try:
                sub_concepts = generate_subconcepts_llm(llm_client, concept_id, current_graph)
                if sub_concepts:
                    add_expansion_to_graph(current_graph, concept_id, sub_concepts, depth)
            except Exception as e:
                logging.error(f"Error expanding concept {concept_id}: {e}")
        # Mark expanded nodes to prevent re-expansion in the same run if selection logic doesn't handle it
        for concept_id in concepts_to_expand:
             if 'expansion_depth' not in current_graph.nodes[concept_id]:
                  current_graph.nodes[concept_id]['expansion_depth'] = depth

    logging.info(f"Graph expansion finished. Final size: {current_graph.number_of_nodes()} nodes, {current_graph.number_of_edges()} edges.")
    return current_graph


if __name__ == "__main__":
    from src.policy_extractor import extract_axioms
    from src.config import POLICY_FILE_PATH, GRAPH_FILE
    from src.llms import OpenAILLMClient
    from src.policy_loader import load_regulation_text
    from src.policy_extractor import policy_extraction
    from src.graph_builder import build_akg, export_graph

    # --- Build graph ---
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Initialize LLM client
    policy_llm_client = OpenAILLMClient("gpt-4o")
    expansion_llm_client = OpenAILLMClient("gpt-4.1")
    
    # Load policy text
    policy_text = load_regulation_text(POLICY_FILE_PATH)

    # extract rules from policy text
    rules = policy_extraction(policy_llm_client, policy_text)
    
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


    # --- Expand graph ---
    GRAPH_EXPANDED_FILE = "expanded_graph.gml"
    expanded_graph = expand_graph(expansion_llm_client, graph, max_depth=2)
    export_graph(expanded_graph, GRAPH_EXPANDED_FILE)

    # --- Print summary statistics ---
    print(f"Expanded graph built with {expanded_graph.number_of_nodes()} nodes and {expanded_graph.number_of_edges()} edges.")
    print(f"Node types: {set(nx.get_node_attributes(expanded_graph, 'node_type').values())}")
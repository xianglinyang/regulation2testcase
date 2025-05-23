'''
Policy Extraction Stage (Phase 1 -> Axiom Nodes):
Goal: Extract the most complete representation of each rule possible.
Target Schema: Use the comprehensive schema, aiming to populate fields for:
- ID: Essential for referencing the axiom in the graph.
- Source Location: Good practice for traceability back to the original document.
- Source Text: Crucial for verification and understanding the exact wording.

- Subject: Who is the actor responsible for adhering to the rule, or who is being regulated? (e.g., "User", "Service Provider", "AI Model", "Data Controller")
- Action: The core verb or activity being performed or regulated. (e.g., "collect", "share", "generate", "display", "use", "process", "deploy").
- Object: What is the direct recipient or target of the action? (e.g., "personal data", "user content", "harmful output", "biometric information", "consent").
- Modality: The force or obligation of the rule (e.g., "MUST", "MUST_NOT", "SHOULD", "MAY", "DEFINITION").
- Method: How the action is performed, constrained, or qualified. This captures adverbs, prepositional phrases indicating means, or specific techniques. (e.g., "without consent", "using encryption", "via automated means", "in a deceptive manner", "unless anonymized").
- Domain: The scope, location, or specific situation where the rule applies. (e.g., "in public spaces", "for law enforcement purposes", "within the EU", "regarding children", "during employment").
- Temporal: Specific time elements.
- Purpose: The stated Why.
- Condition: A flexible list to handle various conditional logic (if, unless, except) clearly, separating the type from the clause text.
- Keywords: Key terms associated with the axiom.
'''
import logging
from typing import List, Dict
from src.llms import LLMClient, OpenAILLMClient
from src.utils import parse_json_response
from src.logging_utils import setup_logging
from src.policy_loader import load_regulation_text

from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Axiom:
    ID: str
    SourceLocation: str
    SourceText: str
    Subject: str
    Action: str
    Object: str
    Modality: str
    Method: str
    Domain: str
    Temporal: str
    Purpose: str
    Condition: List[str]
    Keywords: List[str]

POLICY_EXTRACT_SYSTEM_PROMPT = """You are a helpful policy extraction model to identify actionable policies from organizational safety guidelines. \
    Your task is to exhaust all the potential policies from the provided organization handbook \
    which sets restrictions or guidelines for user or entity behaviors in this organization. \
    You will extract specific elements from the given guidelines to produce structured and actionable outputs."""

POLICY_EXTRACT_PROMPT = """As a policy extraction model to clean up policies from openai, your tasks are:
1. Read and analyze the provided safety policies carefully, section by section.
2. Exhaust all axioms of actionable policy rules, each axiom is a tuple, e.g., a = <ID, Source, Subject, Predicate, Object, Modality, Condition, Keywords>.
3. For each axiom, extract the following eight elements:
    - ID: Unique identifier.
    - SourceLocation: The location of the axiom in the document.
    - SourceText: The summary of the axiom.
    - Subject: Who is the actor responsible for adhering to the rule, or who is being regulated? (e.g., "User", "Service Provider", "AI Model", "Data Controller")
    - Action: The core verb or activity being performed or regulated. (e.g., "collect", "share", "generate", "display", "use", "process", "deploy").
    - Object: What is the direct recipient or target of the action? (e.g., "personal data", "user content", "harmful output", "biometric information", "consent").
    - Modality: The force or obligation of the rule (e.g., "MUST", "MUST_NOT", "SHOULD", "MAY", "DEFINITION").
    - Method: How the action is performed, constrained, or qualified. This captures adverbs, prepositional phrases indicating means, or specific techniques. (e.g., "without consent", "using encryption", "via automated means", "in a deceptive manner", "unless anonymized").
    - Domain: The scope, location, or specific situation where the rule applies. (e.g., "in public spaces", "for law enforcement purposes", "within the EU", "regarding children", "during employment").
    - Temporal: Specific time elements.
    - Purpose: The stated Why.
    - Condition: A flexible list to handle various conditional logic (if, unless, except) clearly, separating the type from the clause text.
    - Keywords: Key terms associated with the axiom.

#### Extraction Guidelines:
• Do not summarize, modify, or simplify any part of the original policy. Copy the exact descriptions.
• Ensure each extracted axiom is self-contained and can be fully interpreted by looking at its Definition, Scope, and Policy Description.
• If any of the elements in the axiom is unclear, leave the value as None.
• Avoid grouping multiple axioms into one block. Extract axioms as individual pieces of statements.

Provide the output in the following JSON format:
```json
[
{
"ID": "Unique identifier.",
"SourceLocation": "The location of the axiom in the document.",
"SourceText": "The summary of the axiom.",
"Subject": "Who is the actor responsible for adhering to the rule, or who is being regulated? (e.g., 'User', 'Service Provider', 'AI Model', 'Data Controller').",
"Action": "The core verb or activity being performed or regulated. (e.g., 'collect', 'share', 'generate', 'display', 'use', 'process', 'deploy').",
"Object": "What is the direct recipient or target of the action? (e.g., 'personal data', 'user content', 'harmful output', 'biometric information', 'consent').",
"Modality": "The force or obligation of the rule (e.g., 'MUST', 'MUST_NOT', 'SHOULD', 'MAY', 'DEFINITION').",
"Method": "How the action is performed, constrained, or qualified. This captures adverbs, prepositional phrases indicating means, or specific techniques. (e.g., 'without consent', 'using encryption', 'via automated means', 'in a deceptive manner', 'unless anonymized').",
"Domain": "The scope, location, or specific situation where the rule applies. (e.g., 'in public spaces', 'for law enforcement purposes', 'within the EU', 'regarding children', 'during employment').",
"Temporal": "Specific time elements.",
"Purpose": "The stated Why."
"Condition": "List of textual description of circumstances under which the rule applies (e.g., 'if data is transferred outside EU', 'unless anonymized').",
"Keywords": "List of key terms associated with the axiom."
},
...
]
```

#### Output Requirement:
- Each policy must focus on explicitly restricting or guiding behaviors.
- Ensure policies are actionable and clear.
- Do not combine unrelated statements into one policy block.
- Try to be specific and concise in each field.

#### Policy Document:
"""

def extract_axioms(rules):
    axioms = []
    for rule in rules:
        axiom = Axiom(
            ID=rule["ID"],
            SourceLocation=rule["SourceLocation"],
            SourceText=rule["SourceText"],
            Subject=rule["Subject"],
            Action=rule["Action"],
            Object=rule["Object"],
            Modality=rule["Modality"],
            Method=rule["Method"],
            Domain=rule["Domain"],
            Temporal=rule["Temporal"],
            Purpose=rule["Purpose"],
            Condition=rule["Condition"],
            Keywords=rule["Keywords"]
        )
        axioms.append(axiom)
    return axioms
    
def policy_extraction(llm_client: LLMClient, regulation_text):
    prompt = POLICY_EXTRACT_PROMPT+regulation_text
    response = llm_client.invoke(prompt, system_prompt=POLICY_EXTRACT_SYSTEM_PROMPT)
    rules = parse_json_response(response)
    return rules

def pretty_print_axioms(axioms: List[Axiom]):
    # pretty print the axioms
    for axiom in axioms:
        logging.info(f"ID: {axiom.ID}")
        logging.info(f"SourceLocation: {axiom.SourceLocation}")
        logging.info(f"SourceText: {axiom.SourceText}")
        logging.info(f"Subject: {axiom.Subject}")
        logging.info(f"Action: {axiom.Action}")
        logging.info(f"Object: {axiom.Object}")
        logging.info(f"Modality: {axiom.Modality}")
        logging.info(f"Method: {axiom.Method}")
        logging.info(f"Domain: {axiom.Domain}")
        logging.info(f"Temporal: {axiom.Temporal}")
        logging.info(f"Purpose: {axiom.Purpose}")
        logging.info(f"Condition: {axiom.Condition}")
        logging.info(f"Keywords: {axiom.Keywords}")
        logging.info("-"*100)


if __name__ == "__main__":
    setup_logging(task_name="policy_extractor")

    regulation_text = load_regulation_text("/home/ljiahao/xianglin/git_space/regulation2testcase/docs/openai.txt")
    
    llm_client = OpenAILLMClient(model_name="gpt-4o")
    rules = policy_extraction(llm_client, regulation_text)
    logging.info(f"Extracted {len(rules)} rules from the regulation text.")
    
    axioms = extract_axioms(rules)
    logging.info(f"Extracted {len(axioms)} axioms from the rules.")
    # pretty print the axioms
    pretty_print_axioms(axioms)


import logging
import json
from typing import List
from src.llms import LLMClient, OpenAILLMClient
from src.utils import parse_json_response

from dataclasses import dataclass

@dataclass
class Axiom:
    id: str
    source: str
    subject: str
    predicate: str
    object: str
    modality: str
    condition: str
    keywords: List[str]

POLICY_EXTRACT_SYSTEM_PROMPT = """You are a helpful policy extraction model to identify actionable policies from organizational safety guidelines. \
    Your task is to exhaust all the potential policies from the provided organization handbook \
    which sets restrictions or guidelines for user or entity behaviors in this organization. \
    You will extract specific elements from the given guidelines to produce structured and actionable outputs."""

POLICY_EXTRACT_PROMPT = """As a policy extraction model to clean up policies from openai, your tasks are:
1. Read and analyze the provided safety policies carefully, section by section.
2. Exhaust all axioms of actionable policy rules, each axiom is a tuple, e.g., a = <ID, Source, Subject, Predicate, Object, Modality, Condition, Keywords>.
3. For each axiom, extract the following eight elements:
    - ID: Unique identifier.
    - Source: Pointer to the text span in d ∈ RC.
    - Subject: The entity primarily responsible or affected (e.g., 'Data Controller', 'User').
    - Predicate: The core action or relationship (e.g., 'must obtain', 'prohibited from sharing', 'is defined as').
    - Object: The entity acted upon or related to (e.g., 'Explicit Consent', 'Personal Data', 'Hate Speech Definition').
    - Modality: The nature of the rule (e.g., MUST, MUST_NOT, SHOULD, MAY, DEFINITION).
    - Condition: Textual description of circumstances under which the rule applies (e.g., 'if data is transferred outside EU', 'unless anonymized').
    - Keywords: Key terms associated with the axiom.

#### Extraction Guidelines:
• Do not summarize, modify, or simplify any part of the original policy. Copy the exact descriptions.
• Ensure each extracted policy is self-contained and can be fully interpreted by looking at its Definition, Scope, and Policy Description.
• If the Definition or Scope is unclear, leave the value as None.
• Avoid grouping multiple policies into one block. Extract policies as individual pieces of statements.

Provide the output in the following JSON format:
```json
[
{
"ID": "Unique identifier.",
"Source": "Pointer to the text span in d ∈ RC.",
"Subject": "The entity primarily responsible or affected (e.g., 'Data Controller', 'User').",
"Predicate": "The core action or relationship (e.g., 'must obtain', 'prohibited from sharing', 'is defined as').",
"Object": "The entity acted upon or related to (e.g., 'Explicit Consent', 'Personal Data', 'Hate Speech Definition').",
"Modality": "The nature of the rule (e.g., MUST, MUST_NOT, SHOULD, MAY, DEFINITION).",
"Condition": "Textual description of circumstances under which the rule applies (e.g., 'if data is transferred outside EU', 'unless anonymized').",
"Keywords": "Key terms associated with the axiom.",
},
...
]
```

#### Output Requirement:
- Each policy must focus on explicitly restricting or guiding behaviors.
- Ensure policies are actionable and clear.
- Do not combine unrelated statements into one policy block.

#### Policy Document:
"""

def extract_axioms(rules):
    axioms = []
    for rule in rules:
        axiom = Axiom(
            id=rule["ID"],
            source=rule["Source"],
            subject=rule["Subject"],
            predicate=rule["Predicate"],
            object=rule["Object"],
            modality=rule["Modality"],
            condition=rule["Condition"],
            keywords=rule["Keywords"]
        )
        axioms.append(axiom)
    return axioms
    
def policy_extraction(llm_client: LLMClient, regulation_text):
    prompt = POLICY_EXTRACT_PROMPT+regulation_text
    response = llm_client.invoke(prompt, system_prompt=POLICY_EXTRACT_SYSTEM_PROMPT)
    rules = parse_json_response(response)
    return rules


if __name__ == "__main__":
    from src.policy_loader import load_regulation_text
    regulation_text = load_regulation_text("/home/ljiahao/xianglin/git_space/regulation2testcase/docs/openai.txt")
    llm_client = OpenAILLMClient(model_name="gpt-4o")
    rules = policy_extraction(llm_client, regulation_text)
    print(rules)


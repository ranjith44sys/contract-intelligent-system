import re
import json
from ..utils import client
from ..config import logger

class ReferenceExtractor:
    def __init__(self):
        pass

    def extract(self, text: str) -> list:
        """
        Extracts section references (e.g. 3.1(f)) and defined term references.
        Uses a hybrid approach (Regex for sections, LLM for terms).
        """
        references = []
        
        # 1. Regex for Section References (e.g., Section 4.2, Article III, Clause 8.1(a))
        section_pattern = r'(?:Section|Article|Clause|Paragraph)\s+([0-9]+(?:\.[0-9]+)?(?:\([a-z0-9]\))?)'
        matches = re.findall(section_pattern, text, re.IGNORECASE)
        for val in set(matches):
            references.append({"type": "section", "value": val})

        # 2. LLM for Defined Term References
        system_prompt = (
            "You are a legal document analyst. "
            "Identify all specific references to other defined terms (e.g. 'Approval Order', 'Closing Date') in the text. "
            "Return a JSON array of objects: [{'type': 'defined_term', 'value': '...'}]"
        )
        
        response = client.get_completion(system_prompt, text, json_mode=True)
        
        if response:
            try:
                llm_refs = json.loads(response)
                if isinstance(llm_refs, list):
                    references.extend(llm_refs)
            except Exception:
                pass

        return references

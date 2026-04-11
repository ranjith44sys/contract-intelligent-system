import re
import json
from ..utils import client
from ..config import logger

class ClauseSplitter:
    def __init__(self):
        # We check semantic complexity if it looks long or has many sub-points
        self.split_threshold_chars = 1500 # Approx 300-400 tokens

    def split_if_needed(self, clause: dict) -> list:
        """
        Detects if a clause is overloaded and splits it into granular sub-clauses.
        """
        text = clause.get("text", "")
        
        # Simple heuristic to trigger LLM check: long text or many (a), (b), (c) patterns
        has_enumerations = bool(re.search(r'\([a-z0-9]\)', text))
        is_long = len(text) > self.split_threshold_chars

        if not (is_long or has_enumerations):
            return [clause]

        logger.info(f"Checking for semantic splitting: {clause.get('clause_title')}")
        
        system_prompt = (
            "You are a legal document structure specialist. "
            "Analyze the following clause and split it into smaller, logically independent sub-clauses IF it contains multiple distinct topics or sections. "
            "Preserve exact numbering (e.g. 8.1(a)) and the original text. "
            "If the clause is already granular, return it as a single item in the array. "
            "Return a JSON array of objects: [{'clause_title': '...', 'text': '...', 'section_number': '...'}]"
        )
        
        response = client.get_completion(system_prompt, text, json_mode=True)
        
        if not response:
            return [clause]

        try:
            sub_clauses = json.loads(response)
            if not isinstance(sub_clauses, list):
                sub_clauses = [sub_clauses]
                
            # Copy over metadata from parent
            for sub in sub_clauses:
                sub["page_number"] = clause.get("page_number")
                sub["contract_type"] = clause.get("contract_type")
                sub["raw_text"] = sub.get("text", "") # Initial raw text for sub-clause
            
            return sub_clauses
        except Exception as e:
            logger.error(f"Clause splitting failed: {e}")
            return [clause]

import re # Need re for the heuristic check

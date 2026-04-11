import json
from ..utils import client
from ..config import logger

class TypeNormalizer:
    def __init__(self):
        self.taxonomy = [
            "Parties", "Recitals", "Definitions", "Bankruptcy", "Financing",
            "Sales Terms", "Agent Rights", "Payment Terms", "Amendments",
            "Waiver", "Execution", "Interpretation", "Confidentiality",
            "Termination", "Liability", "Other"
        ]

    def normalize_type(self, raw_type: str, clause_text: str) -> dict:
        """
        Maps a fuzzy clause type into the predefined legal taxonomy.
        """
        system_prompt = (
            "You are a legal classification expert. "
            "Map the provided clause into EXACTLY ONE of the following predefined categories: "
            f"{', '.join(self.taxonomy)}. "
            "If it does not fit perfectly, pick 'Other'. "
            "Return a JSON object: {'clause_type': '...', 'confidence': float}"
        )
        
        user_prompt = f"RAW TYPE: {raw_type}\nTEXT: {clause_text[:500]}..."
        
        response = client.get_completion(system_prompt, user_prompt, json_mode=True)
        
        if not response:
            return {"clause_type": "Other", "confidence": 0.0}

        try:
            data = json.loads(response)
            mapped_type = data.get("clause_type", "Other")
            if mapped_type not in self.taxonomy:
                mapped_type = "Other"
            
            return {
                "clause_type": mapped_type,
                "confidence": data.get("confidence", 0.0)
            }
        except Exception:
            return {"clause_type": "Other", "confidence": 0.0}

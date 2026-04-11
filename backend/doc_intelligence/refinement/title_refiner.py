import json
from ..utils import client
from ..config import logger

class TitleRefiner:
    def __init__(self):
        pass

    def refine(self, clause_text: str, context: str = "") -> str:
        """
        Uses LLM to generate a precise semantic title for a clause.
        Uses the document context for better accuracy.
        """
        system_prompt = (
            "You are a legal document expert. "
            "Generate a precise and meaningful clause title based on the provided text and document context. "
            "Avoid generic titles like 'Agreement' or 'Recitals'. "
            "If the clause is about specific terms like 'Interest Rate' or 'Governing Law', reflect that exactly. "
            "Return ONLY the improved title string."
        )
        
        user_prompt = f"DOCUMENT CONTEXT:\n{context[:2000]}...\n\nCLAUSE TEXT:\n{clause_text}\n\nREFINED TITLE:"
        
        response = client.get_completion(system_prompt, user_prompt)
        
        if response:
            return response.strip().strip('"').strip("'")
        return "Untitled Clause"

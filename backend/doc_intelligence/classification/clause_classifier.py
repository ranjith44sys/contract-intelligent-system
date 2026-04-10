import json
from ..utils import client
from ..config import logger

class ClauseClassifier:
    def __init__(self):
        pass

    def classify(self, clause_text: str):
        """Classifies an individual clause into a taxonomy label."""
        system_prompt = (
            "You are a professional legal clause classifier. "
            "Categorize the following clause text into a specific standardized category (e.g., Governing Law, Indemnification, Termination). "
            "Return a JSON object with 'clause_type' and 'confidence' (0.0 to 1.0). "
            "Use Zero-Shot Learning: identify the clause type based on its inherent legal meaning without a predefined list."
        )
        
        response = client.get_completion(system_prompt, clause_text, json_mode=True)
        
        if not response:
            logger.warning("No response from clause classification API. Using default.")
            return {"clause_type": "Unknown", "confidence": 0.0}

        try:
            data = json.loads(response)
            return {
                "clause_type": data.get("clause_type", "Unknown"),
                "confidence": data.get("confidence", 0.0)
            }
        except Exception as e:
            logger.error(f"Error parsing clause classification response: {e}")
            return {"clause_type": "Others", "confidence": 0.0}

    def batch_classify(self, clauses):
        """Classifies a list of clause objects."""
        for clause in clauses:
            logger.info(f"Classifying clause: {clause.get('clause_title', 'Unnamed')}")
            result = self.classify(clause["text"])
            clause.update(result)
        return clauses

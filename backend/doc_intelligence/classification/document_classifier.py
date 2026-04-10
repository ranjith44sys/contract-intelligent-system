import json
from ..utils import client
from ..config import logger

class DocumentClassifier:
    def __init__(self):
        pass

    def classify(self, full_text):
        """Classifies the document into a contract type."""
        system_prompt = (
            "You are a professional legal document classifier. "
            "Categorize the contract type based on its inherent characteristics (e.g., Master Service Agreement, NDA, Employment Contract). "
            "Return a JSON object with 'contract_type' and a 'confidence' score (0.0 to 1.0). "
            "Base your decision on the preamble, title, and key provisions. "
            "Use Zero-Shot Learning: categorize without a predefined list."
        )
        
        # Use first 8000 tokens for context
        truncated_text = " ".join(full_text.split()[:8000])
        
        logger.info("Classifying document type...")
        response = client.get_completion(system_prompt, truncated_text, json_mode=True)
        
        if not response:
            logger.warning("No response from classification API. Using default.")
            return {"contract_type": "Unknown", "confidence": 0.0}

        try:
            data = json.loads(response)
            return {
                "contract_type": data.get("contract_type", "Unknown"),
                "confidence": data.get("confidence", 0.0)
            }
        except Exception as e:
            logger.error(f"Doc classification failed: {e}")
            return {"contract_type": "Others", "confidence": 0.0}

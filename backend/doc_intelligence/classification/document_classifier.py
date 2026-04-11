import json
from ..utils import client
from ..config import logger

class DocumentClassifier:
    def __init__(self):
        pass

    def classify(self, full_text):
        """Classifies the document into a contract type with strict intelligence rules."""
        system_prompt = (
            "You are a professional legal document classifier. "
            "Analyze the following contract text, which may contain OCR noise or missing characters. "
            "1. Categorize the contract type based on its inherent characteristics (e.g., Master Service Agreement, NDA, Lease). "
            "2. Interpret noisy or incomplete text intelligently. "
            "3. NEVER return 'Unknown' for 'contract_type' unless there is absolutely no readable text. "
            "4. Return a JSON object with 'contract_type' and 'confidence' (0.0 to 1.0)."
        )
        
        # Use first 8000 tokens for context
        truncated_text = " ".join(full_text.split()[:8000])
        
        logger.info("Classifying document type with strict rules...")
        response = client.get_completion(system_prompt, truncated_text, json_mode=True)
        
        if not response:
            logger.warning("No response from classification API. Using default.")
            return {"contract_type": "Legal Document", "confidence": 0.1}


        try:
            data = json.loads(response)
            return {
                "contract_type": data.get("contract_type", "Unknown"),
                "confidence": data.get("confidence", 0.0)
            }
        except Exception as e:
            logger.error(f"Doc classification failed: {e}")
            return {"contract_type": "Others", "confidence": 0.0}

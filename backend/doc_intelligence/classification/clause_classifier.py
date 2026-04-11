import json
from ..utils import client
from ..config import logger

class ClauseClassifier:
    def __init__(self):
        pass

    def classify(self, clause_text: str):
        """Analyzes an individual clause to extract type, summary, and key points."""
        system_prompt = (
            "You are a professional legal analyst. Analyze the following clause text, which may contain OCR noise. "
            "1. Categorize into a standardized legal category. "
            "2. Provide a short, concise 'summary' of the clause. "
            "3. Extract a list of 'key_points' (obligations or conditions). "
            "4. Even if unclear, DO NOT return empty fields. Use 'Not Found' if something is unextractable. "
            "5. Interpret noisy text intelligently to infer meaning. "
            "Return JSON with 'clause_type', 'summary', and 'key_points' (array)."
        )
        
        logger.info("Extracting clause intelligence with strict rules...")
        response = client.get_completion(system_prompt, clause_text, json_mode=True)
        
        if not response:
            logger.warning("No response from clause intelligence API. Using defaults.")
            return {
                "clause_type": "Other", 
                "summary": "Clause content extracted but summary generation failed.", 
                "key_points": ["Review required"]
            }

        try:
            data = json.loads(response)
            return {
                "clause_type": data.get("clause_type", "Other"),
                "summary": data.get("summary", "Not Found"),
                "key_points": data.get("key_points") if isinstance(data.get("key_points"), list) and data.get("key_points") else ["Not Found"]
            }

        except Exception as e:
            logger.error(f"Error parsing clause classification response: {e}")
            return {"clause_type": "Others", "confidence": 0.0}

    def batch_classify(self, clauses):
        """Classifies a list of clause objects in parallel to reduce ingestion latency."""
        from concurrent.futures import ThreadPoolExecutor
        
        def process_clause(clause):
            logger.info(f"Classifying clause: {clause.get('clause_title', 'Unnamed')}")
            result = self.classify(clause["text"])
            return clause, result

        # Using a pool of 5 to balance speed and rate limits
        with ThreadPoolExecutor(max_workers=5) as executor:
            job_results = list(executor.map(process_clause, clauses))
            
        for clause, result in job_results:
            clause.update(result)
            
        return clauses

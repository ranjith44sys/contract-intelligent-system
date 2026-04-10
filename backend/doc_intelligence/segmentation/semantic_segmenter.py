import json
from ..utils import client
from ..config import logger

class SemanticSegmenter:
    def __init__(self):
        pass

    def segment_text(self, document_text):
        """Advanced semantic segmentation for clauses."""
        system_prompt = (
            "You are a legal document structure specialist. "
            "Segment the following text into logically complete clauses. "
            "For each clause, extract: "
            "1. clause_title: The name of the section or heading. "
            "2. text: The full content of the clause. "
            "3. section_number: The numbering prefix (e.g. 1.1, 2.a). "
            "4. page_number: The physical page number referenced in the text or metadata. "
            "Return a JSON array: [{'clause_title': '...', 'text': '...', 'section_number': '...', 'page_number': int}]"
        )
        
        logger.info("Performing advanced semantic segmentation...")
        
        # Implementation Detail: Process in overlapping windows if text is too large
        # Here we use a 10kb window for demonstration
        max_len = 10000
        text_segments = [document_text[i:i + max_len] for i in range(0, len(document_text), max_len)]
        
        all_clauses = []
        for i, segment in enumerate(text_segments):
            logger.info(f"Segmenting window {i+1}/{len(text_segments)}...")
            response = client.get_completion(system_prompt, segment, json_mode=True)
            
            if not response:
                logger.warning(f"No response from segmentation API for window {i+1}. Skipping.")
                continue

            try:
                data = json.loads(response)
                # Handle gpt-4o-mini occasionally wrapping in a "clauses" key
                if isinstance(data, dict):
                    data = data.get("clauses", [data])
                
                all_clauses.extend(data)
            except Exception as e:
                logger.error(f"Segmentation failed on window {i+1}: {e}")
                
        return all_clauses

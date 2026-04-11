import json
from ..utils import client
from ..config import logger

class DefinitionExtractor:
    def __init__(self):
        pass

    def extract(self, text: str) -> list:
        """
        Extracts defined terms and their meanings from the text.
        """
        # Only check if text looks like it has definitions (e.g., quotes followed by 'means' or 'shall mean')
        if not re.search(r'["\'][\w\s]+["\']\s+(?:means|shall mean)', text, re.IGNORECASE):
            return []

        system_prompt = (
            "You are a legal document analyst. "
            "Extract all defined terms and their specific meanings from the provided text. "
            "Return a JSON array of objects: [{'term': '...', 'definition': '...'}]"
        )
        
        response = client.get_completion(system_prompt, text, json_mode=True)
        
        if not response:
            return []

        try:
            data = json.loads(response)
            if isinstance(data, dict):
                data = data.get("definitions", [data] if "term" in data else [])
            return data
        except Exception:
            return []

import re

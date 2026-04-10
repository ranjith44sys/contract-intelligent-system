import re
from ..config import logger

class TextCleaner:
    def __init__(self):
        pass

    def clean(self, text):
        """Performs production-grade text cleaning."""
        if not text:
            return ""

        # 1. Normalize Whitespace (replace multiple spaces with one, multiple newlines with two)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)

        # 2. Fix broken words (hyphenation across lines)
        # Matches "word-\n  subword"
        text = re.sub(r'(\w+)-\n\s*(\w+)', r'\1\2', text)

        # 3. Remove OCR noise (common invalid symbols)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text) # Remove non-ASCII (simple OCR noise filter)
        
        # 4. Remove obvious noise patterns like random strings
        # e.g. "|||", "---"
        text = re.sub(r'\|{3,}', ' ', text)
        text = re.sub(r'-{4,}', ' ', text)

        return text.strip()

    def clean_document_data(self, extracted_data):
        """Cleans all pages in a list of extracted data objects."""
        for page in extracted_data:
            page["text"] = self.clean(page["text"])
        return extracted_data

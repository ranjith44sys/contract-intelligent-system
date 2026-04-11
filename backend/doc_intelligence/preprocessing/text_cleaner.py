import re
from ..config import logger

class TextCleaner:
    def __init__(self):
        pass

    def clean(self, text):
        """Standardizes Markdown text from Docling."""
        if not text:
            return ""

        # 1. Normalize Horizontal spacing (preserve Markdown structure)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # 2. Fix broken words (hyphenation)
        text = re.sub(r'(\w+)-\n\s*(\w+)', r'\1\2', text)

        # 3. Clean up excessive empty lines (preserve double newlines for sections)
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)

        # 4. Remove obvious OCR artifacts that might leak from Vision fallback
        text = re.sub(r'\|{3,}', ' ', text)
        
        return text.strip()

    def clean_document_data(self, extracted_data):
        """Cleans all pages/segments in the extracted data."""
        for segment in extracted_data:
            segment["text"] = self.clean(segment["text"])
        return extracted_data

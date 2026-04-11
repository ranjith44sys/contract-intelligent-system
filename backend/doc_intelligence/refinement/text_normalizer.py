import re
import unicodedata

class TextNormalizer:
    def __init__(self):
        pass

    def normalize(self, text: str) -> str:
        """
        Cleans and normalizes text for consistent processing.
        Fixes unicode quotes, normalizes spacing, and removes common noise.
        """
        if not text:
            return ""

        # 1. Normalize Unicode characters (e.g., smart quotes to standard quotes)
        text = unicodedata.normalize('NFKC', text)
        
        # 2. Fix specific unicode quotes and dashes
        replacements = {
            '\u201c': '"', '\u201d': '"',  # Smart double quotes
            '\u2018': "'", '\u2019': "'",  # Smart single quotes
            '\u2013': '-', '\u2014': '-',  # Dashes
            '\u2022': '*',                  # Bullets
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        # 3. Normalize whitespace (preserve double newlines for sections)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        
        # 4. Remove common OCR artifacts (e.g., random dots or pipe sequences)
        text = re.sub(r'\|\s*\|+', ' ', text)
        text = re.sub(r'\.\s*\.\s*\.', '...', text)

        return text.strip()

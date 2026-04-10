import re
from ..config import logger

class RegexSegmenter:
    def __init__(self):
        # Patterns for section headers
        # 1. Section Numbering: 1, 1.1, 2.3.4
        # 2. Keywords: Section, Clause, Article
        # 3. All Caps Headings: TITLE OF SECTION
        self.patterns = [
            r'^(?:Section|Clause|Article)\s+\d+(?:\.\d+)*',  # Section 1, Clause 1.1
            r'^\d+(?:\.\d+)+\s+[A-Z]',                        # 1.1 Header
            r'^[A-Z][A-Z\s]{5,}(?:\n|$)',                     # ALL CAPS HEADING
        ]
        self.combined_pattern = re.compile('|'.join(self.patterns), re.MULTILINE)

    def segment(self, text: str):
        """Identifies potential clause boundaries using regex."""
        segments = []
        matches = list(self.combined_pattern.finditer(text))
        
        if not matches:
            logger.warning("No regex segment boundaries found. Returning full text as one segment.")
            return [{"title": "Full Document", "text": text, "start": 0, "end": len(text)}]

        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i+1].start() if i + 1 < len(matches) else len(text)
            
            title = match.group().strip().split('\n')[0]
            segment_text = text[start:end].strip()
            
            segments.append({
                "clause_title": title,
                "text": segment_text,
                "start": start,
                "end": end
            })
            
        return segments

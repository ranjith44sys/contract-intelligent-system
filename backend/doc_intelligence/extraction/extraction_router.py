from .hybrid_extractor import HybridExtractor
from ..config import logger

class ExtractionRouter:
    def __init__(self, file_path):
        self.file_path = file_path
        self.hybrid_extractor = HybridExtractor(file_path)

    def extract_full_document(self, progress_callback=None):
        """
        Orchestrates extraction using the v4 Hybrid Engine:
        PyMuPDF (Text) + Camelot (Tables) + GPT-4o Vision (Images).
        """
        logger.info(f"Starting v4 Hybrid Extraction for: {self.file_path}")
        
        try:
            extracted_pages = self.hybrid_extractor.extract_all(progress_callback=progress_callback)
            return extracted_pages
        except Exception as e:
            logger.error(f"Hybrid Extraction failed: {e}")
            return [{"page_number": 1, "text": f"Extraction failed: {e}"}]

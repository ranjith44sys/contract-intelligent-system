from .pymupdf_extractor import PyMuPDFExtractor
from .llm_fallback_extractor import LLMFallbackExtractor
from ..config import logger

class ExtractionRouter:
    def __init__(self, file_path, threshold=50):
        self.file_path = file_path
        self.threshold = threshold
        self.pymu_extractor = PyMuPDFExtractor(file_path)
        self.llm_extractor = LLMFallbackExtractor(file_path)

    def extract_full_document(self):
        """Orchestrates extraction for the entire document page by page."""
        page_count = self.pymu_extractor.get_page_count()
        logger.info(f"Processing {page_count} pages with ExtractionRouter.")
        
        full_extracted_data = []

        for i in range(1, page_count + 1):
            # 1. Try PyMuPDF
            text = self.pymu_extractor.extract_page_text(i)
            has_images = self.pymu_extractor.has_images(i)
            
            # 2. Decision Logic
            use_fallback = False
            if len(text) < self.threshold:
                logger.info(f"Low text content on page {i} ({len(text)} chars).")
                use_fallback = True
            elif has_images:
                logger.info(f"Page {i} contains images. Routing to LLM Vision fallback for high precision.")
                use_fallback = True
            
            # 3. Route
            if use_fallback:
                final_text = self.llm_extractor.extract_page_text_vision(i)
                # If vision fails, keep the raw text as fallback
                if not final_text:
                    logger.warning(f"Vision fallback failed on page {i}. Using PyMuPDF text.")
                    final_text = text
            else:
                final_text = text

            full_extracted_data.append({
                "page_number": i,
                "text": final_text
            })

        return full_extracted_data

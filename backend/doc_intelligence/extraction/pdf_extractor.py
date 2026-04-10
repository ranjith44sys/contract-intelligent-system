import pdfplumber
from ..config import logger
from ..utils import client

class PDFExtractor:
    def __init__(self, file_path):
        self.file_path = file_path

    def extract_pages(self):
        """Extracts raw text page by page and refines it using LLM."""
        extracted_data = []
        
        try:
            with pdfplumber.open(self.file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_number = i + 1
                    logger.info(f"Extracting raw text from page {page_number}...")
                    
                    raw_text = page.extract_text()
                    if not raw_text:
                        logger.warning(f"No text found on page {page_number}.")
                        refined_text = ""
                    else:
                        refined_text = self._refine_text_with_llm(raw_text, page_number)
                    
                    extracted_data.append({
                        "page": page_number,
                        "text": refined_text,
                        "raw_text": raw_text
                    })
        except Exception as e:
            logger.error(f"Failed to open/process PDF {self.file_path}: {e}")
            
        return extracted_data

    def _refine_text_with_llm(self, raw_text, page_number):
        """Uses LLM to clean up raw text (removing headers/footers/noise)."""
        system_prompt = (
            "You are a document extraction assistant. "
            "Your task is to take raw text from a PDF page and return a clean, "
            "logically structured version of it. "
            "Remove obvious headers, footers, page numbers, and artifacts. "
            "Maintain the original meaning and legal structure. "
            "If the page contains a table that is poorly formatted, try to preserve its rows and columns in a readable text format."
        )
        user_prompt = f"Page Number: {page_number}\n\nRaw Text:\n{raw_text}"
        
        refined = client.get_completion(system_prompt, user_prompt)
        return refined if refined else raw_text

if __name__ == "__main__":
    # Test snippet (can be run with: python -m doc_intelligence.extraction.pdf_extractor)
    import sys
    if len(sys.argv) > 1:
        extractor = PDFExtractor(sys.argv[1])
        pages = extractor.extract_pages()
        for p in pages:
            print(f"--- Page {p['page']} ---")
            print(p['text'][:500] + "...")

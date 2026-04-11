import fitz
import camelot
import os
import warnings
from .llm_fallback_extractor import LLMFallbackExtractor
from ..config import logger

# Suppress noisy Camelot warnings (Ghostscript, etc.)
warnings.filterwarnings("ignore", category=UserWarning, module="camelot")

class HybridExtractor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.vision_extractor = LLMFallbackExtractor(file_path)

    def extract_all(self, progress_callback=None):
        """
        Orchestrates the Triple-Engine extraction:
        1. PyMuPDF (Text)
        2. Camelot (Tables)
        3. GPT-4o Vision (Images/Complex Pages)
        """
        doc = fitz.open(self.file_path)
        total_pages = len(doc)
        logger.info(f"Hybrid Extraction started for {total_pages} pages...")
        
        full_extracted_data = []

        for i, page in enumerate(doc):
            page_num = i + 1
            if progress_callback:
                progress_callback(float(i+1)/total_pages, f"Parsing Page {page_num}/{total_pages}")
            
            logger.info(f"Processing Page {page_num}/{total_pages}...")
            
            # Detect Images
            image_list = page.get_images(full=True)
            has_images = len(image_list) > 0
            
            # 1. Image Case -> GPT-4o Vision (High Quality)
            if has_images:
                logger.info(f"Page {page_num}: Images detected. Triggering Vision Engine...")
                page_text = self.vision_extractor.extract_page_text_vision(page_num)
                full_extracted_data.append({
                    "page_number": page_num,
                    "text": page_text,
                    "type": "vision"
                })
                continue

            # 2. Text & Table Case
            # Extract standard text first
            text = page.get_text("text").strip()
            
            # Detect Tables with Camelot (Stream flavor for robust GS-less extraction)
            logger.info(f"Page {page_num}: Checking for tables with Camelot...")
            try:
                tables = camelot.read_pdf(self.file_path, pages=str(page_num), flavor='stream')
                if tables.n > 0:
                    logger.info(f"Page {page_num}: Found {tables.n} tables.")
                    table_texts = []
                    for table in tables:
                        table_texts.append(table.df.to_markdown(index=False))
                    
                    # Interleave table data
                    combined_text = text + "\n\n### TABLE DATA ###\n\n" + "\n\n".join(table_texts)
                    full_extracted_data.append({
                        "page_number": page_num,
                        "text": combined_text,
                        "type": "hybrid_text_table"
                    })
                else:
                    # Clean text only
                    full_extracted_data.append({
                        "page_number": page_num,
                        "text": text,
                        "type": "text"
                    })
            except Exception as e:
                logger.warning(f"Page {page_num}: Camelot failed ({e}). Falling back to pure text.")
                full_extracted_data.append({
                    "page_number": page_num,
                    "text": text,
                    "type": "text"
                })

        doc.close()
        return full_extracted_data

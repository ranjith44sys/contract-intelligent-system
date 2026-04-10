import fitz
from ..config import logger

class PyMuPDFExtractor:
    def __init__(self, file_path):
        self.file_path = file_path

    def extract_page_text(self, page_number):
        """Extracts raw text from a specific page using PyMuPDF (fitz)."""
        try:
            doc = fitz.open(self.file_path)
            page = doc.load_page(page_number - 1)
            text = page.get_text("text") # "text" mode preserves layout better for legal
            doc.close()
            return text.strip()
        except Exception as e:
            logger.error(f"PyMuPDF failed on page {page_number}: {e}")
            return ""

    def get_page_count(self):
        try:
            doc = fitz.open(self.file_path)
            count = doc.page_count
            doc.close()
            return count
        except Exception as e:
            logger.error(f"Failed to get page count for {self.file_path}: {e}")
            return 0

    def has_images(self, page_number):
        """Checks if a page contains images."""
        try:
            doc = fitz.open(self.file_path)
            page = doc.load_page(page_number - 1)
            images = page.get_images(full=True)
            doc.close()
            return len(images) > 0
        except Exception as e:
            logger.error(f"Failed to check images on page {page_number}: {e}")
            return False

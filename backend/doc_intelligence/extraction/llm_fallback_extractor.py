import fitz
import base64
import io
from PIL import Image
from ..utils import client
from ..config import logger

class LLMFallbackExtractor:
    def __init__(self, file_path):
        self.file_path = file_path

    def extract_page_text_vision(self, page_number):
        """Converts a page to an image and uses GPT-4o Vision for extraction."""
        logger.info(f"Triggering LLM Vision fallback for page {page_number}...")
        
        try:
            # 1. Convert page to image using PyMuPDF
            doc = fitz.open(self.file_path)
            page = doc.load_page(page_number - 1)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) # Higher resolution for better OCR
            img_data = pix.tobytes("png")
            doc.close()

            # 2. Encode to base64
            base64_image = base64.b64encode(img_data).decode('utf-8')

            # 3. Call OpenAI Vision
            system_prompt = (
                "You are an advanced legal document OCR system. "
                "Extract all text from the document image accurately. "
                "Preserve headings, clauses, and formatting. "
                "Return ONLY the raw text without any commentary."
            )
            
            # Note: We need a utility that supports vision. I'll update utils.py or add a local method.
            # Using the existing client for now assuming I add vision support to it.
            
            response = self._get_vision_completion(system_prompt, base64_image)
            return response if response else ""

        except Exception as e:
            logger.error(f"LLM Vision fallback failed on page {page_number}: {e}")
            return ""

    def _get_vision_completion(self, system_prompt, base64_image):
        """Helper to call OpenAI with an image."""
        try:
            from openai import OpenAI
            from ..config import OPENAI_API_KEY, OPENAI_BASE_URL
            
            # We use gpt-4o for vision as requested, with the correct base_url for OpenRouter
            client_vision = OpenAI(
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL
            )
            response = client_vision.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract the text from this legal document page image."},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                            }
                        ]
                    }
                ],
                max_tokens=2048,
                temperature=0
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Vision API call failed: {e}")
            return None

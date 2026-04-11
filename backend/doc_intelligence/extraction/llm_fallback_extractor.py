import pypdfium2 as pdfium
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
            # 1. Convert page to image using pypdfium2
            pdf = pdfium.PdfDocument(self.file_path)
            page = pdf[page_number - 1]
            # Render at 2x scale for better OCR quality
            bitmap = page.render(scale=2)
            pil_image = bitmap.to_pil()
            
            # Save to bytes
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='PNG')
            img_data = img_byte_arr.getvalue()
            
            pdf.close()

            # 2. Encode to base64
            base64_image = base64.b64encode(img_data).decode('utf-8')

            # 3. Call OpenAI Vision
            system_prompt = (
                "You are an advanced legal document OCR system. "
                "Extract all text from the document image accurately. "
                "Preserve headings, clauses, and formatting. "
                "Return ONLY the raw text without any commentary."
            )
            
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
            
            client_vision = OpenAI(
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL,
                default_headers={
                    "HTTP-Referer": "https://github.com/ranjith44sys/contract-intelligent-system",
                    "X-Title": "Contract Intelligent System",
                    "Authorization": f"Bearer {OPENAI_API_KEY}" # Explicitly set for OpenRouter stability
                }
            )
            response = client_vision.chat.completions.create(
                model="openai/gpt-4o",
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
                max_tokens=1024,
                temperature=0
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Vision API call failed: {e}")
            return None

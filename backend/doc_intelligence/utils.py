import os
import time
from openai import OpenAI
from .config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, OPENAI_TEMPERATURE, logger

class OpenAIClient:
    def __init__(self):
        self.client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            default_headers={
                "HTTP-Referer": "https://github.com/ranjith44sys/contract-intelligent-system",
                "X-Title": "Contract Intelligent System",
                "Authorization": f"Bearer {OPENAI_API_KEY}" # Explicitly set for OpenRouter stability
            }
        )

    def get_completion(self, system_prompt: str, user_prompt: str, json_mode: bool = False):
        max_retries = 3
        retry_delay = 5
        timeout = 60
        
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"API Completion Attempt {attempt}/{max_retries}...")
                response_format = {"type": "json_object"} if json_mode else {"type": "text"}
                response = self.client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=OPENAI_TEMPERATURE,
                    response_format=response_format,
                    max_tokens=4096,
                    timeout=timeout
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"API Attempt {attempt} failed: {e}")
                if attempt < max_retries:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"All {max_retries} attempts failed. Returning None.")
                    return None

# Singleton-like instance
client = OpenAIClient()

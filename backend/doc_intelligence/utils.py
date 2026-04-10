import os
from openai import OpenAI
from .config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, OPENAI_TEMPERATURE, logger

class OpenAIClient:
    def __init__(self):
        self.client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL
        )

    def get_completion(self, system_prompt: str, user_prompt: str, json_mode: bool = False):
        try:
            response_format = {"type": "json_object"} if json_mode else {"type": "text"}
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=OPENAI_TEMPERATURE,
                response_format=response_format
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return None

# Singleton-like instance
client = OpenAIClient()

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load the same .env file the app uses
load_dotenv('.env')
key = os.getenv('OPENAI_API_KEY')

print(f"DEBUG: Key exists: {bool(key)}")
if key:
    print(f"DEBUG: Key starts with: {key[:15]}...")

client = OpenAI(
    api_key=key,
    base_url='https://openrouter.ai/api/v1',
    default_headers={
        'HTTP-Referer': 'http://localhost',
        'X-Title': 'Diagnostic Test'
    }
)

print("Testing OpenRouter connection...")
try:
    resp = client.chat.completions.create(
        model='openai/gpt-4o-mini',
        messages=[{'role': 'user', 'content': 'hi'}]
    )
    print("SUCCESS: Connection established.")
    print(f"RESPONSE: {resp.choices[0].message.content}")
except Exception as e:
    print(f"FAILURE: {e}")

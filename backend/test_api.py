import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
BASE_URL = "http://127.0.0.1:8000/api"
API_KEY = os.getenv("DOC_INTEL_API_KEY", "your_secret_key")
SAMPLE_PDF = "doc_intelligence/sample_contract.pdf" # Change this to a real path

def test_health():
    response = requests.get("http://127.0.0.1:8000/")
    print(f"Health Check: {response.status_code} - {response.json()}")

def test_extraction():
    payload = {
        "file_path": SAMPLE_PDF,
        "api_key": API_KEY
    }
    
    print(f"Testing Extraction API with file: {SAMPLE_PDF}")
    response = requests.post(f"{BASE_URL}/extract", json=payload)
    
    if response.status_code == 200:
        print("Success!")
        print(json.dumps(response.json(), indent=2)[:1000] + "...")
    else:
        print(f"Failed: {response.status_code} - {response.text}")

if __name__ == "__main__":
    # Ensure uvicorn is running first!
    try:
        test_health()
        test_extraction()
    except Exception as e:
        print(f"Connection error: {e}. Is the server running (uvicorn app.main:app)?")

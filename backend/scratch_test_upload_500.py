import requests
import json

url = "http://localhost:8000/api/pipeline/upload"
files = [('files', ('test.pdf', b'%PDF-1.4 test content', 'application/pdf'))]

try:
    response = requests.post(url, files=files)
    print(f"Status Code: {response.status_code}")
    print(f"Content-Type: {response.headers.get('Content-Type')}")
    print(f"Response Body: {response.text}")
except Exception as e:
    print(f"Request failed: {e}")

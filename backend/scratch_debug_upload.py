import requests

url = "http://localhost:8000/api/pipeline/upload"
files = [('files', ('test.pdf', b'%PDF-1.4 test content', 'application/pdf'))]

try:
    response = requests.post(url, files=files)
    print(f"Status Code: {response.status_code}")
    print(f"Response Body: {response.text[:500]}")
except Exception as e:
    print(f"Request failed: {e}")

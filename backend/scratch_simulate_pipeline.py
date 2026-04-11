import sys
import os
import json
from doc_intelligence.main import run_advanced_pipeline

# Add current dir to path
sys.path.append(os.getcwd())

test_pdf = "contract_018.pdf"

print(f"--- SIMULATING FULL PIPELINE ON: {test_pdf} ---")
try:
    # Run pipeline (it might take time due to 402 errors in vision if it hits them)
    # But PyMuPDF should succeed first!
    result = run_advanced_pipeline(test_pdf, api_key="test_key_placeholder")
    print("PIPELINE EXECUTION FINISHED.")
    print(f"RESULT KEYS: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
    if "error" in result:
        print(f"ERROR RETURNED: {result['error']}")
except Exception as e:
    print(f"PIPELINE CRASHED: {e}")

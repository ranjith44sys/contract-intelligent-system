import sys
import os
import traceback

# Add current dir to path
sys.path.append(os.getcwd())

print("--- DIAGNOSTIC: IMPORT CHECK ---")
try:
    from doc_intelligence.main import run_advanced_pipeline
    print("SUCCESS: doc_intelligence.main imported successfully.")
except Exception:
    print("FAILURE: Error importing doc_intelligence.main")
    traceback.print_exc()

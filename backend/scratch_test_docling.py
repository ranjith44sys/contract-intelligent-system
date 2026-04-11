import sys
import os
from doc_intelligence.extraction.docling_extractor import DoclingExtractor

# Add current dir to path
sys.path.append(os.getcwd())

test_pdf = "contract_018.pdf"

if not os.path.exists(test_pdf):
    # Try alternate path
    test_pdf = os.path.join("doc_intelligence", "data", "sample_contract.pdf")

print(f"--- TESTING DOCLING FIX ON: {test_pdf} ---")
try:
    extractor = DoclingExtractor(test_pdf)
    markdown = extractor.extract_as_markdown()
    if markdown:
        print(f"SUCCESS: Extracted {len(markdown)} characters.")
    else:
        print("FAILURE: Extraction returned empty string.")
except Exception as e:
    print(f"CRITICAL ERROR: {e}")

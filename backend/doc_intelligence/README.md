# Document Intelligence System

A modular Python-based system for high-performance legal document extraction, classification, and segmentation.

## Features
1. **Text Extraction**: Uses `pdfplumber` for raw extraction and `gpt-4o-mini` for intelligent refinement.
2. **Table Extraction**: Uses `Camelot` to extract structured data from digital PDFs.
3. **Document Classification**: Broad legal taxonomy classification.
4. **Hybrid Segmentation**: Combines Regex (structure detection) with Semantic (LLM) refinement.
5. **Clause Classification**: Maps clauses to a rich legal taxonomy.
6. **Clause Graph**: Generates a hierarchical JSON representation of the document.

## Prerequisites
- Python 3.8+
- **Ghostscript**: Required by `camelot-py`.
  - **Windows**: Download and install from [Ghostscript Downloads](https://ghostscript.com/releases/gsdnld.html). Ensure the `bin` and `lib` folders are in your PATH.
  - **Linux**: `sudo apt install ghostscript`
  - **macOS**: `brew install ghostscript`

## Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - Copy `.env.example` to `.env` (if provided) or create a `.env` file.
   - Add your `OPENAI_API_KEY`.

## Usage
Run the pipeline against a PDF file:
```bash
python main.py --file path/to/document.pdf
```

## Folder Structure
- `extraction/`: PDF and Table extractors.
- `classification/`: Document and Clause classifiers with taxonomy/normalization.
- `segmentation/`: Regex and Semantic segmenters.
- `graph/`: JSON graph builder.
- `output/`: Processed JSON results.

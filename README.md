# Contract Intelligent System (V4)

An enterprise-grade Document Intelligence system for deep analysis and structuring of complex legal documents.

## 🚀 Recent Updates (V4)
- **Refinement Layer**: Added 7-stage post-processing for high-fidelity clause analysis.
- **Docling Integration**: Superior layout-aware extraction for tables and hierarchies.
- **Reference Linking**: Automatic extraction of cross-section references and defined terms.
- **Semantic Splitting**: Intelligently decomposes large clauses into granular RAG-friendly chunks.
- **Parallel Processing**: 5x faster post-processing using concurrent execution.

## 📂 Project Structure
```text
doc_intelligence/
  extraction/      # Docling, PyMuPDF, and Vision fallback
  preprocessing/   # Text cleaning and normalization
  segmentation/    # Semantic clause detection
  classification/  # Zero-shot document and clause typing
  refinement/      # [NEW] 7-stage refinement & definition extraction
  graph/           # Knowledge graph construction
  validation/      # JSON schema and integrity checks
```

## 🛠 Setup & Installation

1. **Environment**:
   - Python 3.10+
   - Configure `.env` with `OPENAI_API_KEY` (OpenRouter supported).

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Application**:
   ```bash
   uvicorn app.main:app --reload
   ```

## 📊 API Overview

### POST `/api/extract`
Upload a PDF to receive a structured Knowledge Graph of the contract.

**Key Refinement Features**:
- **Semantic Titles**: meaningful titles generated via LLM.
- **Definitions**: Nested terms and meanings per clause.
- **References**: Linked cross-references.
- **Confidence**: Weighted multi-stage confidence scoring.

## 📄 Approach
For a deep dive into the technology stack and processing logic, refer to [approach.md](./doc_intelligence/approach.md).

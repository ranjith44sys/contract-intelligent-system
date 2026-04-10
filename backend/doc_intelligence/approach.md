# Implementation Approach: Document Intelligence (V3 Advanced)

This document outlines the architecture, tools, and methodologies used in the current production-grade implementation of the Document Intelligence system.

## 1. Objective
To build a high-performance system that transforms complex, unstructured legal PDFs into a structured "Clause Graph" (JSON). The system is optimized for RAG (Retrieval-Augmented Generation) and capable of handling image-only pages.

## 2. Tools & Technologies
- **Main Logic**: Python 3
- **Core Extraction**: `PyMuPDF` (High-speed text and layout extraction).
- **Vision Fallback**: OpenAI `gpt-4o` (Vision) for OCR and high-precision extraction of image-heavy pages.
- **LLM Engine**: OpenAI/OpenRouter (Support for `gpt-4o-mini`).
- **Processing**: `tiktoken` (Exact token counting for RAG) and `jsonschema` (Output validation).
- **Architecture**: Granular modular pipeline with distinct stages for extraction, cleaning, chunking, and validation.

## 3. Core Approach

### A. Hybrid Extraction Layer
The system uses an **Intelligent Extraction Router**:
- **Standard Path**: Uses PyMuPDF for fast, structural text extraction.
- **Vision Fallback**: If a page is detected as image-only or has low text density, the system converts it to a high-resolution image and uses GPT-4o Vision for precise transcription.

### B. Production-Grade Preprocessing
Raw text is rarely clean. The `TextCleaner` module performs:
- **Noise Filtering**: Removes OCR artifacts and random symbols.
- **Hyphenation Correction**: Automatically rejoins words that were split across line breaks.
- **Normalization**: Standardizes whitespaces and structural markers for downstream processing.

### C. Zero-Shot Classification
To escape the limitations of hardcoded taxonomies, the system now uses **Zero-Shot Learning (ZSL)**:
- **Dynamic Categorization**: The LLM categorizes contracts and clauses based on their inherent legal meaning without being confined to a predefined list.
- **High Flexibility**: The system automatically adapts to niche contract types and novel legal clauses.

### D. RAG-Ready Semantic Chunking
- **Token-Based**: Text is split using a sliding window approach optimized for specific token limits (GPT context windows).
- **Semantic Mapping**: Every chunk is explicitly tagged with its parent `Clause ID`, `Location (Page)`, and `Contract Type`, ensuring high-precision retrieval during the guidance/reasoning phase.

### E. Resilient Graph Generation & Validation
- **Schema Enforcement**: The final output is validated against a strict JSON schema to ensure consumption by the backend is always error-free.
- **Retry Logic**: All LLM-based stages use exponential backoff with jitter to handle intermittent API failures or rate limits, ensuring maximum reliability.

## 4. Integration
The system is integrated into a **FastAPI backend**, exposed via a secure `/api/extract` endpoint with internal API key authentication and support for custom base URLs (e.g., OpenRouter).

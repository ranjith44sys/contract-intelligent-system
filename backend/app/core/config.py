import os
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from doc_intelligence.utils import client
from doc_intelligence.config import OPENAI_MODEL

# Base Directory Handling
# Current file is in backend/app/core/config.py
# Project root is backend/
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Data Directories
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "doc_intelligence" / "output"

# Ensure directories exist
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# LLM 
MODEL_NAME_GENERATION = OPENAI_MODEL

# Retrieval Models
logger_name = "app.core.config"
import logging
logger = logging.getLogger(logger_name)

logger.info(f"Initializing Paths - BASE: {BASE_DIR}")
logger.info("Initializing Dense Model (all-MiniLM-L6-v2)...")
dense_model = SentenceTransformer('all-MiniLM-L6-v2')

logger.info("Initializing Cross-Encoder (ms-marco-MiniLM-L-6-v2)...")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

__all__ = ["client", "MODEL_NAME_GENERATION", "dense_model", "cross_encoder", "BASE_DIR", "UPLOADS_DIR", "OUTPUT_DIR"]

import os
from pathlib import Path
from dotenv import load_dotenv
import logging

# Determine the project root (backend folder)
BASE_DIR = Path(__file__).resolve().parent.parent
env_path = BASE_DIR / ".env"

# Load environment variables from backend/.env (Override existing to ensure .env is source of truth)
load_dotenv(dotenv_path=env_path, override=True)

# OpenAI Configuration - NO HARDCODING
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0))

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("doc_intelligence")

if not OPENAI_API_KEY:
    logger.warning(f"OPENAI_API_KEY not found in {env_path}. Ensure it is set in the backend/.env file.")

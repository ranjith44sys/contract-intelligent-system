import os
from .config import logger

def verify_api_key(api_key: str) -> bool:
    """Verifies the provided API key against the one in .env."""
    expected_key = os.getenv("DOC_INTEL_API_KEY") # Separate key for internal auth
    if not expected_key:
        logger.warning("DOC_INTEL_API_KEY not set in .env. Authentication bypassed.")
        return True
    
    return api_key == expected_key

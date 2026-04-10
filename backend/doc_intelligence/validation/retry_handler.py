import time
import random
from ..config import logger

def retry_with_exponential_backoff(
    func,
    max_retries=3,
    initial_delay=1,
    backoff_factor=2,
    jitter=True
):
    """Retries a function call with exponential backoff and optional jitter."""
    retries = 0
    delay = initial_delay
    
    while retries < max_retries:
        try:
            return func()
        except Exception as e:
            retries += 1
            if retries >= max_retries:
                logger.error(f"Max retries reached. Error: {e}")
                raise e
            
            # Calculate sleep time
            sleep_time = delay * (backoff_factor ** (retries - 1))
            if jitter:
                sleep_time += random.uniform(0, 0.1 * sleep_time)
            
            logger.warning(f"Retry {retries}/{max_retries} following error: {e}. Sleeping for {sleep_time:.2f}s...")
            time.sleep(sleep_time)
            
    return None

import sys
import os

# Add the backend directory to sys.path
sys.path.append(os.getcwd())

try:
    print("Testing imports in app.api.routes...")
    from app.api.routes import router
    print("SUCCESS: Routes imported.")
except ImportError as e:
    print(f"IMPORT ERROR: {e}")
except Exception as e:
    print(f"ERROR: {e}")

# config.py
import os
from dotenv import load_dotenv

def get_google_api_key():
    load_dotenv()
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "GOOGLE_API_KEY not set.\n"
            "Set it as an environment variable or in a .env file."
        )
    return key

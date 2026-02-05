# config.py
import os
from dotenv import load_dotenv

ENV_FILE = ".env"

def get_google_api_key():
    load_dotenv()

    key = os.getenv("GOOGLE_API_KEY")
    if key:
        return key

    # First-time setup
    key = input("Enter your Google API key: ").strip()
    if not key:
        raise RuntimeError("API key cannot be empty.")

    # Save to .env
    with open(ENV_FILE, "a") as f:
        f.write(f"\nGOOGLE_API_KEY={key}\n")

    os.environ["GOOGLE_API_KEY"] = key
    return key

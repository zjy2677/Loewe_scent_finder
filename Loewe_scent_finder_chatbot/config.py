# config.py
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from google.genai.errors import ClientError

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

GEMINI_MODEL_FALLBACK = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-3-flash",
]

def get_gemini_llm(temperature=0.5):
    last_error = None

    for model in GEMINI_MODEL_FALLBACK:
        try:
            return ChatGoogleGenerativeAI(
                model=model,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=temperature,
            )
        except ClientError as e:
            # Quota / rate limit / model unavailable
            last_error = e
            if "RESOURCE_EXHAUSTED" in str(e) or "quota" in str(e).lower():
                print(f"[WARN] {model} exhausted, switching model…")
                continue
            else:
                raise  # real error, don't hide it

    raise RuntimeError(
        "All Gemini models are exhausted. Please try again later."
    ) from last_error

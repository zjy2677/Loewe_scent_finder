# import all necessary packages
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import json
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import time
import os

# import all helper functions from other document 
from config import get_google_api_key
from intent import route_intent
from handlers import (
    handle_recommendation,
    handle_order,
    handle_complaint,
    handle_product_info,
    handle_out_of_scope,
    greet_user,
)
from retrieval import build_bm25

try:
    import langchain
except ImportError:
    raise RuntimeError(
        "Dependencies not installed.\n"
        "Run: pip install -r requirements.txt"
    )

def main():
    print(greet_user())
    # Build BM25 index ONCE
    json_path = "data/perfumes.json"
    data, bm25 = build_bm25(json_path)
    history = []
    while True:
        user_text = input("You (type 'exit' to quit, 'history' to view): ").strip()

        if user_text.lower() == "exit":
            break

        if user_text.lower() == "history":
            if not history:
                print("(no history yet)")
            else:
                for i, (u, a) in enumerate(history, 1):
                    print(f"\n[{i}] You: {u}\n    Assistant: {a}")
            continue

        user_intent = route_intent(user_text)

        if user_intent == "recommendation":
            assistant_text = handle_recommendation(user_text, history, data, bm25)

        elif user_intent == "order":
            assistant_text = handle_order()

        elif user_intent == "complaint":
            assistant_text = handle_complaint()

        elif user_intent == "product_info":
            assistant_text = handle_product_info()

        else:
            assistant_text = handle_out_of_scope()

        print("\nAssistant:", assistant_text)
        history.append((user_text, assistant_text))

if __name__ =="__main__":
  api_key = input("Please provide your api key first:")
  os.environ["GOOGLE_API_KEY"]="api_key"
  load_dotenv()
  print(f"Your Google API KEY {os.getenv("GOOGLE_API_KEY")} is verified")
  main()
      


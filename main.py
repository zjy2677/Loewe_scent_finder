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
from config.py import xxxx
from intent.py import xxxx
from retrieval.py import xxxx
from response_handler.oy import xxxx

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
            assistant_text = demosimple(user_text, history, data, bm25)

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
  main()
      


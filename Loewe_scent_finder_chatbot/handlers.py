import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from Loewe_scent_finder_chatbot.retrieval import bm25_retrieve, candidate_to_context
from Loewe_scent_finder_chatbot.config import get_gemini_llm

def handle_recommendation(user_text: str, history, data, bm25):
   return demosimple(user_text, history, data, bm25)
  

def handle_order():
  return (
        "I can help with orders, shipping, and returns.\n"
        "Please provide your order number or tell me what you'd like to check."
    )


def handle_complaint():
  return (
        "I'm sorry to hear that. Could you describe the issue in more detail?\n"
        "For example: damaged item, wrong product, or delivery delay."
    )

def handle_out_of_scope():
   return (
        "I do not understand your questions. I'm here to help with perfume recommendations and product support.\n"
        "Could you rephrase your question or tell me what you're looking for?"
    )
   
def handle_product_info():
  return(
      "I "
  )
   
def greet_user():
    return (
        "Hello 👋 Welcome!\n"
        "I can help you:\n"
        "• find a perfume recommendation\n"
        "• answer product questions\n"
        "• help with orders or issues\n\n"
        "Just tell me what you're looking for 😊"
    )

def demosimple(user_text: str, history, data, bm25):
    # Optional: use a little history to enrich retrieval query
    last_user_turns = " ".join([u for u, _ in history[-2:]]) if history else ""
    query = (last_user_turns + " " + user_text).strip()

    # Retrieve top 20 from your JSON using BM25
    candidates = bm25_retrieve(query, data, bm25, k=20)
    context = "\n---\n".join(candidate_to_context(p) for p in candidates)

    prompt = ChatPromptTemplate.from_template(
        """You are a perfume recommendation assistant.
You MUST recommend only from the CANDIDATES provided.
If the user asks for something not supported by the candidates, say so and pick the closest matches.

User request:
{query}

CANDIDATES:
{context}

Return exactly 4 recommendations as:

1) Name — Brand (ID)
   Why it matches (2 bullets)
   Evidence (quote 1 short line from the candidate text)

Be concise and do not invent notes/accords.
"""
    )

   '''
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.5,
    )
    '''
    llm = get_gemini_llm(temperature=0.5)

    chain = prompt | llm
    resp = chain.invoke({"query": user_text, "context": context}).content
    return resp

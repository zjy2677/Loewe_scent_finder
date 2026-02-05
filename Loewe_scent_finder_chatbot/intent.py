import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from Loewe_scent_finder_chatbot.intent_config import INTENT_KEYWORDS, INTENTS, INTENT_PROMPT
from Loewe_scent_finder_chatbot.config import get_gemini_llm

def route_intent(text: str) -> str:
    text = text.lower()
    scores = {k: 0 for k in INTENT_KEYWORDS}

    for intent, keywords in INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                scores[intent] += 1
    best_intent, best_score = max(scores.items(), key=lambda x: x[1])
    CONFIDENCE_THRESHOLD = 2
    # set up a llm here to deal with more complex intent
    if best_score >= CONFIDENCE_THRESHOLD:
      return best_intent
    '''
    llm_intent_decider = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.0,
    )
    '''
    llm_intent_decider = get_gemini_llm(temperature=0.0)
    chain = INTENT_PROMPT | llm_intent_decider
    llm_intent_resp = chain.invoke({"text": text}).content.strip().lower()

    if llm_intent_resp in INTENTS:
        return llm_intent_resp

    return "out_of_scope"


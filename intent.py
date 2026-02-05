INTENT_KEYWORDS = {
    "recommendation": [
        "recommend", "suggest", "looking for", "similar to",
        "perfume", "fragrance", "scent", "smell"
    ],
    "order": [
        "order", "shipping", "delivery", "return", "refund",
        "price", "cost", "available", "in stock"
    ],
    "complaint": [
        "broken", "damaged", "wrong", "late", "complaint"
    ],
    "product_info": [
        "what is", "tell me about", "notes", "accords", "brand"
    ],
}

INTENTS = {
    "recommendation",
    "order",
    "complaint",
    "product_info",
    "out_of_scope",
}

INTENT_PROMPT = ChatPromptTemplate.from_template(
    """Classify the user's message into exactly ONE of the following categories:

- recommendation
- order
- complaint
- product_info
- out_of_scope

Message:
"{text}"

Return ONLY the category name.
"""
)

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

    llm_intent_decider = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.0,
    )
    chain = INTENT_PROMPT | llm_intent_decider
    llm_intent_resp = chain.invoke({"text": text}).content.strip().lower()

    if llm_intent_resp in INTENTS:
        return llm_intent_resp

    return "out_of_scope"


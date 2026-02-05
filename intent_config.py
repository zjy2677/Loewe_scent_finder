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

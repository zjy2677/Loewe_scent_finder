import json, re
from rank_bm25 import BM25Okapi

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def perfume_to_search_text(p: dict) -> str:
    md = p.get("matched_data1") or {}
    notes = md.get("notes") or {}

    parts = [
        p.get("title",""),
        p.get("description",""),
        md.get("name",""),
        md.get("brand",""),
        md.get("sub-brand",""),
        " ".join(md.get("main accords", []) or []),
        " ".join(notes.get("top notes", []) or []),
        " ".join(notes.get("middle notes", []) or []),
        " ".join(notes.get("base notes", []) or []),
        p.get("product_type",""),
        p.get("level_1",""),
        p.get("level_2",""),
    ]
    return normalize(" ".join([x for x in parts if x]))

def build_bm25(json_path: str):
    data = json.load(open(json_path, "r", encoding="utf-8"))
    corpus = [perfume_to_search_text(p) for p in data]
    tokenized = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized)
    return data, bm25


def bm25_retrieve(query: str, data, bm25, k=20):
    q = normalize(query).split()
    scores = bm25.get_scores(q)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [data[i] for i in top_idx]

def candidate_to_context(p: dict) -> str:
    md = p.get("matched_data1") or {}
    notes = (md.get("notes") or {})
    return (
        f"- {md.get('name', p.get('title',''))} | {md.get('brand','')}"
        f" | accords: {', '.join(md.get('main accords', [])[:8])}"
        f" | top: {', '.join(notes.get('top notes', [])[:6])}"
        f" | mid: {', '.join(notes.get('middle notes', [])[:6])}"
        f" | base: {', '.join(notes.get('base notes', [])[:6])}"
        f" | price: {p.get('price','')}"
        f" | link: {p.get('link','')}"
    )

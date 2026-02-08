from recommender.rerank import rerank_similar_scents
from recommender.similarity import get_similar_scents
from recommender.utils import build_terms_by_scent, list_canonical_scents

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

# ---- 1. Load data -------------------------------------------------
scents = pd.read_pickle(BASE_DIR / "data" / "scents.pkl")
olfactory_terms_long = pd.read_pickle(
    BASE_DIR / "data" / "olfactory_terms_long.pkl")
terms_by_scent = build_terms_by_scent(olfactory_terms_long)

# ---- 2. Choose query ----------------------------------------------
available_scents = list_canonical_scents(scents)

print("\nCanonical scents (sample):")
for s in available_scents[:20]:
    print(" -", s)
print(f"\nTotal available scents: {len(available_scents)}")

QUERY_SCENT = input(
    "\nType a canonical title (copy-paste recommended): ").strip()

print("=" * 60)
print(f"Query scent: {QUERY_SCENT}")
print("=" * 60)

# ---- 3. Stage 1: text similarity ----------------------------------
try:
    candidates = get_similar_scents(
        canonical_title_query=QUERY_SCENT,
        scents_df=scents,
        top_k=10,
    )
except ValueError as e:
    print(f"\nError: {e}")
    raise SystemExit(1)

print("\nTop candidates after TEXT similarity:\n")
print(candidates[["canonical_title", "desc_similarity_score",
      "has_structured"]].head(10).to_string(index=False))

# ---- 4. Stage 2: structured reranking -----------------------------
reranked = rerank_similar_scents(
    similar_scents_df=candidates,
    canonical_title_query=QUERY_SCENT,
    terms_by_scent=terms_by_scent,
    alpha=0.7,
)

print("\nTop recommendations after RERANKING:\n")
print(reranked[["canonical_title", "final_similarity_score"]].head(
    4).to_string(index=False))

print("\nDemo completed successfully.")

This project implements a perfume recommender for LOEWE fragrances, combining semantic similarity from perfume descriptions with structured olfactory-note reranking.
## Recommender setup
1. User inputs a perfume  
2. Resolve input to `canonical_title`  
3. Description retrieval: output `desc_similarity_score` using cosine similarity based on perfume description embeddings  
   - 768-dimensional vectors from `sentence-transformers/all-mpnet-base-v2`
4. Retrieve top-10 candidate scents  
5. Structured reranking: output `notes_similarity_score` using weighted Jaccard similarity on:
   - main accords
   - top notes
   - middle notes
   - base notes
6. Return top-4 recommendations 

## How to run the demo 
```bash
python demo_recommender.py
```

## Inputs and outputs

1. Input: canonical perfume title, ie, extracted perfume titles without volume information
2. Output: a ranked list of recommended perfumes with similarity scores, which is a weighted sum of description similarity score and notes similarity scores. Weight is predefined.

## Notes

- Embeddings are precomputed and stored in `scents.pkl`
- Structured reranking applies only when the olfactory data for both query item and candidate items are available; if not, recommendations are based solely on `desc_similarity_score`
- This prototype is to test core logic. No UI or free-text search is implemented yet



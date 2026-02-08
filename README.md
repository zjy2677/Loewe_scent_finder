This project implements a perfume recommender for LOEWE fragrances, combining semantic similarity from perfume descriptions with structured olfactory-note reranking.
## Recommender setup
1. User selects a perfume  
2. Resolve input to `canonical_title`  
3. Description retrieval: output `desc_similarity_score` that measures semantic similarity based on perfume description embeddings  
   - 768-dimensional vectors from `sentence-transformers/all-mpnet-base-v2`
4. Retrieve top-10 candidate scents  
5. Structured reranking: `output notes_similarity_score` using weighted Jaccard similarity on:
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

1. Input: canonical perfume title, ie, perfume titles without volume information
2. Output: a ranked list of recommended perfumes with similarity scores, a weighted sum of description similarity score and notes similarity scores

## Notes

- Embeddings are precomputed and stored in `scents.pkl`
- Structured reranking applies only when olfactory data is available
- No UI or free-text search is implemented yet

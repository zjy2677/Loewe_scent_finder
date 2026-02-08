import pandas as pd
import numpy as np


def rerank_similar_scents(similar_scents_df: pd.DataFrame, canonical_title_query: str, terms_by_scent: dict, alpha=0.6) -> pd.DataFrame:
    '''
    Input: 
    scents_df: containing canonical_title, description, sex, has_structured, desc_similarity_score
    terms_by_scent: a dict of dict, where the first level key is canonical_title, and the value is another dict {olfactory_term: weight, ...} 
    alpha: weight on description similarity
    '''
    query_terms = terms_by_scent.get(canonical_title_query, {})
    aggregated_scents = similar_scents_df.copy()
    # 1. if query terms is empty, return desc_similarity_score
    if not query_terms:
        aggregated_scents['final_similarity_score'] = aggregated_scents['desc_similarity_score']
        return aggregated_scents.sort_values('final_similarity_score', ascending=False)

    note_similarity_scores = []
    # always have values
    candidate_scents = similar_scents_df['canonical_title'].tolist()
    for can in candidate_scents:
        candidate_terms = terms_by_scent.get(can, {})
        if not candidate_terms:
            # to exclude candidates with no olfactory terms
            note_similarity_scores.append(np.nan)
            continue
        note_similarity_score = compute_weighted_jaccard_similarity(
            query_terms, candidate_terms)
        note_similarity_scores.append(note_similarity_score)

    aggregated_scents['note_similarity_score'] = note_similarity_scores
    aggregated_scents['final_similarity_score'] = np.where(
        aggregated_scents['note_similarity_score']. notna(),
        alpha * aggregated_scents['desc_similarity_score'] +
        (1 - alpha) * aggregated_scents['note_similarity_score'],
        aggregated_scents['desc_similarity_score']
    )
    reranked_scents = aggregated_scents.sort_values(
        'final_similarity_score', ascending=False)
    return reranked_scents


def compute_weighted_jaccard_similarity(dict1: dict, dict2: dict) -> float:
    '''
    Compute the weighted Jaccard similarity between two dictionaries of olfactory terms and their associated weights. 
    The keys of the dictionary are the olfactory terms, and the values are the weights (importance) of those terms for the scent. 
    The weighted Jaccard similarity is defined as: 
        sum of min(weights of shared terms)) / sum of max(weights of all terms in both scents))

    Input: 
    dict1, dict2: two dictionaries of the form {term: weight, ...}
    '''
    # 1. If either of the scents has no olfactory terms, similarity is 0 -> not dict is True when it's empty
    if not dict1 or not dict2:
        return 0.0
    # 2. Get the set of all unique terms across both scents
    all_terms = set(dict1.keys()).union(set(dict2.keys()))
    # 3. Compute the sum of min weights for shared terms and sum of max weights
    sum_min_weights = 0
    sum_max_weights = 0
    for term in all_terms:
        weight1 = dict1.get(term, 0)
        weight2 = dict2.get(term, 0)
        sum_min_weights += min(weight1, weight2)
        sum_max_weights += max(weight1, weight2)
    # 4. Compute weighted Jaccard similarity
    weighted_jaccard = sum_min_weights / sum_max_weights if sum_max_weights > 0 else 0
    return weighted_jaccard

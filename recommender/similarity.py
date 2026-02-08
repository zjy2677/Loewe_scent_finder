import pandas as pd
import numpy as np


def get_similar_scents(canonical_title_query: str, scents_df: pd.DataFrame, top_k=5):
    ''' 
    Given the canonical_title_query, return the top_k most similar scents based on cosine similarity 
    between the user query's description embedding and the scent description embeddings

    Input:
        canonical_title_query: the canonical_title of the query scent. Do not accept free-form text input / fuzzy name input. Raise value error 
        if the input is not found in the scents_df['canonical_title'].
        scents_df: the dataframe containing the canonical scents informaion
        top_k: how many similar scents tp return

    Output: a dataframe with top_k most similar scents, sorted by similarity score in descending order.
    '''
    # 1. Validate input and get embedding for the query scent
    if canonical_title_query not in scents_df['canonical_title'].values:
        raise ValueError(
            f"Unknown canonical_title_query: {canonical_title_query}.Query should be one of: {scents_df['canonical_title'].values}"
        )
    query_embedding = np.array(scents_df.loc[scents_df['canonical_title'] ==

                                             canonical_title_query, 'description_embeddings'].iloc[0])
    query_index = scents_df[scents_df['canonical_title']
                            == canonical_title_query].index[0]

    # 2. Compute cosine similarity between query embedding and embedding matrix
    embedding_matrix = np.vstack(
        scents_df['description_embeddings'].to_numpy())
    similarity_scores = np.dot(
        embedding_matrix, query_embedding)

    # 3. Get top_k most similar scents (excluding the query scent itself)
    similarity_scores[query_index] = -1
    top_k_indices = np.argsort(similarity_scores)[-top_k:][::-1]
    # 4. Return the top_k most similar scents as a dataframe, storing their similarity scores
    similar_scents = scents_df.loc[top_k_indices, ['canonical_title', 'description_clean',
                                                   'sex', 'has_structured']].rename(columns={'description_clean': 'description'})
    similar_scents['desc_similarity_score'] = similarity_scores[top_k_indices]

    return similar_scents

from collections import defaultdict
import pandas as pd


def list_canonical_scents(scents_df: pd.DataFrame) -> list:
    """
    Returns a list of canonical titles from the scents dataframe.
    """
    return scents_df['canonical_title'].unique().tolist()


def build_terms_by_scent(olfactory_terms_df: pd.DataFrame) -> dict:
    terms_by_scent = defaultdict(dict)
    for row in olfactory_terms_df.itertuples(index=False):
        canonical_title = row.canonical_title
        term = row.olfactory_term
        weight = row.term_weight
        terms_by_scent[canonical_title][term] = weight
    return terms_by_scent

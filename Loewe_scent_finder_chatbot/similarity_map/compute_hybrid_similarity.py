# compute_hybrid_similarity.py
r"""
Compute a hybrid similarity model for Loewe perfumes and export:

- output/neighbors_topk.json
- output/scent_map.json
- output/scent_map.html               (optional)
- output/scent_map_editorial.html     (optional)

It expects axis scores produced by compute_axis_scores.py.

Hybrid similarity:
  S = w_axes * cos(axes_vectors)
    + w_struct * struct_similarity(accords+notes+type)
    + w_text  * cos(TFIDF(description+pros+cons))

Default weights: w_axes=0.60, w_struct=0.25, w_text=0.15 (must sum to 1.0)
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Helpers
# ----------------------------
HTML_ENT_RE = re.compile(r"&ndash;|&mdash;|&amp;|&quot;|&apos;", re.IGNORECASE)

def norm_text(s: str) -> str:
    s = s or ""
    s = HTML_ENT_RE.sub(" ", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize_simple(s: str) -> List[str]:
    s = norm_text(s).lower()
    s = re.sub(r"[^a-z0-9\s\-\/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return [t for t in s.split(" ") if t]

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def safe_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []

def safe_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}

def join_pipe(items: List[str]) -> str:
    return " | ".join([i.strip() for i in items if isinstance(i, str) and i.strip()])

def flatten_opinions(p: Dict[str, Any]) -> Tuple[str, str]:
    pros = p.get("pros")
    cons = p.get("cons")
    if isinstance(pros, str) or isinstance(cons, str):
        return (norm_text(str(pros or "")), norm_text(str(cons or "")))

    m = safe_dict(p.get("matched_data1"))
    opinions = safe_dict(m.get("opinions"))
    pros_list = safe_list(opinions.get("pros"))
    cons_list = safe_list(opinions.get("cons"))

    pros_txt, cons_txt = [], []
    for it in pros_list:
        if isinstance(it, dict) and isinstance(it.get("text"), str):
            pros_txt.append(it["text"])
        elif isinstance(it, str):
            pros_txt.append(it)
    for it in cons_list:
        if isinstance(it, dict) and isinstance(it.get("text"), str):
            cons_txt.append(it["text"])
        elif isinstance(it, str):
            cons_txt.append(it)

    return (norm_text(join_pipe(pros_txt)), norm_text(join_pipe(cons_txt)))

def extract_notes_and_accords(p: Dict[str, Any]) -> Tuple[List[str], List[str], List[str], List[str]]:
    m = safe_dict(p.get("matched_data1"))
    accords = [a for a in safe_list(m.get("main accords")) if isinstance(a, str)]
    notes_obj = safe_dict(m.get("notes"))
    top = [n for n in safe_list(notes_obj.get("top notes")) if isinstance(n, str)]
    mid = [n for n in safe_list(notes_obj.get("middle notes")) if isinstance(n, str)]
    base = [n for n in safe_list(notes_obj.get("base notes")) if isinstance(n, str)]
    return accords, top, mid, base

TYPE_BUCKET = {
    "male": "masc",
    "men": "masc",
    "man": "masc",
    "female": "fem",
    "women": "fem",
    "woman": "fem",
    "unisex": "uni",
}

def extract_type(p: Dict[str, Any]) -> str:
    m = safe_dict(p.get("matched_data1"))
    t = m.get("type")
    if isinstance(t, str):
        return TYPE_BUCKET.get(t.strip().lower(), t.strip().lower())
    t2 = p.get("type")
    return TYPE_BUCKET.get(t2.strip().lower(), t2.strip().lower()) if isinstance(t2, str) else ""

# ----------------------------
# Text sanitization (remove canonical titles + generic perfume words)
# ----------------------------

GENERIC_PERFUME_STOPWORDS = {
    # English
    "fragrance", "fragrances", "perfume", "perfumes", "scent", "scents", "smell", "smells",
    "aroma", "aromas", "olfactory", "spray", "sprays", "eau", "parfum", "toilette", "cologne",
    "bottle", "bottles", "flask", "flasks", "atomiser", "atomizer", "mist",
    "collection", "range", "line", "edition", "limited", "new", "classic",
    # Common stopwords
    "and", "the",

    # French-ish (optional, since your site copy may contain some)
    "parfum", "parfums", "odeur", "odeurs", "senteur", "senteurs", "flacon", "flacons",
}

def build_title_patterns(perfumes: List[Dict[str, Any]]) -> List[re.Pattern]:
    """
    Build regex patterns to remove canonical titles from text.
    We compile multiple patterns because titles vary (LOEWE, EDP/EDT, punctuation…).
    """
    patterns: List[re.Pattern] = []
    seen = set()

    for p in perfumes:
        raw = str(p.get("canonical_title") or p.get("title") or "").strip()
        if not raw:
            continue

        t = norm_text(raw)

        # Remove very generic tokens that often appear in titles
        t = re.sub(r"\bLOEWE\b", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\b(EAU\s+DE\s+PARFUM|EAU\s+DE\s+TOILETTE|EDP|EDT|PARFUM|ELIXIR)\b", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s+", " ", t).strip()

        if not t or len(t) < 3:
            continue

        key = t.lower()
        if key in seen:
            continue
        seen.add(key)

        # Escape, but allow flexible whitespace
        esc = re.escape(t)
        esc = esc.replace(r"\ ", r"\s+")
        patterns.append(re.compile(rf"\b{esc}\b", re.IGNORECASE))

    # Sort longer first to avoid partial removals
    patterns.sort(key=lambda r: len(r.pattern), reverse=True)
    return patterns

def scrub_text(text: str, title_patterns: List[re.Pattern]) -> str:
    """
    Remove all canonical titles + clean whitespace.
    """
    s = norm_text(text)
    for pat in title_patterns:
        s = pat.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_stopwords(perfumes: List[Dict[str, Any]], title_patterns: List[re.Pattern]) -> set:
    """
    Stopwords = sklearn English + generic perfume words + tokens from canonical titles.
    """
    stop = set(GENERIC_PERFUME_STOPWORDS)

    # Also stop the tokens from titles (e.g., "esencia", "aire", "solo", "001")
    for p in perfumes:
        raw = str(p.get("canonical_title") or p.get("title") or "")
        cleaned = scrub_text(raw, title_patterns)  # will remove other titles too; still fine
        toks = tokenize_simple(cleaned)
        for t in toks:
            if len(t) >= 2:
                stop.add(t)

    return stop


# ----------------------------
# Axis scores loading
# ----------------------------
RADAR_AXES = [
    "Fresh & Aquatic",
    "Fruity & Citrusy",
    "Floral & Green",
    "Woody & Spicy",
    "Sweet & Gourmand",
    "Powdery & Musky",
]

def load_axis_scores(path: Path) -> Dict[str, Dict[str, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("axis_scores.json must be a dict keyed by perfume id.")
    # light validation
    out: Dict[str, Dict[str, float]] = {}
    for pid, scores in data.items():
        if not isinstance(scores, dict):
            continue
        out[pid] = {ax: float(scores.get(ax, 0.0)) for ax in RADAR_AXES}
    return out

# ----------------------------
# Similarities
# ----------------------------
def build_text_tfidf(perfumes: List[Dict[str, Any]]) -> Tuple[TfidfVectorizer, Any]:
    # Build patterns once so removal is consistent across all items
    title_patterns = build_title_patterns(perfumes)
    custom_stopwords = build_stopwords(perfumes, title_patterns)

    texts = []
    for p in perfumes:
        # Scrub canonical title occurrences from everything
        title = scrub_text(str(p.get("canonical_title") or p.get("title") or ""), title_patterns)
        desc = scrub_text(str(p.get("description") or ""), title_patterns)
        pros, cons = flatten_opinions(p)
        pros = scrub_text(pros, title_patterns)
        cons = scrub_text(cons, title_patterns)

        texts.append("\n".join([
            f"title: {title}",
            f"description: {desc}",
            f"pros: {pros}",
            f"cons: {cons}",
        ]))

    vec = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        stop_words=sorted(custom_stopwords)
    )
    X = vec.fit_transform(texts)
    return vec, X

def struct_similarity(perfumes: List[Dict[str, Any]]) -> np.ndarray:
    n = len(perfumes)

    acc_sets = []
    top_sets, mid_sets, base_sets = [], [], []
    types = []

    for p in perfumes:
        accords, top, mid, base = extract_notes_and_accords(p)
        acc_sets.append(set([norm_text(a).lower() for a in accords if a]))
        top_sets.append(set([norm_text(x).lower() for x in top if x]))
        mid_sets.append(set([norm_text(x).lower() for x in mid if x]))
        base_sets.append(set([norm_text(x).lower() for x in base if x]))
        types.append(extract_type(p))

    def jacc(a: set, b: set) -> float:
        if not a and not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 0.0

    def weighted_notes(i: int, j: int) -> float:
        jt = jacc(top_sets[i], top_sets[j])
        jm = jacc(mid_sets[i], mid_sets[j])
        jb = jacc(base_sets[i], base_sets[j])
        return 0.40 * jt + 0.35 * jm + 0.25 * jb

    S = np.zeros((n, n), dtype=float)
    for i in range(n):
        S[i, i] = 1.0
        for j in range(i + 1, n):
            a = jacc(acc_sets[i], acc_sets[j])
            nscore = weighted_notes(i, j)

            ti, tj = types[i], types[j]
            type_bonus = 0.0
            if ti and tj:
                if ti == tj:
                    type_bonus = 0.08
                elif "uni" in (ti, tj):
                    type_bonus = 0.04

            s_ij = clamp(0.55 * a + 0.45 * nscore + type_bonus, 0.0, 1.0)
            S[i, j] = s_ij
            S[j, i] = s_ij

    return S

def compute_topk_neighbors(ids: List[str], sim_matrix: np.ndarray, k: int) -> Dict[str, List[Dict[str, Any]]]:
    n = len(ids)
    out: Dict[str, List[Dict[str, Any]]] = {}
    for i in range(n):
        sims = sim_matrix[i].copy()
        sims[i] = -1.0
        top_idx = np.argsort(-sims)[:k]
        out[ids[i]] = [{"id": ids[j], "score": float(sims[j])} for j in top_idx if sims[j] > -0.5]
    return out

# ----------------------------
# 2D map
# ----------------------------
def compute_2d_map(embedding: np.ndarray, random_state: int = 42) -> np.ndarray:
    try:
        import umap  # type: ignore
        reducer = umap.UMAP(
            n_neighbors=12,
            min_dist=0.12,
            metric="cosine",
            random_state=random_state,
        )
        return reducer.fit_transform(embedding)
    except Exception:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=random_state)
        return pca.fit_transform(embedding)

def label_clusters_kmeans(X, vectorizer: TfidfVectorizer, k: int, random_state: int = 42):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    cluster_ids = km.fit_predict(X)

    terms = np.array(vectorizer.get_feature_names_out())
    centroids = km.cluster_centers_

    def humanize(t: str) -> str:
        t = t.replace("_", " ").strip()
        return t[:1].upper() + t[1:] if t else t

    cluster_labels: Dict[int, str] = {}
    for cid in range(k):
        weights = centroids[cid]
        top_idx = np.argsort(-weights)[:20]
        top_terms = []
        for i in top_idx:
            if weights[i] <= 0:
                continue
            term = terms[i]
            if term in {"title", "description", "pros", "cons"}:
                continue
            if len(term) <= 2:
                continue
            top_terms.append(humanize(term))
            if len(top_terms) >= 3:
                break
        cluster_labels[cid] = " / ".join(top_terms) if top_terms else f"Region {cid+1}"

    return cluster_ids, cluster_labels

# ----------------------------
# Plotting
# ----------------------------
def write_interactive_plot_raw(map_rows: List[Dict[str, Any]], out_html: Path) -> None:
    import plotly.express as px  # type: ignore
    df = pd.DataFrame(map_rows)
    fig = px.scatter(
        df,
        x="x",
        y="y",
        hover_name="title",
        hover_data={
            "id": True,
            "type": True,
            "cluster_label": True,
            "price": True,
            "availability": True,
            "x": False,
            "y": False,
        },
    )
    fig.update_layout(title="Loewe Scent Map (2D projection) — Raw")
    fig.write_html(str(out_html), include_plotlyjs="cdn")

def write_interactive_plot_editorial(map_rows: List[Dict[str, Any]], out_html: Path) -> None:
    import plotly.express as px  # type: ignore
    import plotly.graph_objects as go  # type: ignore

    df = pd.DataFrame(map_rows)
    df["type_clean"] = df["type"].fillna("unknown").astype(str)
    df["cluster_label"] = df["cluster_label"].fillna("Unknown").astype(str)

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="cluster_label",
        symbol="type_clean",
        hover_name="title",
        hover_data={
            "id": True,
            "type_clean": True,
            "price": True,
            "availability": True,
            "cluster_label": True,
            "x": False,
            "y": False,
        },
    )

    centroids = (
        df.groupby("cluster_label")[["x", "y"]]
        .mean()
        .reset_index()
        .to_dict(orient="records")
    )
    for c in centroids:
        fig.add_trace(
            go.Scatter(
                x=[c["x"]],
                y=[c["y"]],
                mode="text",
                text=[c["cluster_label"]],
                textposition="middle center",
                showlegend=False,
            )
        )

    fig.update_layout(
        title="Loewe Scent Map — Editorial Regions (clusters + legend)",
        legend_title_text="Scent Region",
        margin=dict(l=40, r=40, t=70, b=40),
    )
    fig.write_html(str(out_html), include_plotlyjs="cdn")


# ----------------------------
# Hybrid embedding for mapping
# ----------------------------
def build_hybrid_embedding(
    axes_mat: np.ndarray,
    struct_mat: np.ndarray,
    text_mat,
    w_axes: float,
    w_struct: float,
    w_text: float,
) -> Tuple[np.ndarray, np.ndarray]:
    sim_axes = cosine_similarity(axes_mat)
    sim_text = cosine_similarity(text_mat)
    sim_struct = struct_mat.copy()

    sim_hybrid = (w_axes * sim_axes) + (w_struct * sim_struct) + (w_text * sim_text)
    sim_hybrid = np.clip(sim_hybrid, 0.0, 1.0)

    # Dense embedding for map: axes + truncated SVD of text
    try:
        from sklearn.decomposition import TruncatedSVD
        ncomp = min(32, text_mat.shape[1] - 1) if text_mat.shape[1] > 2 else 2
        svd = TruncatedSVD(n_components=ncomp, random_state=42)
        text_dense = svd.fit_transform(text_mat)
    except Exception:
        text_dense = text_mat.toarray() if hasattr(text_mat, "toarray") else np.asarray(text_mat)

    def l2norm(A: np.ndarray) -> np.ndarray:
        denom = np.linalg.norm(A, axis=1, keepdims=True) + 1e-9
        return A / denom

    axes_n = l2norm(axes_mat.astype(float))
    text_n = l2norm(text_dense.astype(float))

    embed = np.concatenate([
        math.sqrt(max(w_axes, 1e-6)) * axes_n,
        math.sqrt(max(w_text, 1e-6)) * text_n,
    ], axis=1)

    return sim_hybrid, embed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perfumes", type=str, required=True, help="Path to loewe_perfumes_deduped.json")
    ap.add_argument("--outdir", type=str, default="output", help="Output directory (default: output)")
    ap.add_argument("--axis_scores", type=str, default="", help="Path to axis_scores.json (default: outdir/axis_scores.json)")
    ap.add_argument("--topk", type=int, default=12, help="Top-K neighbors per perfume")
    ap.add_argument("--make_html", action="store_true", help="Write HTML maps")
    ap.add_argument("--clusters", type=int, default=8, help="Number of editorial regions (KMeans clusters)")
    ap.add_argument("--random_state", type=int, default=42)

    ap.add_argument("--w_axes", type=float, default=0.60)
    ap.add_argument("--w_struct", type=float, default=0.25)
    ap.add_argument("--w_text", type=float, default=0.15)
    args = ap.parse_args()

    w_axes, w_struct, w_text = args.w_axes, args.w_struct, args.w_text
    w_sum = w_axes + w_struct + w_text
    if not (0.999 <= w_sum <= 1.001):
        raise ValueError(f"Weights must sum to 1.0. Got {w_sum:.4f}")

    perfumes_path = Path(args.perfumes)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    perfumes = json.loads(perfumes_path.read_text(encoding="utf-8"))
    if not isinstance(perfumes, list):
        raise ValueError("Perfumes JSON must be a list/array of objects.")
    perfumes = [p for p in perfumes if isinstance(p, dict) and str(p.get("id", "")).strip()]
    if len(perfumes) < 2:
        raise ValueError("Need at least 2 perfumes with valid 'id'.")

    ids = [str(p["id"]).strip() for p in perfumes]
    titles = [norm_text(str(p.get("canonical_title") or p.get("title") or "")) for p in perfumes]

    # Load axis scores
    axis_path = Path(args.axis_scores) if args.axis_scores else (outdir / "axis_scores.json")
    if not axis_path.exists():
        raise FileNotFoundError(
            f"axis_scores.json not found at {axis_path}. "
            "Run compute_axis_scores.py first (or pass --axis_scores)."
        )
    axis_scores = load_axis_scores(axis_path)

    # Build axes matrix in id order
    axes_mat = np.zeros((len(ids), len(RADAR_AXES)), dtype=float)
    missing = 0
    for i, pid in enumerate(ids):
        scores = axis_scores.get(pid)
        if not scores:
            missing += 1
            scores = {ax: 0.0 for ax in RADAR_AXES}
        axes_mat[i] = np.array([float(scores.get(ax, 0.0)) for ax in RADAR_AXES], dtype=float)
    if missing:
        print(f"[WARN] Missing axis scores for {missing} perfumes (filled with 0).")

    # Structured similarity
    S_struct = struct_similarity(perfumes)

    # Text similarity
    tfidf_vec, X_text = build_text_tfidf(perfumes)

    # Hybrid similarity + map embedding
    sim_hybrid, embed_for_map = build_hybrid_embedding(
        axes_mat=axes_mat,
        struct_mat=S_struct,
        text_mat=X_text,
        w_axes=w_axes,
        w_struct=w_struct,
        w_text=w_text,
    )

    # Top-K neighbors
    neighbors = compute_topk_neighbors(ids, sim_hybrid, k=args.topk)
    (outdir / "neighbors_topk.json").write_text(
        json.dumps(neighbors, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Map coords
    coords = compute_2d_map(embed_for_map, random_state=args.random_state)

    # Clusters + editorial labels (cluster on TF-IDF for interpretability)
    k = max(2, min(args.clusters, len(perfumes) - 1))
    cluster_ids, cluster_labels = label_clusters_kmeans(X_text, tfidf_vec, k=k, random_state=args.random_state)

    # Build scent_map rows
    map_rows = []
    for i, p in enumerate(perfumes):
        pros, cons = flatten_opinions(p)
        t = extract_type(p)

        pid = ids[i]
        row = {
            "id": pid,
            "title": titles[i],
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "type": t,
            "description": norm_text(str(p.get("description") or "")),
            "pros": pros,
            "cons": cons,
            "cluster_id": int(cluster_ids[i]),
            "cluster_label": cluster_labels.get(int(cluster_ids[i]), f"Region {int(cluster_ids[i])+1}"),
            "axis_scores": axis_scores.get(pid, {ax: 0.0 for ax in RADAR_AXES}),
            "link": p.get("link", ""),
            "image_link": p.get("image_link", ""),
            "price": p.get("price", ""),
            "availability": p.get("availability", ""),
            "canonical_title": p.get("canonical_title", ""),
        }
        map_rows.append(row)

    (outdir / "scent_map.json").write_text(
        json.dumps(map_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.make_html:
        write_interactive_plot_raw(map_rows, outdir / "scent_map.html")
        write_interactive_plot_editorial(map_rows, outdir / "scent_map_editorial.html")

    print("✅ Done.")
    print(f"✅ Output folder: {outdir.resolve()}")
    print("✅ Files written:")
    print("  - neighbors_topk.json")
    print("  - scent_map.json")
    if args.make_html:
        print("  - scent_map.html")
        print("  - scent_map_editorial.html")
    print("✅ Weights:")
    print(f"  w_axes={w_axes:.2f}, w_struct={w_struct:.2f}, w_text={w_text:.2f}")


if __name__ == "__main__":
    main()

# python .\compute_hybrid_similarity.py --perfumes .\loewe_perfumes_deduped.json --outdir output --topk 15 --make_html --clusters 8 --w_axes 0.60 --w_struct 0.25 --w_text 0.15

# python .\compute_hybrid_similarity.py --perfumes .\loewe_perfumes_deduped.json --outdir output --axis_scores .\output\axis_scores.json

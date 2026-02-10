# compute_axis_scores.py
"""
Compute 6-axis radar scores for Loewe perfumes and export:

- output/axis_scores.json
- output/radar_scores.json          (same as axis_scores, kept for compatibility)
- output/radar_index.html           (optional)
- output/radar_<ID>.html            (optional, one per perfume)

INPUT: a *deduped* JSON array (loewe_perfumes_deduped.json).
No taxonomy.json required.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------
# Helpers
# ----------------------------
HTML_ENT_RE = re.compile(r"&ndash;|&mdash;|&amp;|&quot;|&apos;", re.IGNORECASE)

def norm_text(s: str) -> str:
    s = s or ""
    s = HTML_ENT_RE.sub(" ", s)
    s = re.sub(r"<[^>]+>", " ", s)  # strip HTML
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
    """
    Returns (pros_str, cons_str) as "a | b | c".
    Uses p["pros"]/p["cons"] if present (deduped output),
    else matched_data1.opinions.pros/cons.
    """
    pros = p.get("pros")
    cons = p.get("cons")
    if isinstance(pros, str) or isinstance(cons, str):
        return (norm_text(str(pros or "")), norm_text(str(cons or "")))

    m = safe_dict(p.get("matched_data1"))
    opinions = safe_dict(m.get("opinions"))
    pros_list = safe_list(opinions.get("pros"))
    cons_list = safe_list(opinions.get("cons"))

    pros_txt: List[str] = []
    for it in pros_list:
        if isinstance(it, dict) and isinstance(it.get("text"), str):
            pros_txt.append(it["text"])
        elif isinstance(it, str):
            pros_txt.append(it)

    cons_txt: List[str] = []
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


# ----------------------------
# 6 radar axes
# ----------------------------
RADAR_AXES = [
    "Fresh & Aquatic",
    "Fruity & Citrusy",
    "Floral & Green",
    "Woody & Spicy",
    "Sweet & Gourmand",
    "Powdery & Musky",
]

AXIS_NOTE_KEYWORDS = {
    "Fresh & Aquatic": [
        # “fresh/clean/soapy” language (very common in your pros/cons)
        "fresh", "clean", "crisp", "bright", "sparkling", "bubbly", "airy", "light",
        "watery", "water", "rain", "wet", "dewy", "cool", "cooling", "refreshing",
        "soapy", "soap", "shower", "shower gel", "gel", "laundry", "linen", "cotton",
        "detergent", "wet wipes", "wipe", "spa", "high-end spa", "morning", "early morning", "morning light", "dawn", "daybreak",
        "new beginnings", "new beginning", "beginning", "beginnings",
        "light", "luminous", "uplifting", "pure", "fresh and pure", "to share", "share",

        # aquatic/marine terms
        "aquatic", "marine", "sea", "seaside", "ocean", "beach", "seaweed",
        "salt", "salty", "briny", "ozone", "ozonic", "mist",

        # “summer heat” usage appears a lot in reviews
        "high heat", "hot", "summer", "vacation", "daytime",
    ],

    "Fruity & Citrusy": [
        # citrus family (shows up constantly)
        "citrus", "zesty", "tart", "juicy",
        "bergamot", "lemon", "lime", "grapefruit", "orange", "mandarin", "tangerine",
        "yuzu", "kumquat", "pomelo", "bitter orange", "petitgrain",

        # fruits actually present in your notes/reviews
        "pear", "apple", "peach", "guava", "lychee", "mango",
        "berry", "berries", "red currant", "black currant", "cassis",
        "pineapple", "tropical",

        # coconut shows up a lot in Paula’s Ibiza
        "coconut", "coconut water",
    ],

    "Floral & Green": [
        # florals (lots in your rows)
        "floral", "flower", "flowers", "petals", "bouquet",
        "rose", "jasmine", "magnolia", "neroli", "orange blossom",
        "lily", "lily-of-the-valley", "muguet", "lotus", "cyclamen",
        "tuberose", "ylang", "mimosa", "marigold",

        # green/herbal/aromatic (very present in Aire / Solo / Paula’s)
        "green", "leaf", "leaves", "tomato leaf", "grass", "meadow",
        "galbanum", "green notes",
        "tea", "matcha",
        "herb", "herbal", "aromatic",
        "sage", "thyme", "rosemary", "mint", "basil", "chamomile", "clary", "artemisia",

        # nature/park vocabulary (Un Paseo por Madrid descriptions)
        "park", "garden", "rose garden", "trees", "wildlife", "oasis", "hunting estate",
    ],

    "Woody & Spicy": [
        # woods/resins/incense are everywhere in 7 / Earth / Solo / Madrid
        "woody", "wood", "woods", "woodsy", "oakwood", "oak",
        "cedar", "cedarwood", "sandalwood", "vetiver", "patchouli",
        "oud", "driftwood", "cypress", "pine",
        "oakmoss", "moss",
        "resin", "resinous", "benzoin", "labdanum",

        # incense family (spelled many ways)
        "incense", "olibanum", "frankincense",

        # spices
        "spice", "spicy",
        "pepper", "pink pepper", "red pepper", "pepper berries",
        "ginger", "cinnamon", "nutmeg", "clove", "saffron",

        # darker facets
        "smoky", "smoke", "leather",
        "earthy", "earth", "mineral", "temple", "ancient", "egyptian",
    ],

    "Sweet & Gourmand": [
        # sweet/gourmand words in reviews
        "sweet", "sugary", "candy", "cotton candy",
        "vanilla", "roasted vanilla",
        "caramel", "butterscotch",
        "tonka", "praline",
        "chocolate", "cocoa",
        "honey", "maple",
        "gourmand",
        "boozy", "cognac",

        # amber/balsamic shows up a lot in accords + copy
        "amber", "ambery", "balsamic", "resinous", "warm", "rich",
    ],

    "Powdery & Musky": [
        # powder/makeup language appears constantly in cons
        "powdery", "powder", "makeup", "lipstick", "cosmetic", "vintage",

        # musk/skin/soft
        "musk", "musky", "white musk", "skin", "skin-like", "skin scent", "soft",
        "creamy", "milky", "lactonic", "cashmere",

        # iris/violet/orris are the big “powder” signals in your dataset
        "iris", "orris", "orris root", "violet",
    ],
}


ACCORD_TO_AXIS = {
    # Fresh & Aquatic
    "fresh": ("Fresh & Aquatic", 2.2),
    "aquatic": ("Fresh & Aquatic", 2.8),
    "marine": ("Fresh & Aquatic", 2.8),
    "ozonic": ("Fresh & Aquatic", 2.4),
    "soapy": ("Fresh & Aquatic", 2.2),

    # Fruity & Citrusy
    "citrus": ("Fruity & Citrusy", 2.8),
    "fruity": ("Fruity & Citrusy", 2.2),
    "tropical": ("Fruity & Citrusy", 2.2),
    "coconut": ("Fruity & Citrusy", 1.8),

    # Floral & Green
    "floral": ("Floral & Green", 2.6),
    "white floral": ("Floral & Green", 2.8),
    "yellow floral": ("Floral & Green", 2.4),
    "rose": ("Floral & Green", 2.2),
    "green": ("Floral & Green", 2.2),
    "aromatic": ("Floral & Green", 2.0),
    "herbal": ("Floral & Green", 2.0),
    "lavender": ("Floral & Green", 2.0),

    # Woody & Spicy
    "woody": ("Woody & Spicy", 2.8),
    "oud": ("Woody & Spicy", 2.6),
    "patchouli": ("Woody & Spicy", 2.4),
    "earthy": ("Woody & Spicy", 1.8),
    "smoky": ("Woody & Spicy", 2.2),
    "leather": ("Woody & Spicy", 2.0),
    "warm spicy": ("Woody & Spicy", 2.4),
    "fresh spicy": ("Woody & Spicy", 1.6),
    "soft spicy": ("Woody & Spicy", 1.4),

    # Sweet & Gourmand
    "sweet": ("Sweet & Gourmand", 2.6),
    "vanilla": ("Sweet & Gourmand", 2.8),
    "caramel": ("Sweet & Gourmand", 2.4),
    "amber": ("Sweet & Gourmand", 1.8),
    "balsamic": ("Sweet & Gourmand", 1.8),

    # Powdery & Musky
    "powdery": ("Powdery & Musky", 2.8),
    "musky": ("Powdery & Musky", 2.6),
    "iris": ("Powdery & Musky", 2.4),
    "violet": ("Powdery & Musky", 2.2),
    "lactonic": ("Powdery & Musky", 2.0),
}


def compute_axis_scores(p: Dict[str, Any]) -> Dict[str, float]:
    """
    Explainable 0..5 scoring using:
      - accords (strong)
      - notes (medium)
      - title/description/pros/cons (light)
    """
    title = norm_text(str(p.get("canonical_title") or p.get("title") or ""))
    desc = norm_text(str(p.get("description") or ""))
    pros, cons = flatten_opinions(p)

    accords, top, mid, base = extract_notes_and_accords(p)
    notes_all = " ".join([*top, *mid, *base])
    notes_l = " ".join(tokenize_simple(notes_all))
    text_l = " ".join(tokenize_simple(" ".join([title, desc, pros, cons])))

    s = {ax: 0.0 for ax in RADAR_AXES}

    # 1) Accords (strong)
    for a in accords:
        a_l = a.strip().lower()
        if a_l in ACCORD_TO_AXIS:
            ax, w = ACCORD_TO_AXIS[a_l]
            s[ax] += w

    # 2) Notes keywords (medium)
    for ax, kws in AXIS_NOTE_KEYWORDS.items():
        for kw in kws:
            kw_n = norm_text(kw).lower()
            if kw_n and (kw_n in notes_l):
                s[ax] += 0.8

    # 3) Text keywords
    # Default: light boost, because accords/notes are usually better signals.
    # If accords+notes are missing, increase weight so description/pros/cons can still score axes.
    text_boost = 0.25
    if (not accords) and (not top) and (not mid) and (not base):
        text_boost = 0.8

    for ax, kws in AXIS_NOTE_KEYWORDS.items():
        for kw in kws:
            kw_n = norm_text(kw).lower()
            if kw_n and (kw_n in text_l):
                s[ax] += text_boost

    return s


# ----------------------------
# Optional: radar HTML
# ----------------------------
def write_radar_html(perfume_id: str, title: str, scores: Dict[str, float], out_html: Path) -> None:
    import plotly.graph_objects as go  # type: ignore

    r = [scores[a] for a in RADAR_AXES]
    theta = RADAR_AXES

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=r, theta=theta, fill="toself", name=title))
    fig.update_layout(
        title=f"Radar Profile — {title}",
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig.write_html(str(out_html), include_plotlyjs="cdn")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perfumes", type=str, required=True, help="Path to loewe_perfumes_deduped.json")
    ap.add_argument("--outdir", type=str, default="output", help="Output directory (default: output)")
    ap.add_argument("--make_html", action="store_true", help="Write per-perfume radar HTML + index")
    args = ap.parse_args()

    perfumes_path = Path(args.perfumes)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    perfumes = json.loads(perfumes_path.read_text(encoding="utf-8"))
    if not isinstance(perfumes, list):
        raise ValueError("Perfumes JSON must be a list/array of objects.")

    perfumes = [p for p in perfumes if isinstance(p, dict) and str(p.get("id", "")).strip()]
    if len(perfumes) < 1:
        raise ValueError("Need at least 1 perfume with valid 'id'.")

    ids = [str(p["id"]).strip() for p in perfumes]
    titles = [norm_text(str(p.get("canonical_title") or p.get("title") or "")) for p in perfumes]

    axis_scores: Dict[str, Dict[str, float]] = {}
    for i, p in enumerate(perfumes):
        axis_scores[ids[i]] = compute_axis_scores(p)

    (outdir / "axis_scores.json").write_text(
        json.dumps(axis_scores, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # compatibility
    (outdir / "radar_scores.json").write_text(
        json.dumps(axis_scores, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.make_html:
        radar_index = []
        for i, pid in enumerate(ids):
            title = titles[i] or pid
            scores = axis_scores[pid]
            radar_html = outdir / f"radar_{pid}.html"
            write_radar_html(pid, title, scores, radar_html)
            radar_index.append((pid, title, radar_html.name))

        index_html = outdir / "radar_index.html"
        lines = ["<html><head><meta charset='utf-8'></head><body>"]
        lines.append("<h2>Radar Profiles</h2><ul>")
        for pid, title, fname in radar_index:
            lines.append(f"<li><a href='{fname}' target='_blank'>{pid} — {title}</a></li>")
        lines.append("</ul></body></html>")
        index_html.write_text("\n".join(lines), encoding="utf-8")

    print("✅ Done.")
    print(f"✅ Output folder: {outdir.resolve()}")
    print("✅ Files written:")
    print("  - axis_scores.json")
    print("  - radar_scores.json")
    if args.make_html:
        print("  - radar_index.html")
        print("  - radar_<ID>.html (one per perfume)")


if __name__ == "__main__":
    main()

# python .\compute_axis_scores.py --perfumes .\loewe_perfumes_deduped.json --outdir output --make_html

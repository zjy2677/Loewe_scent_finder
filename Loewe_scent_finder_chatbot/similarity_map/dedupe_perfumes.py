import json
import re
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Any, Tuple


# ----------------------------
# Helpers
# ----------------------------

VOLUME_RE = re.compile(
    r"""
    (\b\d+\s?(ml|ML|mL)\b) |
    (\b\d+\s?(oz|OZ)\b) |
    (\b\d+\s?fl\.?\s?oz\b)
    """,
    re.IGNORECASE | re.VERBOSE,
)

def strip_volume(title: str) -> str:
    """Remove bottle size only (keeps EDP/EDT/Elixir words)."""
    if not title:
        return ""
    t = VOLUME_RE.sub("", title)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def normalize_for_key(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def detect_concentration(title: str) -> str:
    t = (title or "").lower()
    if "elixir" in t:
        return "elixir"
    if re.search(r"\bedp\b", t) or "eau de parfum" in t:
        return "edp"
    if re.search(r"\bedt\b", t) or "eau de toilette" in t:
        return "edt"
    if re.search(r"\bparfum\b", t):
        return "parfum"
    return "unknown"

def get_matched(p: Dict[str, Any]) -> Dict[str, Any]:
    m = p.get("matched_data1") or {}
    return m if isinstance(m, dict) else {}

def norm_text(s: str) -> str:
    s = s or ""
    s = re.sub(r"&ndash;|&mdash;|&amp;|&quot;|&apos;", " ", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Filter out non-perfume SKUs (bundles / sample kits / cases)
EXCLUDE_TITLE_RE = re.compile(r"\b(case|sample|set)\b", re.IGNORECASE)
def should_exclude_title(canonical_title: str) -> bool:
    """
    Exclude bundles and non-single-scent items like:
      - Duo Set, Sample Set/Box, Crafted Collection Sample Set, Wood Case, etc.
    """
    if not canonical_title:
        return True
    return EXCLUDE_TITLE_RE.search(canonical_title) is not None

def list_to_str(x: Any, sep: str = "; ") -> str:
    if not x:
        return ""
    if isinstance(x, list):
        return sep.join([str(i) for i in x if i is not None])
    return str(x)

def extract_notes(matched: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
    notes_obj = matched.get("notes") or {}
    if not isinstance(notes_obj, dict):
        return [], [], []
    top = notes_obj.get("top notes") or []
    mid = notes_obj.get("middle notes") or []
    base = notes_obj.get("base notes") or []
    top = [n for n in top if isinstance(n, str)]
    mid = [n for n in mid if isinstance(n, str)]
    base = [n for n in base if isinstance(n, str)]
    return top, mid, base

def extract_opinions(matched: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Return pros/cons as lists of text strings.
    Original structure:
      opinions: { pros: [{text, votes}], cons: [{text, votes}] }
    """
    opinions = matched.get("opinions") or {}
    if not isinstance(opinions, dict):
        return [], []
    pros_raw = opinions.get("pros") or []
    cons_raw = opinions.get("cons") or []
    pros: List[str] = []
    cons: List[str] = []

    if isinstance(pros_raw, list):
        for item in pros_raw:
            if isinstance(item, dict):
                txt = item.get("text")
                if isinstance(txt, str) and txt.strip():
                    pros.append(txt.strip())
            elif isinstance(item, str) and item.strip():
                pros.append(item.strip())

    if isinstance(cons_raw, list):
        for item in cons_raw:
            if isinstance(item, dict):
                txt = item.get("text")
                if isinstance(txt, str) and txt.strip():
                    cons.append(txt.strip())
            elif isinstance(item, str) and item.strip():
                cons.append(item.strip())

    # de-dup while preserving order
    def dedup(seq: List[str]) -> List[str]:
        seen = set()
        out = []
        for s in seq:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    return dedup(pros), dedup(cons)

def data_richness_score(p: Dict[str, Any]) -> int:
    """
    Higher score = better representative among same scent different volumes.
    Rewards:
      - description length
      - accords
      - notes
      - type present
      - pros/cons (and votes presence)
    """
    score = 0

    desc = p.get("description") or ""
    if isinstance(desc, str):
        score += min(len(desc), 600) // 60  # up to ~10

    matched = get_matched(p)

    accords = matched.get("main accords") or []
    if isinstance(accords, list):
        score += len([a for a in accords if isinstance(a, str)]) * 2

    top, mid, base = extract_notes(matched)
    score += (len(top) + len(mid) + len(base)) * 2

    # type present
    mtype = matched.get("type")
    if isinstance(mtype, str) and mtype.strip():
        score += 4

    # pros/cons richness
    opinions = matched.get("opinions") or {}
    if isinstance(opinions, dict):
        pros = opinions.get("pros") or []
        cons = opinions.get("cons") or []
        if isinstance(pros, list):
            score += len(pros) * 3
            for pr in pros:
                if isinstance(pr, dict):
                    if isinstance(pr.get("text"), str) and pr.get("text", "").strip():
                        score += 1
                    if isinstance(pr.get("votes"), dict):
                        score += 1
        if isinstance(cons, list):
            score += len(cons) * 3
            for co in cons:
                if isinstance(co, dict):
                    if isinstance(co.get("text"), str) and co.get("text", "").strip():
                        score += 1
                    if isinstance(co.get("votes"), dict):
                        score += 1

    # slight preference for in-stock
    if str(p.get("availability", "")).lower().strip() == "in stock":
        score += 1

    return score


# ----------------------------
# Deduplication logic (size variants only)
# ----------------------------
def dedupe_size_variants(perfumes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Groups perfumes that are the same scent but different volumes.
    Keeps EDT/EDP/Elixir distinct.
    """
    groups: Dict[str, Dict[str, Any]] = {}

    
    for p in perfumes:
        title = str(p.get("title", "") or "")
        matched = get_matched(p)

        canonical_title = strip_volume(title)

        # Skip bundles / sample kits / cases (not a single perfume profile)
        if should_exclude_title(canonical_title):
            continue

        concentration = detect_concentration(canonical_title)

        frag_name = matched.get("name")
        if isinstance(frag_name, str) and frag_name.strip():
            identity = f"frag::{normalize_for_key(frag_name)}::{concentration}"
        else:
            identity = f"title::{normalize_for_key(canonical_title)}::{concentration}"

        score = data_richness_score(p)

        if identity not in groups:
            groups[identity] = {
                "canonical_title": canonical_title,
                "concentration": concentration,
                "representative": p,
                "rep_score": score,
                "variants": []
            }
        else:
            g = groups[identity]
            g["variants"].append({
                "id": p.get("id"),
                "title": p.get("title"),
                "price": p.get("price"),
                "availability": p.get("availability"),
                "score": score,
            })

            if score > g["rep_score"]:
                old = g["representative"]
                g["variants"].append({
                    "id": old.get("id"),
                    "title": old.get("title"),
                    "price": old.get("price"),
                    "availability": old.get("availability"),
                    "score": g["rep_score"],
                })
                g["representative"] = p
                g["rep_score"] = score

    cleaned: List[Dict[str, Any]] = []
    for g in groups.values():
        rep = g["representative"].copy()

        # add canonical fields
        rep["canonical_title"] = g["canonical_title"]
        rep["concentration"] = g["concentration"]
        rep["variants"] = g["variants"]
        rep["_rep_score"] = g["rep_score"]

        cleaned.append(rep)

    cleaned.sort(key=lambda x: (normalize_for_key(x.get("canonical_title", "")), x.get("concentration", "")))
    return cleaned


# ----------------------------
# CSV output
# ----------------------------
def write_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    fields = [
        "id",
        "canonical_title",
        "concentration",
        "type",
        "description",
        "main_accords",
        "top_notes",
        "middle_notes",
        "base_notes",
        "pros",
        "cons",
    ]

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for p in rows:
            matched = get_matched(p)

            accords = matched.get("main accords") or []
            if not isinstance(accords, list):
                accords = []
            accords = [a for a in accords if isinstance(a, str)]

            top, mid, base = extract_notes(matched)
            pros, cons = extract_opinions(matched)

            w.writerow({
                "id": p.get("id", ""),
                "canonical_title": p.get("canonical_title", strip_volume(str(p.get("title", "") or ""))),
                "concentration": p.get("concentration", detect_concentration(p.get("canonical_title", ""))),
                "type": matched.get("type", "") if isinstance(matched.get("type"), str) else "",
                "description": norm_text(str(p.get("description", "") or "")),
                "main_accords": list_to_str(accords),
                "top_notes": list_to_str(top),
                "middle_notes": list_to_str(mid),
                "base_notes": list_to_str(base),
                # CSV-friendly: join opinions text
                "pros": list_to_str(pros, sep=" | "),
                "cons": list_to_str(cons, sep=" | "),
            })


# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="loewe_perfumes.json")
    ap.add_argument("--out_csv", default="loewe_perfumes_clean.csv", help="Output CSV path")
    ap.add_argument("--out_json", default="", help="Optional: also write deduped JSON (debug/trace)")
    args = ap.parse_args()

    inp = Path(args.input)
    with open(inp, "r", encoding="utf-8") as f:
        perfumes = json.load(f)
    if not isinstance(perfumes, list):
        raise ValueError("Input JSON must be a list/array of objects")

    cleaned = dedupe_size_variants(perfumes)

    out_csv = Path(args.out_csv)
    write_csv(cleaned, out_csv)

    if args.out_json:
        out_json = Path(args.out_json)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, ensure_ascii=False, indent=2)
        print(f"✅ Wrote JSON: {out_json}")

    print(f"✅ Input SKUs: {len(perfumes)}")
    print(f"✅ Unique scents (size-deduped, concentration kept): {len(cleaned)}")
    print(f"✅ Wrote CSV: {out_csv}")


if __name__ == "__main__":
    main()

# python .\dedupe_perfumes.py --input .\loewe_perfumes.json --out_csv .\loewe_perfumes_clean.csv --out_json .\loewe_perfumes_deduped.json

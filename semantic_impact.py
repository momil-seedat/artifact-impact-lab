#!/usr/bin/env python3
"""Semantic (no-edges) impact ranking for artifacts.

A second, independent approach to impact analysis. Where `extract_artifacts.py`
follows the dependency *graph*, this tool ignores edges entirely and ranks
artifacts purely by **semantic similarity** of their text: names, descriptions,
docstrings, config values and file names.

It reads the `artifacts.json` produced by `extract_artifacts.py`, builds a
small TF-IDF model over every artifact's text (identifier-aware: it splits
camelCase, snake_case and dotted/slashed names into words), and ranks all other
artifacts by cosine similarity to a chosen target.

Stdlib only -- runs anywhere, on any project that has an artifacts.json.

Usage:
    python3 semantic_impact.py --target R1
    python3 semantic_impact.py --target payment.timeout --top 5
    python3 semantic_impact.py --target R1 --artifacts path/to/artifacts.json
"""

import argparse
import json
import math
import os
import re
from collections import Counter


STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "by", "is",
    "are", "be", "must", "should", "shall", "may", "not", "no", "within", "if",
    "when", "that", "this", "it", "its", "as", "at", "with", "from", "into",
    "via", "all", "any", "each", "up", "out", "do", "does", "done", "return",
    "returns", "uses", "use", "used", "pretend", "single", "dict", "file",
}


def split_identifier(token):
    """Break camelCase / PascalCase / snake_case runs into sub-words."""
    parts = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+", token)
    return parts or [token]


def tokenize(text):
    """Lowercased, identifier-aware token list with stopwords removed."""
    tokens = []
    for raw in re.findall(r"[A-Za-z0-9]+", text or ""):
        for part in split_identifier(raw):
            part = part.lower()
            if len(part) > 1 and part not in STOPWORDS:
                tokens.append(part)
    return tokens


def document_for(node):
    """The text blob representing an artifact for similarity scoring."""
    file_words = os.path.splitext(os.path.basename(node.get("file", "")))[0]
    return " ".join([
        node.get("id", ""),
        node.get("name", ""),
        node.get("text", ""),
        file_words,
        node.get("type", ""),
    ])


def build_tfidf(docs):
    """docs: {id: token_list} -> {id: {term: tfidf_weight}}, plus idf map."""
    n = len(docs)
    df = Counter()
    for tokens in docs.values():
        for term in set(tokens):
            df[term] += 1
    idf = {term: math.log((n + 1) / (count + 1)) + 1
           for term, count in df.items()}

    vectors = {}
    for art_id, tokens in docs.items():
        if not tokens:
            vectors[art_id] = {}
            continue
        counts = Counter(tokens)
        total = len(tokens)
        vectors[art_id] = {t: (c / total) * idf[t] for t, c in counts.items()}
    return vectors, idf


def cosine(a, b):
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def shared_terms(vec_a, vec_b, limit=4):
    """Top overlapping terms between two vectors, by combined weight."""
    common = set(vec_a) & set(vec_b)
    ranked = sorted(common, key=lambda t: vec_a[t] + vec_b[t], reverse=True)
    return ranked[:limit]


def rank(nodes, target_id, top=None):
    docs = {node["id"]: tokenize(document_for(node)) for node in nodes}
    if target_id not in docs:
        raise KeyError(target_id)

    vectors, _ = build_tfidf(docs)
    target_vec = vectors[target_id]
    by_id = {node["id"]: node for node in nodes}

    results = []
    for art_id, vec in vectors.items():
        if art_id == target_id:
            continue
        score = cosine(target_vec, vec)
        terms = shared_terms(target_vec, vec)
        if terms:
            reason = ("shares terms (%s) with %s [%s]"
                      % (", ".join(terms), target_id, by_id[art_id]["type"]))
        else:
            reason = ("no shared vocabulary with %s; low semantic overlap"
                      % target_id)
        results.append({"artifact": art_id,
                        "score": round(score, 3),
                        "reason": reason})

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top] if top else results


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", required=True,
                        help="artifact id whose change you want to assess")
    parser.add_argument("--artifacts", default=os.path.join(here, "artifacts.json"),
                        help="path to artifacts.json (default: ./artifacts.json)")
    parser.add_argument("--top", type=int, default=None,
                        help="only show the top N results")
    parser.add_argument("--out", default=None,
                        help="also write the ranked JSON to this path")
    args = parser.parse_args()

    with open(args.artifacts) as handle:
        data = json.load(handle)
    nodes = data["nodes"]

    try:
        results = rank(nodes, args.target, args.top)
    except KeyError:
        ids = ", ".join(sorted(n["id"] for n in nodes))
        parser.error("unknown artifact %r. Known: %s" % (args.target, ids))

    text = json.dumps(results, indent=2)
    print(text)

    out = args.out
    if out is None:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", args.target)
        out = os.path.join(here, "semantic_impact_%s.json" % safe)
    with open(out, "w") as handle:
        handle.write(text + "\n")
    print("\nWrote %s" % os.path.relpath(out, here), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Proposed method: embeddings + dependency graph + matrix propagation.

The baseline (embedding_impact.py) ranks artifacts purely by embedding cosine
similarity to the changed artifact. It misses transitively-impacted artifacts
whose *text* is not similar (e.g. completeOrder is unrelated to the word
"timeout", yet it calls processPayment which reads payment.timeout linked to R1).

This proposed method keeps the SAME embeddings but adds a reasoning layer:

    Artifacts
       v
    Embeddings              (semantic prior s: cosine to the changed artifact)
       v
    Dependency Discovery    (edges from artifacts.json)
       v
    Matrix M                (row-normalised change-propagation adjacency)
       v
    Propagation             (Katz-style multi-hop spread of impact)

Final impact score blends the semantic prior with graph propagation:

    impact = lambda * semantic_similarity + (1 - lambda) * propagated_reach

so completeOrder inherits a high score by being reachable from R1 through the
dependency path  R1 -> payment.timeout -> processPayment -> completeOrder,
even though its embedding similarity to R1 is low.

Fair experimental setup (same embeddings, different reasoning layer):
    Baseline  : SentenceTransformer + cosine            (embedding_impact.py)
    Proposed  : SentenceTransformer + graph + matrix M   (this file)

Backends (so the propagation layer runs even without the model / internet):
    --semantic embedding   SentenceTransformer all-MiniLM-L6-v2 (default)
    --semantic lexical     stdlib TF-IDF prior, no download

Usage:
    python3 graph_propagation_impact.py --target R1
    python3 graph_propagation_impact.py --target R1 --semantic lexical
    python3 graph_propagation_impact.py --target R1 --lam 0.5 --decay 0.7 --show-matrix
"""

import argparse
import json
import os
import re

import numpy as np

from embedding_impact import build_artifact_text
from semantic_impact import tokenize, build_tfidf, cosine as tfidf_cosine


# --------------------------------------------------------------------------- #
# Semantic prior: cosine similarity of every artifact to the changed one.
# --------------------------------------------------------------------------- #
def semantic_prior_embedding(artifact_text, target, model_name):
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Missing dependency: %s\nInstall with:\n"
                         "    pip install -r requirements.txt" % exc)
    try:
        model = SentenceTransformer(model_name)
    except OSError as exc:  # pragma: no cover - network dependent
        raise SystemExit(
            "Could not load model %r (downloads from huggingface.co on first "
            "run). Use --semantic lexical to run offline.\nError: %s"
            % (model_name, exc))

    vectors = {aid: model.encode(text) for aid, text in artifact_text.items()}
    query = vectors[target]
    return {aid: float(cosine_similarity([query], [vec])[0][0])
            for aid, vec in vectors.items()}


def semantic_prior_lexical(artifact_text, target):
    docs = {aid: tokenize(text) for aid, text in artifact_text.items()}
    vectors, _ = build_tfidf(docs)
    query = vectors[target]
    return {aid: tfidf_cosine(query, vec) for aid, vec in vectors.items()}


# --------------------------------------------------------------------------- #
# Dependency matrix M and propagation
# --------------------------------------------------------------------------- #
def propagation_matrix(node_ids, edges):
    """Row-normalised adjacency where M[i, j] = impact flows from i to j.

    Trace edges point R->C->S, caller->callee and test->service. A *change*
    propagates R->C->S->caller and service->test, so call/test edges are
    reversed (changing the callee/target impacts its callers/tests).
    """
    index = {aid: i for i, aid in enumerate(node_ids)}
    n = len(node_ids)
    M = np.zeros((n, n))
    for edge in edges:
        src, dst, kind = edge["from"], edge["to"], edge["kind"]
        if kind in ("service_to_service", "test_to_service"):
            src, dst = dst, src
        if src in index and dst in index:
            M[index[src], index[dst]] += 1.0
    # Row-normalise so each node distributes its impact among its successors.
    row_sums = M.sum(axis=1, keepdims=True)
    np.divide(M, row_sums, out=M, where=row_sums != 0)
    return M, index


def propagate(seed, M, decay, steps):
    """Katz-style multi-hop spread: sum_k decay^k * seed @ M^k (k=1..steps)."""
    total = np.zeros_like(seed)
    current = seed.copy()
    for _ in range(steps):
        current = decay * (current @ M)
        total += current
    return total


def rank(nodes, edges, target, semantic, model_name,
         lam=0.5, decay=0.7, steps=6, top=None):
    artifact_text = build_artifact_text(nodes)
    if target not in artifact_text:
        raise KeyError(target)

    if semantic == "embedding":
        sim = semantic_prior_embedding(artifact_text, target, model_name)
    else:
        sim = semantic_prior_lexical(artifact_text, target)

    node_ids = [n["id"] for n in nodes]
    M, index = propagation_matrix(node_ids, edges)

    seed = np.zeros(len(node_ids))
    seed[index[target]] = 1.0
    prop = propagate(seed, M, decay, steps)

    # Normalise both signals to [0, 1] for a fair blend.
    prop_max = prop.max() if prop.max() > 0 else 1.0
    by_id = {n["id"]: n for n in nodes}

    results = []
    for aid in node_ids:
        if aid == target:
            continue
        s = max(0.0, sim.get(aid, 0.0))
        p = prop[index[aid]] / prop_max
        score = lam * s + (1 - lam) * p
        results.append({
            "artifact": aid,
            "score": round(float(score), 3),
            "semantic": round(float(s), 3),
            "structural": round(float(p), 3),
            "reason": _reason(aid, by_id[aid]["type"], s, p, lam),
        })
    results.sort(key=lambda r: r["score"], reverse=True)
    return (results[:top] if top else results), (M, index)


def _reason(aid, art_type, sem, struct, lam):
    if struct > 0 and sem < 0.2:
        driver = "driven by dependency propagation (low text similarity)"
    elif sem > 0 and struct == 0:
        driver = "driven by semantic similarity only (no dependency path)"
    elif struct > 0 and sem > 0:
        driver = "supported by both semantics and dependency propagation"
    else:
        driver = "weakly related"
    return ("[%s] %s; semantic=%.3f structural=%.3f (lambda=%.2f)"
            % (art_type, driver, sem, struct, lam))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", required=True, help="artifact id that changed")
    parser.add_argument("--artifacts", default=os.path.join(here, "artifacts.json"))
    parser.add_argument("--semantic", choices=("embedding", "lexical"),
                        default="embedding", help="semantic prior backend")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--lam", type=float, default=0.5,
                        help="weight on semantic vs structural (0..1)")
    parser.add_argument("--decay", type=float, default=0.7,
                        help="per-hop decay in propagation")
    parser.add_argument("--steps", type=int, default=6, help="propagation hops")
    parser.add_argument("--top", type=int, default=None)
    parser.add_argument("--show-matrix", action="store_true",
                        help="print the propagation matrix M")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    with open(args.artifacts) as handle:
        data = json.load(handle)
    nodes, edges = data["nodes"], data["edges"]

    try:
        results, (M, index) = rank(
            nodes, edges, args.target, args.semantic, args.model,
            args.lam, args.decay, args.steps, args.top)
    except KeyError:
        ids = ", ".join(sorted(n["id"] for n in nodes))
        parser.error("unknown artifact %r. Known: %s" % (args.target, ids))

    if args.show_matrix:
        labels = list(index)
        print("Propagation matrix M (row -> col, impact flow):")
        for label in labels:
            row = M[index[label]]
            nz = {labels[j]: round(float(row[j]), 2) for j in range(len(labels))
                  if row[j] > 0}
            print("  %-45s -> %s" % (label, nz or "{}"))
        print()

    text = json.dumps(results, indent=2)
    print(text)

    out = args.out
    if out is None:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", args.target)
        out = os.path.join(here, "graph_propagation_impact_%s.json" % safe)
    with open(out, "w") as handle:
        handle.write(text + "\n")
    print("\nWrote %s" % os.path.relpath(out, here))


if __name__ == "__main__":
    main()

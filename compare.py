#!/usr/bin/env python3
"""Compare the similarity baseline against the matrix-propagation method.

Runs both impact rankings for one changed artifact and lines them up so you can
see exactly which artifacts the propagation layer promotes:

    Baseline : cosine similarity only            (same prior, no edges)
    Proposed : similarity + dependency matrix M  (propagation)

Both use the SAME semantic prior (chosen with --semantic) so the only
difference is the reasoning layer -- a fair comparison.

It prints a side-by-side table (similarity score/rank vs proposed score/rank,
and the rank change), reports the Spearman rank correlation between the two
orderings, and writes comparison_<target>.json + comparison_<target>.md.

Usage:
    python3 compare.py --target R1                    # embeddings (needs model)
    python3 compare.py --target R1 --semantic lexical # offline, no download
"""

import argparse
import json
import os
import re

from embedding_impact import build_artifact_text
from graph_propagation_impact import (
    semantic_prior_embedding, semantic_prior_lexical, rank as proposed_rank,
)


def ranks_from_scores(scores):
    """{id: score} -> {id: 1-based rank} (highest score = rank 1)."""
    ordered = sorted(scores, key=lambda k: scores[k], reverse=True)
    return {aid: i + 1 for i, aid in enumerate(ordered)}


def spearman(scores_a, scores_b):
    """Spearman rank correlation between two {id: score} maps."""
    ids = [k for k in scores_a if k in scores_b]
    n = len(ids)
    if n < 2:
        return float("nan")
    rank_a, rank_b = ranks_from_scores(scores_a), ranks_from_scores(scores_b)
    d2 = sum((rank_a[i] - rank_b[i]) ** 2 for i in ids)
    return 1 - (6 * d2) / (n * (n * n - 1))


def compare(nodes, edges, target, semantic, model_name, lam, decay, steps):
    artifact_text = build_artifact_text(nodes)
    if target not in artifact_text:
        raise KeyError(target)

    if semantic == "embedding":
        sim = semantic_prior_embedding(artifact_text, target, model_name)
    else:
        sim = semantic_prior_lexical(artifact_text, target)
    sim = {aid: max(0.0, v) for aid, v in sim.items() if aid != target}

    proposed, _ = proposed_rank(nodes, edges, target, semantic, model_name,
                                lam=lam, decay=decay, steps=steps)
    prop_score = {r["artifact"]: r["score"] for r in proposed}

    sim_rank = ranks_from_scores(sim)
    prop_rank = ranks_from_scores(prop_score)
    by_id = {n["id"]: n for n in nodes}

    rows = []
    for aid in prop_score:
        rows.append({
            "artifact": aid,
            "type": by_id[aid]["type"],
            "similarity_score": round(sim.get(aid, 0.0), 3),
            "similarity_rank": sim_rank.get(aid),
            "proposed_score": round(prop_score[aid], 3),
            "proposed_rank": prop_rank[aid],
            "rank_change": sim_rank.get(aid, 0) - prop_rank[aid],
        })
    rows.sort(key=lambda r: r["proposed_rank"])
    rho = spearman(sim, prop_score)
    return rows, rho


def render_table(rows, rho, target, semantic):
    head = ("Comparison for target %r  (semantic prior: %s)\n"
            "  + rank_change means the proposed method ranks it HIGHER than "
            "pure similarity.\n" % (target, semantic))
    cols = "%-46s %-6s %8s %6s %8s %6s %7s" % (
        "artifact", "type", "sim", "simR", "prop", "propR", "Δrank")
    lines = [head, cols, "-" * len(cols)]
    for r in rows:
        lines.append("%-46s %-6s %8.3f %6d %8.3f %6d %+7d" % (
            r["artifact"], r["type"][:6], r["similarity_score"],
            r["similarity_rank"], r["proposed_score"], r["proposed_rank"],
            r["rank_change"]))
    lines.append("\nSpearman rank correlation (similarity vs proposed): %.3f"
                 % rho)
    return "\n".join(lines)


def render_markdown(rows, rho, target, semantic):
    lines = ["# Impact comparison: similarity vs matrix propagation", "",
             "Target (changed artifact): `%s`  " % target,
             "Semantic prior: `%s`  " % semantic,
             "Spearman rank correlation: **%.3f**" % rho, "",
             "`rank_change` > 0 means the proposed (propagation) method ranks "
             "the artifact higher than pure similarity does.", "",
             "| Artifact | Type | Similarity (score / rank) | Proposed (score / rank) | Δrank |",
             "|----------|------|---------------------------|-------------------------|------|"]
    for r in rows:
        lines.append("| `%s` | %s | %.3f / %d | %.3f / %d | %+d |" % (
            r["artifact"], r["type"], r["similarity_score"],
            r["similarity_rank"], r["proposed_score"], r["proposed_rank"],
            r["rank_change"]))
    return "\n".join(lines) + "\n"


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", required=True)
    parser.add_argument("--artifacts", default=os.path.join(here, "artifacts.json"))
    parser.add_argument("--semantic", choices=("embedding", "lexical"),
                        default="embedding")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--decay", type=float, default=0.7)
    parser.add_argument("--steps", type=int, default=6)
    args = parser.parse_args()

    with open(args.artifacts) as handle:
        data = json.load(handle)
    nodes, edges = data["nodes"], data["edges"]

    try:
        rows, rho = compare(nodes, edges, args.target, args.semantic,
                            args.model, args.lam, args.decay, args.steps)
    except KeyError:
        ids = ", ".join(sorted(n["id"] for n in nodes))
        parser.error("unknown artifact %r. Known: %s" % (args.target, ids))

    print(render_table(rows, rho, args.target, args.semantic))

    safe = re.sub(r"[^A-Za-z0-9._-]", "_", args.target)
    json_path = os.path.join(here, "comparison_%s.json" % safe)
    md_path = os.path.join(here, "comparison_%s.md" % safe)
    with open(json_path, "w") as handle:
        json.dump({"target": args.target, "semantic": args.semantic,
                   "spearman": rho, "rows": rows}, handle, indent=2)
        handle.write("\n")
    with open(md_path, "w") as handle:
        handle.write(render_markdown(rows, rho, args.target, args.semantic))
    print("\nWrote %s and %s"
          % (os.path.relpath(json_path, here), os.path.relpath(md_path, here)))


if __name__ == "__main__":
    main()

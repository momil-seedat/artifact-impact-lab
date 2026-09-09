#!/usr/bin/env python3
"""Embedding-based (no-edges) impact ranking.

The "classic embedding approach": turn every artifact into a short natural-
language sentence, embed each sentence with a sentence-transformer
(all-MiniLM-L6-v2 -> R^384), then rank artifacts by cosine similarity to the
changed artifact. Graph edges are ignored entirely -- relevance is purely
semantic.

This is a research baseline meant to be compared against:
  * extract_artifacts.py   -- structural / dependency-graph impact
  * semantic_impact.py     -- lexical TF-IDF similarity (stdlib, no model)
  * this file              -- transformer embedding similarity

Requires (see requirements.txt):
    pip install sentence-transformers scikit-learn

Usage:
    python3 embedding_impact.py --target R1
    python3 embedding_impact.py --target R1 --top 5 --model all-MiniLM-L6-v2
    python3 embedding_impact.py --dump-text      # just print artifact_text
"""

import argparse
import json
import os
import re


def _module_words(module):
    """'payment_service' -> 'payment service'."""
    return module.replace("_", " ")


def humanize(node):
    """Convert an artifact node into a short natural-language description.

    Examples:
        R1                              -> "Payment must time out"
        payment.timeout                 -> "payment timeout configuration ..."
        payment_service.processPayment  -> "Function processPayment in payment service ..."
        order_service.completeOrder     -> "Function completeOrder in order service ..."
    """
    art_type = node.get("type")
    name = node.get("name", node["id"])
    text = (node.get("text") or "").strip()

    if art_type == "requirement":
        # Drop a leading "R1 - " / "R1 — " style prefix from the title.
        title = re.sub(r"^\S+\s*[-—:]\s*", "", name)
        sentence = title or name
    elif art_type == "config":
        words = node["id"].replace(".", " ")
        sentence = "%s configuration property" % words
    elif art_type == "service":
        module = node["id"].rsplit(".", 1)[0]
        sentence = "Function %s in %s" % (name, _module_words(module))
    elif art_type == "test":
        module = node["id"].rsplit(".", 1)[0]
        sentence = "Test %s in %s" % (name, _module_words(module))
    else:
        sentence = name

    # Append the stored description/docstring/value for a richer embedding.
    if text and text.lower() not in sentence.lower():
        sentence = "%s. %s" % (sentence, text)
    return re.sub(r"\s+", " ", sentence).strip()


def build_artifact_text(nodes):
    return {node["id"]: humanize(node) for node in nodes}


def rank(artifact_text, target, model_name, top=None):
    # Imported lazily so --dump-text works without the heavy deps installed.
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "Missing dependency: %s\nInstall with:\n"
            "    pip install sentence-transformers scikit-learn" % exc)

    try:
        model = SentenceTransformer(model_name)
    except OSError as exc:  # pragma: no cover - network dependent
        raise SystemExit(
            "Could not load model %r (needs to download from huggingface.co "
            "on first run).\nRun once with internet access, or pre-download the "
            "model and use offline mode.\nUnderlying error: %s"
            % (model_name, exc))

    ids = list(artifact_text)
    vectors = {aid: model.encode(artifact_text[aid]) for aid in ids}
    query = vectors[target]

    results = []
    for aid in ids:
        if aid == target:
            continue
        score = float(cosine_similarity([query], [vectors[aid]])[0][0])
        results.append({
            "artifact": aid,
            "score": round(score, 3),
            "reason": "embedding cosine similarity to %r (text: %r)"
                      % (target, artifact_text[aid]),
        })
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top] if top else results


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", help="artifact id that changed")
    parser.add_argument("--artifacts", default=os.path.join(here, "artifacts.json"),
                        help="path to artifacts.json (default: ./artifacts.json)")
    parser.add_argument("--model", default="all-MiniLM-L6-v2",
                        help="sentence-transformers model name")
    parser.add_argument("--top", type=int, default=None, help="show top N only")
    parser.add_argument("--out", default=None, help="also write ranked JSON here")
    parser.add_argument("--dump-text", action="store_true",
                        help="print the generated artifact_text mapping and exit")
    args = parser.parse_args()

    with open(args.artifacts) as handle:
        nodes = json.load(handle)["nodes"]
    artifact_text = build_artifact_text(nodes)

    if args.dump_text:
        print(json.dumps(artifact_text, indent=2))
        return

    if not args.target:
        parser.error("--target is required (or use --dump-text)")
    if args.target not in artifact_text:
        parser.error("unknown artifact %r. Known: %s"
                     % (args.target, ", ".join(sorted(artifact_text))))

    results = rank(artifact_text, args.target, args.model, args.top)
    text = json.dumps(results, indent=2)
    print(text)

    out = args.out
    if out is None:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", args.target)
        out = os.path.join(here, "embedding_impact_%s.json" % safe)
    with open(out, "w") as handle:
        handle.write(text + "\n")
    print("\nWrote %s" % os.path.relpath(out, here))


if __name__ == "__main__":
    main()

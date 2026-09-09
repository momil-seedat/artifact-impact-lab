#!/usr/bin/env python3
"""Evaluate impact-analysis methods against labelled scenarios.

Each file in scenarios/ describes a change and its ground-truth ("actual")
impact set. For every scenario this script runs each method, turns its ranked
scores into a predicted impact set (score > threshold), and computes
precision / recall / F1 against the ground truth. It then reports a macro
average across all scenarios -- the numbers to quote in the paper.

Methods compared (all share the SAME semantic prior; only the reasoning layer
differs, controlled by lambda in graph_propagation_impact.rank):

    semantic      lambda = 1.0   pure similarity, no edges        (baseline)
    proposed      lambda = 0.5   similarity + graph propagation   (proposed)
    structural    lambda = 0.0   pure dependency propagation

Definitions:
    predicted = { artifact : score > threshold }
    precision = |predicted & actual| / |predicted|
    recall    = |predicted & actual| / |actual|
    F1        = 2 * precision * recall / (precision + recall)

Usage:
    python3 evaluate.py --semantic lexical          # offline, no model
    python3 evaluate.py --semantic embedding        # MiniLM (needs internet)
    python3 evaluate.py --semantic lexical --threshold 0.05
"""

import argparse
import glob
import json
import os

from graph_propagation_impact import rank as gp_rank


METHODS = [
    ("semantic", 1.0),
    ("proposed", 0.5),
    ("structural", 0.0),
]


def load_scenarios(scenarios_dir):
    scenarios = []
    for path in sorted(glob.glob(os.path.join(scenarios_dir, "*.json"))):
        with open(path) as handle:
            scenarios.append(json.load(handle))
    return scenarios


def prf(predicted, actual):
    tp = len(predicted & actual)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(actual) if actual else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return precision, recall, f1


def method_scores(nodes, edges, target, semantic, model, lam, decay, steps):
    results, _ = gp_rank(nodes, edges, target, semantic, model,
                         lam=lam, decay=decay, steps=steps)
    return {r["artifact"]: r["score"] for r in results}


def evaluate(nodes, edges, scenarios, semantic, model, threshold, decay, steps):
    per_scenario = []
    totals = {name: {"p": 0.0, "r": 0.0, "f1": 0.0} for name, _ in METHODS}

    for scenario in scenarios:
        target = scenario["changed"]
        actual = set(scenario["actual_impact"])
        row = {"scenario": scenario["id"], "title": scenario["title"],
               "changed": target, "n_actual": len(actual), "methods": {}}
        for name, lam in METHODS:
            scores = method_scores(nodes, edges, target, semantic, model,
                                   lam, decay, steps)
            predicted = {a for a, s in scores.items() if s > threshold}
            p, r, f1 = prf(predicted, actual)
            row["methods"][name] = {
                "precision": round(p, 3), "recall": round(r, 3),
                "f1": round(f1, 3), "n_predicted": len(predicted),
            }
            totals[name]["p"] += p
            totals[name]["r"] += r
            totals[name]["f1"] += f1
        per_scenario.append(row)

    n = len(scenarios)
    averages = {name: {"precision": round(v["p"] / n, 3),
                       "recall": round(v["r"] / n, 3),
                       "f1": round(v["f1"] / n, 3)}
                for name, v in totals.items()}
    return per_scenario, averages


def render(per_scenario, averages, semantic, threshold):
    lines = ["Impact-analysis evaluation",
             "  semantic prior: %s   |   predicted = score > %.3f"
             % (semantic, threshold), ""]
    for row in per_scenario:
        lines.append("%s (%s) - changed %r, |actual|=%d"
                     % (row["scenario"], row["title"], row["changed"],
                        row["n_actual"]))
        lines.append("  %-12s %9s %7s %5s %5s" %
                     ("method", "precision", "recall", "F1", "#pred"))
        for name, _ in METHODS:
            m = row["methods"][name]
            lines.append("  %-12s %9.3f %7.3f %5.3f %5d" %
                         (name, m["precision"], m["recall"], m["f1"],
                          m["n_predicted"]))
        lines.append("")
    lines.append("MACRO AVERAGE across %d scenarios" % len(per_scenario))
    lines.append("  %-12s %9s %7s %5s" % ("method", "precision", "recall", "F1"))
    for name, _ in METHODS:
        a = averages[name]
        lines.append("  %-12s %9.3f %7.3f %5.3f" %
                     (name, a["precision"], a["recall"], a["f1"]))
    return "\n".join(lines)


def render_markdown(per_scenario, averages, semantic, threshold):
    lines = ["# Impact-analysis evaluation", "",
             "Semantic prior: `%s`  |  predicted set = `score > %.3f`"
             % (semantic, threshold), "",
             "## Macro average across %d scenarios" % len(per_scenario), "",
             "| Method | Precision | Recall | F1 |",
             "|--------|-----------|--------|----|"]
    for name, _ in METHODS:
        a = averages[name]
        lines.append("| %s | %.3f | %.3f | %.3f |"
                     % (name, a["precision"], a["recall"], a["f1"]))
    lines += ["", "## Per scenario", ""]
    for row in per_scenario:
        lines.append("### %s — %s (changed `%s`, |actual|=%d)"
                     % (row["scenario"], row["title"], row["changed"],
                        row["n_actual"]))
        lines += ["", "| Method | Precision | Recall | F1 | #pred |",
                  "|--------|-----------|--------|----|-------|"]
        for name, _ in METHODS:
            m = row["methods"][name]
            lines.append("| %s | %.3f | %.3f | %.3f | %d |"
                         % (name, m["precision"], m["recall"], m["f1"],
                            m["n_predicted"]))
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts", default=os.path.join(here, "artifacts.json"))
    parser.add_argument("--scenarios", default=os.path.join(here, "scenarios"))
    parser.add_argument("--semantic", choices=("embedding", "lexical"),
                        default="lexical")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="an artifact is predicted impacted if score > this")
    parser.add_argument("--decay", type=float, default=0.7)
    parser.add_argument("--steps", type=int, default=6)
    args = parser.parse_args()

    with open(args.artifacts) as handle:
        data = json.load(handle)
    nodes, edges = data["nodes"], data["edges"]
    scenarios = load_scenarios(args.scenarios)
    if not scenarios:
        parser.error("no scenarios found in %s" % args.scenarios)

    per_scenario, averages = evaluate(
        nodes, edges, scenarios, args.semantic, args.model,
        args.threshold, args.decay, args.steps)

    print(render(per_scenario, averages, args.semantic, args.threshold))

    out = {"semantic": args.semantic, "threshold": args.threshold,
           "scenarios": per_scenario, "macro_average": averages}
    json_path = os.path.join(here, "evaluation_results.json")
    md_path = os.path.join(here, "evaluation_results.md")
    with open(json_path, "w") as handle:
        json.dump(out, handle, indent=2)
        handle.write("\n")
    with open(md_path, "w") as handle:
        handle.write(render_markdown(per_scenario, averages,
                                     args.semantic, args.threshold))
    print("\nWrote %s and %s" % (os.path.relpath(json_path, here),
                                 os.path.relpath(md_path, here)))


if __name__ == "__main__":
    main()

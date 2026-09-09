#!/usr/bin/env python3
"""Extract artifact dependencies for impact analysis.

This tool statically analyses the project and recovers the traceability graph
between four kinds of artifacts:

    R  Requirement   (requirements/*.md, via @id / @config annotations)
    C  Config        (configs/*.properties keys)
    S  Service       (functions/methods in services/*.py)
    T  Test          (test functions in tests/*.py)

It recovers these edges (directions match the hand-drawn traceability model):

    R -> C   requirement traces to a config property      (requirement_to_config)
    C -> S   config property is read inside a service fn   (config_to_service)
    S -> S   a service function calls another             (service_to_service)
    T -> S   a test exercises a service function          (test_to_service)

Everything is stdlib only (ast, json, re). Outputs:

    artifacts.json          nodes + edges (primary, machine + human readable)
    traceability_matrix.md  human-readable edge table grouped by kind
    artifacts.dot           Graphviz graph (render: dot -Tpng artifacts.dot -o g.png)

Usage:
    python extract_artifacts.py                  # extract + write all outputs
    python extract_artifacts.py --impact payment.timeout
                                                 # what does changing this affect?
    python extract_artifacts.py --root some/dir  # analyse a different project root
"""

import argparse
import ast
import glob
import json
import os
import re
from collections import deque


# --------------------------------------------------------------------------- #
# Parsing: Config (C)
# --------------------------------------------------------------------------- #
def parse_config(config_dir):
    """Return {key: {value, file, line}} for every .properties key."""
    keys = {}
    for path in sorted(glob.glob(os.path.join(config_dir, "*.properties"))):
        with open(path) as handle:
            for lineno, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, value = stripped.split("=", 1)
                keys[key.strip()] = {
                    "value": value.strip(),
                    "file": os.path.relpath(path),
                    "line": lineno,
                }
    return keys


# --------------------------------------------------------------------------- #
# Parsing: Requirements (R) and R -> C edges
# --------------------------------------------------------------------------- #
def parse_requirements(req_dir):
    """Parse @id / @config annotations from requirement markdown files.

    Returns (nodes, edges). A `@config:` annotation attaches to the most
    recently seen `@id:`.
    """
    id_re = re.compile(r"@id:\s*(\S+)")
    config_re = re.compile(r"@config:\s*(\S+)")
    title_re = re.compile(r"^#+\s*(.+)$")

    nodes, edges = {}, []
    for path in sorted(glob.glob(os.path.join(req_dir, "*.md"))):
        current = None
        last_title = None
        with open(path) as handle:
            for lineno, line in enumerate(handle, start=1):
                title_match = title_re.match(line.strip())
                if title_match:
                    last_title = title_match.group(1).strip()
                id_match = id_re.search(line)
                if id_match:
                    current = id_match.group(1)
                    nodes[current] = {
                        "id": current,
                        "type": "requirement",
                        "name": last_title or current,
                        "file": os.path.relpath(path),
                        "line": lineno,
                        "text": last_title or "",
                    }
                config_match = config_re.search(line)
                if config_match and current:
                    edges.append({
                        "from": current,
                        "to": config_match.group(1),
                        "kind": "requirement_to_config",
                    })

                # Accumulate prose under a requirement as its semantic text.
                stripped = line.strip()
                is_meta = (id_match or config_match or not stripped
                           or stripped.startswith(("#", "<!--", "---")))
                if current and not is_meta:
                    nodes[current]["text"] += " " + stripped
    return nodes, edges


# --------------------------------------------------------------------------- #
# Parsing: Services / Tests via AST
# --------------------------------------------------------------------------- #
def _module_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def _iter_function_defs(tree):
    """Yield (qualified_name, node) for every def, including methods."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node.name, node


def build_function_registry(py_files):
    """Map a bare function/method name -> its qualified 'module.name' id."""
    registry = {}
    for path in py_files:
        module = _module_name(path)
        with open(path) as handle:
            tree = ast.parse(handle.read(), filename=path)
        for name, _ in _iter_function_defs(tree):
            registry[name] = "%s.%s" % (module, name)
    return registry


def _called_names(func_node):
    """Bare names of everything called inside a function body."""
    names = set()
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Name):
                names.add(target.id)
            elif isinstance(target, ast.Attribute):
                names.add(target.attr)
    return names


def _string_constants(func_node):
    """All string-literal values appearing inside a function body."""
    values = set()
    for node in ast.walk(func_node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            values.add(node.value)
    return values


def analyse_python(py_files, node_type, config_keys, registry):
    """Extract nodes + edges from a set of python files.

    node_type is "service" or "test". Produces:
      - one node per function defined in the files
      - config_to_service edges (only for services)
      - service_to_service / test_to_service call edges
    """
    nodes, edges = {}, []
    for path in py_files:
        module = _module_name(path)
        with open(path) as handle:
            tree = ast.parse(handle.read(), filename=path)
        for name, func_node in _iter_function_defs(tree):
            qualified = "%s.%s" % (module, name)
            docstring = ast.get_docstring(func_node) or ""
            nodes[qualified] = {
                "id": qualified,
                "type": node_type,
                "name": name,
                "file": os.path.relpath(path),
                "line": func_node.lineno,
                "text": " ".join((name, docstring)).strip(),
            }

            # Config usage: literal config keys read inside this function.
            if node_type == "service":
                for literal in _string_constants(func_node):
                    if literal in config_keys:
                        edges.append({
                            "from": literal,
                            "to": qualified,
                            "kind": "config_to_service",
                        })

            # Call edges to known service functions.
            for called in _called_names(func_node):
                callee = registry.get(called)
                if callee is None or callee == qualified:
                    continue
                kind = ("test_to_service" if node_type == "test"
                        else "service_to_service")
                edges.append({"from": qualified, "to": callee, "kind": kind})
    return nodes, edges


# --------------------------------------------------------------------------- #
# Impact analysis
# --------------------------------------------------------------------------- #
def build_propagation_graph(edges):
    """Direction a *change* propagates ("change here -> review there").

    Trace edges point R->C->S and caller->callee and test->service. A change
    propagates: requirement->config->service->caller and service->test, so we
    keep R->C and C->S as-is but reverse the call/test edges.
    """
    graph = {}
    for edge in edges:
        src, dst, kind = edge["from"], edge["to"], edge["kind"]
        if kind in ("service_to_service", "test_to_service"):
            src, dst = dst, src  # reverse: changing the callee/target impacts it
        graph.setdefault(src, set()).add(dst)
    return graph


def impacted_by(start, edges):
    """Breadth-first set of nodes impacted by a change to `start`."""
    graph = build_propagation_graph(edges)
    seen, order, queue = {start}, [], deque([start])
    while queue:
        node = queue.popleft()
        for neighbour in sorted(graph.get(node, ())):
            if neighbour not in seen:
                seen.add(neighbour)
                order.append(neighbour)
                queue.append(neighbour)
    return order


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #
def write_json(path, nodes, edges):
    payload = {
        "nodes": sorted(nodes.values(), key=lambda n: (n["type"], n["id"])),
        "edges": sorted(edges, key=lambda e: (e["kind"], e["from"], e["to"])),
    }
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


KIND_LABELS = {
    "requirement_to_config": "Requirement -> Config (R -> C)",
    "config_to_service": "Config -> Service (C -> S)",
    "service_to_service": "Service -> Service (S -> S)",
    "test_to_service": "Test -> Service (T -> S)",
}


def write_markdown(path, nodes, edges):
    lines = ["# Traceability Matrix", "",
             "Auto-generated by `extract_artifacts.py`. Do not edit by hand.", ""]
    lines += ["## Artifacts", "", "| ID | Type | Name | Source |",
              "|----|------|------|--------|"]
    for node in sorted(nodes.values(), key=lambda n: (n["type"], n["id"])):
        loc = "%s:%s" % (node["file"], node["line"])
        lines.append("| `%s` | %s | %s | `%s` |"
                      % (node["id"], node["type"], node["name"], loc))

    lines += ["", "## Dependencies", ""]
    for kind, label in KIND_LABELS.items():
        kind_edges = [e for e in edges if e["kind"] == kind]
        if not kind_edges:
            continue
        lines += ["### %s" % label, "", "| From | To |", "|------|----|"]
        for edge in sorted(kind_edges, key=lambda e: (e["from"], e["to"])):
            lines.append("| `%s` | `%s` |" % (edge["from"], edge["to"]))
        lines.append("")
    with open(path, "w") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")


DOT_STYLE = {
    "requirement": "shape=box, style=filled, fillcolor=\"#cfe8ff\"",
    "config": "shape=note, style=filled, fillcolor=\"#fff2cc\"",
    "service": "shape=ellipse, style=filled, fillcolor=\"#d5e8d4\"",
    "test": "shape=hexagon, style=filled, fillcolor=\"#f8cecc\"",
}


def write_dot(path, nodes, edges):
    lines = ["digraph artifacts {", "  rankdir=LR;",
             "  node [fontname=\"Helvetica\"];"]
    for node in sorted(nodes.values(), key=lambda n: n["id"]):
        style = DOT_STYLE.get(node["type"], "")
        label = "%s\\n%s" % (node["id"], node["name"])
        lines.append("  \"%s\" [label=\"%s\", %s];" % (node["id"], label, style))
    for edge in edges:
        lines.append("  \"%s\" -> \"%s\" [label=\"%s\"];"
                     % (edge["from"], edge["to"], edge["kind"].replace("_", " ")))
    lines.append("}")
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def extract(root):
    config_keys = parse_config(os.path.join(root, "configs"))
    req_nodes, req_edges = parse_requirements(os.path.join(root, "requirements"))

    service_files = sorted(glob.glob(os.path.join(root, "services", "*.py")))
    service_files = [p for p in service_files if not p.endswith("__init__.py")]
    test_files = sorted(glob.glob(os.path.join(root, "tests", "*.py")))
    test_files = [p for p in test_files if not p.endswith("__init__.py")]

    # Registry must span every service function so cross-module calls resolve.
    registry = build_function_registry(service_files)

    svc_nodes, svc_edges = analyse_python(
        service_files, "service", config_keys, registry)
    test_nodes, test_edges = analyse_python(
        test_files, "test", config_keys, registry)

    nodes = {}
    nodes.update(req_nodes)
    for key, meta in config_keys.items():
        nodes[key] = {"id": key, "type": "config", "name": key,
                      "file": meta["file"], "line": meta["line"],
                      "text": "%s %s" % (key, meta["value"])}
    nodes.update(svc_nodes)
    nodes.update(test_nodes)

    edges = req_edges + svc_edges + test_edges
    # De-duplicate identical edges (e.g. an annotation shown twice in a doc).
    seen, unique = set(), []
    for edge in edges:
        key = (edge["from"], edge["to"], edge["kind"])
        if key not in seen:
            seen.add(key)
            unique.append(edge)
    return nodes, unique


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)),
                        help="project root to analyse (default: this script's dir)")
    parser.add_argument("--impact", metavar="NODE",
                        help="print artifacts impacted by a change to NODE and exit")
    args = parser.parse_args()

    nodes, edges = extract(args.root)

    if args.impact:
        if args.impact not in nodes:
            parser.error("unknown artifact %r. Known: %s"
                         % (args.impact, ", ".join(sorted(nodes))))
        affected = impacted_by(args.impact, edges)
        print("Impact of changing %r:" % args.impact)
        if not affected:
            print("  (nothing else depends on it)")
        for node_id in affected:
            print("  - %s (%s)" % (node_id, nodes[node_id]["type"]))
        return

    json_path = os.path.join(args.root, "artifacts.json")
    md_path = os.path.join(args.root, "traceability_matrix.md")
    dot_path = os.path.join(args.root, "artifacts.dot")
    write_json(json_path, nodes, edges)
    write_markdown(md_path, nodes, edges)
    write_dot(dot_path, nodes, edges)

    print("Extracted %d artifacts and %d dependencies." % (len(nodes), len(edges)))
    print("Wrote:")
    for path in (json_path, md_path, dot_path):
        print("  - %s" % os.path.relpath(path, args.root))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit the checked-in Lean IR modules used by the correctness verifier.

This is deliberately a reader for generated Lean, not a certificate reader. It
makes the M0 executable-node inventory reproducible; semantic provenance is
reported only when it is exported by the Lean analyzer.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import re
import sys
from dataclasses import dataclass


ROOT = pathlib.Path(__file__).resolve().parents[1]
INPUTS = (
    ROOT / "crates/correctness/lean/MxxCorrectness/Generated/ToyExample/Ir.lean",
    ROOT / "crates/we/lean/MxxWe/Generated/DiamondWeFamily/Ir.lean",
)
EXPECTED_WORKFLOW_HASHES = {
    "ToyExample": "50d3fa2842746284fe6afc1d2b1004bec7e32982cdca391b9f6170b57877c374",
    "DiamondWeFamily": "60b0219e7732db469820f1b3a636c9d04528f62dc47ec09a8dc5cb54820dd01b",
}
EXPECTED_SOURCE_PATHS = {
    "ToyExample": (
        "crates/correctness/Cargo.toml",
        "crates/correctness/examples/emit_correctness.rs",
        "crates/correctness/src",
        "crates/dsl/Cargo.toml",
        "crates/dsl/src",
        "crates/ir-core/Cargo.toml",
        "crates/ir-core/src",
    ),
    "DiamondWeFamily": (
        "crates/bgg/Cargo.toml",
        "crates/bgg/src",
        "crates/correctness/Cargo.toml",
        "crates/correctness/src",
        "crates/dsl/Cargo.toml",
        "crates/dsl/src",
        "crates/gadgets/Cargo.toml",
        "crates/gadgets/src",
        "crates/ir-core/Cargo.toml",
        "crates/ir-core/src",
        "crates/we/Cargo.toml",
        "crates/we/examples/emit_correctness.rs",
        "crates/we/src",
    ),
}
EXPECTED_GENERATOR = "mxx-correctness-emitter-v5"
DEFAULT_ANALYSIS_FACTS = ROOT / "target/correctness/m0-analysis-facts.json"

ALLOWED_KINDS = {
    "bitExtract",
    "boolToInt",
    "concat",
    "constantBool",
    "constantInt",
    "constantMatrix",
    "dimension",
    "evaluateInt",
    "extractCoefficient",
    "familyGetDynamic",
    "familyGetStatic",
    "gadgetDecompose",
    "gadgetMatrix",
    "gaussianSample",
    "hashSample",
    "identityMatrix",
    "input",
    "intBinary",
    "intCompare",
    "matrixAdd",
    "matrixMultiply",
    "matrixNegate",
    "matrixScale",
    "matrixSubtract",
    "parallelLoop",
    "preimageSample",
    "reshape",
    "select",
    "sequentialLoop",
    "slice",
    "thresholdDecodeBool",
    "trapdoorSample",
    "uniformSample",
    "zeroMatrix",
}

FORBIDDEN_KINDS = {"familyPack", "subgraphCall", "trapdoorPublic"}
SPECIAL_KINDS = FORBIDDEN_KINDS | {"matrixMultiply", "matrixScale", "reshape"}

PROGRAM_RE = re.compile(r"^def ([A-Za-z0-9_]+) : Mxx\.Ir\.Prog :=")
METADATA_STRING_RE = re.compile(
    r'^def ([A-Za-z0-9_]+)_(generatorVersion|protocolSourceHash|workflowHash|toolkitHash) '
    r': String := "([^"]+)"'
)
SOURCE_PATHS_RE = re.compile(
    r"^def ([A-Za-z0-9_]+)_protocolSourcePaths : List String := (\[.*\])$"
)
SCOPE_RE = re.compile(r'^\s*\("([^"]+)",\s*$')
KIND_RE = re.compile(r"kind := \.([A-Za-z0-9_]+)")
ARGUMENT_RE = re.compile(r"\{ node := ([0-9]+), port := ([0-9]+) \}")
INPUT_NAME_RE = re.compile(r'\.input "([^"]+)"')
OUTPUT_RE = re.compile(r'\("([^"]+)", \{ node := ([0-9]+), port := ([0-9]+) \}\)')
ARTIFACT_BINDING_RE = re.compile(
    r'\("([^"]+)", \.artifact "([^"]+)" "([^"]+)"\)'
)


@dataclass(frozen=True)
class Node:
    source: pathlib.Path
    program: str
    scope: str
    node: int
    kind: str
    arguments: tuple[tuple[int, int], ...]
    text: str


@dataclass
class Freshness:
    generator_version: str | None = None
    protocol_source_paths: tuple[str, ...] | None = None
    protocol_source_hash: str | None = None
    workflow_hash: str | None = None
    toolkit_hash: str | None = None


def parse(
    path: pathlib.Path,
) -> tuple[dict[str, Freshness], list[Node], dict[tuple[str, str], tuple[int, int]]]:
    metadata: dict[str, Freshness] = {}
    nodes: list[Node] = []
    outputs: dict[tuple[str, str], tuple[int, int]] = {}
    program: str | None = None
    next_scope = "__root"
    scope = "__root"
    node_index = 0

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        metadata_match = METADATA_STRING_RE.match(raw_line)
        if metadata_match:
            name, field, value = metadata_match.groups()
            freshness = metadata.setdefault(name, Freshness())
            setattr(
                freshness,
                {
                    "generatorVersion": "generator_version",
                    "protocolSourceHash": "protocol_source_hash",
                    "workflowHash": "workflow_hash",
                    "toolkitHash": "toolkit_hash",
                }[field],
                value,
            )
        source_paths_match = SOURCE_PATHS_RE.match(raw_line)
        if source_paths_match:
            name, encoded_paths = source_paths_match.groups()
            metadata.setdefault(name, Freshness()).protocol_source_paths = tuple(
                json.loads(encoded_paths)
            )

        program_match = PROGRAM_RE.match(raw_line)
        if program_match:
            program = program_match.group(1)
            next_scope = "__root"
            scope = "__root"
            node_index = 0
            continue

        scope_match = SCOPE_RE.match(raw_line)
        if scope_match and program is not None:
            next_scope = scope_match.group(1)
            continue

        if "{ nodes := [" in raw_line and program is not None:
            scope = next_scope
            node_index = 0
            continue

        kind_match = KIND_RE.search(raw_line)
        if kind_match and program is not None:
            nodes.append(
                Node(
                    source=path,
                    program=program,
                    scope=scope,
                    node=node_index,
                    kind=kind_match.group(1),
                    arguments=tuple(
                        (int(node), int(port))
                        for node, port in ARGUMENT_RE.findall(raw_line)
                    ),
                    text=raw_line.strip(),
                )
            )
            node_index += 1

        if program is not None and scope == "__root" and "outputs :=" in raw_line:
            for name, output_node, port in OUTPUT_RE.findall(raw_line):
                outputs[(program, name)] = (int(output_node), int(port))

    return metadata, nodes, outputs


def hash_files(paths: list[pathlib.Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        relative = path.relative_to(ROOT).as_posix().encode()
        contents = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(contents).to_bytes(8, "little"))
        digest.update(contents)
    return digest.hexdigest()


def source_hash(relative_paths: tuple[str, ...]) -> str:
    sources: list[pathlib.Path] = []
    for relative in relative_paths:
        path = ROOT / relative
        if path.is_dir():
            sources.extend(candidate for candidate in path.rglob("*") if candidate.is_file())
        elif path.is_file():
            sources.append(path)
        else:
            raise FileNotFoundError(path)
    return hash_files(sources)


def toolkit_hash() -> str:
    return hash_files(list((ROOT / "lean/Mxx").rglob("*.lean")))


def location(node: Node) -> str:
    return f"{node.program}/{node.scope}/{node.node}"


def input_name(node: Node) -> str | None:
    match = INPUT_NAME_RE.search(node.text)
    return match.group(1) if match else None


def typed_origin_paths(value: object) -> list[object]:
    """Read origin leaves already emitted by Lean; do not derive semantic facts here."""
    if isinstance(value, list):
        return [origin for item in value for origin in typed_origin_paths(item)]
    if not isinstance(value, dict):
        return []
    origins: list[object] = []
    value_reference = value.get("valueInstanceRef")
    if isinstance(value_reference, dict):
        origins.append(value_reference)
    if value.get("kind") == "loopResult":
        origins.append(
            {
                "kind": "loopResult",
                "recurrence": value.get("recurrence"),
                "path": value.get("path"),
            }
        )
    for key, child in value.items():
        if key != "valueInstanceRef":
            origins.extend(typed_origin_paths(child))
    return origins


def operand_description(
    multiply: Node,
    operand: tuple[int, int],
    nodes: dict[tuple[str, str, int], Node],
    analyzer_facts: dict[tuple[str, str, int, int], dict[str, object]],
) -> tuple[str, str]:
    operand_node, port = operand
    source = nodes.get((multiply.program, multiply.scope, operand_node))
    if source is None:
        return (f"node {operand_node}:{port} (missing from scope)", "unknown")
    name = input_name(source)
    suffix = f' name="{name}"' if name is not None else ""
    analyzer_fact = analyzer_facts.get(
        (multiply.program, multiply.scope, operand_node, port)
    )
    classification = (
        str(analyzer_fact.get("primary")) if analyzer_fact is not None else "unknown"
    )
    matrix_fact = analyzer_fact.get("matrixFact") if analyzer_fact is not None else None
    primary_form = matrix_fact.get("primaryForm") if isinstance(matrix_fact, dict) else None
    origins = typed_origin_paths(primary_form)
    return (
        f"node {operand_node}:{port} kind={source.kind}{suffix} "
        f"analyzer-primary={classification} "
        f"typed-origins={json.dumps(origins, sort_keys=True, separators=(',', ':'))}",
        classification,
    )


def load_analysis_facts(
    path: pathlib.Path,
) -> tuple[dict[str, object] | None, dict[tuple[str, str, int, int], dict[str, object]], list[str]]:
    if not path.is_file():
        return None, {}, [f"analyzer fact table is missing: {path.relative_to(ROOT)}"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return None, {}, [f"cannot read analyzer fact table {path}: {error}"]
    failures: list[str] = []
    if payload.get("schema") != "mxx-analysis-facts-v1":
        failures.append("analyzer fact table schema is not mxx-analysis-facts-v1")
    facts: dict[tuple[str, str, int, int], dict[str, object]] = {}
    wire_facts = payload.get("wireFacts")
    if not isinstance(wire_facts, list) or not wire_facts:
        failures.append("analyzer fact table contains no wire facts")
        wire_facts = []
    for fact in wire_facts:
        try:
            key = (
                str(fact["program"]),
                str(fact["scope"]),
                int(fact["node"]),
                int(fact["port"]),
            )
        except (KeyError, TypeError, ValueError):
            failures.append(f"malformed analyzer wire fact: {fact!r}")
            continue
        if key in facts:
            failures.append(f"duplicate analyzer wire fact: {key!r}")
        if fact.get("primary") not in {"exact", "bounded", "affine"}:
            failures.append(f"wire fact has unsupported primary classification: {key!r}")
        if "valueInstanceRef" not in fact:
            failures.append(f"wire fact lacks ValueInstanceRef: {key!r}")
        if fact.get("primary") in {"exact", "bounded", "affine"}:
            matrix_fact = fact.get("matrixFact")
            if not isinstance(matrix_fact, dict):
                failures.append(f"matrix wire fact lacks typed matrixFact: {key!r}")
            elif not isinstance(matrix_fact.get("primaryForm"), dict):
                failures.append(f"matrix wire fact lacks typed primaryForm: {key!r}")
        facts[key] = fact
    for field in ("families", "recurrences", "semanticAnchors"):
        if not isinstance(payload.get(field), list):
            failures.append(f"analyzer fact table lacks {field} array")
    return payload, facts, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true", help="fail unless every M0 acceptance condition is established"
    )
    parser.add_argument(
        "--analysis-facts",
        type=pathlib.Path,
        default=DEFAULT_ANALYSIS_FACTS,
        help="JSON fact table exported directly from the Lean analyzer",
    )
    args = parser.parse_args()

    all_metadata: dict[str, Freshness] = {}
    all_nodes: list[Node] = []
    all_outputs: dict[tuple[str, str], tuple[int, int]] = {}
    for path in INPUTS:
        metadata, nodes, outputs = parse(path)
        all_metadata.update(metadata)
        all_nodes.extend(nodes)
        all_outputs.update(outputs)

    print("Canonical generated metadata:")
    for name, freshness in sorted(all_metadata.items()):
        print(f"  {name}:")
        print(f"    generator: {freshness.generator_version}")
        print(f"    workflow: {freshness.workflow_hash}")
        print(f"    protocol sources: {freshness.protocol_source_hash}")
        print(f"    toolkit: {freshness.toolkit_hash}")
        print(f"    source paths: {freshness.protocol_source_paths}")

    analysis_payload, analyzer_facts, analysis_failures = load_analysis_facts(
        args.analysis_facts
    )
    print("\nAnalyzer fact table:")
    if analysis_payload is None:
        print("  UNAVAILABLE")
    else:
        print(f"  source: {args.analysis_facts}")
        print(f"  wire facts: {len(analyzer_facts)}")

    print("\nNodeKind union:")
    counts = collections.Counter(node.kind for node in all_nodes)
    for kind, count in sorted(counts.items()):
        print(f"  {kind}: {count}")

    print("\nSpecial nodes:")
    for node in all_nodes:
        if node.kind in SPECIAL_KINDS:
            print(f"  {location(node)}: {node.text}")

    node_map = {(node.program, node.scope, node.node): node for node in all_nodes}
    unresolved_multiplications: list[str] = []
    print("\nMultiply operand provenance:")
    for multiply in (node for node in all_nodes if node.kind == "matrixMultiply"):
        descriptions = [
            operand_description(multiply, operand, node_map, analyzer_facts)
            for operand in multiply.arguments
        ]
        classes = [classification for _, classification in descriptions]
        if classes == ["exact", "exact"]:
            rule_shape = "X*X"
        elif classes == ["bounded", "exact"]:
            rule_shape = "L*X"
        elif len(classes) == 2 and classes[0] == "affine" and classes[1] in {"exact", "bounded"}:
            rule_shape = "A*R"
        elif classes == ["bounded", "affine"]:
            rule_shape = "L*A"
        elif classes == ["bounded", "bounded"]:
            rule_shape = "L*R"
        elif classes in (["exact", "affine"], ["affine", "affine"]):
            rule_shape = "FORBIDDEN"
            unresolved_multiplications.append(location(multiply))
        else:
            rule_shape = "UNRESOLVED (Lean analyzer fact absent)"
            unresolved_multiplications.append(location(multiply))
        print(f"  {location(multiply)}: {rule_shape}")
        for index, (description, _) in enumerate(descriptions):
            print(f"    operand[{index}]: {description}")

    diamond_text = INPUTS[1].read_text(encoding="utf-8")
    artifact_bindings = {
        consumer: (producer_stage, producer_output)
        for consumer, producer_stage, producer_output in ARTIFACT_BINDING_RE.findall(diamond_text)
    }
    residual_artifacts = (
        "diamond_decoder_preimage",
        "diamond_k_preimage",
        "diamond_one_preimage",
        "diamond_r_decomposed",
    )
    missing_artifact_paths: list[str] = []
    print("\nDiamond residual direct artifact origins:")
    for consumer in residual_artifacts:
        binding = artifact_bindings.get(consumer)
        consumer_node = next(
            (
                node
                for node in all_nodes
                if node.program == "DiamondWeFamily_stage_decrypt"
                and node.scope == "__root"
                and input_name(node) == consumer
            ),
            None,
        )
        if binding is None or consumer_node is None:
            missing_artifact_paths.append(consumer)
            print(f"  {consumer}: MISSING binding or consumer root input")
            continue
        producer_stage, producer_output = binding
        producer_program = f"DiamondWeFamily_stage_{producer_stage}"
        producer_wire = all_outputs.get((producer_program, producer_output))
        if producer_wire is None:
            missing_artifact_paths.append(consumer)
            print(f"  {consumer}: MISSING producer output")
            continue
        print(
            f"  {consumer}: {producer_program}/__root/{producer_wire[0]}:{producer_wire[1]} "
            f"-> {location(consumer_node)}:0"
        )

    print("\nDiamond residual typed origin paths:")
    semantic_anchors = analysis_payload.get("semanticAnchors", []) if analysis_payload else []
    residual_fact = next(
        (
            fact
            for fact in semantic_anchors
            if fact.get("label") == "diamond.decoder.residual"
        ),
        None,
    )
    residual_semantic_failure = True
    if residual_fact is None:
        print("  UNAVAILABLE in Lean analyzer fact table")
        print(
            "  required anchor: diamond.decoder.residual with analyzer-emitted "
            "coefficient and basis origins for both subtraction operands"
        )
    else:
        print("  typed anchor fact is present")
        values = residual_fact.get("values")
        inputs = residual_fact.get("inputs")
        output_matrix = (
            values[0].get("matrixFact")
            if isinstance(values, list) and len(values) == 1 and isinstance(values[0], dict)
            else None
        )
        input_matrices = (
            [entry.get("matrixFact") for entry in inputs]
            if isinstance(inputs, list) and len(inputs) == 2
            and all(isinstance(entry, dict) for entry in inputs)
            else []
        )

        def signal_identities(matrix_fact: object) -> object:
            if not isinstance(matrix_fact, dict):
                return None
            primary = matrix_fact.get("primaryForm")
            if not isinstance(primary, dict) or not isinstance(primary.get("terms"), list):
                return None
            identities = []
            for term in primary["terms"]:
                if not isinstance(term, dict) or not isinstance(term.get("coefficient"), dict):
                    return None
                identities.append(
                    (
                        term["coefficient"].get("expression"),
                        term.get("basis"),
                        term.get("mode"),
                    )
                )
            return identities

        output_signals = signal_identities(output_matrix)
        left_signals = signal_identities(input_matrices[0]) if len(input_matrices) == 2 else None
        right_signals = signal_identities(input_matrices[1]) if len(input_matrices) == 2 else None
        residual_semantic_failure = not (
            output_signals == []
            and left_signals is not None
            and left_signals == right_signals
        )
        print(
            "  semantic cancellation comparison: "
            + ("ESTABLISHED" if not residual_semantic_failure else "FAILED")
        )

    failures: list[str] = []
    failures.extend(analysis_failures)
    current_toolkit_hash = toolkit_hash()
    for name, expected_workflow_hash in EXPECTED_WORKFLOW_HASHES.items():
        freshness = all_metadata.get(name)
        if freshness is None:
            failures.append(f"generated metadata is missing for {name}")
            continue
        expected_paths = EXPECTED_SOURCE_PATHS[name]
        if freshness.generator_version != EXPECTED_GENERATOR:
            failures.append(f"{name} generator version changed")
        if freshness.protocol_source_paths != expected_paths:
            failures.append(f"{name} protocol source path set changed")
        elif freshness.protocol_source_hash != source_hash(expected_paths):
            failures.append(f"{name} protocol source hash is stale")
        if freshness.workflow_hash != expected_workflow_hash:
            failures.append(
                f"{name} workflow hash changed: expected {expected_workflow_hash}, "
                f"observed {freshness.workflow_hash}"
            )
        if freshness.toolkit_hash != current_toolkit_hash:
            failures.append(f"{name} toolkit hash is stale")
    if analysis_payload is not None and analysis_payload.get("workflowHash") != (
        all_metadata.get("DiamondWeFamily").workflow_hash
        if all_metadata.get("DiamondWeFamily") is not None
        else None
    ):
        failures.append("analyzer fact table does not match the Diamond workflow hash")
    unsupported = sorted(set(counts) - ALLOWED_KINDS)
    if unsupported:
        failures.append(f"allowlist-external kinds: {', '.join(unsupported)}")
    present_forbidden = sorted(set(counts) & FORBIDDEN_KINDS)
    if present_forbidden:
        failures.append(f"forbidden kinds: {', '.join(present_forbidden)}")

    scales = [node for node in all_nodes if node.kind == "matrixScale"]
    invalid_scales = [node for node in scales if ".matrixScale (.constant (1 : Int))" not in node.text]
    if invalid_scales:
        failures.append(
            "non-identity MatrixScale at " + ", ".join(location(node) for node in invalid_scales)
        )

    reshapes = [node for node in all_nodes if node.kind == "reshape"]
    expected_reshape = (
        "DiamondWeFamily_stage_encrypt/__root/42",
        '.reshape (.parameter "diamond_digit_count") (.constant (1 : Int))',
    )
    if len(reshapes) != 1 or location(reshapes[0]) != expected_reshape[0] or expected_reshape[1] not in reshapes[0].text:
        failures.append("the single expected bounded decomposition reshape changed")

    if unresolved_multiplications:
        failures.append(
            "multiply semantic provenance is unavailable at "
            + ", ".join(unresolved_multiplications)
        )
    if residual_semantic_failure:
        failures.append("Diamond residual typed-origin cancellation is not established")
    if missing_artifact_paths:
        failures.append(
            "Diamond residual artifact bindings are missing for "
            + ", ".join(missing_artifact_paths)
        )

    if failures:
        print("\nM0 audit: FAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1 if args.check else 0

    print("\nM0 audit: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

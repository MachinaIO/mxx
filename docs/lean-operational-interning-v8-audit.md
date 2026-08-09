# Lean Operational Interning v8 Audit

## Scope

This audit evaluates whether source-level constant interning is sufficient to make the local Tall
BGG+ parameter simulation practical. It does not change derivation rules, operational bounds,
checker acceptance logic, or theorem statements.

## Baseline

The depth-one Tall candidate uses ring dimension 8, 10-bit CRT moduli, 5-bit gadget base, scale
1024, and one parameter-simulation worker.

- Generated prepared module: 69,170,940 bytes
- Total emitted nodes: 117,175
- Earlier exact whole-node duplicate measurement: 34,042 unique node texts out of 117,175
- Cold elaboration: more than 82 minutes without completion

## Measurements

An intermediate local-let experiment reduced the generated module to 36,854,414 bytes. The v8
top-level `NodeKind` and output-type interning experiment reduced it further to 30,057,787 bytes.

The emitter histogram for the same depth-one candidate reported:

```text
total_nodes=117175
distinct_node_shapes=33198
distinct_node_kinds=33075
distinct_output_type_lists=75
```

The distinct-shape ratio is therefore 28.33%. This exceeds the session gate of 5%, so literal
interning is not expected to reduce elaboration enough by itself. The probe was stopped before a
full cold build. No Tall `.olean` was produced and no warm-cache timing is available.

## Semantic checks completed during the experiment

- The workflow hash remained unchanged.
- The derivation hash was restored to its pre-transport value by hashing the logical derivation
  list rather than the Array-vs-List Lean syntax.
- The compact node constructor was definitionally equal to the corresponding `Node` structure
  literal.
- No `sorry`, `admit`, axiom, or additional `native_decide` was introduced.
- The correctness Rust unit suite passed before the final measurement.
- `lake build MxxCorrectness` passed for the generated Toy module.
- The prepared Toy checker evaluated two requests in approximately six seconds.

## Decision

Per the session stop condition, do not proceed to the intermediate or full Tall timing gates with
literal interning. The next design to review is a compact numeric table plus a shared Lean decoder,
which avoids elaborating tens of thousands of distinct literal expressions.

The v8 source changes remain an uncommitted experiment. They are not a completed implementation
and must not be described as a successful Tall parameter-simulation optimization.

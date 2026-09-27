# mxx-ir-core

`mxx-ir-core` defines the program that every other mxx crate builds, executes, or analyzes: a
typed dataflow graph whose nodes are primitive lattice operations. A protocol is written once
as such a graph (usually through `mxx-dsl`), validated under concrete parameters, and then run by
the CPU or GPU backend in `mxx-backends` or exported to Lean. This crate depends on no other
workspace crate.

## Contents

| Module | What it defines |
| --- | --- |
| `graph` | Construction handles and scopes, subgraph sealing and captures, freezing, and serialization. |
| `node` | `NodeKind`, the complete vocabulary of executable operations. |
| `types` | Node and wire identities, symbolic and concrete wire types. |
| `ring` | Rings as ordered CRT bases, their resolution, and CRT basis generation. |
| `expr` | Compile-time integer and real expressions, and parameter bindings (`ParamEnv`). |
| `constraints` | Parameter-only constraints derived from a graph. |
| `validate`, `checks` | Structural and concrete validation producing a `ValidatedGraph`. |
| `encoding` | Canonical JSON and specification hashes. |
| `artifact` | Production identities and artifact manifests. |
| `protocol` | Multi-stage protocol declarations, ideal specifications, and predicates. |
| `lean` | Export of validated graphs and protocol claims to Lean. |
| `inventory` | A structural snapshot of a graph for checkers, without evaluation. |

The handwritten Lean package that generated relations build on lives in `lean/`; see
`lean/README.md`.

## Design

```text
construction code (mxx-dsl)  --freeze-->  Graph  --validate(bindings)-->  ValidatedGraph
                                                                            |-> CPU executor (mxx-backends)
                                                                            |-> GPU runtime  (mxx-backends)
                                                                            '-> Lean export  (lean)
```

- **Construction is not execution.** Building a graph performs no arithmetic or sampling; that
  happens only when a backend executes the validated graph.
- **Compile parameters versus runtime values.** Shapes, loop counts, moduli, and sampler
  parameters are compile expressions resolved during validation. Runtime integers and Booleans
  can select family members or candidates but never change a shape or a loop count.
- **Structure is kept, not unrolled.** Subgraph and loop bodies are stored and validated once.
  Executors instantiate them per call or per loop index, and runtime identities carry the
  instantiation path.
- **Sampler placement is semantics.** A sampler outside a loop is shared by every instance; a
  sampler inside a loop body produces a fresh value per executed instance. Samplers carry their
  integer coefficient cutoffs.
- **Artifacts link stages.** A graph can export outputs as artifacts identified by a hash of the
  graph and its bindings plus an execution nonce, and another graph imports them by that
  identity. Protocol declarations link stages, ideal specifications, and requirements.

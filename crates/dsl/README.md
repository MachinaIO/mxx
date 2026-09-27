# mxx-dsl

`mxx-dsl` is the Rust-based DSL for writing lattice protocols. You write a protocol with ring
elements, matrices, samplers, trapdoors, and loops as ordinary Rust values and operators. The
code records an `mxx-ir-core` graph that a backend in `mxx-backends` can validate and execute.
The DSL library depends only on `mxx-ir-core`; its examples also use `mxx-backends`.

[SPEC.md](SPEC.md) lists every value type and operation, grouped by purpose, with how to call
it.

## Contents

| Item | What it provides |
| --- | --- |
| `DslContext`, `BuiltGraph` | Declaring compile parameters, inputs, and outputs; building, validating, and drawing a graph (`render_html`). |
| `Ring` | Rings as ordered CRT bases, and the inputs, constants, and samplers over a ring. |
| `Mat`, `SmallMatrix`, `Preimage`, `Trapdoor`, `BoundedMatrix` | Matrices, bounded matrices, trapdoors, and their operations. |
| `Int`, `Bool`, `Bytes` | Runtime scalars and byte strings. |
| `Family` | Ordered collections with one element schema. |
| `parallel`, `iterate`, `select`, `Subgraph` | Independent loops, loops with carried state, runtime selection, and reusable named bodies. |
| `HashTag`, `tag!` | Typed hash tags for hash-derived samples. |
| `GraphValue`, `GraphValueSchema` | Flattening tuples, vectors, and records into graph wires. |
| `examples/rlwe_encrypt.rs` | A Ring-LWE round trip on the GPU (`--features gpu`). |

## Design

- **Recording, not computing.** Every operation creates a graph node at once, and nothing is
  computed until a backend executes the graph. Rust control flow runs once, while the graph is
  built, and cannot branch on a runtime `Bool`; `select` chooses between values instead.
- **Handles share values.** Cloning a handle shares the underlying node, so cloning a sample
  never draws a new one. Where a sampler is written decides whether it is shared or fresh per
  loop instance.
- **Loops are recorded once.** `parallel` and `iterate` run their closure once to record the
  body. Outer values the body reads become explicit loop arguments, either one family member per
  instance or a value shared by all instances.
- **Parameters stay symbolic.** Shapes, loop counts, moduli, and sampler widths can be
  `IntExpr` or `RealExpr` variables declared with `int_parameter` and `real_parameter`, and bound
  when the graph is validated.

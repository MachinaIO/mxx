# Workspace architecture

This repository is a virtual Cargo workspace. It intentionally has no root package or compatibility facade: consumers depend on the crate that owns the required abstraction.

## Dependency layers

Arrows below point from a consumer to one of its dependencies:

```text
mxx-runtime          -> mxx-ir-core, mxx-primitives
mxx-ir-symbolic    -> mxx-ir-core
mxx-noise-simulator -> mxx-ir-symbolic, mxx-primitives
mxx-bench-estimator  -> mxx-ir-core, mxx-runtime
mxx-gadgets          -> mxx-primitives, mxx-noise-simulator
mxx-bgg              -> mxx-ir-core, mxx-gadgets
application crates   -> graph, execution, primitive, gadget, and BGG layers
```

Dependencies follow these rules:

- `mxx-ir-core` has no dependency on another workspace crate.
- `mxx-runtime` depends on `mxx-ir-core` and `mxx-primitives`.
- `mxx-ir-symbolic` depends only on `mxx-ir-core`.
- `mxx-noise-simulator` depends on `mxx-ir-symbolic` and `mxx-primitives`.
- `mxx-bench-estimator` depends on `mxx-ir-core` and reuses the runtime liveness schedule.
- `mxx-primitives` has no dependency on another workspace crate.
- `mxx-gadgets` depends on `mxx-primitives` and re-exports the historical norm
  kernel from `mxx-noise-simulator` for existing callers.
- `mxx-bgg` depends on `mxx-ir-core` and `mxx-gadgets`.
- `mxx-func-enc`, `mxx-we`, and `mxx-io` may depend on the graph,
  execution, primitive, gadget, and BGG compiler layers.
- Application crates must not depend on one another.

Code shared by two application crates belongs in the lowest layer that can express it naturally. For example, the native polynomial-matrix benchmark interface and common iO/WE error-simulation helpers live in `mxx-gadgets`, while CUDA matrix measurements live beside the reusable benchmark interface instead of in `mxx-io`.

## Directory ownership

### `mxx-primitives`

`crates/primitives/` owns operations that are broadly useful across lattice-cryptography schemes:

| Path | Responsibility |
| --- | --- |
| `crates/primitives/src/element/` | Ring element abstractions. |
| `crates/primitives/src/poly/` | Polynomial traits, DCRT parameters, CPU polynomials, and GPU polynomial wrappers. |
| `crates/primitives/src/matrix/` | Generic matrices, DCRT matrices, memory/disk backing, and GPU matrices. |
| `crates/primitives/src/sampler/` | Uniform, hash, trapdoor, and GPU samplers plus generic sampling bounds. |
| `crates/primitives/src/rlwe_enc.rs` | Low-level RLWE encryption helpers. |
| `crates/primitives/src/env.rs` | Primitive-operation configuration. |
| `crates/primitives/cuda/` | Native CUDA headers and kernels. |
| `crates/primitives/build.rs` | OpenFHE and CUDA compilation/linking. |
| `crates/primitives/benches/` | Primitive CPU/GPU matrix and preimage benchmarks. |

GPU implementations of primitive operations stay in this crate even when a higher-level crate is their main caller.

### Graph infrastructure

| Crate | Path | Responsibility |
| --- | --- | --- |
| `mxx-ir-core` | `crates/ir-core/` | Executable typed graph structure, exact compile expressions, concrete type and shape validation, canonical identities, and runtime artifact manifests. This is the core Graph IR. |
| `mxx-ir-symbolic` | `crates/ir-symbolic/` | Optional symbolic atom and term identities, source provenance, elaboration, rewrite machinery, and cross-graph symbolic manifests. It retains only the structural `Large`/`Bounded` classification and performs no numerical bound calculation. It does not define BGG-, Diamond-, or AKY-specific invariants. |
| `mxx-noise-simulator` | `crates/noise-simulator/` | Target-driven numerical analysis of elaborated symbolic graphs. It owns all magnitude arithmetic, declared virtual-atom metadata, stable statistical dependencies, selection branch joins, threshold-decode reports, and the historical `PolyNorm`/`PolyMatrixNorm` rules. |
| `mxx-runtime` | `crates/runtime/` | CPU/GPU graph execution over existing primitive APIs, transcript recording/replay, shared liveness, indexed artifact-family persistence, and manifest production. |
| `mxx-bench-estimator` | `crates/bench-estimator/` | Per-node measurement composition, binding-sensitive subgraph reuse, critical paths, parallel waves, and persistent/workspace peak modeling. |
| `mxx-bgg` | `crates/bgg/` | BGG+ public-key and encoding wire bundles plus recursive `PolyCircuit` graph lowering. Lookup and slot-transfer gates receive an explicit scheme-specific lowering context because `PolyCircuit` does not own their preprocessing state. Concrete BGG arithmetic remains owned by `mxx-gadgets`. |

Trapdoor types and sampling nodes carry `gadget_base` and `digit_count`
explicitly. The runtime validates those values against the concrete backend
parameters before calling the unchanged trapdoor or preimage sampler.

`GadgetDecompose` and decomposed hash nodes normally derive their digit count
from the modulus and base. DCRT small decomposition is different: its width is
defined per CRT tower and cannot be recovered from the aggregate modulus.
Those nodes therefore accept an optional compile-time `digit_count`; DCRT
small-gadget graph builders must set it explicitly.

#### Symbolic overlays

`mxx-ir-symbolic` can optionally apply a `SymbolicOverlay` while elaborating
an unchanged executable Graph IR. An overlay never becomes a graph node and
does not affect the Graph IR spec hash or runtime behavior.

- A **fold** replaces selected terms with a derived fold atom whose
  `DefExpr::Fold` retains the exact replaced expression. It is an identity by
  construction and is therefore a symbolic fact.
- An **unfold** replaces a wire description with an assumed term list that may
  contain shared virtual atoms. It is an axiom. The local assumption hash and
  all transitively imported assumption digests are retained in elaboration
  results and symbolic manifests.

This fact/axiom distinction is the overlay trust boundary. An empty
`assumption_digests` set is the only indication that a result is
assumption-free; merely importing and re-exporting a result cannot erase its
provenance.

Fold validation:

| Check | Result |
| --- | --- |
| Expected and actual canonical term lists differ | Error with both descriptions |
| Groups overlap, omit a position, or use an invalid position | Error |
| A signal group lacks a common non-scalar suffix or has a large prefix | Error |
| A noise group contains a large atom | Error |
| A whole-signal group (`suffix_len = 0`) contains standalone noise | Error |
| A selector targets a non-matrix wire or overlaps another selector | Error |
| A folded prefix carries preimage references | Warning |
| A selector matches no concrete wire | Warning |

Unfold validation:

| Check | Result |
| --- | --- |
| A factor chain does not compose to the selected wire type | Error |
| An assumed preimage does not satisfy `uniform × preimage = target` by type | Error |
| Assumed preimage targets form a cycle | Error |
| Bounded/large character differs from the current description | Error |
| A non-source description is replaced without `replace_derived` | Error |
| A non-source description is replaced with `replace_derived` | Warning |
| A preimage description is discarded | Warning |
| A virtual declaration is invalid or references an undeclared virtual | Error |
| A virtual atom, assumed term list, or selector is unused | Warning |

Symbolic manifests use content-derived atom and term-list identifiers, retain
the original production namespace of re-exported records, and export the
complete closure of definitions, dependencies, uniforms, and preimage targets.
Multiple projections of one production merge only when their interpretation
digests and every overlapping record agree.

Symbolic manifest format version 3 exports self-contained source parameters,
assumed bounded metadata, and selection-domain roles. Locally declared
dependency labels and selection domains are qualified by their production
identity during export, and re-export preserves that origin.

#### Noise simulation

`mxx-noise-simulator::simulate` evaluates only matrix graph outputs and the
inputs of instantiated `ThresholdDecode` nodes. Atom results are memoized.
Reports distinguish signal from noise and state that their values are
high-probability coefficient envelopes using the existing CLT eligibility
rules, not worst-case bounds.

- Gaussian samples use the existing `6.5 * sigma` envelope.
- Preimage samples derive sigma from the explicit trapdoor sigma, gadget base,
  digit count, public-matrix row count, and target block-row count, then apply
  the same `6.5` envelope.
- Balanced gadget digits retain
  `6.5 * sqrt((base^2 + 2) / 12)`.
- Selection joins use the maximum branch norm. Dependencies are unioned,
  `is_const_poly` is combined with logical AND, equal `zero_rows` metadata is
  retained, and the joined result is not CLT-ready.
- Exact rational `RealExpr` values convert directly to `BigDecimal`; square
  roots use `BigDecimal::sqrt` without an intermediate `f64`.
- A bounded external input requires imported manifest metadata or an unfold
  assumption. The simulator does not guess a bound.
- Unknown modulus-switch shapes are reported as unsupported instead of being
  assigned a new universal formula.

Opaque tensor nodes are unsupported in the initial symbolic elaborator.
Concat, reshape, and folds reject any operation that would hide a signal/noise
mixture inside one atom. The range-versus-modulus comparison for
`UniformSample` remains in `mxx-ir-symbolic` because it determines the
structural `Large`/`Bounded` classification rather than a numerical noise
bound.

### `mxx-gadgets`

`crates/gadgets/` owns reusable cryptographic components assembled from primitives:

| Path | Responsibility |
| --- | --- |
| `crates/gadgets/src/bgg/` | BGG public keys, encodings, samplers, and vector/polynomial variants. |
| `crates/gadgets/src/circuit/` | Circuit IR, serialization, analysis, and evaluators. |
| `crates/gadgets/src/circuit_gadgets/` | Arithmetic, convolution, FHE/Ring-GSW, PRG, NTT, and other circuit gadgets. |
| `crates/gadgets/src/lookup/` | LWE and GGH15 lookup mechanisms. |
| `crates/gadgets/src/decoder/` | Decoder artifacts, mask circuits, PRG helpers, simulation, and estimates. |
| `crates/gadgets/src/noise_refresh/` | Reusable noise-refresh circuits and models. |
| `crates/gadgets/src/input_injector/` | Diamond input injection. |
| `crates/gadgets/src/slot_transfer/` | Public-key and encoding slot transfer. |
| `crates/gadgets/src/commit/` | Commitment components, including WEE25. |
| `crates/gadgets/src/bench_estimator/` | Reusable circuit and native-operation benchmark models. |
| `crates/gadgets/src/simulator/` | Lattice-security and shared application simulation utilities, plus compatibility re-exports of the generic norm kernel owned by `mxx-noise-simulator`. |
| `crates/gadgets/src/storage/` | Artifact storage helpers. |

The former circuit-gadget directory is now `crates/gadgets/src/circuit_gadgets/` so that the crate name `mxx-gadgets` describes the full layer rather than one nested directory.

### Application crates

| Crate | Paths | Responsibility |
| --- | --- | --- |
| `mxx-func-enc` | `crates/func-enc/src/` | Functional-encryption trait and constructions. The AKY24 module remains disabled as it was before the split. |
| `mxx-we` | `crates/we/src/diamond_we*` | Witness-encryption trait, Diamond WE, simulation, and estimates. |
| `mxx-io` | `crates/io/src/aky24_io*`, `crates/io/src/diamond_io*` | iO trait, AKY24 iO models, Diamond iO, simulations, and estimates. |

Diamond WE provides key-generation and evaluation graph builders. Diamond iO
and the maintained AKY24 model provide obfuscation and evaluation graph
builders for their common artifact-backed state-transition path. These core
graphs remain executable without optional symbolic elaboration.

## Features and native build ownership

Crates with concrete storage or device behavior define `disk` or `gpu`
features as needed. Upper crates only forward those features to their
dependencies. `mxx-primitives` is the sole owner of:

- OpenFHE linkage;
- `cc` build dependencies;
- CUDA source compilation;
- `libc` and `memmap2` for primitive matrix backing.

Use workspace-wide commands when checking a cross-cutting change:

```sh
cargo check --workspace
cargo check --workspace --features disk
cargo check --workspace --features gpu
```

## Tests

Unit tests remain beside the implementation they test. Integration tests are grouped by owner:

- `crates/gadgets/tests/` for lookup, circuit-gadget, input-injector, and related regression tests;
- `crates/we/tests/` for Diamond WE;
- `crates/io/tests/` for AKY24 iO and Diamond iO.

Integration tests are not part of routine validation and must only be run when a task explicitly requests them. Compiling their targets with `cargo check --workspace --tests` is permitted for boundary validation.

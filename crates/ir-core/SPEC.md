# mxx-ir-core Lean correctness specification

This document specifies how a protocol written in the DSL becomes a Lean theorem to prove: what
the protocol declaration contains, which Lean modules `mxx_ir_core::lean::protocol::export`
writes, what the generated statement `GeneratedClaim.CorrectnessClaim` says, and how a proof
package checks a proof of it. It is written for readers who know lattice cryptography and Lean
but not mxx. The DSL itself is specified in `crates/dsl/SPEC.md`.

## 1. Concepts

**Statement compiler.** `export` reads a protocol declaration and deterministically writes the
Lean statement of its correctness. The statement is a function of the declaration only: the
same graphs and bindings always give byte-identical modules. Nobody writes or edits the
statement by hand, so a reader who trusts the exporter and the Lean kernel only has to check that
a fixed theorem, `GeneratedClaim.CorrectnessClaim`, is proved.

**Stage.** One graph that runs as a unit, such as key generation or decryption, identified by a
`StageId`. A protocol is an ordered list of stages, producers before consumers.

**Ideal functionality.** A separate graph that computes, without cryptography, what the protocol
should output, for example the product of two plaintexts. It reads the same external inputs as
the stages.

**External input.** A value the caller chooses, such as a message or a public hash key, as
opposed to a value a stage computes. Each one has a *contract*, the assumption the statement
makes about it, and *destinations*, the graph inputs it feeds.

**Link.** A stage input fed by an earlier stage's output, as the caller passes one execution's
output to the next.

**Endpoint.** The stage output compared with the ideal output of the same meaning, for example
the decrypted bit.

**Correctness claim.** The statement that the endpoint equals the ideal output, for every
execution or except with a bounded probability. It states nothing else: noise bounds, decoder
margins, and other intermediate facts are steps of the proof, not part of the statement.

**Proof package.** A Lean package, owned by the application, that proves the claim and checks
the proof with a certificate.

## 2. Declaring a protocol

A protocol is a `mxx_ir_core::protocol::ProtocolDecl`. `ProtocolDecl::new` validates it.

| Field | Meaning |
| --- | --- |
| `params` | The compile parameters every graph of the protocol declares, with their kinds. |
| `bindings` | The values of `params` at which the claim is stated. |
| `failure_probability_log2` | `None` claims that every execution is correct; `Some(k)` claims that an execution fails with probability at most `2^-k` (section 6). |
| `bundle.workflow` | The stages in order and the entrypoint stage; each stage has its graph and its links. |
| `bundle.ideal` | The ideal functionality, an `IdealSpec`; it has no samplers. |
| `bundle.input_contract`, `bundle.input_bindings` | One contract and one destination list per external input. |
| `bundle.comparator`, `bundle.endpoints`, `bundle.endpoint_specs` | The compared endpoint: `ComparatorSpec::Equality` with one `EndpointSpecId::Exact` endpoint naming the stage output and the ideal output. |
| `bundle.requirements`, `bundle.precondition_spec` | Optional requirement graphs whose Boolean outputs must be true in an execution. |
| `bundle.operational_decoder_targets` | Optional decoder descriptions; the claim does not read them. |

**Links.** A `ProtocolStage` lists its links as `ArtifactBinding`s from a producer stage output
to a consumer input. The consumer input may be a plain graph input, fed directly, or an artifact
input, whose name and availability must then match the producer output. Leaves of a composite
value are linked one by one; `mxx_dsl::artifact_bindings` names them from the value's schema.

**External inputs.** Every graph input that no link feeds must be the destination of exactly one
external input. An input can feed several destinations, typically a stage input and the ideal
input of the same meaning, so both sides see the same value. The exporter turns these contracts
into Lean predicates:

| `InputValueContract` | Lean predicate on the value `x` |
| --- | --- |
| `IntegerRange { lower, upper }` | `lower ≤ x ∧ x ≤ upper` |
| `Boolean` | `True` |
| `Bytes { length }` | `x.size = length` |
| `Family { count, element }` | `∀ i : Fin count, P (x i)` for the element predicate `P` |

Matrix and trapdoor contracts are rejected by the exporter.

**Endpoint.** The stage output and the ideal output must have the same type: a Boolean, an
integer, or a family of either.

## 3. Exporting

```rust
mxx_ir_core::lean::protocol::export(&protocol, &directory)?;
```

`export` validates the declaration, validates every graph at `bindings` (stages against the
manifests of the artifacts their producers export), and writes into `directory`:

| Module | Contents |
| --- | --- |
| `Stage_<id>.lean` | The relation of stage `<id>` (section 4). Stage IDs are nonempty ASCII letters, digits, and underscores. |
| `Requirement_<i>.lean` | The relation of requirement graph `i`. |
| `Ideal.lean` | The relation of the ideal functionality. |
| `Backend.lean` | The gadget layouts the graphs use (section 5). |
| `Claim.lean` | The linked statement `GeneratedClaim.CorrectnessClaim` (section 6). |

The application writes nothing into the generated modules. Its own Lean files, the proof and
the certificate, live beside them in the proof package (section 7).

## 4. Stage modules

Each module `M` describes one graph as a Lean relation, not as a function, so that sampled and
hashed values are existential witnesses constrained by what the runtime guarantees about them.

- `M.Params` is a structure with one field per compile parameter; `Claim.lean` instantiates it
  with `bindings`.
- `M.generatedRoot` relates the graph's inputs and outputs: `M.generatedRoot params inputs
  outputs` holds exactly when some choice of witnesses (sampled values, hash outputs, decoded
  integers) satisfies every node's constraint. Inputs and outputs are right-nested tuples in the
  order of the graph's input and output nodes; names do not appear.
- Each subgraph and loop body has its own relation, called from its parent's; loops are not
  unrolled, and families are functions on `Fin count`.
- Every node contributes the relation of its primitive, defined in the `MxxRuntime` Lean library
  of `crates/ir-core/lean`: for example `matrixAdd`, `thresholdDecode`, `gaussianSample`, or
  `gadgetDecomposeRuns`.
- A polynomial constant with more than 64 coefficients is a packed natural number, defined at
  top level as `M.<scope>.table_<node>` so that proofs can refer to it.

**Samplers.** A sampler node relates its output to the sampler's support only: a Gaussian sample
is any value whose centered coefficients lie within the cutoff, and a uniform sample is any
value in its interval or residue ring. Because the runtime resamples any draw beyond the cutoff,
these relations hold for every execution, and a deterministic claim (`None`) holds for every
sample. With a failure probability, each sampled coefficient is instead read from a sampling
tape (section 6).

**Hashes.** A hash node reads `hashModel`, a `MxxRuntime.HashModel` the claim quantifies over
universally: the statement holds for every function from keys to outputs, so it assumes nothing
about the hash function.

## 5. The backend module

Gadget operations (decomposition, gadget matrices, trapdoors, preimages) read the regular gadget
layout of their ring from `Backend.backend`, a `MxxRuntime.BackendContext`. `export` derives one
layout per ring from the gadget nodes exactly as the runtime backend derives it from the same
declarations:

- `digitsPerTower = ceil(bits / log2 base)`, where `bits` is the bit length of the widest CRT
  modulus, which the runtime's `crt_bits` always equals;
- a regular decomposition with `D` digits keeps the leading `D / digitsPerTower` towers and drops
  the rest (`droppedModuli`), an approximate gadget whose error `MxxRuntime.RegularLayout.errorBound`
  bounds;
- a small decomposition must have `digitsPerTower` digits and reads every tower.

Export fails when a digit count does not keep whole towers, or when two nodes of one ring
declare different layouts. Lean checks every layout's obligations: coprime towers whose product
is the modulus, and enough digits to cover each tower.

## 6. The claim

`Claim.lean` links the root relations and states the claim:

```lean
structure ExternalInputs where      -- one field per external input
  input_0 : ...
def ValidExternals (external : ExternalInputs) : Prop := ...   -- the contracts
structure Execution where           -- one field per root: stages, requirements, ideal
  «stage_0» : ...
  «ideal» : ...
def Runs (hashModel) (external) [tape] (execution) : Prop :=
  ValidExternals external ∧
  Stage_<id>.generatedRoot ... stage_0_params (<inputs>) execution.«stage_0» ∧ ... ∧
  <requirement outputs> = true ∧ ... ∧
  Ideal.generatedRoot ideal_params (<inputs>) execution.«ideal»
```

Each root input is an external input field or, for a link, the producer's output projected
from `execution`. The claim compares the endpoint projections:

```lean
-- failure_probability_log2 = None
def CorrectnessClaim : Prop :=
  ∀ hashModel external execution, Runs hashModel external execution →
    <stage endpoint> = <ideal endpoint>

-- failure_probability_log2 = Some k
def CorrectnessClaim : Prop :=
  ∀ hashModel external,
    MxxRuntime.tapeMeasure {tape | ∃ execution, Runs hashModel external tape execution ∧
      ¬ (<stage endpoint> = <ideal endpoint>)} ≤ (2 : ENNReal)⁻¹ ^ k
```

**Sampling tape.** A `MxxRuntime.SampleTape` assigns an integer to every `SampleKey`: a site path,
a row, a column, a coefficient, and the sampler's law. Root `i` reads site prefix `[i]`; a
subgraph call appends its node, a loop body its node and iteration, and a sampler node reads
`path ++ [node]`, so distinct occurrences read distinct keys and the same occurrence always reads
the same key. `MxxRuntime.tapeMeasure` is the product measure that draws every key independently
from its law: the truncated discrete Gaussian, a uniform integer interval, or a uniform residue.
This is the ideal-sampler assumption. The claim bounds the measure of the tapes on which some
execution fails, for every hash model and every external input. Trapdoor and preimage samplers
have no tape semantics, so a probabilistic claim rejects them.

## 7. Proof packages

A proof package is a Lake package with three parts:

- a library over `generated/`, the modules `export` writes;
- the handwritten proof, which imports `Claim` and proves `GeneratedClaim.CorrectnessClaim`;
- `Certificate.lean`, which checks the proof against the generated statement and reports the
  axioms it uses:

```lean
import Claim
import MyProof

theorem certificate : GeneratedClaim.CorrectnessClaim := MyProof.correctness

#print axioms certificate
```

The package requires `mxx-ir-core` (`crates/ir-core/lean`) and Mathlib by path. `lake build` in
the package directory checks everything; an accepted proof prints only `propext`,
`Classical.choice`, and `Quot.sound`. Since the certificate names the generated statement, a
reviewer checks only these three lines and the axiom report, not the proof.

`MxxRuntime` supplies protocol-independent tools: lemmas that restate sampler, hash, and
arithmetic relations (`RuntimeLemmas.lean`), exponential-moment, Chernoff, and Hoeffding bounds
over the tape (`RuntimeProbability.lean`), and the sub-Gaussian moment of the truncated discrete
Gaussian (`RuntimeGaussian.lean`).

## 8. Examples

| Protocol | Declared in | Claim | Proof package |
| --- | --- | --- | --- |
| RLWE encryption of one bit | `crates/dsl/examples/rlwe_encrypt.rs` | every execution decrypts its message | `crates/dsl/examples/rlwe` |
| TFHE NAND gate | `crates/fhe/tests/gpu_tfhe.rs` | the gate fails with probability at most `2^-128` | `crates/fhe/lean/tfhe` |
| BGV multiply, relinearize, and modulus switch | `crates/fhe/tests/gpu_bgv.rs` | every execution decrypts the slotwise product | `crates/fhe/lean/bgv` |

Each declaration uses the graphs it executes, so the statement describes exactly the program
that runs.

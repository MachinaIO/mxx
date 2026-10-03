# Lean correctness statements

This document explains how to turn a protocol written in the DSL into a Lean theorem, and exactly
which statement that theorem is. It is written for readers who know lattice cryptography and Lean
but not mxx. The DSL itself is specified in `crates/dsl/SPEC.md`.

## 1. Overview

You describe a protocol as a list of DSL graphs (its *stages*) and a separate *ideal
functionality* that computes, without cryptography, what the protocol should output. You pass
this declaration and a directory to `mxx_ir_core::lean::protocol::export`, which writes Lean
modules ending in one proposition, `GeneratedClaim.CorrectnessClaim`:

> At the declared parameters, for every input that meets its contract, the protocol's output
> equals the ideal output, for every execution or except with a declared probability.

The statement is generated deterministically from the declaration: the same graphs and parameters
always give byte-identical modules, and nobody writes or edits the statement by hand. You then
prove the statement in Lean. A reader who trusts the generator and the Lean kernel only has to
check that this one fixed proposition is proved; they do not have to read the proof.

## 2. Declaring a protocol

A protocol is a `mxx_ir_core::protocol::ProtocolDecl`; `ProtocolDecl::new` checks it.

| Field | What you supply |
| --- | --- |
| `params` | The compile parameters, which every graph of the protocol must declare. |
| `bindings` | Their values. The statement is about the protocol at exactly these values. |
| `failure_probability_log2` | `None` to claim that every execution is correct, or `Some(k)` to claim that an execution fails with probability at most `2^-k`. |
| `bundle.workflow` | The stages in execution order, each a graph with an ID, and the entrypoint stage. |
| `bundle.ideal` | The ideal functionality, an `IdealSpec` (a graph without samplers). |
| `bundle.input_contract`, `bundle.input_bindings` | The external inputs: for each, its contract and where it goes. |
| `bundle.comparator`, `bundle.endpoints`, `bundle.endpoint_specs` | The compared output: `ComparatorSpec::Equality` with one `EndpointSpecId::Exact` endpoint naming a stage output and the ideal output it must equal. |
| `bundle.requirements`, `bundle.precondition_spec` | Optional graphs whose Boolean outputs the statement assumes true. |
| `bundle.operational_decoder_targets` | Optional; the statement does not use them. |

Every input of every graph must be fed by exactly one of the following:

- **An earlier stage's output (a link).** Each stage lists links from producer outputs to its
  inputs, as you pass one execution's output to the next. `mxx_dsl::artifact_bindings` builds
  them for a composite value from its schema.
- **An external input.** A value the caller chooses, such as a message or a 32-byte hash key.
  It can feed several inputs, typically a stage input and the ideal input with the same meaning,
  so that both sides see the same value. Its contract is the only assumption the statement makes
  about it:

| `InputValueContract` | Assumption on the value `x` |
| --- | --- |
| `IntegerRange { lower, upper }` | `lower ≤ x ≤ upper` |
| `Boolean` | none |
| `Bytes { length }` | `x` has `length` bytes |
| `Family { count, element }` | each of the `count` members meets `element` |

The compared stage output and ideal output must have the same type: a Boolean, an integer, or a
family of either.

## 3. Exporting

```rust
mxx_ir_core::lean::protocol::export(&protocol, &directory)?;
```

This writes into `directory` one module per stage (`Stage_<id>.lean`, so stage IDs are ASCII
letters, digits, and underscores), one per requirement (`Requirement_<i>.lean`), `Ideal.lean`,
`Backend.lean`, and `Claim.lean`, which holds the statement. Regenerate the modules whenever the
graphs or parameters change; never edit them.

## 4. The generated statement

`Claim.lean` builds the statement from four definitions. The RLWE example
(`crates/dsl/examples/rlwe_encrypt.rs`) declares one stage `rlwe` that encrypts and decrypts a
bit, external inputs `seed` (32 bytes) and `message` (in `[0, 1]`), and an ideal functionality
that returns the message; it generates this `Claim.lean` (matrix types and parameter records
abbreviated):

```lean
structure ExternalInputs where
  input_0 : ByteArray           -- seed
  input_1 : Int                 -- message

def ValidExternals (external : ExternalInputs) : Prop :=
  (external.input_0).size = 32 ∧
  ((0 : Int) ≤ external.input_1 ∧ external.input_1 ≤ (1 : Int))

structure Execution where
  «stage_0» : ExactMatrix Q 4096 1 1 × Bool × Unit   -- outputs of stage rlwe
  «ideal» : Bool                                      -- output of the ideal functionality

def Runs (hashModel : MxxRuntime.HashModel) (external : ExternalInputs)
    (execution : Execution) : Prop :=
  ValidExternals external ∧
  Stage_rlwe.generatedRoot hashModel stage_0_params
    ((external.input_0, external.input_1, ())) execution.«stage_0» ∧
  Ideal.generatedRoot ideal_params (external.input_1) execution.«ideal»

def CorrectnessClaim : Prop :=
  ∀ hashModel external execution, Runs hashModel external execution →
    execution.«stage_0».2.1 = execution.«ideal»
```

**`ExternalInputs`** has one field per external input, `input_0`, `input_1`, ..., in the order of
`input_bindings`. **`ValidExternals`** is the conjunction of their contracts.

**`Execution`** has one field per graph: `stage_0`, `stage_1`, ... for the stages in workflow
order, `requirement_0`, ... for the requirements, and `ideal`. Each field holds that graph's
outputs. A graph with several outputs gives a right-nested tuple ending in `()`, ordered by output
name; a graph with one output gives the value itself. So `execution.«stage_0».2.1` above is the
second output of `rlwe` by name, `decrypted`.

**`Runs`** says that `execution` is a possible run of the protocol on `external`. It conjoins the
contracts, one relation per graph, and `= true` for each requirement output. Each graph reads its
inputs as a tuple in the order of its input nodes, which the signature of its `generatedRoot`
shows: an external input reads `external.input_i`, and a link reads the producer's output from
`execution`. `stage_0_params` holds the parameter bindings.

**`M.generatedRoot params inputs outputs`** is the relation of graph `M`: it holds exactly when
the graph can map `inputs` to `outputs`. Deterministic operations fix their results, and the
relation leaves free only what the runtime does not determine:

- **A sampled value** may be any value the sampler can return: any value whose centered
  coefficients lie within the Gaussian cutoff, or any value in a uniform sampler's range. The
  runtime resamples any draw beyond the cutoff, so every real execution is among these.
- **A hashed value** is `hashModel` applied to the hash input. The statement quantifies over every
  `hashModel`, so it assumes nothing about the hash function.

**`CorrectnessClaim`** compares the declared stage output with the ideal output. With
`failure_probability_log2 = None` it is the worst-case statement above: every run, with every
possible sampled value and every hash function, is correct. With `Some(k)` the sampled values are
random instead:

```lean
def CorrectnessClaim : Prop :=
  ∀ hashModel external,
    MxxRuntime.tapeMeasure {tape | ∃ execution, Runs hashModel external tape execution ∧
      ¬ (<stage output> = <ideal output>)} ≤ (2 : ENNReal)⁻¹ ^ k
```

Here `Runs` also takes a `tape` that fixes every sampled value, and `MxxRuntime.tapeMeasure`
draws the tape so that every sampled coefficient of every sampler occurrence is an independent
draw from that sampler's distribution: the discrete Gaussian truncated at its cutoff, or a
uniform integer or residue. This is the ideal-sampler assumption. The statement says that, for
every hash function and every valid external input, the sampled values on which some run fails
have probability at most `2^-k`. Trapdoor and preimage samplers have no such distribution, so a
protocol that uses them can only make the worst-case statement.

## 5. Proving the statement

Put the proof in a Lake package beside the generated modules:

| Path | Contents |
| --- | --- |
| `lakefile.toml` | A library over `generated/`, a library for your proof, and requirements on `mxx-ir-core` (`crates/ir-core/lean`) and Mathlib by path. |
| `generated/` | The modules `export` writes. |
| your proof | A module that imports `Claim` and proves `GeneratedClaim.CorrectnessClaim`. |
| `Certificate.lean` | The three lines below. |

```lean
import Claim
import MyProof

theorem certificate : GeneratedClaim.CorrectnessClaim := MyProof.correctness

#print axioms certificate
```

`lake build` in the package directory checks everything. An accepted proof uses only the axioms
`propext`, `Classical.choice`, and `Quot.sound`. Since the certificate names the generated
statement, a reviewer reads only these lines and the axiom report.

In a proof, destructure `Runs` into the graph relations and each relation into its witnesses
and constraints. The definitions they refer to are in the `MxxRuntime` library of
`crates/ir-core/lean`, which also provides lemmas for common steps: bounding Gaussian and uniform
samples, reading coefficients of reduced polynomials, and, for probabilistic statements,
Chernoff and Hoeffding bounds over the sampling tape and the sub-Gaussian moment of the truncated
Gaussian. Graphs that use gadget decomposition read their layouts from `Backend.backend`, and a
long constant polynomial is a named definition `M.<scope>.table_<node>` that a proof can refer to.

## 6. Examples

| Protocol | Declared in | Statement | Proof |
| --- | --- | --- | --- |
| RLWE encryption of one bit | `crates/dsl/examples/rlwe_encrypt.rs` | every execution decrypts its message | `crates/dsl/examples/rlwe` |
| TFHE NAND gate | `crates/fhe/tests/gpu_tfhe.rs` | the gate fails with probability at most `2^-128` | `crates/fhe/lean/tfhe` |
| BGV multiply, relinearize, and modulus switch | `crates/fhe/tests/gpu_bgv.rs` | every execution decrypts the slotwise product | `crates/fhe/lean/bgv` |

Each declaration uses the graphs it executes, so the statement is about exactly the program that
runs.

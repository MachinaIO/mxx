# mxx-fhe

`mxx-fhe` builds fully homomorphic encryption with the mxx DSL: TFHE with NAND bootstrapping over
integer LWE, and leveled BGV with SIMD slots, rotations, relinearization, and hybrid RNS key
switching. Its methods build graphs rather than encrypting eagerly, so the same key generation,
encryption, evaluation, and decryption can run on the CPU or a supported GPU backend. It depends on `mxx-dsl`,
`mxx-ir-core`, and `mxx-backends`.

## Contents

| Item | What it provides |
| --- | --- |
| `FheCommonParams` | Ring, secret, and Gaussian parameters shared by both schemes. |
| `tfhe` | TFHE and everything that depends on it: parameters, ciphertexts, keys, public-key encryption, the bootstrapping stages, its `spec::Fhe` implementation with the NAND gate (`tfhe::scheme`), and the CUDA blind-rotation kernel registration (`tfhe::gpu`). |
| `BgvParams`, `BgvCiphertext` | BGV parameters, SIMD encoding, arithmetic, rotations, and key switching. |
| `spec` | The implementation-agnostic public-key HE syntax: the circuits of two-input gates and `Depth` (`spec/circuit.rs`), the `Fhe` trait and `Eval` (`spec/scheme.rs`), statements and reductions (`spec/statement.rs`), the graphs of a scheme and a circuit (`spec/graphs.rs`), and the Lean statement generator (`spec/lean/`). |
| `bgv` | BGV and everything that depends on it: the scheme, its `spec::Fhe` implementation with the slot-wise multiplication gate (`bgv::scheme`), and test fixtures. |
| `utils` | Parameter helpers, including the standard TFHE Boolean profile used in tests. |
| `lean/` | Lean packages: the FHE definitions (`spec`), the BGV correctness and security proofs (`bgv`), and the TFHE statements and proofs (`tfhe`). |
| `cuda/` | The CUDA-only native TFHE blind-rotation kernel. |
| `tests/` | GPU runs of the `spec::graphs` of TFHE and BGV: the graphs the Lean statements name. |
| `scripts/` | A GPU BGV comparison with PhantomFHE: building a pinned external checkout, the comparison driver, and validating and summarizing its measurements. |

## Design

- **Schemes as graph builders.** Each operation adds nodes to a `DslContext`. A protocol stage,
  such as key generation or one bootstrapped gate, is one graph that is planned once and run
  many times, with keys kept on the GPU between runs.
- **Public noise tracking.** Ciphertexts carry public noise bounds that evaluation propagates,
  and `can_decrypt` checks a conservative condition without inspecting secrets.
- **A native kernel where it pays.** TFHE blind rotation is a named subgraph whose GPU execution
  is one cooperative CUDA kernel, while the CPU runs its DSL body; tests check that both agree
  bit for bit.

## Implementation-agnostic HE statements

Correctness and IND-CPA security of public-key homomorphic encryption are defined once, in Lean,
by the package `lean/spec` (library `MxxFhe`), at the level of abstraction of the literature:
`KeyGen` returns a public key, an evaluation key, and a secret key; `Enc` encrypts a message under
the public key; `Eval` evaluates a circuit on ciphertexts with the evaluation key; `Dec` decrypts
with the secret key. Correctness is stated for one circuit (`MxxFhe.Evaluation`: its plaintext
value and `Eval` of it). Nothing in it depends on a scheme. A scheme enters a statement only
through the trait `spec::Fhe`, which gives each algorithm as a function on DSL values, with
associated types for the messages, the keys, and the ciphertexts. TFHE and BGV implement the same
trait, and everything from it to the Lean statements is shared. The circuits are the same for
every scheme: a `spec::Circuit` is a list of two-input gates in topological order, and a scheme
gives only its gate. `spec::evaluate` and `spec::plain_eval` run a circuit gate by gate.

| Syntax | `Fhe` |
| --- | --- |
| depth | `depth() -> spec::Depth`: `Bounded(d)` for a leveled scheme (BGV), `Unbounded` for a bootstrapped one (TFHE) |
| message space | `message_space()`: the message type and the messages correctness covers |
| `KeyGen` | `key_generation() -> (PublicKey, EvaluationKey, SecretKey)` |
| `Enc` | `encrypt(pk, message) -> Ciphertext` |
| gate | `plain_gate(x, y) -> Message` on messages, `eval_gate(ek, x, y) -> Ciphertext` on ciphertexts |
| `Eval` | `spec::evaluate(scheme, ek, circuit, ciphertexts)`: `eval_gate` at each gate |
| `Dec` | `decrypt(sk, ciphertext) -> Message` |

`spec::graphs` and `spec::lean::export` reject a circuit deeper than the scheme's depth
(`BindingError::Depth`). The type of a ciphertext may depend on the gates that returned it, such
as the level of BGV; the ciphertext value carries it, so no algorithm takes it as an argument. The trait holds only the
syntax: what a scheme claims is a `spec::Statement` (the kind of correctness, the allowed
assumptions with their parameters, and a cost bound), and the simulators of a security proof are
`spec::Reduction`s, both passed to `spec::lean::export`.

`spec::lean::export` takes a scheme, one circuit, a statement, and one reduction per assumption,
builds the graphs of `KeyGen`, `Enc`, `Eval` of the circuit on fresh ciphertexts, `Dec` of each
ciphertext type, `plain_eval` of the circuit, and the simulators, exports them with their sampling
plans (`mxx_ir_core::lean::graph_modules`), and writes `Scheme.lean`, which assembles the
algorithms into an `MxxFhe.Scheme` and the circuit's `MxxFhe.Evaluation`, and states

- `CorrectnessClaim := MxxFhe.WorstCaseCorrect scheme evaluation`: on valid messages, every
  execution decrypts the circuit's evaluation to its plaintext value; or, for
  `Correctness::AverageCase`, `MxxFhe.AverageCaseCorrect scheme evaluation ε`, where decryption
  fails with probability at most `ε = 2^-k`;
- `BoundedSecurityClaim := ∀ backend, MxxFhe.SecureWithin backend scheme codec assumptions bound
  0`: for every adversary given the public and evaluation keys, the IND-CPA advantage is at most
  twice the sum of the advantages of reductions against the allowed assumptions
  (`mxx_ir_core::security::AssumptionId`), for reductions whose simulators are some IR programs
  (`MxxSecurity.IR.Program`) of cost at most `Statement::cost_bound`, so the statement itself
  bounds the work of each reduction. The programs run with `backend` (its gadget layouts and
  default preimage widths) and with any hash function, chosen after the simulators. `codec`
  converts between the scheme's values and IR values by type.

The simulators are not part of a statement. Each reduction has two simulator graphs: `setup`
reads the assumption's challenge and writes the keys, and `challenge` encrypts the chosen
message. The proof modules hold them: `Reductions.lean` as computations, the fixed template
`MxxFhe.reduce` running the adversary once around them, with `SimulatedSecurity :=
MxxFhe.Secure scheme uses 0`, IND-CPA with these reductions; and `Simulators.lean` as IR programs,
the evidence for `BoundedSecurityClaim`.

`bgv::scheme::BgvScheme` implements `spec::Fhe` for BGV, with slot-wise multiplication as its
gate and depth `Bounded` by the number of levels but one. A ciphertext carries its level, and the
gate, on arguments of levels `i` and `j`, switches the higher one down to level `max(i, j)`, then
multiplies, relinearizes, and switches down `modswitch_steps` CRT levels. `BgvScheme::security` gives its statement and three reductions, in the order of the
game hops:
circular key-switching security of the relinearization key, then decision RLWE for the public
key and for the ciphertext. Every gate relinearizes with the one key of the top level:
`BgvParams::relinearize` reduces a key of a higher level to the ciphertext's level and scales the
value so that the key's gadget factors recombine its digits, with the noise of a key generated at
that level. The evaluation key the adversary sees is therefore the same at every depth.

`lean/bgv` holds the generated statements (`generated/`, written by
`cargo run -p mxx-fhe --example export_claims -- <dir>` into `<dir>/bgv`, for the circuit
`x₀ · (x₁ · x₂)`) and their proofs. `Certificate.lean` proves `CorrectnessClaim` (worst
case, at the fixture parameters) and `BoundedSecurityClaim` (from circular security of the
relinearization key and decision RLWE, with the simulators of `Simulators.lean` as IR programs of
cost at most `2^30`); its `#print axioms` lists only `propext`, `Classical.choice`, and
`Quot.sound`.

| Module | Proves |
| --- | --- |
| `BgvForms`, `BgvEncryptForms`, `BgvSetupForms` | closed forms of the generated encryption and simulator graphs |
| `BgvFirstHop` | closed form of key generation; the real world equals the circular assumption's real world |
| `BgvHops` | the hops from the circular assumption to the two `DecisionRlwe` uses |
| `BgvSecurity` | the last hop and `SimulatedSecurity` |
| `BgvBounded` | the cost of the simulator programs against the bound of `BoundedSecurityClaim` |
| `BgvNtt`, `BgvPrograms` | the IR slot encoding (the inverse NTT of `MxxSecurity.IR.Eval`) is the runtime's encoding, and the encryption program of the key simulators, run as an IR program, is BGV encryption |
| `BgvReductions` | the setup programs and the ciphertext simulator's encryption program; each reduction run through `MxxFhe.reduceProgram` equals the generated one; `BoundedSecurityClaim` |
| `BgvGate0`, `BgvGate1` | the graphs of one multiplication at level 0 and at level 1, which the proof reads the circuit's graph as (not part of any statement) |
| `BgvCorrectness` | key generation, fresh encryption, decryption at level 0, the circuit's plaintext value, and the gate on two level-0 arguments |
| `BgvDepth` | the gate on level-1 arguments with the reduced top-level key, the switch from level 0 to level 1, the phase invariant of every level, and `CorrectnessClaim` for the circuit |

## TFHE

`tfhe::scheme::TfheScheme` implements the same `spec::Fhe` for TFHE, with NAND on bits as its
gate and `Depth::Unbounded`, since every gate bootstraps. `KeyGen` samples the binary LWE secret `s` and ring secret `z`, the bootstrapping key
(ring-GSW encryptions of `s` under `z`), the key-switch key (LWE encryptions of `z` under `s`, with
masks drawn as uniform residues of a ring of about `2^162` reduced modulo `q`), and the public
key, the ring-LWE sample `(a, a z + e)`. `Enc` encrypts a bit as the ring-LWE ciphertext
`(u a + e₁, u b + e₂ + μ)` of the constant `μ = ±⌊Q/8⌋` with a binary `u`, then sample-extracts
and key-switches it to an LWE ciphertext under `s`, as the end of a bootstrap does. Every gate
bootstraps, so a circuit has no depth limit.

Because `Enc` key-switches, both NAND inputs carry key-switching noise. The default parameters
(`utils::tfhe_params`) therefore lower the LWE noise to sigma `2^15` and raise the LWE dimension to
720, which the lattice estimator puts at about `2^95` operations, above the original `n = 630`,
sigma `2^17` profile. With them, the proven bound on the input noise, which takes every
key-switch digit at its worst value, is about `2^-135`, below the claimed `2^-100`; at sigma `2^17`
the same bound is only about `2^-26`.

`TfheScheme::statement` claims average-case correctness of one NAND gate on fresh ciphertexts and
no security statement yet. `lean/tfhe` holds its generated statement (`<dir>/tfhe` of
`export_claims`) and its proof. `Certificate.lean` proves `CorrectnessClaim`: on valid messages,
decryption of the NAND of two fresh encryptions fails with probability at most `2^-100`; its
`#print axioms` lists only `propext`, `Classical.choice`, and `Quot.sound`.

The proof reads one run as a function of a sampling tape. A run fails only if the input noise
(the key-switching errors weighted by the two encryptions' digits, plus the rounding errors of the
NAND input mask to `2 N`) or the output noise (the bootstrapping-key errors weighted by the
blind-rotation digits, plus the output key-switching errors) is large. The rounding errors are
bounded on average: each mask entry is a uniform key-switching mask entry times a nonzero digit sum
plus terms independent of it, so its rounding error has mean within `2` of zero. The output noise
is bounded for each of the `2048` initial rotations of the lookup table, since the rotation reads
the key-switching errors through the fresh ciphertexts' bodies.

| Module | Proves |
| --- | --- |
| `TfheBase`, `TfheBridge` | the sampling sites of one run; the failure probability is the tape measure of `{ω \| failR ω}` (via `MxxSecurity.Tape`) |
| `TfheStages`, `TfheEncrypt`, `TfheKeygen`, `TfheEval`, `TfheRun` | closed forms of decryption, encryption, key generation and the bootstrapped NAND gate, and of one run |
| `TfhePhase`, `TfheRotation`, `TfheFresh` | the phase of an extracted and key-switched ciphertext, the phase of the blind rotation, and the phase of a fresh encryption |
| `TfheNand` | the lookup-table half the NAND input selects, and `run_correct`: small input and output noise give a correct run |
| `TfheMS`, `TfheInput` | the moment of the rounding errors over the near-uniform key-switching masks, column by column, and the tail of the input noise |
| `TfheBrNoise`, `TfheOutput` | the blind-rotation error as a linear form whose coefficients read only earlier errors, and the tail of the output noise |
| `TfheCorrectness` | the union over the initial rotations, the numerical bound, and `CorrectnessClaim` |

## Regenerating and checking

Regenerate the statements without a GPU, then check the proofs inside each package directory:

```bash
cargo run -p mxx-fhe --example export_claims -- <dir>
```

```bash
cd crates/fhe/lean/bgv && lake build
```

`scripts/check_lean_claims.py` regenerates every statement into a temporary directory and compares
it byte for byte with the committed `generated/` directories. The GPU tests run the same graphs
(integration tests require explicit authorization):

```bash
MXX_GPU_BACKEND=cuda cargo test -r -p mxx-fhe --features gpu --test gpu_tfhe --test gpu_bgv
```

## GPU backend scope

`MXX_GPU_BACKEND=cuda|hip` is selected by `mxx-backends` and propagated through Cargo metadata;
FHE does not choose a separate native compiler. BGV uses the common backend on either build.
The TFHE blind-rotation kernel remains CUDA-only: HIP builds skip its native compilation and
`TfheParams::gpu_blind_rotation_kernel` returns `None`. AMD TFHE execution is deferred and is
not a supported or validated path in this change; CPU TFHE APIs remain available.

HIP workspace/FHE compilation passed for gfx1100 and gfx942 with ROCm 7.0.0; AMD BGV device
execution remains unverified. Five CUDA FHE device cases passed in each of three smoke repetitions; the whole device
gate still has a pending long KHE probe after the compiled planner/lowering correction. See [`AMD_GPU_VALIDATION.md`](../../AMD_GPU_VALIDATION.md). The external PhantomFHE comparison tools
under `crates/fhe/scripts/` target CUDA; they do not validate HIP or establish cross-vendor
performance equivalence. See the [backend support evidence](../backends/README.md#gpu-backend-selection).

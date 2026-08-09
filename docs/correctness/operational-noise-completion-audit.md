# Operational Noise Checker Completion Audit

## Scope

This audit records the completion evidence for the execution-aligned operational noise checker.
The completed milestone is the graph-derived deterministic hard-bound calculation used for
parameter search. The later theorem connecting every operational fact to runtime matrix semantics,
and the final Witness Encryption correctness theorem, are intentionally not claimed here.

The normative behavior is documented in:

- `docs/correctness/operational-protocol-inventory.md`;
- `docs/correctness/operational-affine-reuse-audit.md`;
- `docs/correctness/gadget-decomposition-semantics.md`;
- `docs/universal-ir-noise-checker-plan.md`.

## Requirement evidence

| Requirement | Authoritative implementation or evidence | Result |
|---|---|---|
| One flat integer-coefficient sum of ordered products | `lean/Mxx/Certificate/OperationalBounds.lean`: `OperationalFactorKey`, `OperationalProductKey`, `OperationalTerm`, and normalization fixtures | Complete |
| Any positive number of Large factors remains signal | `operationalLargeFactorCount` and the multi-Large fixture in `OperationalBounds.lean` | Complete |
| No operational `opaqueFinite` or `notSmall` fallback | Active-source search over `OperationalBounds.lean` and `MxxWe.DiamondChecker` | Complete |
| Exact merge/cancellation before relation rewrite and bounded compression | `normalizeOperationalTerms`, relation rewrite, bounded-run compression, and focused cancellation/relation fixtures | Complete |
| Deterministic worst-case product arithmetic | `OperationalBoundExpr.matrixProduct`, metadata transfer, and matrix arithmetic fixtures | Complete |
| No CLT or dependency-disjointness heuristic | Active operational source and reuse-audit inspection | Complete |
| Operand-owned preimage and decomposition relations | relation snapshots and checked adjacency rewrite in `OperationalBounds.lean` | Complete |
| One shared source `B` for the two Boolean preimages | `sharedPreimageBaseScope` fixture proves identical public identity and distinct relation owners/targets | Complete |
| Selection uses an exact-one branch maximum | dynamic selection transfer and branch-maximum fixtures | Complete |
| Every Rust and Lean IR primitive has an explicit transfer or normal rejection | exhaustive Rust emission, exhaustive `operationalTransferClass`, nested-enum classifiers, and the inventory test | Complete |
| Every enabled full protocol reaches a generic report without `UnsupportedNode` | generated-Toy operational fixtures in `MxxCorrectness.OperationalToy` and the Diamond GPU integration path | Complete |
| Family, subgraph, parallel-loop, and sequential-loop handling | structural evaluator and positive/negative nested-scope fixtures | Complete |
| Sequential analysis has fixed-size numeric state | `OperationalRecurrenceTransition`, simultaneous transition evaluation, recurrence fixtures, and no symbolic loop unrolling in the active import graph | Complete |
| Generic decoder obligation | `decoderNoiseCheckReport` checks `2 * plaintext_modulus * noise_bound < ciphertext_modulus` with exact integers | Complete |
| Diamond search uses only the generic Lean report | `crates/we/lean/MxxWe/DiamondChecker.lean` and `crates/we/src/diamond/parameter_search.rs` | Complete |
| Rust does not mirror the bound formula | Rust source inspection finds only checker invocation, report parsing, and candidate orchestration | Complete |
| Active checker excludes the retired analyzer and expression arena | active-import search over `Mxx.Certificate`, `MxxCorrectness`, and `MxxWe.DiamondChecker` | Complete |
| Active checker has no proof holes or new axioms | active-source `sorry`/`admit`/axiom scan | Complete |
| Superseded proof work cannot enter the active build accidentally | `OperationalSemantics.lean` and `ToyOperationalAlignment.lean` are explicitly commented out; both elaborate as empty deferred modules | Complete |
| Checked-in generated modules are fresh | both owner emitters pass without `MXX_REGENERATE_CORRECTNESS`; workflow, derivation, and toolkit hashes match | Complete |
| Operational completion is distinguished from final correctness | this audit and `docs/universal-ir-noise-checker-plan.md` | Complete |

## Validation commands

The final local validation used the following commands:

```text
cargo +nightly fmt --all -- --check
cargo test -p mxx-correctness -p mxx-bgg -p mxx-gadgets -p mxx-we --lib
cd lean && lake build Mxx MxxCorrectness MxxWe \
  mxx_diamond_checker mxx_diamond_derivation_checker
git diff --check
```

Results:

- `mxx-correctness`: 25 passed;
- `mxx-bgg`: 42 passed;
- `mxx-gadgets`: 89 passed and 3 intentionally ignored long-running tests;
- `mxx-we`: 11 passed and 1 environment-dependent estimator test ignored;
- all listed active Lean targets built successfully;
- formatting, diff whitespace, active import, and active trust scans passed.

The explicitly requested GPU integration command also passed:

```text
cargo test -p mxx-we --features gpu --test test_gpu_diamond_we -- \
  --ignored --exact test_gpu_diamond_we_parameter_search_estimate_and_round_trip --nocapture
```

It completed parameter search, Lean checking, GPU cost estimation, encryption, decryption, and a
real message round trip in one execution. The selected smoke-test candidate had ring dimension 32,
Boolean-circuit depth 2, a 120-bit ciphertext modulus, and an estimated lattice security level of
11 bits. The generic noise bound was `96689699662452587413635072`; the accepted Lean candidate
check took about 2.77 seconds and the complete integration test took about 42.22 seconds.

## Deferred proof milestone

Proof-oriented source retained for later work is not imported by the active operational checker.
The superseded `OperationalSemantics.lean` and `ToyOperationalAlignment.lean` implementations are
kept inside explicit outer comments so they neither compile stale proof declarations nor silently
reuse old `.olean` artifacts. In particular, passing the operational report establishes that the
implemented deterministic calculation accepts the candidate; it does not yet prove that every
calculated fact denotes the runtime value or that the Witness Encryption round trip succeeds for
every input satisfying the protocol preconditions. Those statements require the separate
local-soundness and end-to-end Lean proof milestone and must reuse these executable definitions
rather than introduce another analyzer.

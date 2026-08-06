# M0 Correctness IR Audit

## Scope and reproduction

This report records the canonical emitter-v5 surface. It covers the complete programs contained
in:

- `crates/correctness/lean/MxxCorrectness/Generated/ToyExample/Ir.lean`
- `crates/we/lean/MxxWe/Generated/DiamondWeFamily/Ir.lean`

The Diamond input injector is not a third generated protocol. It is embedded in
the Diamond decrypt stage. Its outer operations are the initial-state expansion
at root node 2 and the state recurrence at root node 8; the recurrence body is
scope `sequential:__root:8` and its nested parallel scopes.

Regenerate the inputs and repeat the structural audit with:

```sh
MXX_REGENERATE_CORRECTNESS=1 cargo run -p mxx-correctness --example emit_correctness
MXX_REGENERATE_CORRECTNESS=1 cargo run -p mxx-we --example emit_correctness
(cd lean && lake build mxx_analysis_facts)
lean/.lake/build/bin/mxx_analysis_facts /tmp/mxx-diamond-analysis-facts-20260806.json
python3 scripts/audit_correctness_ir.py --check
```

`mxx_analysis_facts` calls the Lean analyzer directly and serializes its `AnalysisResult`; it does
not mirror inference in Rust or Python. Analyzer failure is a nonzero exit with the first
`VerifyError`, and no replacement JSON is written. The audit rejects missing, malformed, or empty
fact tables.

Both generated modules identify `mxx-correctness-emitter-v5`. Their closed-workflow hashes are:

| Closed workflow | Workflow hash |
|---|---|
| Toy example | `1cffc358ce796800fad2b9513f767c5df8e2eaf8e62bca1329a8b0012874b0df` |
| Diamond WE family, including the input injector | `ce6395d760eba47744a2e1ee2e692af2a316a95e60d9943589c3d0cef891de27` |

The audit does not trust copied source or toolkit hashes. It reads each generated
`protocolSourcePaths`, recomputes `protocolSourceHash` over that exact source set, and recomputes
`toolkitHash` over `lean/Mxx/**/*.lean`. The canonical Diamond source set contains the Cargo
manifests and complete `src` trees of `bgg`, `correctness`, `dsl`, `gadgets`, `ir-core`, and `we`,
plus the WE emitter source. Including `correctness` ensures that a change to the Rust Lean emitter
invalidates Diamond's generated module. The toy source set contains the corresponding
`correctness`, `dsl`, and `ir-core` inputs. A missing, added, reordered, or stale source entry fails
the audit.

## NodeKind inventory

The following table is the union of workflow stages, loop bodies, requirement
programs, ideal programs, and the toy comparator. Counts are useful for detecting
an accidental shrink or expansion of the audited surface; the allowlist itself
is a set, not a count-based policy.

| Emitted kind | Toy | Diamond | Total |
|---|---:|---:|---:|
| `bitExtract` | 0 | 1 | 1 |
| `boolToInt` | 1 | 152 | 153 |
| `concat` | 0 | 9 | 9 |
| `constantBool` | 0 | 16 | 16 |
| `constantInt` | 0 | 175 | 175 |
| `constantMatrix` | 1 | 3 | 4 |
| `dimension` | 1 | 1 | 2 |
| `evaluateInt` | 0 | 82 | 82 |
| `extractCoefficient` | 0 | 1 | 1 |
| `familyGetDynamic` | 0 | 56 | 56 |
| `familyGetStatic` | 0 | 6 | 6 |
| `gadgetDecompose` | 0 | 3 | 3 |
| `gadgetMatrix` | 0 | 1 | 1 |
| `gaussianSample` | 1 | 2 | 3 |
| `hashSample` | 0 | 3 | 3 |
| `identityMatrix` | 0 | 5 | 5 |
| `input` | 4 | 341 | 345 |
| `intBinary` | 0 | 162 | 162 |
| `intCompare` | 0 | 134 | 134 |
| `matrixAdd` | 1 | 9 | 10 |
| `matrixMultiply` | 0 | 19 | 19 |
| `matrixNegate` | 0 | 1 | 1 |
| `matrixScale` | 0 | 8 | 8 |
| `matrixSubtract` | 0 | 20 | 20 |
| `parallelLoop` | 0 | 131 | 131 |
| `preimageSample` | 0 | 5 | 5 |
| `reshape` | 0 | 1 | 1 |
| `select` | 1 | 38 | 39 |
| `sequentialLoop` | 0 | 10 | 10 |
| `slice` | 0 | 3 | 3 |
| `thresholdDecodeBool` | 1 | 0 | 1 |
| `trapdoorSample` | 0 | 1 | 1 |
| `uniformSample` | 0 | 2 | 2 |
| `zeroMatrix` | 1 | 7 | 8 |

`concat` uses all three axes: five row concatenations, two column
concatenations, and two diagonal concatenations. Consequently the diagonal
embedding rule is part of the observed initial subset and cannot remain a
disabled, speculative constructor.

No `TrapdoorPublic`, `FamilyPack`, or `SubgraphCall` node occurs. No node outside
the M0 allowlist occurs.

## Special transform audit

All eight `MatrixScale` nodes use the literal scalar one. They are identity
materializations, not arithmetic scaling.

There is exactly one `Reshape`:

| Stage/scope/node | Input chain | Output shape |
|---|---|---|
| `encrypt/__root/42` | node 40 `gadgetDecompose` -> node 41 identity `matrixScale` -> node 42 `reshape` | `diamond_digit_count x 1` |

The reshaped value is therefore decomposition materialization, not a value with
an independently introduced signal carrier. The new analyzer must still enforce
this from its input fact; the node inventory alone is not a sound replacement
for the `reshapeAffine` precondition.

There are 19 multiply sites. `scripts/audit_correctness_ir.py` prints every site,
both operand node/port references, the immediate operand kind, the input name
for formal inputs, and the strongest classification justified by executable
syntax alone. The structurally important root sites are:

| Stage/scope/node | Operands | Intended initial rule shape |
|---|---|---|
| `encrypt/__root/43` | public-key difference x reshaped decomposition | whole-expression bounded product |
| `encrypt/__root/57` | message/secret selector x public-key element | `L * X` |
| `decrypt/__root/11` | final input-injection state x decoder preimage | `A * R` |
| `decrypt/__root/13` | final input-injection state x K preimage | `A * R` |
| `decrypt/__root/15` | final input-injection state x one preimage | `A * R` |
| `decrypt/__root/74` | one-minus-circuit encoding x R decomposition | `A * R` |

The remaining 13 sites are inside parallel or sequential bodies and receive
typed body inputs. The emitter-v5 IR preserves operand order but does not carry
the input fact, family substitution, recurrence result, or affine provenance for
those placeholders. Even the six root sites depend on facts inherited through
families or artifacts. Consequently all 19 sites currently print `UNRESOLVED`.

This is an M0 failure, not an informational warning. `--check` exits nonzero until the Lean
analyzer exports `target/correctness/m0-analysis-facts.json`. The audit accepts only schema
`mxx-analysis-facts-v1`, requires its Diamond workflow hash to equal the generated workflow hash,
and reads each operand's normalized primary form and `ValueInstanceRef` from `wireFacts`. It never
infers semantic facts from node kinds. The analyzer output must supply, for each operand, its
normalized primary form
(`exact`, whole-expression `bounded`, or `affine`) after artifact inheritance and
loop-body substitution. Only then can the audit classify the site as `X*X`,
`L*X`, `A*R`, `L*A`, or `L*R`, and reject `X*A` and general `A*A'` mechanically.
The intended-rule column above is explanatory and is never used as an audit
input.

## Loop-family bound uniformity

The generated Diamond workflow contains no bound expression indexed by a loop
slot. Every repeated sampler is represented once in a loop-body template and
uses one of these lane- and iteration-invariant expressions:

| Sampler family | Occurrences | Bound or range expression |
|---|---:|---|
| Trapdoor | 1 template | `diamond_preimage_max_coefficient_bound` |
| Preimage | 5 sites/templates | `diamond_preimage_max_coefficient_bound` |
| Gaussian error | 2 sites/templates | `diamond_error_max_coefficient_bound` |
| Uniform selector | 2 sites/templates | closed interval `[-1, 1]` |

The gadget decomposition sites likewise share
`diamond_gadget_base` and `diamond_digit_count`. This satisfies the initial
iteration/lane-uniform bound restriction. Any future `.loopIndex` occurrence in
a sampler bound must fail the semantic M0 check as `UnsupportedIndexedBound`;
the current text-level audit is intentionally not treated as the long-term
semantic verifier.

Five sequential-loop bodies contain parallel-family construction. The complete generated-scope
classification is:

| Program and sequential scope | Parallel nodes |
|---|---:|
| Encrypt `sequential:__root:32` | 7 |
| Decrypt `sequential:__root:8` | 5 |
| Decrypt `sequential:__root:67` | 41 |
| Requirement 1 `sequential:__root:79` | 6 |
| Requirement 2 `sequential:__root:27` | 7 |

These 66 sites all use the same uniform-template rule. A concrete aggregate retains its
sequential-iteration path as value identity, while family-element schema and hard-bound metadata
are selected by the frozen joint-family identity and output slot. Repeated analyzer entries for
that same frozen key resolve deterministically to the analyzer-owned template; no certificate
chooses the template.

## Artifact and residual origin paths

The current workflow preserves direct artifact wiring for the decoder inputs:

| Artifact | Producer wire | Consumer wire |
|---|---|---|
| `diamond_decoder_preimage` | `encrypt/__root/48:0` | `decrypt/__root/10:0` |
| `diamond_k_preimage` | `encrypt/__root/63:0` | `decrypt/__root/12:0` |
| `diamond_one_preimage` | `encrypt/__root/69:0` | `decrypt/__root/14:0` |
| `diamond_r_decomposed` | `encrypt/__root/42:0` | `decrypt/__root/73:0` |

The residual is assembled at decrypt root nodes 71 through 76:

1. node 71 selects the circuit vector from the Boolean recurrence;
2. node 72 computes one-vector minus circuit-vector;
3. node 74 multiplies that difference by the imported R decomposition;
4. node 75 adds the imported K-preimage product;
5. node 76 subtracts that sum from the decoder-preimage product.

The script checks these four required artifact names, then derives their producer and consumer
wires from generated workflow artifact bindings, root outputs, and consumer input nodes. It
confirms that these imported matrices are not recomputed in decrypt.

Generated executable IR does not serialize or self-report analyzer conclusions. The separate Lean
exporter serializes these values directly from `AnalysisResult`:

- `ValueInstanceRef`, including the common protocol/artifact origin;
- `JointFamilyId` and output slot plus normalized family index;
- `SequentialRecurrenceInstanceRef` for a sequential carried value;
- `MatrixFactPath` for each residual coefficient and basis;
- `BoundFactPath` for the associated coefficient/noise/total bounds.

It also exports an analyzer-owned `semanticAnchors` record attached to the existing
`diamond.decoder.residual` semantic anchor. For each normalized term on both subtraction operands
it contains the coefficient `MatrixExpr`, basis `MatrixExpr`, their complete `ValueInstanceRef`
paths, and any recurrence/family projection. The audit compares these typed values for exact
equality. A human-readable construction trace or a claim that separately computed matrices have
equal values is insufficient.

There is not yet a successful Diamond analyzer result to serialize. The latest completed Phase A
run rejected a parallel-family aggregate under encrypt `sequential:__root:32` because the concrete
aggregate carried a sequential-iteration instance path while the uniform element template was
registered at its frozen joint/output key. The complete 66-site class is recorded above and the
analyzer lookup has been updated coherently; a post-fix exporter run is still required before this
report can claim Phase A acceptance.

## M0 disposition: failed closed

The syntax-only portion passes:

- the observed NodeKind union is within the authorized ceiling;
- `TrapdoorPublic`, `FamilyPack`, and `SubgraphCall` are absent;
- all matrix scales are identity materializations;
- the only reshape is the bounded gadget-decomposition path;
- loop sampler bounds are iteration/lane uniform.

The complete M0 audit fails because:

- all 19 multiply sites lack analyzer-emitted, substituted affine facts, so the
  forbidden `X*A` and `A*A'` cases cannot yet be excluded;
- complete residual cancellation equality lacks typed
  artifact/family/recurrence/result-path identities, despite the four direct
  artifact edges being present;
- the latest completed analyzer run stopped before the exporter could emit the semantic-anchor
  facts consumed by the exact residual-term comparison; the post-fix run is still pending.

The old `trace_*`, `*ConstructionTrace`, and manual certificate APIs have been deleted; repository
search returns no remaining occurrence. The audit command deliberately returns a failure status
now. M1 must not be declared accepted until analyzer output closes the semantic failures.

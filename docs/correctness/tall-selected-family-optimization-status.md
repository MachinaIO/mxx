# Tall Selected-Family Operational Checker Status

## Status

This report records an auditable, incomplete checkpoint of the Lean operational checker work for
the Tall BGG+ nested-RNS parameter simulation. The implementation checkpoint is commit `3ba662a5`
(`Optimize Tall selected-family checking`). It does not claim that the Tall integration test passes
or that the operational checker has been demonstrated to finish within 30 minutes.

The executable Rust/CUDA protocol graph is unchanged by this checkpoint. The changes are confined
to Lean Graph IR decoding, operational-analysis facts, selection handling, relation transport, and
the generated correctness hash.

## Audit Map

The implementation under review is concentrated in these files:

- `lean/Mxx/Certificate/OperationalBounds.lean`: `SelectedMatrixSummary`,
  `SelectedMatrixFamily`, packed-family summaries, relation-aware identity matching, and the
  independent-selection evaluation paths;
- `lean/Mxx/Ir/BinaryFormat.lean`: array and hexadecimal binary decoding without intermediate
  `List.range` allocations;
- `crates/correctness/lean/MxxCorrectness/Generated/ToyExample/Ir.lean`: regenerated source hash
  for the changed shared Lean modules.

Commit `3ba662a5` contains the complete code diff for this checkpoint. This report is intentionally
kept in a separate documentation commit so reviewers can inspect the implementation without mixing
the code diff with the status narrative.

## Measured Workload

The diagnostic Tall candidate uses deliberately small parameters so that the remaining cost is the
checker rather than cryptographic runtime:

- ring dimension: 8;
- CRT depth: 1;
- CRT modulus bits: 10;
- gadget base bits: 5;
- scale: 1024;
- multiplication count: 1;
- parameter-simulation parallelism: 1;
- required security bits: 0.

The graph contains 31 lookup tables, 30,720 lookup preimages, and 48 slot-operation preimages, for
30,768 preprocessing preimages in total. It reports 184 artifacts and these top-level gate counts:

| Gate | Count |
| --- | ---: |
| SubCircuitOutput | 21 |
| Input | 21 |
| LargeScalarMul | 12 |
| Add | 11 |
| Sub | 3 |
| SlotTransfer | 1 |

The emitted source sizes are stable across the measured runs:

| Source | Bytes |
| --- | ---: |
| Operational IR | 38,804,763 |
| Proof IR | 30,057,136 |
| Derivation IR | 29,976,992 |

## Implemented Optimizations

### Array-based binary decoding

`Mxx.Ir.BinaryFormat.readArray` and hexadecimal decoding now build arrays with range iteration
instead of constructing `List.range` values. This removes millions of intermediate list cells when
loading the approximately 38.8 MB operational source.

The generated module cache is prepared in approximately 6.1 to 6.8 seconds when cold and 0.84 to
1.00 seconds when reused. Runtime decoding of the prepared operational workflow is still about
37.6 to 38.3 seconds, but it now completes reliably; before this change it did not reach evaluation
within the earlier observation window.

### Cached selected-family summaries

Each selected matrix family stores one `SelectedMatrixSummary` containing:

- an optional complete uniform schema;
- whether every branch is relation-free;
- a shared last public-matrix identity template;
- a shared first relation public-matrix identity template.

Creating the summary examines all materialized branches once. Later checks for the GGH-style
public-matrix/preimage boundary use only the cached summary and are constant-time with respect to
the logical branch count. Nonuniform families retain exact branches and do not use the envelope
fast path.

### Logical-count envelopes

A `SelectedMatrixFamily` separates its logical branch count from its stored representative array.
Uniform families can therefore retain one representative while recording thousands of logical
alternatives. The summary proves when this is allowed; a nonuniform family cannot be silently
treated as an envelope.

Packed families likewise carry a cached matrix summary. Parallel-loop zip and zip-offset inputs
can create a selected envelope without slicing or copying every packed element. Loop bodies are
evaluated once, and `representsLoopLanes` distinguishes loop-lane alternatives from an ordinary
nested protocol selection.

### Relation-aware template matching

Public identities from different packed families may have different concrete branch binders while
still being selected by the same executable index. Template comparison ignores only the concrete
lane index and preserves the loop slot, selection identity, source identity, gadget parameters, and
matrix parameters. This permits the intended public-key/preimage rewrite without equating unrelated
selections.

Dynamic selection of a uniform relation-bearing preimage family retains a representative relation
under the exact executable selection identity. Preimage sampling for a uniform selected target also
samples one operational representative instead of constructing one Lean fact per logical target.

### Exact handling of independent selections

Two selected operands are still zipped only when their selection identities agree. Different
identities are never positionally aligned.

For matrix multiplication, a new independent selection can be absorbed into every branch of an
existing selected family only after all alternatives for that existing branch have the same complete
operational schema. The absorbed selection is recorded structurally in matrix and relation
identities, while the older nonuniform branch family remains explicit.

For addition and subtraction, a relation-aware fast path reuses an already absorbed selection when
the exact binder is recoverable. Otherwise each independent selected operand is materialized
separately with the existing `selectOperationalPolynomials` rule and the resulting ordinary
polynomials are added or subtracted. This preserves exact-one signal indicators and computes the
maximum of complete branch-noise bounds. It avoids constructing the Cartesian product of the two
selected families.

### Exact nonuniform fallback

An exact packed family with no uniform summary still uses all concrete elements. Relation-bearing
dynamic selection retains exact branch-local facts; relation-free selection uses the existing
exact-one join. Missing summaries no longer cause a nonuniform family to be interpreted as a
uniform representative.

## Current Problem

The checker has progressed from source preparation failures and early 30,720-way SOP expansion to
a small, specific nested-selection boundary in the encoding stage.

Before the independent-selection changes, evaluation consistently reached the parallel-loop body
at root node 376 in about 426 to 433 seconds:

- node 17 multiplies a value selected by an earlier loop with a preimage selected by the current
  loop;
- node 18 adds that result to a term selected only by the current loop.

At node 17, the four explicit combinations each had 88 symbolic terms, 219 retained relation
annotations, and the same hard bound of 504. The two current-preimage alternatives had identical
schemas for each fixed earlier branch, which justified absorbing the current selection inside each
earlier branch. The two earlier branches were not schema-identical and therefore could not be
collapsed.

At node 18, the two operands depended on different selection dimensions. An experimental exact
Cartesian representation passed node 18 but caused the checker to run beyond 30 minutes, so that
approach was rejected and removed. The checkpoint instead materializes the two selections
independently using exact-one signal terms and branch-wise noise maxima. The final Tall run of this
new path was interrupted before completion, so its runtime and next failure boundary are not yet
known.

The remaining risk is therefore not the 30,720-branch LUT family itself. That family is represented
by a uniform envelope. The risk is growth of the ordinary signal polynomial after several small,
independent selections are materialized. The next run must measure term counts and relation-rewrite
cost after node 18 and prove that they stay below the 30-minute target.

## Rejected Approaches

### Eager SOP for large packed LUTs

Expanding every packed LUT branch before relation rewriting creates a very large polynomial and is
not viable. Large uniform families must remain envelopes until their branch-specific relation has
been consumed.

### Positional zip of different selections

Equal branch counts do not imply correlated choices. Zipping different selection identities would
silently prove the wrong relation and remains rejected.

### Persistent Cartesian selected families

Keeping four explicit branches at node 18 preserved semantics but exceeded the 30-minute target.
It also permits exponential growth as additional independent selection dimensions are introduced.
The code no longer uses this fallback.

### Unconditional representative selection

Equal hard bounds alone are insufficient. Branches may have different relation schemas,
dependencies, or symbolic signal terms. Representative collapse is allowed only with a complete
uniform-schema proof, or when the existing exact-one materialization rule explicitly preserves the
signal alternatives and takes a complete-branch noise maximum.

## Remaining Work

1. Run the checkpointed independent-materialization path to completion or its next explicit
   fail-closed boundary, with a 30-minute wall-clock limit.
2. Record signal-term counts, bounded-summary counts, and relation-rewrite time after node 18.
3. If ordinary signal terms still grow too quickly, add analysis-only sharing for independently
   selected signal sums. Do not add a semantic Graph IR node and do not restore `fold`.
4. Add focused Lean fixtures for:
   - cached summary construction versus an explicitly enumerated family;
   - uniform envelope logical counts;
   - GGH public-matrix/preimage template matching;
   - independent multiply selection absorption;
   - independent add/sub materialization without a Cartesian family;
   - nonuniform fail-closed behavior.
5. Remove any optimization that cannot be justified by those fixtures or by the exact operational
   relation semantics.
6. Re-run the Tall parameter-simulation integration test and require a successful checker result in
   less than 30 minutes before declaring this work complete.

## Validation at the Checkpoint

The following validations passed for commit `3ba662a5`:

```text
lake build Mxx.Certificate.OperationalBounds Mxx.Ir.BinaryFormat
cargo test -p mxx-correctness --lib operational_runner::tests
cargo test -p mxx-correctness --lib \
  operational_runner::tests::runs_generated_toy_workflow -- \
  --ignored --exact --nocapture
cargo +nightly fmt --all
git diff --check
```

The first correctness command passed three tests and left the Lean-compiler test ignored. The
second command ran that ignored test explicitly and passed it. The generated ToyExample module was
regenerated through the repository emitter after formatting.

## Audit Boundary

This is intentionally an intermediate checkpoint. It proves that the modified Lean modules compile,
the generated correctness module is current, and the small operational runner works. It does not
prove that Tall parameter selection succeeds, that the checker meets the 30-minute target, or that
the final noise bound agrees with runtime noise. Those claims remain open until the Tall integration
test completes successfully.

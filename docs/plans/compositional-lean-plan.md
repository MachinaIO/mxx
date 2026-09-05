# Implementation plan: compositional Lean correctness

## 1. Scope and starting point

Implement [the specification](../correctness/compositional-lean-spec.md) in bounded stages after
design approval. This task itself adds documentation only. It does not implement the stages below.

The source baseline is `codex/new-IR` at
`d84669f79ad2cd4c72c88970da92043187043244`, pulled on 2026-09-05 into
`codex/compositional-lean-design`. The prior work is preserved at
`codex/app-specific-lean`, `7cb9d089d`; its handoff can be read with:

```sh
git show 7cb9d089d:docs/app-specific-lean-correctness-handoff.md
```

Do not merge the prior branch or PR 147 wholesale. Selective reuse is permitted. Treat the previous
trace implementation as a source of lessons and reusable mathematics, not a list of tasks to finish.

The desired outcome is a theorem generated from the actual Diamond frozen IR, with application-owned
noise mathematics, constant-size family proof structure, and a candidate checker whose work follows
stored syntax and numeric bit size rather than logical lane count.

## 2. Design review conclusions

The former plan begins by repairing `reachedParallelGridLanePrimitiveInsideRootGrid`, then building
additional actual-run wrappers. That is no longer the proposed next step. It would continue investing
in an execution-trace representation that the theorem does not need.

The simpler next step is a small direct relational extraction prototype. It must expose the actual
operands of a sampled preimage and its consuming multiplication directly. If that does not lead to
a short local proof, stop before extending the implementation to Diamond.

Generate ordinary Lean relation definitions from the Rust frozen graph. Pure nodes become `let`
bindings, samplers introduce existential variables constrained by fixed primitive relations,
parallel loops quantify over one index, and sequential loops use one shared `IterRuns` relation.
Do not add a second Lean graph AST, graph interpreter, or intrinsically typed compiler. Rust export
remains an explicit trust boundary, as it was in the old design.

Several earlier requirements are reclassified:

- Full source/target identity remains essential. A separate carrier registry is unnecessary when
  the primitive relation already refers to the actual operand and result variables.
- Source-bearing concat/slice bans were abstract-interpreter limitations. The new exact semantics
  permits ordinary concat/slice and proves whatever local identities the application requires.
- A source may vary across outer groups. A branch-uniform source assumption applies only within the
  scope of the lemma that needs it.
- BGG+ contains both a public mask term and a gadget payload term. Preserve
  `c=s*A-x*(t*G)+e`, and preserve both error summands in a full BGG multiplication.
- A theorem for every family element does not require a proof object for every concrete element.
- A fixed-size proof of a loop does not automatically give a fast numerical bound. Both must be
  addressed and measured.
- The newer Rust checker already preserves structural parallel bodies on important paths. Do not
  attribute universal per-lane expansion to it. The verified issue is the large symbolic machinery
  and iteration-dependent sequential analysis; identify individual complexity claims from source.

## 3. Stage A: smallest viable semantic interface

Owners: the exporter in `crates/ir-core/src/lean` and relation combinators in
`crates/ir-core/lean`, with a tiny primitive domain in `crates/runtime/lean` or a test module.

Implement only enough direct extraction and primitive relations to exercise:

- one input tuple and output tuple;
- SSA sharing with two uses of one value;
- ordinary matrix addition/multiplication;
- a trapdoor sampler with coupled public-matrix and token output ports;
- a preimage sampler with its three ordered arguments: explicit public matrix, trapdoor, target;
- one child call with parameter bindings;
- a structural parallel loop with two output ports; and
- a sequential loop with two simultaneously updated carried values.

Use the current Rust payload and argument semantics. No old `ParallelGrid` adapter is allowed.
Reusable algebra and loop rules must be parametric in dimensions and counts. The initial extracted
graph may fix candidate geometry, while retaining loop indices as symbolic binders. Test at least
two candidate geometries using the same application proof template; do not hard-code a fixture
value into the mathematical lemmas.

Required demonstrations:

1. Eliminating the generated relation gives `B*K=T` for the exact `K` used by both consumers.
2. A graph computing `x-x` shares the same bound input value; separate sampler invocations are not
   incorrectly identified by a static site ID.
3. The child receives the actual parent arguments, with bindings evaluated simultaneously from the
   snapshot after inserting the loop index.
4. A parallel multi-output result keeps related outputs in one lane-local tuple.
5. A sequential body's next state is not mistaken for its current state, and one iteration's sampled
   value cannot be substituted for another without a proved equality.
6. Application proofs do not require `HEq`, a trace search, a caller-provided semantic value, or an
   application-specific structural view.
7. One named scope called with two different compile bindings is exported once with formal
   parameters, and both calls retain their own types and values. A representative environment from
   `validate.rs::collect_scope_bindings` must not replace the original call expressions.
8. The public matrix carried by the trapdoor matches the explicit first preimage argument; a
   mismatched-public fixture is rejected. A public gadget trapdoor uses deterministic decomposition,
   while a sampled secret trapdoor uses the bounded sampler relation. Neither output coupling nor
   this distinction may be postponed until the full primitive inventory.

This stage is an implementation feasibility gate, not a toy correctness acceptance path. Its tests
must inspect the real semantic rules, not redefine a production claim as `True`.

Stop condition: if implementing these examples requires a second Lean AST/compiler, pervasive application
type wrappers, or a large occurrence registry, revise the generic interface before adding operations.
Do not hide the problem behind increasing Lean heartbeats.

## 4. Stage B: faithful export and graph identity

Owner: `crates/ir-core/src/lean`; protocol linking remains in the current workflow-owning layer.

Build one generic exporter over the validated frozen graph. Emit ordinary Lean scope relations from:

- static scopes and child references exactly once;
- node payloads, ordered arguments, every output port, scope input/output wires, and effect roots;
- parameter expressions and lexical loop bindings;
- exact linked producer/consumer references and shared external input mappings; and
- named output references for the application claim.

Those items become ordinary typed binders, applications of primitive semantic functions/relations,
and generated child-relation calls. Output references are tuple projections. A source-map sidecar
records each source scope/node/port. Do not also emit a deep Lean graph object and interpret it;
the generated relation is the single Lean protocol definition.

The first exporter may resolve uniform type dimensions from candidate parameters once. It must keep
loop indices symbolic; parameterized definitions can be used where convenient. Export literal arrays as literal
data only when they really occur in the frozen graph. Do not enumerate family extents to discover
node instances or validate a body at one representative lane and silently assume all lanes.

The exporter must not construct a second expected Diamond graph, invoke the DSL graph builder again,
or emit a `HasDiamondGraphShape` hierarchy. Optional generated one-node equations are proved from
the exported scope relation and share that definition.

Required checks:

- constructor coverage for all operations reachable in the initial Diamond inventory;
- wrong operand order, wrong result port, wrong child scope, omitted effect root, wrong artifact
  producer, and parameter-binding mutation cause verification or well-formedness failure;
- a changed graph changes the generated theorem target and artifact identity;
- output references identify the actual named noisy/decoded roots;
- repeated calls share scope syntax without conflating invocation-local values; and
- round-trip export/regeneration is deterministic.

The emitter remains an explicit Rust-to-Lean trust boundary. Mechanical export and mutation tests
reduce accidental mismatch; they are not a formal proof of the Rust exporter or GPU runtime.
The final theorem must be visibly about the relations extracted from that graph. A program hash
does not prove the translation correct. The guarantee is mechanical single-source generation with
an explicit translator trust assumption.

## 5. Stage C: family and loop rules with scaling checks

Owner: generic Lean semantics in `crates/ir-core/lean`.

Prove the pointwise parallel rule and the sequential invariant rule once. Cover `Broadcast`, `Zip`,
and checked `ZipOffset` exactly. Support empty families through `Fin 0`, zero-iteration carried state,
and nested lexical bindings without flattening their logical domains.

For selection and gather, prove the useful rule by applying a universally quantified relation at
the actual validated index. Include repeated selectors and two successive gathers. No special
provenance datatype may be introduced for these cases.

Add a generated family fixture with logically related outputs, such as `(K i, T i)`, and check that
mixing indices cannot discharge the desired relation. Include a dynamic signed index to ensure
negative values do not become zero through `Int.toNat`.

Scaling matrix, with the same static syntax and symbolic theorem:

| Axis | Values | Required observation |
| --- | --- | --- |
| Family length `N` | `1`, `16`, `2^10`, `2^20` | No body/node/proof-instance count proportional to `N` |
| Nested family sizes | `(16,16)`, `(2^10,2^10)` | No Cartesian expansion |
| Sequential count `L` | `0`, `1`, `2^10`, `2^20` | One step lemma and one induction application |
| Explicit `FamilyPack` arity | Small growing arities | Work may follow actual stored arguments |

Record generated bytes, declarations, static node visits, instantiated lane proofs, peak memory,
and Lean kernel/elaboration time. Numeric literal length is allowed to grow. Do not claim a kernel
complexity theorem from wall-clock measurements; use both static counters and measured evidence.

This stage is independent of the full Diamond algebra and must pass before that port begins.

## 6. Stage D: reuse the mathematical core

Owners: primitive, gadget, and BGG crates.

Selectively port from `7cb9d089d`:

| Prior files | Action |
| --- | --- |
| `crates/primitives/lean/MxxPrimitives/Negacyclic.lean`, `Matrix.lean`, `Reduction.lean` | Reuse quotient-ring and coefficient-view foundations after checking interfaces |
| `Bounds.lean` | Retain the one-output-coefficient `n` bound and constant-side improvements |
| `Preimage.lean`, `Radix.lean` | Reuse local equations and reconstruction facts where they match current primitives |
| `crates/bgg/lean/MxxBgg/Encoding.lean`, `Multiplication.lean`, `Boolean.lean` | Reuse full encoding and local gate algebra; remove dependencies on old runtime trace certificates |
| Old `ScopeInvariant`, `Injector*Execution`, `BggActualReplay`, `*Trace` machinery | Do not port trace extraction or occurrence wrappers |
| Old Diamond `Model`/emitter role hierarchies | Replace with direct generated-scope proofs |
| Old caches/runners | Reuse only small process/error-handling utilities after a real theorem succeeds |

Minimize public predicates to `CoeffBound` and `Approx`; use exact equality for preimage relations.
Avoid parallel `NoiseState`, `MagnitudeFact`, `BoundedLift`, and `Carrier` records that store the same
data. A private proof may introduce an error witness through `Exists`; it need not expose an
application-independent error-expression tree.

Prove:

- ring reduction commutes with addition and multiplication;
- one coefficient of a negacyclic product contains `n` contributing products;
- matrix multiplication bound is `innerDimension*n*A_B*B_B`;
- constant-polynomial and identity multiplications use their tighter bounds;
- exact digit reconstruction for the current decomposition convention;
- `(L*B+e)*K = L*P + L*E + e*K`, with its dimension-correct integer error witness;
- target noise may be nonzero and target public structure survives consumption; and
- full BGG multiplication has `e_L*D + x_L*e_R`.

Use mathlib's existing algebra. Do not replace quotient rings with coefficient functions unless a
measured, concrete proof-cost problem justifies proving a representation equivalence. Functions are
the preferred family representation; they need not also become a new polynomial implementation.

## 7. Stage E: primitive semantics for the Diamond inventory

Owner: `crates/runtime/lean`, with lower-level facts in their mathematical crates.

Complete the semantic rules for exactly the operations reached by Diamond and its validity/circuit
predicates. Maintain an inventory generated from those actual scopes. Each reached operation has a
rule, or export rejects it explicitly. Do not represent unknown operations as an unconstrained output
that could accidentally be treated as a known primitive by an application lemma.

Pay particular attention to:

- `MatrixMulSmallRhs` operand order: its DSL receiver is sometimes the preimage, while the actual
  multiplication is left-value times that preimage;
- `MatrixMulAccumulate`: all coefficient signs, bias, and argument pairs;
- canonical versus centered coefficient extraction;
- signed/canonical gadget digits, base, count, and matrix layout;
- exact versus runtime division and rounding;
- preimage equation against the actual trapdoor public matrix;
- one fixed hash interpretation with complete tag encoding and decomposed variants;
- eager `Select` dependencies and unused-but-executed effects; and
- whole producer-family values linked to consumer artifacts without sampling them again.

Bounded sampling semantics covers successful returned samples. It supplies no proof that a sampler
always terminates or that a particular cutoff admits a preimage. State these limitations explicitly
in theorem documentation. A later implementation-refinement theorem is separate work.

## 8. Stage F: application proofs, in dependency order

Owner: `crates/we/lean`, with reusable injector/BGG lemmas below it.

### F1. Input injector

Prove the generated preprocessing scope's target equation and the initial online state. Prove one
selected transition using the same target/source/preimage variables from the sampled relation.
Use the arbitrary-index family theorem to select the actual branch, then the sequential invariant
to compose the input loop.

Keep the invariant mathematical: actual state equals a bounded prefix times its public source plus
bounded error. It is a predicate in this application proof, not a new generic carrier datatype.

### F2. Initial BGG values

Tie the actual instance and witness selectors to the initial encodings, public keys, and messages.
Prove zero padding and constant-zero/one cases with the same full BGG predicate. Do not manufacture
`0*G` terms or carrier annotations.

### F3. Gate and layer

Prove the six Boolean gate cases using the actual key/vector/decomposition operations. Treat the
six-way opcode split as a fixed number of cases, not as a case split over all family indices.
Vector/key/message selection must remain coupled.

Prove one generic layer step and then induction on depth. The invariant states that every active
encoding represents the corresponding value of the circuit prefix. The circuit data remains a
runtime function-valued input. The proof does not evaluate every possible circuit or generate one
theorem per gate slot.

### F4. Projection and decoder

Derive the actual final residual from the accepting BGG output, K encoding, and projection
preimages. Preserve the `ceil(q/2)` encryption convention and derive its modular
`floor(q/2)` consequence under subtraction.

Prove the exact inclusive decoder interval from the generated graph. Prove the whole-polynomial
approximation and use its coefficient-zero consequence to obtain decoded-message equality.

### F5. Public theorem

Prove the generated `DiamondCorrect` proposition for the concrete exported candidate satisfying the
application acceptance predicate, using reusable parameterized algebra and proof templates. A fully
parameterized final theorem is welcome when simpler, but is not a prerequisite. It must quantify
over both messages, all valid runtime circuits,
all accepting witnesses, and all successful bounded-support executions of the exact linked graph.

The final theorem may not accept a final output equation, final noise assumption, arbitrary
caller-selected wire, or a free application execution record. Local input assumptions and ordinary
ghost invariants are allowed and must be discharged at their enclosing scope boundary.

## 9. Stage G: application bound and fast candidate checking

This stage has a mathematical part that can proceed alongside F1-F3 and an acceptance integration
that must wait for F5.

Derive a fixed-dimensional nonnegative affine majorant for each uniform loop. Track only numeric
quantities actually needed by the application's proof, such as prefix magnitude, accumulated error,
and a constant coordinate. Use matrices of fixed dimension, not a generic expression language.

Prove each step's bound, monotonicity, composition, and the matrix-power summary. Prove any uniform
gate majorant from the six case bounds. Record when it loses tightness; compare it with a short
explicit recurrence on small shapes before accepting it for parameter search.

Implement and prove capped nonnegative arithmetic with `cap=C=decoderRadius(q)`. Strict acceptance
is `cappedBound<C`. Capping at `C` is sufficient: `min(B,C)<C` if and only if `B<C`; an extra
overflow-sentinel datatype or `C+1` convention is unnecessary.

Prove capping commutes with addition and multiplication and with the fixed matrix-power evaluator.
Preserve multiplication by zero even after an input saturates. This proof is required because a
premature "saturated means reject" shortcut can be wrong when a later product has a zero factor.

Use verified binary exponentiation. Compare Rust and Lean on exact small cases and threshold
boundaries. For large rejected cases, intermediates stay bounded by `C`; record bit size, integer
operation count, and kernel checking cost. No `native_decide` or unchecked native evaluator may
certify acceptance.

Integrate with `DiamondParameterSearch` only once the real generated theorem exists. Return either
a conservative bound rejection, a Lean-verified candidate, or a distinct infrastructure error.
Every returned candidate must match the graph/parameters/theorem artifact used by Lean. Cache only
after correctness is demonstrated without a cache.

Do not claim CRT-depth minimality or exhaustive failure from binary search without monotonicity.
Finding and independently verifying one candidate is the initial requirement.

## 10. Stage H: remove obsolete correctness paths

The final Diamond acceptance path has one authority: its generated Lean theorem and proved bound.
Remove its dependency on generic symbolic operational-noise acceptance once the new path passes.

This does not authorize deleting `mxx-correctness` wholesale while other applications or shared
protocol declarations still depend on it. Keep generic workflow declarations/linking if useful;
remove unused symbolic normalization incrementally by dependency, without maintaining a fallback
that silently accepts a candidate when Lean fails.

Do not merge the old app-specific branch merely to delete its trace wrappers afterward. Import the
selected mathematical files directly. Historical files remain recoverable in Git.

Update `docs/architecture.md`, `docs/diamond-we.md`, and the relevant public APIs to describe actual
verified behavior only after integration. The present design documents do not change their baseline
behavior claims.

## 11. Test and audit gates

Tests should cover semantic faults, not just implementation-shaped snapshots:

| Mutation or edge case | Expected result |
| --- | --- |
| Replace the selected preimage or target by another same-shaped value | Local proof no longer applies |
| Use two different gather indices for related outputs | Relation is not derivable without an equality proof |
| Drop the target-error term | Nonzero target-error fixture fails |
| Drop `x_L*e_R` from full BGG multiplication | Nonzero RHS-error fixture fails |
| Treat two sampler invocations as one constant | Sharing/freshness semantic fixture fails |
| Change a `ZipOffset` to modular addition | Boundary index fixture fails |
| Change a decoder endpoint from inclusive to exclusive | Threshold boundary fixture fails |
| Change the exported noisy wire or artifact producer | Graph identity/application proof fails |
| `N=0` or `L=0` | Empty family or unchanged carried tuple, with no phantom sample |
| Saturated bound multiplied by zero | Exact capped semantics returns zero for that product |
| Bound equals the radius | Reject |
| Huge logical `N` with unchanged syntax | No lane enumeration |
| Unsupported primitive, stale source, or Lean timeout | Infrastructure error, never acceptance |

At each stage, run only narrow relevant unit/Lean checks. At final integration, require full
crate-local Lean builds and standard-axiom inspection. Generated files must be recreated through the
implemented official exporter command; the plan does not invent a currently available command.

GPU integration tests require explicit authorization for the execution task. When authorized,
use the exact Lean-verified candidate and report message equality and observed noise separately.
The current baseline test only asserts the former for one message. Any added diagnostic readback
must preserve the algorithm and be explicit about its cost.

## 12. Review process and decision gates

Use Luna with reasoning effort `high` for routine independent reviews. Sol performs the final large
design/implementation review after Luna acceptance. Review can request revisions where a concrete
semantic or scalability condition is missing; it should not expand the project for stylistic polish.

Accept a stage only on explicit reviewer acceptance and its stated validation evidence. A narrow
algebra theorem, a successful Rust compile, a numeric bound acceptance, and a generated end-to-end
Lean theorem are different milestones.

Review checkpoints:

1. semantic interface prototype, before more primitive coverage;
2. exporter and uniform family/loop scaling, before Diamond;
3. preimage and BGG local algebra, including both BGG errors;
4. completed generated Diamond theorem and bound calculation; and
5. final integration, negative tests, complexity measurements, and dependency cleanup.

## 13. Definition of done

The implementation is complete when a fresh checkout can export the current DSL-generated Diamond
program, prove its application-specific theorem, and return a candidate only with matching Lean
verification. The theorem covers both messages, accepting witnesses, the actual noisy/decoded output
wires, the whole-polynomial error bound, and the exact decoder condition.

Family cardinality changes must not create more scope bodies, generated node equations, or lane proof
instances. Uniform loop depth changes must not create more step proofs; numeric bound checking uses
capped repeated squaring with measured bit cost. Explicit syntax/data size remains a legitimate cost.

The proof must use neither a duplicate handwritten protocol nor global exact-term normalization,
carrier reconstruction, trace lookup, or application-specific semantic value callbacks. Reusable
mathematics stays in the owning crates. All agreed build, axiom, mutation, and review gates pass.

## 14. Evidence from this design task

Completed here:

- read the prior handoff and inspected its source branch;
- created a design branch from `origin/codex/new-IR` and ran
  `git pull --ff-only origin codex/new-IR` successfully;
- inspected the pulled IR, runtime bindings, BGG multiplication, Diamond decoder, parameter search,
  existing family representations, and historical failure records;
- obtained a separate read-only Luna audit of family/runtime semantics and complexity;
- obtained explicit design acceptance from Luna with reasoning effort `high`, followed by an
  independent final design acceptance from Sol; both are document/design reviews, not implementation
  validation;
- researched the official Lean verification-condition interface as an optional tool; and
- created this plan and the replacement specification.

Additional finite arithmetic sanity checks passed: for `4 ≤ q ≤ 128`, the decoder-radius
inequalities agreed and 85,376 accepted residual/message cases decoded correctly; 12,340 small
cases checked that capping commutes with nonnegative addition and multiplication. These in-memory
checks are not Lean proofs, production checker tests, or scaling measurements. The required general
proofs remain in Stages D, F, and G.

Not performed here: source implementation, source deletion, Lean prototype compilation, parameter
simulation, benchmarks, integration/GPU tests, commits, or pushes. All implementation milestones and
complexity measurements above are future acceptance gates, not observed results.

# Lean Checker and Diamond WE Parallelization Plan

## 1. Status

This document is an implementation plan. It does not authorize implementation by itself and does
not record completed work. The first implementation change may begin only after this plan receives
an independent reviewer decision of `accept`.

Backward compatibility is not required. Generated Lean modules must never be edited by hand; any
change to them must be produced by the owning Rust generator.

## 2. Goal

Reduce the wall-clock cost of deriving and checking Diamond Witness Encryption hard bounds while
preserving the current trust boundary:

- the executable protocol remains the frozen Core IR DAG;
- all hard-bound formulas and acceptance rules remain in Lean;
- parallel scheduling changes evaluation order only, never accepted facts;
- one symbolic loop body represents every lane of a `parallelLoop`;
- fresh samples, selection domains, and artifact origins remain distinct at each instantiated lane;
- the WE parameter search reuses Phase A and parallelizes only genuinely independent work; and
- the final implementation must still pass the real WE parameter search and a runtime round trip.

This plan addresses both the checker and the protocol graph. Merely starting several copies of the
current slow checker is not sufficient.

## 3. Non-goals

- Completing the end-to-end Lean correctness theorem is not part of this optimization.
- No Rust implementation of a Lean bound formula is permitted.
- No probabilistic or CLT bound replaces the current hard-bound rules.
- A `parallelLoop` is not unrolled to its evaluated count.
- Worker count, chunk size, or scheduling order is not part of the protocol hash or certificate.
- The implementation must not add axioms, `sorry`, `admit`, or unchecked acceptance paths.
- This work does not re-enable AKY24 iO or AKY24 FE.

## 4. Current-state audit

Before modifying behavior, M0 must check in a reproducible performance report containing:

1. the exact commit, command, machine CPU count, and parameter environment;
2. elapsed time and peak RSS for parsing, Phase A, recurrence resolution, static-obligation
   checking, and result serialization;
3. counts of scopes, nodes by kind, parallel loops, sequential loops, facts, expression-arena
   entries, symbolic forms, recurrences, and static obligations;
4. the number of unique loop definitions and unique loop-analysis keys defined in Section 7;
5. cacheable repeated work and the five most expensive definitions; and
6. a one-worker reference result serialized in the canonical comparison format from Section 11.

The current code already analyzes a `parallelLoop` body once using a symbolic loop index. It does
not analyze every concrete lane. An implementation that adds per-lane analyzer tasks is therefore
incorrect. The generated Diamond family currently contains many distinct parallel-loop nodes; M0
must report their exact count rather than relying on a number copied into this plan.

The current analyzer also stores ordered facts and arena entries in lists, performs linear fact
lookup, and repeatedly appends to those lists. M0 must separately measure this data-structure cost
before attributing all time to symbolic arithmetic.

## 5. Lessons retained from the previous simulator

The simulator on `origin/main` is a reference for scheduling, not for formulas. Its useful design
patterns are:

1. execute independent nodes in topological waves;
2. compute immutable results in parallel and commit them in deterministic order;
3. summarize a reusable subcomputation once and instantiate its summary at call sites;
4. key reuse by structural identity and normalized input metadata;
5. process large groups in bounded chunks; and
6. release or stop retaining intermediate data after its last consumer.

The old simulator's affine-error formulas, dependency heuristics, global cache implementation, and
concrete batch constants are not copied into the Lean checker.

## 6. Required architecture

### 6.1 Pure semantic core and execution-only scheduler

Every existing semantic operation remains a deterministic pure function. Parallel execution is an
optional runner over those functions:

```text
Frozen bundle
    -> pure dependency and reuse plan
    -> immutable work items
    -> bounded parallel evaluation
    -> stable ordered commit
    -> canonical AnalysisResult
    -> Phase B
```

The one-worker runner is the reference implementation. The parallel runner must produce the same
canonical result or the check fails. Scheduler failures, panics, missing results, duplicate results,
or merge conflicts are errors, never permission to fall back to a weaker fact.

The execution mechanism is bounded Lean tasks over pure work items. Rust may keep a checker process
alive and submit environments, but it does not execute analyzer work items. Each task constructs a
dependent result of this conceptual form:

```lean
structure CertifiedWorkResult (item : AnalysisWorkItem) where
  value : Except VerifyError NodeAnalysisDelta
  valid : value = evaluateWorkItem item
```

The proof is constructed together with the pure call and erased by compiled execution. A pure
collector checks that every planned work ID occurs exactly once, rejects extras and duplicates,
restores stable order, and commits only certified values. Scheduling decides only when those values
are computed.

The effectful scheduler is not equated to a pure function. It only transports dependent values. The
pure collector produces the proof-facing result:

```lean
structure CertifiedAnalysisResult (bundle : ClosedProtocolBundle) where
  analysis : AnalysisResult
  accepted : analyzeProtocol bundle = .ok analysis
```

Only this structure, not an arbitrary task result, may enter existing bundle soundness composition.

`ParallelismConfig` is execution-only. It may select worker and chunk counts, but semantic
functions must not inspect it. The implementation must use bounded chunks and the repository's
environment-based concurrency convention; it must not install a process-global Rayon thread pool.

### 6.2 Indexed, append-efficient state without changing proof-facing order

Replace repeated linear lookup as an implementation milestone before adding broad concurrency. If
M0 finds append or interning material, analysis uses an `AnalysisStateBuilder` whose facts,
expression entries, symbolic forms, and bound witnesses are stored in Lean `Array`s, or an
equivalent representation with amortized constant-time append. The proof-facing `AnalysisState` and
`AnalysisResult` may retain their existing lists: the builder freezes each array to a list exactly
once, in insertion order, when it constructs the result.

Arena references remain insertion indices, so the existing `refsBefore` discipline is preserved.
Indexes point to builder-array positions. The builder additionally maintains analyzer-owned indexes
from frozen identity or structural key to position:

- `CoreWireRef -> fact position`;
- `JointFamilyId -> family position`;
- recurrence identity -> recurrence position; and
- expression/form structural key -> arena position.

An insertion appends exactly once to the builder collection and updates its index. Duplicate frozen
identities are rejected. The indexes are derived accelerators and are excluded from equality,
serialization, protocol hashes, and protocol-correctness theorem conclusions. Tests must compare
indexed lookup with the existing list lookup on every fixture before the latter is removed.

Although indexes are not serialized, their consistency is theorem-relevant. Define an
`AnalysisIndexes.WellFormed` invariant and prove that empty construction, insertion, and stable
merge preserve it. Prove that every indexed lookup equals canonical-list lookup; analyzer
soundness may use an index only through those lemmas.

Also prove that builder append, indexed lookup, expression/form/bound interning, reference validity,
and final array-to-list freeze equal the current list semantics. The `runAnalysisPlan` refinement in
Section 8.0 includes this freeze. If M0 shows only lookup is material and append is not, M1 may retain
list storage, but the M0 report must state the measured reason; the same index-refinement lemmas are
still required.

### 6.3 Node-analysis delta

Independent tasks must not mutate or clone a complete `AnalysisState`. A task reads an immutable
snapshot and returns a `NodeAnalysisDelta` containing only:

- produced facts with their frozen output wires;
- uninterned expressions, forms, and bound witnesses needed by those facts;
- new family or recurrence summaries;
- static obligations; and
- deterministic diagnostics tagged with stage, scope, and node.

Tasks use local temporary references. The owner commits successful deltas in ascending frozen node
order, interns their expressions in that same order, remaps local references, and updates indexes.
This deliberately keeps arena-number allocation serial and deterministic while parallelizing the
expensive rule evaluation. No shared mutable arena, mutex-protected analyzer state, or scheduling-
dependent expression ID is allowed.

If a wave contains any error, the reported error is the lexicographically first error by stage,
scope path, node, output port, and obligation index. Later deltas are discarded.

Work is scheduled in bounded stable-ID chunks. A chunk shares one immutable pre-wave builder
snapshot. After all results in that chunk arrive, the collector orders them, commits each delta, and
releases it immediately. It does not retain all completed deltas for an arbitrarily large wave.

## 7. Analyze and reuse loop templates once

### 7.1 One-lane meaning

A parallel loop is analyzed over its existing symbolic index exactly once. The resulting joint
family template is valid for every in-range lane by the existing loop rule. The analyzer must not
enumerate the evaluated loop count and must not copy one lane's concrete identity to other lanes.

### 7.2 Alpha-normalized input signature and loop-analysis key

`ValueFactSchema` alone is not a valid reuse key. `(x, x)` and `(x, y)` can have the same schemas,
while a body computing `left - right` cancels only in the first case.

Construct an `AlphaInputFactSignature` that preserves all information inspected by analysis while
replacing only caller-local loop identities with typed placeholders. It contains:

- the equality and alias partition among arguments;
- exact and affine expression trees;
- every term's coefficient, basis, factor structure, and product mode;
- bound trees, sampler/dependency metadata, constant-polynomial status, and zero rows;
- complete relation endpoints and relation kinds;
- integer/Boolean expressions, coefficient representation, and intervals;
- selection-domain structure;
- family counts, element facts, and joint-port relationships; and
- relevant external identities as typed, non-renamable anchors.

Signature construction recursively resolves every expression, symbolic form, and bound-witness
arena reference. Missing, ill-typed, or cyclic references reject caching for that body. Alias
partitions are explicit for external wires/values, sampler sources, selection domains, artifact
producers, families, and recurrences. An identity may be alpha-renamed only after structural
analysis proves it is generated inside this summarized loop instance.

Alpha-normalization may rename only argument wires, the loop site, instance path, and loop index
covered by typed substitution. It may not erase an external producer, sampler, artifact,
recurrence, selection domain, or parameter identity.

Different loop nodes may reuse a summary only when this complete structural key is equal:

```text
LoopAnalysisKey {
    frozen_definition,
    transitive_definition_closure_hash,
    loop_kind,
    index_slot,
    binding_modes,
    alpha_input_fact_signature,
    argument_matrix_types,
    output_types,
    carried_count_and_schemas,
    normalized_parameter_expressions,
    enabled_rule_set,
}
```

The key never contains evaluated matrix values or a caller-asserted semantic label. A frozen
definition reference and closure hash are both checked so a name collision cannot cause reuse. The
closure hash covers the direct body plus every transitively resolved nested definition from the
frozen definitions table. Changing a nested body therefore misses the cache even when its outer
definition is unchanged. The
initial version may omit no listed field merely because current Diamond fixtures happen to match.

A future weaker key is allowed only for a definition with a Lean theorem proving analysis is
parametric over every omitted field. The initial implementation uses the complete signature.

### 7.3 Relative summary

`LoopAnalysisSummary` stores results relative to placeholders for the loop site, instance path,
arguments, and index. It contains output templates, local obligations, relative family templates,
relative recurrence templates, and local expression/form trees. Instantiation performs typed
substitution and then the deterministic commit from Section 6.3.

Instantiation must create site-specific provenance:

- a sampler source identity is derived from the concrete loop site, instance path, source node, and
  lane path;
- an indicator/selection domain is similarly re-namespaced;
- artifact and producer identities remain tied to their concrete frozen origins; and
- recurrence references include their concrete recurrence instance path.

Thus formulas and schemas are reused, but independent random variables are never identified.
Summaries containing an unsupported site-dependent construct fail closed rather than reusing an
incomplete key.

Before a cache hit may enter `AnalysisState`, prove in Lean that instantiating the relative summary
with the matched complete signature equals direct body analysis followed by the same canonical
commit. Differential tests support this theorem but do not replace it.

### 7.4 Deterministic cache construction

Do not build a process-global semantic cache. For one bundle:

1. discover loop-analysis keys in stable protocol order;
2. deduplicate equal keys;
3. form dependency waves between unique summaries;
4. analyze independent unique summaries in bounded parallel chunks; and
5. commit immutable summaries in key-discovery order.

This is a per-analysis cache. A persistent checker process may retain a cache only for the exact
frozen bundle hash and analyzer semantic version. Changing either drops the cache. Concurrent
requests for the same uncached key are coalesced into one scheduled work item without a lock around
the analysis itself.

## 8. Phase A scheduling

### 8.0 Required refinement theorems

Differential tests do not justify parallel Phase A by themselves. Before its result can feed a
public correctness API, Lean must prove and use this chain (local naming may differ):

1. `indexedLookup_eq_canonicalLookup` under `AnalysisIndexes.WellFormed`;
2. `freezeBuilder_eq_listAnalysis`: builder operations and one final freeze equal the existing list
   analyzer semantics;
3. `commitWave_eq_sequentialNodes`: stable commit of a dependency-valid independent wave equals
   sequential analysis in frozen node order;
4. `runAnalysisPlan_eq_analyzeProtocol`: the pure planned algorithm, including builder freeze,
   returns exactly the same
   `Except VerifyError AnalysisResult` as the current `analyzeProtocol`;
5. `collectCertifiedWorkResults_eq_plan`: the complete-ID collector returns the planned result
   independently of task completion order; and
6. `collectCertifiedAnalysisResult`: item equalities and the refinement theorems let the pure
   collector construct `CertifiedAnalysisResult bundle`.

The existing sequential `analyzeProtocol` remains the specification until this chain is complete.
No theorem equates an `IO` or `Task` computation with the pure analyzer. The effectful runner only
transports dependent values to the pure collector and may not introduce a weaker proposition.

### 8.1 Scope waves

For each static scope, derive dependencies from actual input wire references and schedule nodes in
topological waves. Input nodes are seeded before the first wave. A node may enter a wave only if all
its input facts and required nested summaries are in the frozen pre-wave snapshot.

Leaf arithmetic, comparison, select, family access, sampling, and structural nodes may share a wave
when their dependencies permit. A parallel or sequential loop remains one outer node; its body is a
separate summarized scope. Sequential-loop iterations are never parallelized because carried state
creates a real dependency.

Nested definitions are analyzed in their own dependency order. Workflow stages remain sequential
when artifacts flow between them. Requirements and the ideal program may run concurrently only if
the dependency planner proves that their initial facts come from immutable protocol inputs and that
their frozen namespaces are disjoint. The comparator waits for both required workflow and ideal
outputs. No program is classified as independent from its name alone.

### 8.2 Staged rollout

Wave execution is introduced only after indexed state and sequential summary reuse are validated.
This ordering determines whether caching and data structures already remove the bottleneck before
the more invasive scheduler is enabled.

## 9. Phase B and parameter-search parallelism

Phase A is bundle-structural and must run once for a persistent checker session. Each candidate
environment runs only Phase B.

1. Build a dependency DAG between symbolic recurrences. Resolve independent recurrences in
   topological waves; update all components of one recurrence state simultaneously.
2. After the recurrence table is frozen, evaluate independent static obligations in bounded
   chunks. Each task calls the existing pure Lean `checkOne` logic. The ordered result collector
   accepts only if every obligation returns success.
3. Preserve a pure one-worker `checkAll` path and its soundness theorem. The parallel executable
   runner is a scheduling refinement of the same checked operations, not a second checker.
4. Do not use `native_decide`, unsafe FFI, a Rust result mirror, or worker-produced proof claims to
   justify acceptance.

Before enabling Phase-B tasks, Lean must prove that certified independent recurrence waves equal
`resolveSymbolicRecurrences`, and that the pure complete-ID obligation collector succeeds exactly
when `checkAll` succeeds with the same ordered obligations. The public parallel checker must supply
the unchanged `checkStaticParameters ... = .ok checked` premise used by existing soundness
composition. Workers cannot construct `CheckedStaticObligations` directly.

The pure collector returns:

```lean
structure CertifiedStaticCheck
    (analysis : AnalysisResult)
    (environment : ParamEnvironment) where
  checked : CheckedStaticObligations
  accepted : checkStaticParameters analysis environment = .ok checked
```

Only this certified value enters soundness composition. The effectful scheduler has no semantic
equality theorem.

The WE search uses one long-lived checker session per frozen bundle. It must not reparse or rerun
Phase A for each CRT-depth candidate. Adaptive CRT-depth search stays sequential where the next
candidate depends on the preceding result. Independent lattice-security candidates may be checked
in bounded ascending batches. Replace the mutable callback with a simple
`Fn(candidate) -> Result<SecurityEstimate, Error> + Sync` callback and call it from bounded Rayon
iteration. Add per-worker estimator construction only if M0 finds expensive reusable thread-local
state in the real estimator.

Batching assumes no monotonicity: evaluate every candidate in the current bounded batch, order all
results by candidate, select the first passing candidate, and otherwise report the earliest
candidate error. Physical completion order and later cancelled work cannot affect the answer.

Rust-side process or Rayon parallelism is limited to independent candidate orchestration and
security-estimator calls. Rust never evaluates a hard-bound expression or decides Lean acceptance.

## 10. Diamond WE and reusable protocol-graph changes

Protocol changes must reduce the number of distinct loop bodies while leaving runtime matrix
operations and outputs unchanged.

### 10.1 Multi-output family loops

Use the DSL's tuple-capable parallel output to fuse loops when all of these conditions hold:

- identical iteration domain and binding modes;
- no dependency between the formerly separate loop outputs;
- lane `i` of every output depends only on lane `i` and invariant inputs;
- the outputs are consumed over compatible lifetimes; and
- fusion does not increase the peak materialized matrix or VRAM set.

One fused loop returns a joint family with multiple ports. Runtime evaluation still executes the
same per-output operations for every lane; fusion changes graph structure and symbolic analysis,
not cryptographic arithmetic.

Do not fuse solely because two loops have equal evaluated counts. If domains, provenance, lifetime,
or binding modes differ, rely on summary reuse instead.

Every production fusion is accompanied by a mechanically generated `ParallelFusionMap` recording:

- each old loop site/output port and its fused site/output port;
- the bijection for loop-local sampler and selection-domain identities;
- unchanged external, artifact, and producer identities; and
- semantic-anchor and final-output mappings.

Before removing the unfused production builder, prove a generic Core IR denotation theorem for
fusing independent same-domain loops into one multi-output loop. Also prove an analyzer refinement
lemma: under a valid `ParallelFusionMap`, inferred facts and obligations agree after its provenance
renaming. These are generic DSL/Core/correctness lemmas, not Diamond-specific hand proofs. If they
require a medium redesign, retain the unfused production graph and use summary reuse instead.

### 10.2 BGG encoding bundle

Treat `(vector, public_key, plaintext)` as one lane-local `BggEncodingWire` bundle throughout
Boolean evaluation. For each layer:

1. compute the flattened slot and gather gate metadata;
2. gather the left and right encoding bundles;
3. build the six candidate encoding bundles once per symbolic lane;
4. select the opcode and active/inactive result once; and
5. return vector, public-key, and plaintext families as three ports of one parallel loop.

The current separate vector, public-key, and plaintext maps and selects must not remain as parallel
copies of the same Boolean layer. Public-key decomposition is computed once in the lane and reused
by both the public-key and vector multiplication expressions. This is deterministic decomposition
of the canonical `R_q` representative; it is not sampled data and is not added to ciphertexts or
artifacts.

The sequential circuit-depth loop carries the joint encoding family as one state tuple. It still
executes layers sequentially because layer `d + 1` consumes layer `d`.

### 10.3 Constant family loops

The initial implementation adds no Core IR broadcast operation. Repeated exact zero, one, vector,
public-key, and plaintext constants with the same count are combined as ports of one existing
tuple-capable parallel loop. The loop uses broadcast binding mode for its invariant arguments.

A structural broadcast primitive would change runtime materialization, family identity, liveness,
proofs, and GPU aliasing. It may be proposed only in a separately reviewed amendment if M0 proves
that fused constant loops remain a material bottleneck. Fresh samples and lane-dependent
provenance must never be shared through an invariant binding.

### 10.4 Diamond graph targets

Audit and then rewrite, in this order:

1. zero/one BGG families and their active masks;
2. witness vector/public-key/plaintext families;
3. Boolean circuit layer metadata and candidate selection;
4. input-injector index families that can be returned as multiple ports of one loop;
5. input-injector preprocessing outputs sharing one state or transition domain; and
6. final residual components consumed together by the checker endpoint.

For the input injector, source-state gather and transition gather remain separate dependency steps
when their index outputs differ, but the two index expressions may be produced by one multi-output
loop. The final lane-local matrix multiplication is already one parallel loop and remains so.

The audit must record each candidate as `fuse`, `invariant-binding`, `summary-reuse-only`, or
`retain`, with the reason and measured peak-liveness effect. This prevents graph-size optimization
from silently increasing VRAM consumption.

## 11. Canonical equivalence and trust checks

Define a comparison projection of `AnalysisResult` that includes every semantic field but excludes
derived indexes, cache entries, timings, worker configuration, and allocator-specific numeric IDs.
Arena references are compared after deterministic canonical renumbering.

For every supported fixture, the following must be equal:

```text
analyze(bundle, workers = 1, cache = off)
analyze(bundle, workers = 1, cache = on)
analyze(bundle, workers = 2, cache = on)
analyze(bundle, workers = available, cache = on)
```

Graph fusion changes frozen loop sites, so fused and unfused graphs are compared through an
alpha-normalized semantic-observable projection rather than full `AnalysisResult` equality. It
contains endpoint affine terms with coefficient/carrier structure, endpoint noise and total bounds,
endpoint obligations, relevant recurrence transitions, sampler/dependency alias partitions, and
Phase-B acceptance. The unfused builder remains as a test fixture until this gate passes. Compare
failing, exact-boundary, and passing environments; neither graph is treated as the expected answer.
These tests supplement the generic fusion and analyzer-refinement lemmas in Section 10.1; they do
not replace them.

Required adversarial tests include:

- equal loop bodies with different concrete sites reuse formulas but not sampler identities;
- different binding modes, schemas, parameter expressions, or rule sets do not reuse a summary;
- changing a frozen body while retaining its name misses the cache;
- shuffled task completion produces identical facts, arena contents, obligations, and diagnostics;
- a task failure reports the earliest frozen location;
- simultaneous recurrence updates do not observe partially updated state;
- sequential loops are not treated as independent iterations;
- invariant loop bindings reject attempts to share fresh or lane-dependent identities;
- fused and unfused BGG fixtures have the same semantic-observable projection and runtime outputs;
- public-key decomposition is computed from the same canonical `R_q` value in both forms; and
- no generated Lean file differs after regeneration followed by a clean second regeneration.

Proof/trust audit:

- no `sorry`, `admit`, new axiom, or unchecked certificate field;
- no Rust hard-bound evaluator or acceptance mirror;
- no semantic dependence on worker count or scheduling;
- no hand edit under any `Generated/` directory; and
- all existing checker soundness theorems still build on the pure operations used by workers.

## 12. Resource controls

Use a small execution-only configuration surface:

- one checker analysis worker limit;
- one checker work-chunk limit; and
- one WE independent-candidate batch limit.

Defaults use available parallelism but cap outstanding work by both item count and an estimated
delta-size budget. Environment overrides follow the repository's existing configuration module
and are parsed once by the executable boundary. Invalid values are reported as configuration
errors. No library function creates a new global thread pool.

Large matrix runtime loops keep their existing VRAM-specific concurrency restrictions. Analyzer
parallelism operates on metadata, but summary and delta retention is still bounded to avoid moving
the bottleneck from CPU time to RAM.

## 13. Implementation milestones

### M0: measurement and graph audit

- Add temporary or feature-gated phase instrumentation.
- Produce the report required by Sections 4 and 10.4.
- Freeze canonical one-worker results for toy, BGG, input injector, and Diamond fixtures.

Gate: the report identifies whether lookup, interning, repeated loop analysis, recurrence
resolution, or obligation checking dominates. No optimization proceeds from assumption alone.

### M1: indexed state

- If M0 attributes material time to append or interning, introduce append-efficient builder arrays
  for facts and all three arenas, followed by one stable freeze to existing proof-facing lists.
- If M0 shows append is immaterial, retain list storage and check in the measurement supporting that
  decision.
- Add derived indexes while retaining canonical ordered collections.
- Prove builder/index/freeze refinement and differentially test every indexed lookup and intern.
- Remove linear hot-path lookup only after equivalence passes.

### M2: sequential loop-summary reuse

- Proceed only if M0 finds repeated complete alpha-normalized keys and attributes material time to
  reanalysis. Otherwise record `skipped` with evidence and continue to M3 or M4.
- Implement the complete key, relative summary, typed substitution, and deterministic cache.
- Run with one worker first.
- Prove the direct-analysis refinement lemma and retain differential tests as regression coverage.

### M3: Diamond and BGG graph consolidation

- Do not add a structural broadcast primitive in this milestone.
- Fuse encoding triples, constants, and same-domain index loops using multi-output families.
- Complete the generic fusion-denotation and analyzer-refinement lemmas before replacing the
  production builder. If they trigger a stop condition, keep the production graph unfused.
- Regenerate owned Lean IR; never patch it manually.
- Compare unfused and fused runtime fixtures before removing the unfused test builder.

### M4: Phase A wave runner

- Proceed only if M0 through M3 still attribute material time to independent node evaluation.
  Otherwise record `skipped` and add no scheduler code.
- Introduce dependency waves and `NodeAnalysisDelta`.
- Add bounded workers and stable commit.
- Enable it scope by scope, retaining the one-worker reference path.

### M5: Phase B and search scheduling

- Independently profile recurrence resolution, obligation checking, and security estimation after
  Phase-A reuse. Implement only the schedulers for phases that remain material; mark the others
  `skipped` and add no dormant machinery.
- Where justified, resolve independent recurrence waves and check obligations in bounded chunks
  using the same pure Lean operations.
- Reuse Phase A in one persistent session.
- Batch only independent security candidates in WE search.

### M6: validation and cleanup

- Run formatting and narrow unit tests first.
- Build all owning Lean packages and run theorem/axiom reports.
- Run the real WE parameter search.
- Run the user-required small GPU/runtime WE round trip.
- Remove temporary instrumentation and obsolete unfused protocol builders.

Integration and GPU tests require explicit execution approval at M6; this plan does not run them.

## 14. Performance acceptance

Correctness and deterministic equivalence are mandatory regardless of speed. Performance is then
accepted only from the M0 command repeated on the same machine:

- Phase A occurs once per frozen bundle during a parameter-search session;
- the second candidate records no repeated Phase-A analysis;
- a multi-worker run demonstrates more than one active worker in at least one measured phase;
- peak RSS and Diamond runtime peak VRAM do not regress by more than 10 percent without a documented
  and approved reason; and
- each optimization milestone reports its own elapsed-time delta, so an ineffective complex layer
  can be removed instead of retained speculatively.

No absolute timing target is specified before M0 establishes a reproducible baseline. If earlier
milestones remove a bottleneck, the corresponding later scheduler is skipped and no disabled or
speculative scheduler code is retained.

## 15. Stop conditions

Stop implementation and return to design review if any of the following is discovered:

- correct reuse requires omitting a field from the complete structural key;
- provenance cannot be re-instantiated without changing a proof-facing identity;
- parallel acceptance would require trusting Rust, unsafe Lean execution, or a new axiom;
- deterministic arena merging requires changing protocol or certificate semantics;
- BGG loop fusion changes runtime arithmetic, sampling, artifact contents, or decoder inputs;
- a proposed fusion materially increases peak VRAM and cannot be bounded; or
- a medium-or-larger redesign of Core IR, symbolic recurrence semantics, or the correctness theorem
  is required.

In those cases, do not hide the issue behind a sequential fallback. Report the concrete blocker and
obtain a reviewed design amendment first.

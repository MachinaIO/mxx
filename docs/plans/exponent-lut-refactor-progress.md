# Exponent-LUT refactor implementation evidence

The authorized scope is `exponent-lut-module-noise-proof-refactor.md`, including
its direct `MulPrivate` revision. This is a work log, not a narrower replacement
for that plan. The previous BFV double-scale implementation is the baseline.

## Frozen baseline

`exponent-lut-refactor-inventory.json` records source identities, type declarations,
and 121 proof declaration/template locations before migration. Replacement claim
entries remain empty until the corresponding operational proof has been checked.
No old proof generator has been deleted yet.

| Source group | Rust files | Lines outside test modules | Public/restricted type declarations |
| --- | ---: | ---: | ---: |
| Entire `src` tree | 103 | 44,101 | 309 |
| `src/bfv` | 30 | 6,752 | 84 |
| `src/lean` | 39 | 11,240 | 4 |

These are conservative source-size measurements: complete test modules and GPU
fixture files are excluded, but comments, imports and other test-only items are
included. A test-only import does not terminate the count. Final API counts must
distinguish fully public, restricted and private declarations.

The baseline validation included 29 BFV CPU tests, eight BGG/scheme regressions,
eight native PRF regressions, nine refresh regressions, two digit tests, exact RNS
tests, warning-free CPU/GPU library builds, and two persisted GPU PRF/refresh/
continuation passes. These prove the baseline only; changed operations and final
certificates need their own evidence.

## Ownership migration in progress

- BGG encoding/public-key/RHS operations, the scheme implementation, concrete
  fixtures, PRF and refresh adapters, and decomposition noise formulas now live
  under `src/bgg`. The temporary backend submodules preserve the existing graph
  construction until duplicated orchestration and transport are extracted.
- Root concrete reexports were removed and consumers now import their owners.
  `BggError` owns concrete encoding/import errors. Root `ExponentLutError`
  contains only common declaration/graph errors.
- `src/utils.rs` owns shared LUT polynomial construction and structural sums.
  `src/noise_utils.rs` owns exact nonnegative rationals and polynomial action
  norms. Renaming `AverageVariance` to `NonnegativeRational` does not change its
  serialized fields or BGG's doubled-coordinate convention. Explicit physical
  coordinate conversion and source-aware moments remain to be implemented.
- Root `src/readout.rs` owns table declarations, identities, trace declarations,
  terminal capabilities and common terminal evaluation. BGG preimages and
  concrete readout operations moved to `src/bgg/readout.rs`; its error wraps
  generic declaration errors without making the root depend on BGG artifacts.
- PRF/refresh capability definitions moved to `noise_refresh`; `src/scheme/`
  was removed. Capability consolidation and production setup migration remain.
- The encoding trait now provides `NoiseModel`, `NoiseEvaluator` and its factory.
  Both existing noise entrypoints invoke common `scheme::trace_noise` over the
  frozen Program. BGG's helper model explicitly labels its doubled coordinates;
  BFV uses the candidate's actual parameters. Existing transfer formulas remain
  intact at this checkpoint; covariance/moment completion is not implied by
  exposing their interpreters through the trait.

## Direct multiplication kernel

`src/bfv/mod.rs` now contains fixed low-base RHS preparation and a cached direct
product, with one public block ModDown and one online block ModDown. Its targeted
CPU test checks the executed graph (four plus two scalar polynomial ModDown
nodes, no automorphism/slice/concat/external-product nodes), the exact signed
four-term error relation, a nonterminal output scale, and non-monomial payloads.
The deterministic transfer in `src/bfv/noise.rs` includes only the direct
operation's inherited, fresh and two rounding terms. The test passed.
The same graph/error fixture passed on GPU three consecutive times (2.11 s,
2.04 s, 2.02 s). The workspace GPU library build was warning-free, and
`cargo check -r --workspace --tests --features gpu` also passed. This checks
consumer compilation without executing integration tests.

At that checkpoint the kernel was not yet connected to Program or optimized PRF
selection; the following migration section records the subsequent connection. Structural family preparation/evaluation now uses the same RHS and
prepared-product metadata types parameterized by their actual carriers. Its
graph still has six scalar ModDown nodes, independent of member count. The
CPU fixture checks every member's error; the extended GPU fixture passed three
times (11.88 s, 10.88 s, 12.34 s). This extended fixture observes additional
family members, so its timings do not measure a regression of the scalar path.
Program routing, persistence, source-aware moments and the operational Lean
certificate remain part of the migration.

After ownership extraction, the full CPU library build passed without warnings.
Narrow regressions passed: BGG scheme/cache fixtures (2), common Program (15),
and BGG readout (10). These are intermediate checks, not final refactor validation.
The shared noise entrypoint passed 25 BGG application simulations and two BFV
noise traces. After simplifying the fixed-pair structural reduction, BGG's
scheme/cache fixtures (2) and encoding fixtures (18) passed. The checkpoint CPU
and GPU workspace library builds passed without warnings.

## Unified Program value types and GPU fixtures

The encoding trait now exposes only `PublicKey` and `Encoding` for both Program
and refresh values. `PublicValue` and `EncodingValue` associated types were
removed throughout the workspace. Backend scalar/family representations remain
intact; their differences do not require conversions at the generic PRF,
terminal, or refresh boundary. Concrete arithmetic fixtures still destructure
scalar outputs to inspect their exact cryptographic relations. Dimensions for
families use public metadata and do not evaluate any family member.

BFV GPU fixtures now share `src/bfv/gpu.rs` and a single GPU parameter constructor.
All 16 preexisting GPU tests remain, alongside the new direct-product fixture.
The old nine GPU files contained 605 lines; with the additional direct-product
fixture the consolidated file contains 462 lines. No operation graph, sample,
artifact schema, CPU/GPU placement, or observed bound changed in this consolidation.
The current backend file counts are BGG 9 and BFV 21 (down from 31 after adding
the direct-product fixture). This is an intermediate reduction, not compliance
with the final three-file target. Shared orchestration and transport still need consolidation.

The BFV scalar variants now directly own their key/encoding and optional known
plaintext. The two pass-through scalar structs and six one-use scalar forwarding
methods were removed. Scalar and family gate implementations share the Program
adapter, eliminating `family_adapter.rs`. Public/encoding separation, scalar
fast paths, family ranges, and cached plaintext propagation remain explicit.

Checkpoint validation after unifying the associated types passed warning-free
workspace CPU/GPU library builds and GPU-enabled workspace consumer compilation.
The BGG scheme/cache tests passed (2), and the BFV exact refresh/continuation
CPU fixture passed (121.94 s). After inlining scalar fields, the BFV Program
all-gate test (0.34 s) and whole-Program artifact reopen test (1.66 s) passed;
workspace CPU/GPU library builds and consumer compilation were again warning-free.
GPU direct-product repetitions after fixture consolidation passed twice (10.61 s,
11.33 s), following the user's two-success validation limit. These are focused
intermediate regressions; they do not replace final production/proof coverage.
The final inlined-value Program artifact fixture also passed twice on GPU
(9.40 s, 9.64 s), exercising every cached gate and persisted known plaintext.

## Explicit private products and shared gate/artifact dispatch

Program now has four primitives. `MulPrivate` is a distinct gate and lowering
method; `Lut` requires a `LutId`. The absent-table/absent-RHS representation is
removed. Root and family scopes use one `lower_gate` dispatcher. BGG direct
operations keep their existing decomposition product and modulus; misleading
identity-LUT operation/helper names were replaced with direct-product names.

BFV scalar/family Program operations now use the four-term direct kernel.
Preprocessing binds fixed RHS material; online evaluation retrieves its cached
lift and performs the row product/ModDown without generating or consuming LUT
masks. Family preparation gathers only the active RHS range, retaining the
source/key/material identities and structural representation. The ordinary
Program-driven PRF tests now use this path, including strided inputs. The
separately optimized `PreparedSparseLwrPrf` route still uses its former composed
LUT preparation and remains an explicit migration obligation.

Direct scalar/family operations persist their keys, low-base RHS and prepared
lift in the existing Program operation descriptor enum. The DSL now reopens
`GraphValueSchema` artifacts directly with the same port naming and exact wire
schemas used by `public_output`/`private_output`. BFV family artifact import
uses that shared method instead of rebuilding its own per-port input graph.
No coefficient scan, host family expansion, or online public preparation was
added. Existing LUT/linear artifact names remain unchanged; direct operation
artifacts are new fields in the revised Program descriptor.

The Program artifact fixture retains explicit scalar/family LUT coverage and
shared helper-bank assertions, and adds scalar/family direct operations with
no online helpers. Its consumer now has eight polynomial ModDown nodes: two
for each of those four operations. The initial expected count of four failed
after adding the two cases; it was updated to count the actual declared gates.

Focused CPU checks passed: Program tests (including BFV), BGG scheme/cache (2),
BFV family/direct relations (2), BFV primitive noise traces (2), direct kernel
(1), ordinary/strided PRF (2), and BGG application simulations (25). The updated
Program artifact reopen fixture passed in 2.62 s. These results do not close the
remaining covariance, optimized PRF, production-genericity, or Lean obligations.

The final workspace CPU/GPU library builds were warning-free, and GPU-enabled
workspace consumer compilation and nightly formatting passed. The expanded
Program artifact GPU fixture passed twice (15.95 s, 15.98 s); its two added
cryptographic operations make these timings incomparable with the earlier
smaller artifact fixture. The direct scalar/family equation/graph fixture also
passed twice (12.33 s, 12.34 s) after active-range gathering was added. No
integration tests were executed. Backend file counts remain BGG 9 and BFV 21;
shared dispatch/transport extraction is progress toward, not completion of,
the required three-file backend architecture.

## Remaining completion obligations

The optimized direct-PRF migration, backend module/type consolidation, fully
generic production setup/refresh/PBC, common source-aware noise trace, parameter
search integration, IR-derived operational certificates and fixed Lean proofs
are not complete. The final architecture and public documentation must describe
those completed interfaces, and final CPU/GPU/proof validation must exercise them.
The implementation must not use file moves or old passing tests to close these
obligations.

## Parameter and noise-planning ownership consolidation

Removed `bfv/parameters.rs` and `bfv/refresh_planning.rs`. Concrete parameter,
workload and manifest types, exact-basis construction and artifact validation
now live in `bfv/mod.rs`. LUT/direct/terminal error transfers, chain search,
workload auditing and refresh digit fixed-point planning live in `bfv/noise.rs`.
`BfvTerminalNoise` moved from terminal operation code to the noise module.
All Rust consumers use the new paths; no compatibility module or reexport was
introduced. The existing basis, workload and serialized certificate contracts
remain enforced. These are ownership changes, with no emitted graph changes
or extra runtime validation.

This checkpoint has 9 BGG files and 17 BFV files. It does not yet satisfy the
three-file backend target: shared artifact/readout/refresh orchestration still
needs extraction, and the noise/proof deliverables of the full plan remain open.
Both consolidated files are below 2,000 lines even including tests.

Narrow CPU validation passed six unit tests: two parameter/manifest tests,
three noise/planning tests and the terminal grouped-error equivalence test.
The full CPU library build passed without warnings. Logs:
`exponent-noise-consolidation-cpu-build.log` and
`exponent-noise-consolidation-unit-{0,1,2}.log` (ephemeral local evidence).

The full GPU library build and GPU-enabled workspace test-target compilation
also passed without warnings. Integration tests were not executed.

## Prepared readout owner consolidation

Removed the public `BfvScalarReadout` and `BfvFamilyReadout` wrapper types.
`BfvPreparedReadout` owns the common readout identity and cached public output
once. A private terminal enum owns scalar/family operation fields directly,
including the scalar finalized terminal key or the structural family and its
cached offset. This replaces three public types with one public owner and one
private storage enum, without exposing internal offset types.

Artifact export now constructs common descriptor metadata once after exporting
the scalar or structural ports. Artifact names, descriptor fields and their
serialization remain unchanged. The sum-mode offset is still prepared and
persisted; online evaluation retains its existing terminal computation and
cached offset application. This change does not finish generic readout scheduling
or remove the backend readout file; backend counts remain BGG 9 and BFV 17.

All five existing readout CPU tests passed, covering full and strided signed
native domains, weighted noise, indexed artifact reopening and summed artifact
reopening. Their graph assertion still observes `4 * width + 2` encoding
ModDown nodes with public switching confined to preprocessing. Both full
workspace library builds passed without warnings; integration tests were not run.
Evidence logs: `exponent-readout-owner-unit.log`,
`exponent-readout-owner-cpu-build.log`, and
`exponent-readout-owner-gpu-build.log` (ephemeral local evidence).

## Structural artifact schema ownership

Deleted the BFV-specific `import_family` adapter. LUT, linear, Program,
terminal-family and readout import sites now declare the existing DSL
`FamilyType` schema and invoke `GraphValueSchema::artifact_input` directly.
The DSL owns graph-port reconstruction; the backend supplies only the element
schema, cardinality, production, artifact name and confidentiality. BFV graph
errors use one `From<DslError>` conversion at the backend boundary.

The same schema, family count and public confidentiality are passed to the same
DSL importer as before. This introduces no matrix reconstruction, family-member
host iteration, artifact naming changes, compatibility format, or new generic
trait. Polynomial block transport and backend descriptor orchestration remain
to be consolidated; file counts remain BGG 9 and BFV 17.

Nine CPU unit tests passed across LUT, Program, linear, indexed terminal and
readout artifact reopen paths. Both full workspace library builds passed without
warnings. Evidence: `exponent-schema-import-unit-{0,1,2,3,4}.log`,
`exponent-schema-import-cpu-build.log` and
`exponent-schema-import-gpu-build.log` (ephemeral local evidence).
Integration tests were not executed.

## Backend cryptographic core consolidation

Removed BFV `blocks.rs`, `linear.rs`, and `encoding.rs`. Their polynomial-block
carriers, key/encoding types, sampling, direct block arithmetic, Add/public
multiplication and LUT kernels now live beside the scheme in `bfv/mod.rs`.
Removed BGG `rhs.rs` and `refresh.rs`; setup-fixed RHS definitions and individual
refresh arithmetic capability implementations now live in `bgg/mod.rs`.
Workspace imports, including GPU fixtures, use the owning module directly,
without old-path compatibility reexports. Internal block-schema fields remain
private to the backend and its child modules.

This applies the plan's concrete type/operation ownership table; it does not
move duplicated Program, artifact or readout orchestration into the core. Those
larger remaining modules still need extraction and deletion. No sampler, block
operation, schema ordering, ModDown placement or family traversal changed.
Existing unit fixtures stay with their implementing types, in named test modules.

Measured backend files: BGG 7, BFV 14. `bgg/mod.rs` has 471 lines outside its two
complete test modules; `bfv/mod.rs` has 1,421 outside its six complete test modules.
These conservative counts include comments, imports and test-only methods outside
the test modules; a test-only import does not terminate the count. Both remain
below the approximately 2,000 implementation-line guideline.

Validation passed 13 targeted CPU unit tests: block arithmetic, linear error
relations, three encoding tests, five RHS artifact tests, two BGG scheme tests,
and full BFV Program artifact reopening. Both full workspace library builds
and GPU-enabled workspace test-target compilation passed without warnings.
Evidence: `exponent-backend-core-unit-{0,1,2,3,4,5}.log`,
`exponent-backend-final-core-{cpu,gpu}-build.log` and
`exponent-backend-final-core-consumers.log` (ephemeral local evidence).
Integration tests were not executed. The three-file target, shared production
refresh, complete noise moments and operational proof migration remain open.

## Refresh capability consolidation

Consolidated the seven refresh capability traits into three:

- `ExponentLUTRefreshScheme` owns common public and encoding arithmetic.
- `ExponentLUTRefreshWitnessScheme` owns preprocessing anchors/witnesses and
  online decoder-base issuance for the witness construction.
- `ExponentLUTRefreshDirectScheme` owns direct decoder issuance and weighted
  terminal/digit preparation for the direct construction.

Public setup operations moved into the base contract. Witness online sampling
moved into the witness contract. Direct online, terminal and digit operations
moved into one direct contract. BGG implements the base and witness contracts;
BFV implements the base and direct contracts. Generic refresh scheduling still
performs each operation explicitly, with separate preprocessing and online APIs;
there is no whole-protocol backend callback or dummy anchor/decoder operation.
All Rust consumers use the consolidated names without aliases.

Removed `BfvTerminalMaterialFamily`: shared and indexed storage use the same
`BfvTerminalMaterial<M>` terminal marker. The inner material remains inaccessible
outside the backend, and the terminal-family enum retains shared versus indexed
storage. This removes a redundant wrapper declaration without allowing ordinary
state material to bypass terminal construction.

This completes the numerical capability-count target, not the entire generic
refresh migration. Concrete setup/PBC paths, prepared-context ownership and
remaining module/descriptor duplication still need work. Backend file counts
remain BGG 7 and BFV 14; no noise/proof completion is claimed.

The full CPU/GPU workspace library builds and GPU-enabled workspace test-target
compilation passed without warnings. The nine common refresh tests, BFV exact
fresh-error replacement/continuation test, and indexed terminal artifact reopen
test passed. Native-PRF refresh artifact validation is tracked separately until
its running test completes. Logs use the prefix
`exponent-refresh-capabilities-` (ephemeral local evidence).

Native PRF digit refresh artifact reopening also passed, including the next LUT:
12 CPU unit tests passed in total for this checkpoint. The controller completed
successfully; no validation job remains running. Integration tests were not run.

## One owner for mask and fresh PRF material

Removed `RefreshMaskMaterial` and `RefreshFreshErrorMaterial`, including their
unused accessors and dead-code allowances. `RefreshPrfInputs` retains one
`RefreshPrfFamilyMaterial` instead of one cloned family per CRT mask slot plus
another fresh-error wrapper. Contract, layout and coverage live once on that
family; the redundant family-identity and family-pair fields were copies of
program/layout identities and were removed.

The existing canonical label index and coverage retain distinct per-slot mask
ranges and the single shared fresh range. Mask and fresh aggregation read the
same family using their unchanged structural gathers and routing operations.
Consumption validates the expected coverage, slot count, PRF contract and error
parameters once instead of comparing copies of identical metadata per slot.
No ciphertext validation or online arithmetic was added. The existing nonzero
layout-identity check remains. This reduces ownership/metadata duplication;
the concrete BGG arithmetic in this application module still needs migration.

Validation: all 11 label/coverage/setup identity tests passed; both full workspace
library builds and GPU-enabled test-target compilation passed without warnings.
The shared-fresh-error decoder numeric test requires CUDA internally even in
this workspace CPU test binary. Its sandbox run failed before arithmetic because
no CUDA device was visible; the first run outside the sandbox passed (11.68 s).
This environment failure was not worked around with CPU fallback or parameter
changes. Evidence logs use `exponent-prf-family-owner-` as their prefix.

The identical decoder command also passed on its second run outside the sandbox
(0 failures in 2 runs). No validation job remains running. Integration tests
were not executed.

## Refresh arithmetic in the BFV core

Removed `bfv/refresh.rs`. Its individual row/public operations, CRT extraction,
weighted target preparation and direct decoder implementation now live beside
`BfvScheme` in `bfv/mod.rs`. Shared refresh scheduling remains in
`noise_refresh`; no protocol loop was moved into the backend. Updated GPU fixture
and artifact imports without compatibility aliases.

Backend counts are BGG 7 and BFV 13. The BFV core has 1,784 lines outside complete
test modules (including comments/imports and test-only methods), still below the
approximately 2,000 implementation-line guideline. Further large modules require
shared orchestration extraction before consolidation.

Noise audit: BGG `transfer_add` currently uses the correlation-agnostic upper
bound `2 * (V_left + V_right)`. This already gives `4 V` for equal input variance,
but does not establish independence or track shared ancestor sources. It remains
an upper bound, not an exact general moment transfer. A scalar factor change
would not satisfy the plan's source-DAG and covariance obligations; those remain
unimplemented and must not be reported complete.

The relocated BFV refresh test passed, including exact fresh-error replacement
and next-LUT continuation (121.21 s). CPU/GPU full workspace library builds and
GPU-enabled workspace test-target checking passed without warnings. Evidence:
`exponent-refresh-owner-unit.log`, `exponent-refresh-owner-{cpu,gpu}-build.log`
and `exponent-refresh-owner-check.log` (ephemeral local evidence). Integration
tests were not executed; no job remains running for this checkpoint.

## One prepared linear representation

Removed `BfvPreparedLinearFamily`. `BfvPreparedLinear<K, M>` retains the same
operand/output key bindings for scalar and structural-family storage. Both use
`LinearOperation<M>` with explicit Add and public-multiplier cases; the family
cache no longer encodes its operation through an optional multiplier.

Scalar keys/multipliers remain scalar, while family keys and multipliers remain
structural. No singleton conversion, concatenation, new validation or arithmetic
node was introduced. The preparation boundary still accepts its existing optional
multiplier argument and fixes the explicit cached operation once. Program caches
and artifact import use the same prepared type with the corresponding carriers.
Artifact export shares one operation-to-descriptor mapping for both carriers;
existing operation tags, artifact names and schemas are preserved. Reduction
retains its separate family-to-scalar representation.

Five targeted CPU tests passed: exact linear error/level relations, both family
fixtures, linear artifact reopening and complete Program artifact reopening.
CPU/GPU full workspace library builds and GPU-enabled workspace test-target
compilation passed without warnings. Evidence logs use
`exponent-linear-owner-` (ephemeral local evidence). No integration tests were
executed. Backend file counts remain BGG 7 and BFV 13; full shared setup,
source-aware moments and implementation-derived proof certificates remain open.

## Scheme-independent PRF declarations and tests

Removed the `BggError` dependency from `noise_refresh/prf.rs`. Native profile
validation, Program declaration and input binding now return `PrfError`, carrying
only a native PRF declaration/binding error or the common Program validation
error. BGG translates this error at its adapter boundary; setup retains it as a
PRF error rather than routing it through a concrete encoding error.

Moved the two BFV-backed common-interpreter fixtures into the BFV test module.
Profile, canonical digit and generic Program tests remain with common PRF code.
The entire PRF module now contains no BGG/BFV references, including test imports.
Existing trait-driven Program/selection evaluation and structural public labels
are unchanged. This establishes the PRF module boundary; concrete imports in
other setup/refresh/PBC modules remain and are not covered by this claim.

Validation passed 14 PRF unit tests: six common profile/declaration tests, two
BFV Program evaluation fixtures and six BGG native PRF tests. CPU/GPU full
workspace library builds and GPU-enabled workspace test-target compilation
passed without warnings. Evidence logs use `exponent-generic-prf-` (ephemeral
local evidence). Integration tests were not executed. Backend file counts remain
BGG 7 and BFV 13; this does not complete the full refactor or proof/noise work.

## Public PBC ownership and shared retry orchestration

Removed `noise_refresh/pbc/layout.rs` and `diagnostics.rs`. Public parameters,
layout hashing/rectangularization, active-cell indexing and diagnostics now live
in `pbc/mod.rs`. Existing public facade names are actual definitions; nested
layout references were updated without compatibility aliases.

Production generation and diagnostic measurement now share `schedule::retry_layout`.
The shared loop owns support validation, deterministic attempt seeds, retry
classification, exhaustion and successful schedule return. Statically dispatched
build/matching callbacks let only the diagnostic entrypoint measure elapsed time;
normal generation has no clock calls or extra diagnostic sample construction.
Diagnostic schedule validation remains outside its matching timer, as before.
Success/failure counts and layout identities keep their existing definitions.

PBC has four files remaining: mod, schedule, artifacts and evaluation. The concrete
encrypted evaluation/transport still needs migration to reach the planned three
files and backend-independent application boundary. Backend counts remain BGG 7
and BFV 13. This checkpoint does not complete noise or proof migration.

All 32 PBC unit tests passed, including diagnostic/production outcome agreement,
retry counts and failure classification, support ordering and structural selector
inputs. CPU/GPU full workspace library builds and GPU-enabled workspace test-target
checking passed without warnings. The public PBC module has 1,189 lines outside
complete test modules, including comments/imports. Evidence logs use
`exponent-pbc-owner-` (ephemeral local evidence). Integration tests were not run.


## PBC ownership and BFV schema transport checkpoint

Removed `noise_refresh/pbc/evaluation.rs`: public vector routing and layout-family
ports now belong to `pbc/mod.rs`; the GSW selector importer and its concrete
fixture belong to `bgg/mod.rs`. Public family declarations and manifest helpers
return `PbcError`, with conversion at the BGG adapter boundary. PBC now has the
planned three files. Its artifact operations and some fixtures still depend on
BGG, so this does not complete the generic application boundary. All 31 remaining
PBC unit tests and the relocated BGG selector test passed.

Removed the BFV `export_blocks` traversal. Scalar blocks now use the existing
`DslContext::public_output` / `GraphValue` transport, just like families. The
small typed import binding delegates to `GraphValueSchema::artifact_input`;
it no longer constructs one artifact input per row and column itself. This
covers input keys, ordinary LUT branches, linear caches, Program values,
terminal caches, readout offsets and refresh targets. The new public block names
are `prefix.port` in row-major order, replacing `prefix.row.column`. There is no
legacy format fallback: producers and consumers must be regenerated together.
Each port is still one polynomial; no matrix concatenation, extra sampler,
ciphertext scan or arithmetic ModDown is introduced. Private payload roles and
production bindings remain separate.

The existing scalar/family LUT, linear, all-gate Program, scalar terminal,
indexed terminal and five readout round-trip fixtures passed (10 tests). These
exercise actual artifact storage/reopening and retained graph assertions for
ModDown placement. CPU and GPU full workspace library compilation and GPU-enabled
workspace test-target checking passed without warnings. Integration tests were
not executed. Evidence logs use `exponent-schema-block-`,
`exponent-pbc-boundary-unit` and `exponent-pbc-selector-unit`.

The GPU all-gate Program artifact round-trip passed twice with the identical
command outside the sandbox (15.92 s and 15.73 s). Both runs used the GPU
backend for preparation, persisted artifact reopening and online evaluation.

Backend file counts remain BGG 7 and BFV 13. Common cache traversal, readout
scheduling and descriptor ownership still need extraction before the remaining
backend files can be consolidated. Noise-source/moment tracing and operational
Lean certificates remain outstanding; this checkpoint does not narrow those
requirements or claim completion.


## Shared terminal batch transport checkpoint

Removed `bfv/terminal_artifacts.rs`. `PreparedTerminal::export` and `import`
now live beside preparation/evaluation in the generic root readout module.
They own ordered term traversal, artifact prefixes, nonempty/count contracts
and reconstruction of the prepared owner. `PreparedTerminalDescriptor<D>`
replaces the BFV-specific batch descriptor without changing its serialized
fields. The existing terminal capability supplies individual term bindings
and cached finalized-output transport; no extra capability trait was added.
The BFV term descriptor and concrete term import/export now live in `bfv/mod.rs`.

The finalized public output is imported directly. Term order, expected stage
identities, parameter/level bindings, confidentiality and production ports
remain checked at the same preparation boundary. No public sum or public
ModDown is replayed, and encoding evaluation retains its final ModDown. The
existing terminal fixture retains its eight-node online ModDown assertion,
manifest checks and reuse with independently sampled online secrets.

BFV now has 12 Rust files; BGG remains at 7. BFV `mod.rs` has 1,862 lines outside
complete test modules (including comments/imports); root `readout.rs` has 672.
This is an incremental extraction, not completion of the three-file target or
of the generic refresh/noise/proof requirements.

CPU validation passed the relocated terminal persistence fixture and 16 common,
BGG and BFV readout tests. Full CPU/GPU workspace library builds and GPU-enabled
workspace test-target checking passed without warnings. The GPU terminal
persistence fixture also passed twice under the same command outside the
sandbox. Evidence logs use `exponent-terminal-transport-`. Integration tests
were not executed.


## BFV artifact binding owner checkpoint

Moved `BfvArtifactBinding` and its construction out of Program transport into
`bfv/mod.rs`. Input and direct-refresh descriptors now reuse this same
parameter/context header as Program and PRF descriptors instead of declaring
and checking the two fields independently. Level, output/key identity, label
routing and term bindings retain their distinct roles. Input and refresh JSON
now contain the shared header; regenerate producers and descriptors together,
without legacy format fallback.

The immutable scheme now owns a lazily initialized parameter commitment.
Artifact operations reuse it instead of traversing and hashing every CRT base
for each input/gate/import. The hash domain, fields and byte ordering are
unchanged. Schemes that do not use artifact transport never compute this hash.
Parameters are private to the backend, exposed by an immutable accessor.
No ciphertext coefficient inspection or extra online arithmetic is introduced.

BGG remains at 7 files and BFV at 12; BFV `mod.rs` has 1,899 lines outside complete
test modules. The operation descriptor enum and generic cache/refresh transport
still require further consolidation. This checkpoint does not complete the
file-count, noise-source/moment or operational-certificate deliverables.


Validation passed three input-encoding fixtures, the all-gate Program persistence
fixture, and the persisted native-digit PRF/refresh/continuation fixture. CPU/GPU
full workspace library builds and GPU-enabled workspace test-target checking
were warning-free. Evidence logs use `exponent-binding-owner-`. No integration
tests were executed; this metadata-only change did not alter GPU arithmetic.


## Linear cache operation ownership checkpoint

BFV's Program cache no longer repeats the Add/MulPublic discriminant already
owned by `BfvPreparedLinear::operation`. Its four linear variants are now
`Linear` and `FamilyLinear`, retaining the scalar and structural layouts.
Both Program gate callbacks share one encoding-lowering method. They still
emit different arithmetic according to the frozen prepared operation; the
public gate compiler creates that operation and trusted Program artifact
identity/range bindings retain the cache-to-Program association. No new per-gate
operation validation was added.

Linear descriptor import no longer chooses the same operation twice (once to
restore the prepared operation, once to rebuild the redundant cache tag).
`LinearOperation::import` shares schema-driven multiplier import between scalar
and family values. Reduction remains separate because it changes cardinality;
ordinary Add/MulPublic preserve their levels and scalar/family layouts.
Artifact field names, serialized operation kinds, known plaintext propagation
and arithmetic graph construction are preserved.

File counts remain BGG 7 and BFV 12. This removes duplicate dispatch needed
before cache/transport extraction; it does not complete that extraction or the
full module/noise/proof plan.


Validation passed five CPU fixtures covering scalar/family linear arithmetic,
linear artifact persistence and every Program gate's persisted cache. The GPU
all-gate Program persistence fixture passed twice with the identical command
(15.99 s and 15.85 s). CPU/GPU full workspace library builds and GPU-enabled
workspace test-target checking were warning-free. Evidence logs use
`exponent-linear-cache-`. Integration tests were not executed.


## Refresh protocol owner checkpoint

Removed `noise_refresh/direct.rs` and `noise_refresh/evaluation.rs`. Their
trait-driven preparation/evaluation now belongs to `noise_refresh/refresh.rs`,
as specified by the target tree. All workspace consumers use the owning
module directly; no forwarding modules or compatibility reexports remain.
The combined module has 1,993 lines outside complete test modules, including
comments/imports. Its legacy BGG-specific compiler still requires extraction;
this move does not claim a backend-independent refresh module yet.

`PreparedDirectRefreshSlot` now owns its public affine offset beside its
terminal terms and decoder target. Removed the complete-refresh owner's
parallel offsets vector and the online decoder-preparation offset argument.
Both production preparation and artifact import construct one slot with its
correct offset. Decoder preparation applies the stored offset; the previous
unused no-offset branch is gone. The existing offset artifact names and
arithmetic are unchanged. This preserves the sum-before-ModDown and subsequent
decoder-offset semantics without requiring another runtime alignment check.

Backend file counts remain BGG 7 and BFV 12. Digit scheduling, concrete BGG
setup/transport, backend consolidation, noise tracing and operational proof
migration remain outstanding parts of the full plan.


CPU validation passed nine common refresh tests, the direct fresh-error
replacement/continuation fixture (121.22 s), and the persisted native-digit
PRF/refresh/continuation fixture (116.01 s). Full CPU/GPU workspace library
builds and GPU-enabled workspace test-target checking passed without warnings.
Evidence logs use `exponent-refresh-protocol-`. Integration tests were not run.
GPU direct-refresh repetitions were started under the identical command;
their completion is recorded separately once observed.


GPU validation remains pending at this checkpoint: live exec session `2219`
runs the two identical direct-refresh GPU commands sequentially. The first
log is `exponent-refresh-protocol-gpu-1.log`; the second is
`exponent-refresh-protocol-gpu-2.log`. Poll this existing session before starting
any replacement run. The observed process remained live and CPU-active; there
was no terminal failure or successful result yet. Do not count either repeat
as passed until its actual result is observed.


The user subsequently changed intermediate validation policy: run only the
narrow necessary CPU tests, once; defer GPU validation to the final stage unless
GPU-specific behavior changes. Under that instruction, the intermediate GPU
runner (session `2219`) and its active test were stopped with SIGTERM. The
runner returned exit code 143; neither repetition is recorded as passed or as
a correctness failure. Final GPU validation remains outstanding. Future
intermediate checkpoints should not repeat workspace-wide/GPU checks when a
narrow CPU check suffices; retain the full final validation obligations.


## BGG refresh compiler ownership checkpoint

Moved `RefreshCompiler`, `RefreshSetupManifest`, `RefreshResult`, the BGG-specific
`RefreshDeclaration`, scalar decoder packages and graph-binding helpers from
`noise_refresh/refresh.rs` to `bgg/mod.rs`. Moved the four concrete compiler
fixtures with their implementation. Setup and integration consumers import the
concrete types from BGG directly, with no facade reexports. Shared trait-based
refresh evaluation remains in the application. The setup driver, PRF wire
routing and error conversions still contain concrete dependencies and require
further migration; this is not the completed generic setup boundary.

Removed `RefreshAnchorEquation` and the impossible absent-equation state from
`RefreshAnchor`. Its private constructor always constructs B*K, so the anchor
owns that graph value directly. Decoder binding no longer returns a spurious
missing-equation error, and the obsolete error variant is gone. Exact anchor
and target identity checks remain: a substituted shared decoder base is still
rejected by the unchanged grouped-slot fixture. No arithmetic/graph product or
sampler was replaced.

The backend remains at seven files; BGG `mod.rs` has 1,263 implementation lines
outside complete test modules, while common `refresh.rs` is down to 1,316.
BFV remains at twelve files. Full backend consolidation, generic production
setup, noise-source/moment tracing and operational Lean migration remain open.


Final narrow CPU validation passed all four BGG refresh compiler fixtures once,
including grouped-slot ordering, anchor substitution rejection and the two
symbolic switch-body assertion. The changed library/test target compiled
without warnings. Consumer test-target compilation also completed during the
move; the subsequently removed unused imports were the only warnings there.
Logs use `exponent-bgg-refresh-owner-`. No GPU execution, repetitions or
integration-test execution were performed under the updated validation policy.


## Refresh digit ownership checkpoint

Removed `noise_refresh/digits.rs`; `RefreshDigitPlan`, `RefreshDigitRouting` and
their exact arithmetic/error contracts now live with refresh preparation in
`noise_refresh/refresh.rs`. Updated trait, BFV planner and production call sites
without a compatibility module. Together with the earlier direct/evaluation
moves, the three specified refresh fragments have been eliminated.

Removed the unused duplicate `RefreshDigitPlan::offset_numerator`. Its existing
nonbinary offset assertions now exercise `RefreshDigitRouting::offset_numerator`,
the method used by actual target preparation. Native signed digit support,
separate mask/fresh counts, weights and mask fixed-point logic are unchanged.

`RefreshPrfFamilyMaterial::new` accepts the existing `Family<Mat>` carrier rather
than BGG's `NativePrfEncodingBatchOutputs`. Its coverage/label/size checks remain.
The removed duplicate label-count comparison followed from the BGG output
constructor's exact count contract; the common constructor still checks the
actual family size against the label index. Concrete batch access stays at the
setup adapter boundary, pending migration of that driver.

Common `refresh.rs` has 1,463 lines outside complete test modules. Backend counts
remain BGG 7 and BFV 12. This checkpoint does not complete generic production
setup, backend file consolidation, noise-source/moment tracing or Lean proofs.


Narrow CPU validation passed all seven refresh digit/label/routing fixtures once;
the changed library test target compiled without warnings. Evidence is in
`exponent-refresh-digits-owner-unit.log`. No GPU or integration tests were run.


## Refresh arithmetic boundary checkpoint

`noise_refresh/refresh.rs` now has no BGG/BFV imports, qualified references or
backend types in its signatures/tests. Removed the unused BGG error-conversion
variants from `RefreshError`; they no longer had any producing call sites.
Mask/fresh aggregation returns the existing matrix carrier rather than a BGG
wire with an always-absent plaintext. The BGG compiler assembles its own wires
from those matrices. Shared routing, summation and affine offsets remain in the
application, not hidden inside a backend protocol callback.

Moved segmented family reduction from `bgg/encoding.rs` to `utils.rs`, with a
common error type. BGG public/encoding/readout paths and application routing all
use this implementation. Its structural gather/add/select graph and odd-width
behavior are preserved. Moved the existing scalar-addition comparison fixture
with the helper; it compares widths 1, 3 and 5 and both segments against the
runtime's scalar additions and the existing single-family reduction.

Narrow CPU validation passed the reduction fixture, seven refresh digit/route
fixtures and four BGG refresh compiler fixtures (12 tests, once each). The
library test target compiled warning-free. Evidence logs use
`exponent-refresh-boundary-`. GPU and integration execution were deferred.

This closes the concrete import boundary for the refresh arithmetic module,
not for the whole application: setup, simulation and artifact/PBC paths still
need their remaining concrete dependencies removed. Backend file counts remain
BGG 7 and BFV 12. The full file/concept, noise and proof objectives remain active.

## PBC selector producer boundary checkpoint

Moved the concrete selector producer from the common PBC artifact module to
`ExponentLutEncodingSampler::sample_pbc_selectors` in `bgg/mod.rs`. Removed
`PbcStructuralSelectorFamilies`: the sampler and artifact output method now
exchange the existing `Family<Mat>` carrier directly. Updated production setup
and both PRF round-trip fixtures, with no compatibility reexport. Trusted bit
runtime binding now returns `PbcError` rather than the unrelated `BggError`.
The application still owns the private schedule and runtime bit materialization;
the BGG sampler owns cross-secret RHS encryption.

The sampler body retains the same single structural loop, source/target
capture, sampling tag bytes, ciphertext output and canonical family order.
Artifact names and metadata are unchanged. The two existing concrete selector
graph fixtures moved beside the sampler without weakening their assertions:
they cover the 49,172-member constant-size graph, one runtime family input,
absence of static packing, and graph-identity equality across private schedules.
The shared module no longer exports a concrete sampler function or its wrapper.

The CPU library test target compiled warning-free. Both sampler fixtures and
29 common PBC fixtures passed once (31 tests total); logs are
`exponent-pbc-sampler-build-final.log`, `exponent-pbc-sampler-unit.log` and
`exponent-pbc-common-unit.log`. No GPU or integration tests were executed.
GPU test call sites use the same method and matrix carrier; GPU-specific
operations were not changed and their validation remains at the final boundary.

Backend file counts remain BGG 7 and BFV 12. BGG `mod.rs` has 1,357 non-test
lines after excluding whole test modules; the PBC artifact file has 864.
PBC selector artifact metadata still uses BGG sampler layouts and RHS metadata;
its generic transport boundary, production setup/simulation, final backend
consolidation, source-aware noise models and operational proofs remain open.

## Generic PBC artifact metadata checkpoint

`noise_refresh/pbc` now contains no concrete BGG/BFV imports, type references or
fixed `gsw` role, including tests. `PbcSelectorArtifacts<M>` and its existing
metadata record retain the backend's typed RHS binding as a parameter; no new
capability trait, backend switch, erased JSON payload or default backend was
introduced. The BGG `pbc_selector_metadata` adapter constructs the source/target
binding and concrete matrix type. The common constructor binds that information
to canonical layout order, and transport verifies manifest production, family
shape, confidentiality and content hashes. Callers declare the artifact role.
The generic transport currently carries matrix families; this checkpoint alone
does not establish BFV production setup through the common application.

Removed the nested single-string BGG artifact-name wrapper from selector names;
each public cell now owns its ciphertext name directly. Name construction no
longer accepts irrelevant secret dimensions. For existing BGG callers, canonical
name bytes, metadata serialization fields/order, binding digests and artifact
roles remain unchanged. This is a source API change, without a compatibility
alias or old-format fallback. Samplers, matrix shapes, structural loops and
rounding locations are unchanged.

Metadata has private immutable fields and is constructed from the public layout;
it is not deserialized. This construction contract avoids an additional active
order traversal at the artifact boundary. Existing provenance mismatch checks
remain covered. Moved five BGG artifact fixtures and their shared fixture beside
the adapter; a common trusted-bit fixture now creates the identical ring directly
instead of obtaining it from an unrelated BGG sampler layout.

CPU validation: `cargo test -r -p mxx-exponent-lut --lib pbc` compiled warning-free
and passed all 32 existing related tests once. Evidence is in
`exponent-pbc-metadata-unit.log`. No GPU or integration tests were executed.
Backend file counts remain BGG 7 and BFV 12. Non-test lines are BGG `mod.rs` 1,402,
PBC `artifacts.rs` 822 and PBC `mod.rs` 1,648, excluding whole test modules.
The remaining setup/artifact/simulation backend dependencies, backend file
consolidation, source-aware noise tracing and operational Lean proofs remain
part of the active full goal.

## Setup sampling ownership checkpoint

Moved the BGG public-key family sampler and cached-preimage label sampler out
of application setup into `bgg/mod.rs`. The latter is now
`ExponentLutEncodingSampler::sample_label_from_preimage`. Both return the
existing backend sampling error instead of depending on `RefreshSetupError`.
The application retains the phase schedule: public keys/decompositions are
produced during preprocessing; public label monomials and online secret `s`
enter label encoding during online preparation.

Preserved the hash tag, reserved index zero, structural gather/decomposition
loops, cached preimage product, Gaussian sampling and exact mask/payload use of
`s`. No public matrix is reconstructed from its preimage online. Updated the
production setup and both PRF round-trip fixture call sites; no legacy helper
alias remains. The two existing sampler graph fixtures moved beside their
implementation, retaining their original parameter fixture and assertions.

`cargo test -r -p mxx-exponent-lut --lib setup_sampling_tests` compiled
warning-free and passed both tests once. The fixtures check constant-size graph
construction for 49,172 public keys and use of the same online secret for mask
and payload when setup secret `t` differs. Evidence is
`exponent-setup-sampling-unit.log`. No GPU or integration tests were executed;
GPU-specific behavior was not changed. BGG `mod.rs` has 1,481 non-test lines;
backend file counts remain BGG 7 and BFV 12.

This removes two concrete sampling implementations from setup, not its remaining
concrete containers or whole production workflow. Generic production setup,
remaining backend consolidation, source-aware noise traces and operational Lean
claims remain required by the full active goal.

## Refresh PRF input owner checkpoint

Removed `RefreshPrfInputs`, its forwarding mask method and its constructor-only
revalidation state. The sole caller now constructs `RefreshPrfFamilyMaterial`
directly and calls the existing common routing functions. The removed sigma
and error-bound fields were copied from the same immutable setup immediately
before comparison with that setup; they supplied no independent provenance.
The extra reconstructed coverage and duplicate parameter/contract validation
were likewise redundant after successful construction.

Retained parameter validation, canonical coverage/index/family-count matching,
PRF contract compatibility, CRT slot count and nonempty layout identity.
The family still owns its contract, coverage and layout binding. Mask and fresh
routing bodies, affine lifts, fresh scales, decoder grouping and artifact data
are unchanged. No replacement wrapper or compatibility alias was introduced.

The CPU library target compiled warning-free. The existing
`switched_preprocessing_artifacts_use_only_selected_decoder_rings` unit fixture
passed once; it builds the production benchmark bundle through preprocessing
and online graph construction and verifies decoder-ring selection. This is
construction/graph evidence, not a new numerical end-to-end execution result.
Logs are `exponent-refresh-input-owner-build.log` and
`exponent-refresh-input-owner-unit.log`. No GPU or integration tests were run.
The complete generic setup workflow, backend file consolidation, source-aware
noise models and operational Lean proofs remain open under the full goal.

## Witness decoder group orchestration checkpoint

Expanded the shared single-group witness preparation into
`prepare_refresh_decoders`, which owns the entire setup-supplied group traversal
and canonical slot restoration through `ExponentLUTRefreshWitnessScheme`.
Removed the short-lived `RefreshDecoderGroup` result wrapper; the common
function returns the three existing collections consumed by setup. Backend
parameter resolution still supplies each level, slot list, target-scale family
and anchor parameters. No new trait, dummy BFV witness or whole-protocol callback
was introduced.

Each group retains one anchor, one reduced output key, the same structural
switch/subtract target loop and witness loop. Public anchors remain in group
order; targets and witnesses are restored through the same slot-keyed maps.
The API has no online secret. Group partitioning remains a setup contract;
no new runtime coefficient or partition validation was added. The small
setup-owned CRT group list now materializes deterministic scales before shared
sampling; sampling order and operation dependencies are preserved.

The CPU library compiled warning-free and the existing
`switched_preprocessing_artifacts_use_only_selected_decoder_rings` test passed
once. It constructs the production preprocessing/online bundle and checks the
selected decoder rings and shared trapdoor count. Evidence is
`exponent-decoder-group-unit.log`. No GPU or integration tests were executed.

Inspection also confirmed that the production online PRF entrypoint still uses
BGG-specific cached accumulation and readout carriers. Those must be migrated
with their cache ownership rather than replacing the full refresh protocol with
an opaque callback. That work, the remaining backend consolidation, source-aware
noise tracing and operational proof migration remain required by the full goal.

## Program cache declaration checkpoint

Added the common borrowed `ExponentLutProgram::gate_instances` iterator over
scalar and structural body gates, using existing `ProgramGateInstance` IDs.
It never expands family members or emits synthetic reduction gates. Replaced
setup's BGG declaration expansion/regrouping with direct static-gate naming.
Family-to-bucket ordinals are computed once, including empty/add-only families,
rather than repeatedly scanning each preceding Program prefix.

Removed `ProgramCacheRole`, `ProgramCacheLocation`, `ProgramCacheDeclaration`
and `program_cache_declarations`. These records were consumed only by name
construction and one fixture; actual runtime cache bindings use their own
existing gate/family maps. Scalar/family RHS names, LUT branch-family names,
widths, member counts and output-indexed public products are preserved. No
cryptographic operation, sampler, modulus switch or artifact payload changed.
The original declaration fixture now checks static gate IDs and primitive kinds
in the Program module, preserving its four-gate and family-body coverage.

The CPU library compiled warning-free. The migrated iterator fixture and the
production preprocessing/online bundle-construction fixture each passed once.
Evidence logs are `exponent-cache-declarations-build-final.log`,
`exponent-cache-declarations-iterator-unit.log` and
`exponent-cache-declarations-setup-unit.log`. No GPU or integration tests ran.

Follow-up identified during inspection: `validate_program_cache_names` still
uses scalar-only LUT prefixes and keys uniqueness by step alone, although the
producer names structural LUT caches by `(step, body_gate)`. The native PRF
fixture has direct selections, so it does not cover that separate family-LUT
validation branch. Correct that mismatch before claiming general cache schema
coverage. Remaining generic production setup, backend file consolidation,
source-aware noise and operational proofs are still part of the full goal.

## Native PRF cache schema checkpoint

Reconciled the previously identified family-LUT name-validation mismatch with
the actual transport contract. Production export, import and manifest validation
all unconditionally rejected nonempty scalar/LUT cache lists. Native PRF
accumulation consists of structural direct selection products, so those name
variants were unused protocol states, not supported caches with a broken importer.
Removed `ProgramCacheGateNames`, `ProgramCacheLutNames` and the outer
`ProgramCacheArtifactNames` wrapper. Refresh names now carry
`Vec<ProgramCacheFamilyNames>` directly. General Program primitives and their
backend cache implementations remain intact.

Canonical name construction rejects non-selection cache shapes before export;
manifest counting is the actual family-list length. Deleted the later checks
for the now-unrepresentable scalar/LUT name lists. Preserved per-family counts,
canonical step/body identities, RHS names and shared versus output-indexed
storage. The refresh descriptor's serialized cache field is now an array rather
than an object with empty `gates`/`luts` fields; preprocessing descriptors and
identities must be regenerated. No old-format fallback was added. Ciphertext
operations and persisted matrix shapes are unchanged.

A positive unit fixture constructs the real native PRF Program, serializes and
reloads producer-generated names, and checks every bucket's shared selector and
output-indexed label entries against active layout counts. It and the existing
production preprocessing/online bundle-construction fixture passed once each.
The CPU library compiled warning-free. Evidence is
`exponent-refresh-cache-schema-unit.log` and
`exponent-refresh-cache-schema-production-unit.log`. No GPU or integration
execution was performed. The earlier attempted general-LUT fixture failed on an
invalid toy PBC configuration before reaching cache validation; it was superseded
when inspection established the actual selection-only transport contract.

This closes the dead refresh-cache name path, not generic Program cache transport
or the full refresh migration. Generic production setup, backend file-count
consolidation, source-aware noise and operational Lean proofs remain required.

## Readout cache and storage accounting ownership checkpoint

Removed `noise_refresh/artifacts.rs`. Its gadget-decomposition cache keys,
preimage declarations and lazy readout importer now belong to `bgg/readout.rs`,
beside their concrete consumer. The common manifest-derived storage report and
its byte-accounting helpers moved to `noise_refresh/simulation.rs`. Updated the
BGG PRF consumer and the GPU benchmark's report import. Artifact names, schema
hashes, serialization and runtime operations are unchanged. This is an ownership
change using the existing runtime store, not a new generic transport abstraction.

Preserved all existing fixtures in their owning modules. Three cache/store
fixtures, two storage-accounting fixtures and the imported-readout graph fixture
passed once each; CPU compilation was warning-free. Logs are
`exponent-artifact-ownership-build.log`,
`exponent-artifact-ownership-cache-unit.log`,
`exponent-artifact-ownership-storage-unit.log` and
`exponent-artifact-ownership-readout-unit.log`. No GPU or integration tests ran.
The destination files contain 1,651 and 1,706 non-test lines respectively.
The broader backend consolidation, generic production setup, source-aware noise
and operational proof work remain open; this checkpoint does not complete them.

## Shared public PRF routing checkpoint

Replaced setup's concrete public routing with
`prepare_refresh_public_routes<S: ExponentLUTRefreshWitnessScheme>`. The common
application consumes the canonical `RefreshPrfLabelIndex`, applies the existing
`scale_refresh_public_input` operation, reduces each ragged mask range and
broadcasts the single fresh source range across slots before reduction. The
segmented reducer now accepts any `GraphValue` carrier and an addition operation;
existing matrix callers retain the same addition and odd-tail graph semantics.

BGG preprocessing now passes the exact mask/fresh route decompositions that it
persists for online use. Previously, public routing rebuilt the same corrected
targets and invoked `BggPublicKeyCompiler::matrix_mul`, decomposing them again.
Removed that duplicate construction plus unused nominal scale families that
were discarded whenever corrected scalar families were supplied. No digit
weights, correction formulas, fresh-source sharing, public sum/ModSwitch order,
or online arithmetic changed. The source targets, preimage schemas and artifact
names remain authoritative; generated production graphs have changed and must
be regenerated rather than treated as identical proof artifacts.

Deleted `aggregate_public_refresh_prf`, `route_public_prf_family`,
`route_scale_family`, `reduce_public_family_segments` and the unused local
`family_count_is`. Setup decreased from 5,538 to 5,228 non-test lines; the new
common traversal is approximately 55 lines. Backend count and full setup
genericity targets remain unfinished.

The CPU library compiled warning-free. A new CPU fixture compares cached routing
with the existing uncached BGG primitive for unequal mask ranges and a shared
fresh source. Its actual graph has exactly two decomposition bodies (one per
prepared route family). It passed once, as did the production setup/online graph
construction fixture and two existing structural reduction fixtures. The initial
new fixture used a gadget base inconsistent with the runtime parameters; it now
derives the base, digit count and columns from those parameters. Passed fixtures
were not repeated. Logs: `exponent-public-route-sharing-unit-final.log`,
`exponent-public-route-sharing-setup-unit.log` and
`exponent-public-route-sharing-reduction-unit.log`. No GPU or integration tests
ran. Full operational proof generation and final CPU/GPU validation remain open.

## Generic verification carrier checkpoint

`RefreshVerification<V: GraphValue>` now owns the common threshold-decoder
construction and output transport. The backend supplies its encoding, anchor,
residual and exact scalar residual components. These are a construction
contract, not an assumed correctness certificate; operational proofs must still
establish the supplied relation. Decoder-target extraction remains generic over
the carrier and checks the same frozen scalar operands.

Moved `build_refresh_verification` from setup to `bgg/mod.rs`, preserving its
separate-secret equation, shape checks, gadget construction and column slicing.
The BGG production simulation request now calls that backend builder. Extracted
the existing concrete matrix-type comparison into `utils.rs` for its setup and
backend callers. BGG's scalar artifact output names and decoder ordering remain
unchanged. BFV's block carrier exports individual polynomial ports directly;
there is no dense row conversion in the common verification container.

Two narrow graph unit fixtures passed once, with warning-free CPU compilation:
BGG retains four column decoders and its existing output names; BFV retains two
scalar decoders and two polynomial ports with no Slice or Concat nodes. Evidence:
`exponent-verification-carrier-unit.log`. The BFV fixture checks the common
transport boundary, not a full BFV production refresh certificate. No GPU or
integration tests ran. Setup now has 5,132 non-test lines and BGG mod has 1,583;
production setup genericity, backend consolidation, source-aware noise and
operational Lean proofs remain unfinished.

## Backend route construction and canonical range ownership checkpoint

Moved BGG's gadget-shaped digit target and preimage generation from common
refresh code into `bgg/mod.rs`, inlining its single-use target helper. The
backend accepts source matrix schema metadata instead of an unused public-key
wire and derives the family count from the digit dimensions. It keeps the
two-row target contract. Removed the separate overriding `corrected_scalars`
argument: one scale family is authoritative, and setup explicitly supplies base
one when those scalars already contain the corrected digit powers. Target
products, structural decomposition and persisted route schemas are preserved.
Common `route_coordinates` retains only scheme-independent index arithmetic.

Removed `RefreshPrfCoverage` and its redundant copy of dimensions/digit counts.
The only producer constructed both it and `RefreshPrfLabelIndex` from the same
validated setup parameters and compared those copies. `RefreshPrfFamilyMaterial`
now owns the existing label index, alongside the distinct PRF contract, PBC
identity and actual output family. Its constructor still checks output count.
Public and online aggregation now derive ranges from that same index contract;
duplicate prefix-count arithmetic and the separate stored/passed slot count
are gone. Fresh PRF outputs remain one shared group, and masks retain ragged
slot ranges. Zero-start gathering keeps the prior identity-index graph.

The migrated route-shape fixture and the production preprocessing/online
bundle-construction fixture each passed once. CPU compilation was warning-free;
logs are `exponent-route-backend-unit.log` and
`exponent-route-index-owner-unit.log`. No GPU or integration tests ran.
Common refresh now has 1,291 non-test lines, setup 5,113 and BGG mod 1,661.
This removes a backend target formula and a redundant coverage concept from the
shared layer; full generic production setup, backend file consolidation,
source-aware noise and operational certificates remain required.

## Shared public/online routing traversal checkpoint

Unified public and online digit-family routing in one generic graph-value
traversal. `prepare_refresh_public_routes` supplies public operations;
`aggregate_refresh_prf` supplies online row operations and adds supplied offsets
after reduction. The witness capability gained one individual operation,
`route_refresh_row`, implemented by BGG's existing preimage product. No new
capability trait, dummy BFV witness or whole-protocol backend callback was added.

`RefreshPrfFamilyMaterial<V>` now carries the associated row type. Deleted the
separate mask/fresh online aggregation functions, their route-product helper,
matrix-only reduction wrapper and duplicate range gathering. The common refresh
file no longer imports `Mat`, `Preimage` or `Ring`. BGG's half-offset vector
constructor and its existing nonbinary fixture moved to `bgg/mod.rs`; setup
supplies those same scaled offsets to the common aggregator. Signed-digit
normalization, fresh-source sharing and aggregate-before-offset/ModSwitch order
are preserved. Consolidated traversal removes redundant gathers; it adds no
decomposition, sampling or ciphertext validation.

Extended the existing CPU public-route fixture to exercise online rows against
the trusted BGG encoding primitive, including unequal mask ranges, a shared
fresh source and per-slot offsets. Public and online actual outputs share
exactly two prepared decomposition bodies. The combined fixture, production
setup/online graph construction and migrated base-16/base-32 offset fixture each
passed once. CPU compilation was warning-free. Evidence logs:
`exponent-online-trait-routing-build-final.log`,
`exponent-online-trait-routing-unit.log`,
`exponent-online-trait-routing-setup-unit.log` and
`exponent-online-trait-routing-offset-unit.log`. No GPU or integration tests ran.

Non-test counts are now 1,143 for common refresh, 1,720 for BGG mod and 5,137 for
setup. BFV's direct path still uses its own terminal preparation capability;
this witness-path consolidation does not demonstrate both complete production
setups or discharge the pending noise/proof and final-validation requirements.

## Program noise dependency DAG checkpoint

The common `trace_noise` entrypoint now records a `NoiseDependencyGraph` during
the actual `lower_program` invocation. A private evaluator decorator forwards
all operations to the existing backend while recording binding, Add, public
product, private product, explicit LUT, broadcast and reduction dependencies.
It does not walk the Program a second time. A flat arena and wire-to-node map
preserve sharing without recursively copying expressions or expanding family
members. Noise helper metadata must be cloneable at this entrypoint; both
existing backend models satisfy that bound.

BGG `ProgramNoiseReport` and BFV `BfvNoiseTrace` retain the resulting graph.
Repeated operands reference the same node; table-level helpers, gate helpers
and RHS families retain their binding provenance. Broadcast and reduction keep
the inherited node and actual active range. These IDs describe Program binding
provenance, not unique samplers, and distinct IDs do not imply independence.
No variance reduction, tail assumption or fresh-error distribution was inferred
from the new graph. The existing numerical transfers remain unchanged. Actual
sampler-source identities, moment/covariance transfers, signed error terms and
certificate integration are still required to complete the planned noise model.

Extended the existing BGG family-visitor and BFV Program noise fixtures. They
confirm repeated-Add identity, broadcast ancestry, active reduction range and
compact graph storage; the BGG fixture also round-trips the flat graph through
serialization. Both passed once, and the CPU library compiled warning-free.
Logs: `exponent-noise-dependency-build-final.log`,
`exponent-noise-dependency-bgg-unit.log` and
`exponent-noise-dependency-bfv-unit.log`. No GPU or integration tests ran.
BGG serialized reports now include the dependency graph; no legacy-format
fallback was added. The full structural, mathematical and proof goal remains
active and incomplete.

## Native digit moments checkpoint

`noise_utils::NoiseMoments` now represents an exact signed rational mean and
centered variance in physical coordinates. Its uniform integer interval
calculation is shared by refresh digit planning and the existing BGG digit
variance calculation; the latter explicitly multiplies physical variance by
four for its `2e` convention.

`RefreshDigitPlan` exposes conditional uniform moments for the native weighted
fresh interval and each centered mask interval. `BfvRefreshPlan` retains one
fresh distribution shared across CRT slots and a mask distribution per slot.
The fresh mean is `sum_j b^j / 2`, while each mask mean is `-1/2`. The model
requires independent uniform native digits; these are not CBD distributions,
and the new fields neither establish PRF independence nor change deterministic
spacing/scale acceptance. The serialized BFV plan gains these required fields;
there is no legacy fallback.

The existing base-four fixture now compares exact fresh moments with enumeration
of its actual two-digit support and checks centered masks with unequal counts.
It passed once after a warning-free CPU library build (27.34 seconds). The
existing nonbinary doubled-variance and BFV refresh fixed-point fixtures also
passed once using that binary. No GPU or integration tests ran. The build and
first fixture log is `exponent-digit-moments-unit.log`.

This completes the interval-moment calculation and BFV report connection, not
the per-gate source/covariance model. BGG application mean propagation, actual
sampler provenance, BFV signed-term moment transfers, generic production setup,
backend consolidation and operational proof migration remain required.

## Derived BGG refresh payload moments checkpoint

Removed the independently mutable mask/fresh variance arrays from
`RefreshNoiseParameters`, their setter, setup-side computation, snapshot
presence/denominator checks, and the compact parameter-search assignment.
These arrays described nominal payload distributions, not route-dependent
operational errors. The authoritative base and digit counts now determine
physical moments through the same native-interval helper used by BFV planning.
Changing counts cannot retain stale nominal statistics. Construction still
validates the base, counts and actual topology/route contracts.

BGG average refresh reports now include conditional physical mask/fresh moments
instead of doubled-coordinate nominal variance scalars. Operational variance
fields retain their existing doubled-coordinate convention and acceptance
equations; nominal payload distributions are not added to the error tail.
The parameter-search report transport retains both mean and variance. Snapshot
and diagnostic JSON schemas change directly, without compatibility fallbacks.

Replaced obsolete variance-injection tests with a positive count-derived moment
fixture covering bases 2, 16 and 32, unequal mask counts and fresh mean 17/2.
The existing switched-refresh snapshot fixture still checks its exact residual,
rounding and operational variance, and now checks the payload moment report.
Both CPU fixtures passed once after a warning-free library build (27.94 seconds).
The first build/test log is `exponent-derived-moments-unit.log`; the second used
the built CPU binary. No GPU or integration tests ran. No ciphertext graph,
sampler or device execution changed.

This supersedes the previous checkpoint's BGG digit-variance helper: that helper
is removed because its remaining production consumers needed complete physical
moments. Full per-gate mean/covariance propagation and the remaining structural
and operational-proof deliverables are still incomplete.

## Shared public artifact emission checkpoint

The DSL now provides `DslContext::public_outputs` for an ordered iterator of
named `GraphValue` values. It uses the existing schema-aware insertion path
with public confidentiality and never expands structural family members.
BFV scalar/family direct products, scalar/family linear operations and prepared
LUT branches use it instead of repeating output-context threading and error
conversion. The backend still selects each field, name and ring and owns
binding checks. Artifact names, order, schemas and import behavior are unchanged;
no ciphertext operations, rounding or samplers were added.

Existing CPU artifact round trips passed once each: linear scalar/family
Add/MulPublic/reduction (0.45 seconds), and all Program gate kinds and public
values (2.83 seconds). The library build was warning-free (33.52 seconds); the
first build/test log is `exponent-public-artifact-export-unit.log`. Formatting
and whitespace checks passed. No GPU or integration tests ran.

This removes repeated field-emission machinery, not the remaining backend
artifact modules themselves. Their binding/import orchestration still needs
consolidation, along with generic production setup, source-aware noise transfers
and operational Lean certificates. The full goal remains active.

## Single-owner BFV LUT cache bindings checkpoint

Removed the duplicated `BfvLutBinding` fields from scalar and family
`PreparedGate` variants. Scalar prepared LUTs already own the binding, and
prepared family LUTs own it through their retained material. Program helper
preparation, online helper/RHS/table checks and artifact import now use those
authoritative fields directly. Imports no longer clone descriptor metadata into
a second cache copy. Existing mask-bank sharing and helper validation remain;
there is no new runtime validation or graph operation.

The existing all-gate Program artifact fixture covers scalar and family LUTs,
preparation of shared online helpers and store/reload of the Program cache.
It passed once on CPU with a warning-free library build; the log is
`exponent-bfv-cache-binding-unit.log`. No GPU or integration tests ran.
Artifact descriptor formats are unchanged by this internal ownership change.
Module consolidation, generic production setup, complete source/moment transfers
and operational proof migration remain required for the full goal.

## Operation-owned linear RHS metadata checkpoint

Replaced `BfvLinearKind` plus the independently optional RHS field of
`BfvLinearBinding` with `BfvLinearOperation`: Add and MulPublic each carry a
required RHS identity, while Reduce carries none. Import returns the prepared
operation together with that identity and no longer performs separate missing-
RHS checks. Reduction no longer checks an impossible attached-RHS combination.
Program artifact dispatch and the existing identity-mismatch fixtures use the
new enum. Public-key, multiplier, level, active-range, output and production
bindings retain their previous meaning; no evaluator or ciphertext graph changed.

The serialized linear descriptor changes from separate `kind`/`rhs` fields to
one operation value. Regenerate descriptors and their containing identities;
there is no old-format alias or fallback. The scalar/family linear artifact
round-trip fixture, including reduction and binding mismatch checks, passed
once on CPU (0.47 seconds) after a warning-free library build (28.68 seconds).
Log: `exponent-linear-operation-binding-unit.log`. No GPU or integration tests
ran. This removes one contradictory metadata combination but does not complete
the remaining module, production setup, noise or proof deliverables.

## Linear artifact ownership and range checkpoint

Moved active-range metadata into `BfvLinearOperation`. Add/MulPublic retain an
optional range for their scalar/family representation; Reduce requires a family
input range and always has a scalar output. Program import and cached-plaintext
schema selection derive input/output ranges from this operation rather than a
separate optional field. Reduction without an input range is no longer a
representable descriptor. Serialized binding fields change directly; no old
format fallback is retained.

Following the shared DSL output extraction and removal of duplicated binding
fields, consolidated the remaining linear binding/import/export implementation
with its Program cache artifact dispatcher in `bfv/program_artifacts.rs`.
Deleted `bfv/linear_artifacts.rs` and its module declaration/import. Existing
linear artifact fixtures live in the new owner. The BFV directory now has 11
Rust files (previously 12); the combined artifact owner has 875 non-test lines,
measured after removing complete test modules rather than stopping at the first
test declaration. This is an intermediate consolidation, not the final target
of approximately three backend files or generic Program artifact orchestration.

The range change passed the existing linear scalar/family/reduction round-trip
once (0.47 seconds). After consolidation, the all-gate Program artifact fixture
passed once. Both builds were warning-free; logs are
`exponent-linear-range-binding-unit.log` and
`exponent-linear-artifact-owner-unit.log`. No GPU or integration tests ran.
Artifact polynomial fields and computation graphs are unchanged. The remaining
generic setup, noise/source model and operational Lean migration are incomplete.

## LUT artifact owner consolidation checkpoint

Consolidated input-material and scalar/family LUT descriptor transport with the
Program cache and linear operation transport in `bfv/program_artifacts.rs`.
Deleted `bfv/artifacts.rs`, updated terminal/readout/refresh consumers, and moved
its existing CPU fixture into the new owner. The basic `import_blocks` helper
now lives beside BFV block types in `bfv/mod.rs`, so arithmetic/readout code can
reopen schema-based polynomial blocks without depending on Program persistence.
The branch exporter continues to use the shared DSL field-emission API.

BFV now has 10 Rust files (previously 11). The combined Program artifact owner
has 1,249 non-test lines, counting after removing all complete test modules.
This consolidates already-shared field transport and binding responsibilities;
the remaining generic Program/refresh artifact orchestration has not yet been
extracted and the final backend module target remains outstanding.

No artifact names, descriptors, schemas, confidentiality or evaluation operations
changed. The moved scalar/family prepared-LUT artifact round-trip passed once
on CPU (1.14 seconds). After correcting one relocated `Ring` import, the final
library build was warning-free (28.93 seconds). Log:
`exponent-lut-artifact-owner-unit.log`. Formatting and whitespace checks passed;
no GPU or integration tests ran. Full noise/proof and production-setup work is
still incomplete.

## Common direct-refresh slot artifact transport

`PreparedDirectRefreshSlot` now owns ordered export/import of its cached decoder
target, offset and terminal list. It uses the existing associated `GraphValue`
schemas and individual terminal callbacks, with no concrete backend imports,
new capability trait, public reevaluation or coefficient validation. BFV supplies
the selected ring schemas and its terminal descriptor restore/export operations.

`PreparedDirectRefreshSlotDescriptor<L, D>` keeps each level with its terminal
descriptors. `BfvRefreshDescriptor` stores these slot records instead of parallel
`levels` and `terms` arrays previously consumed through `zip`. Slot association
is structural rather than enforced by an added runtime length check. Artifact
names, public/private roles, label topology and shared fresh inputs are unchanged;
serialized descriptor structure and its trusted identity change without fallback.
Generic production setup and full protocol artifact transport remain unfinished.

The existing native-PRF direct-refresh artifact CPU fixture passed once in
115.19 seconds, including store/reload, shared fresh digits and next-LUT
continuation. The library build was warning-free (29.21 seconds). Log:
`exponent-refresh-slot-artifacts-unit.log`. No additional CPU repetitions,
GPU tests or integration tests ran; formatting and whitespace checks passed.

## Direct-refresh adapter module consolidation

After slot transport moved into the common refresh layer, consolidated the
remaining BFV direct-refresh descriptor/binding adapter into
`bfv/program_artifacts.rs`. Deleted `bfv/refresh_artifacts.rs`, moved its existing
fixtures intact, and updated the GPU wrapper's fixture import path. No runtime,
sampling, schema or descriptor logic changed in this step.

BFV now has 9 Rust files (previously 10). The artifact owner has 1,360 non-test
lines. The CPU library and all its unit fixtures compiled warning-free in 29.05
seconds (`exponent-refresh-artifact-owner-build.log`). The just-passed 115-second
refresh fixture was not repeated for this pure module move. No GPU or integration
tests ran. Generic production setup, remaining backend extraction, source-aware
noise models and operational proof migration are still incomplete.

## Terminal and readout ownership consolidation

Moved shared-mask and shared-ciphertext terminal artifact enumeration onto the
existing DSL public-output iterator, retaining field names and order. Consolidated
scalar and structural terminal implementations, then incorporated both into
`bfv/readout.rs` as requested. Deleted `bfv/terminal.rs` and
`bfv/terminal_family.rs`; all crate consumers and GPU fixture import paths now
refer to readout. Native readout and terminal values retain distinct types and
phase invariants inside this single owner. Structural batching and the single
outer ModDown are unchanged.

The BFV directory now has 7 Rust files (previously 9). The combined readout owner
has 1,661 non-test lines after removing complete test modules. The indexed
terminal artifact fixture passed once after the shared export change (1.61
seconds; `exponent-terminal-owner-unit.log`). The subsequent move into readout
compiled all CPU unit fixtures warning-free in 28.47 seconds
(`exponent-readout-terminal-build.log`); that passing test was not repeated for
the module move. Formatting and whitespace checks passed. No GPU or integration
tests ran. This is still short of the full module/generic-setup/noise/proof goal.

## Terminal family binding ownership

Removed the duplicated binding field from `BfvPreparedTerminalFamily`. Its
material already owns all binding fields. Mask-bank lookup, raw evaluation and
descriptor export now derive this fixed-size metadata from that owner. The
family and prepared-readout binding accessors return it by value; no polynomial
work, allocation, coefficient validation or public preparation was introduced.
Scalar terminal preparation still retains its necessary binding because it does
not retain the source material. Artifact metadata and its meaning are unchanged.

The existing indexed terminal artifact test passed once on CPU, including
reopening with new online secrets (1.57 seconds). The library build was warning-
free (28.55 seconds); log: `exponent-terminal-binding-owner-unit.log`. Formatting
and whitespace checks passed. No GPU or integration tests ran. The full
structural, noise, setup and proof goal remains incomplete.

## Symmetric readout ownership checkpoint

Both backends now place their readout capability implementation and prepared
readout artifacts in their respective readout modules. The BGG implementation
and production readout cache moved from bgg/prf.rs to bgg/readout.rs; setup imports
the cache from its owner. PRF batch carriers remain in bgg/prf.rs, and the moved
implementation uses their existing accessors. BFV scalar and family terminal
operations remain in bfv/readout.rs.

This is an ownership-only change: graph operations, artifact fields, and
preprocessing/online boundaries are unchanged. Excluding inline test modules,
BGG readout has 1,931 lines and BFV readout has 1,658 lines. The broader backend
consolidation and generic setup/proof work remain incomplete.

Validation: cargo +nightly fmt --all; cargo test -r -p mxx-exponent-lut --lib
--no-run passed without warnings in 28.53 seconds after removing unused imports;
git diff --check passed. No test execution or GPU run was repeated for this move.

## BGG route residual arithmetic ownership

Moved ordinary gadget tower-width resolution and even route correction from
noise_refresh/setup.rs to bgg/noise.rs. Setup maps the existing parameter error
messages at its boundary; the arithmetic no longer depends on setup parameters
or setup errors. Public target construction and the even-residual gadget noise
bridge now use one correction function. The bridge supplies the same CRT
residue, and the shared correction selects its even representative of magnitude
at most the dropped product. No graph operations, samplers, or artifact fields
changed.

Moved the existing half-payload/omitted-residue fixture beside its implementing
arithmetic. Both bridge fixtures passed once on CPU: cargo test -r -p
mxx-exponent-lut --lib bgg::noise::bridge_tests (2 passed, 28.65-second warning-free
build). Formatting and git diff --check passed. The common setup still contains
concrete BGG adapters; this checkpoint does not complete the generic production
setup requirement.

## Shared PRF ranges and readout weight representation

Moved PBC active-block range construction into SparseLwrPrfProgram in the common
PRF module. Both ordinary bind_inputs and BGG cached batch public preparation
consume that implementation. Removed the second PBC range traversal definition
from bgg/prf.rs; cryptographic inputs and cached BGG operations remain backend
owned. This is not yet the complete removal of the BGG PRF adapter module.

BFV readout now uses BfvReadoutWeights<R> for both runtime weights and serialized
metadata. Runtime indexed weights contain structural rows; metadata uses unit
rows because the graph artifact carries them. Removed the separate optional
weight and optional indexed coefficient bound, making shared/indexed mutually
exclusive in the serialized representation without another wrapper enum.
Artifact values, dimensions, sampler sharing, and switching are unchanged.
The descriptor JSON shape changed; existing descriptors must be regenerated.
No compatibility path or runtime coefficient validation was added.

Validation: indexed readout artifact reopen passed once on CPU (7.09 seconds);
the existing structural PRF public accumulation fixture passed once. The final
library unit-target build completed warning-free in 28.40 seconds after removing
an unused import. cargo +nightly fmt --all and git diff --check passed. GPU and
integration tests were not run. Generic production setup, further backend
orchestration extraction, and noise/proof completion remain outstanding.

## Cached affine readout offsets

BFV readout exports the prepared affine offset for every representation:
scalar, shared family, indexed family, and summed family. Imports read these
public artifacts directly. Removed the duplicate inverse-of-two, row-weight
multiplication, and indexed offset construction from import; the sum path no
longer constructs an offset family before discarding it for its saved sum.
Preprocessing retains the original calculation and online evaluation adds the
imported offset. Shared offsets remain one row and indexed offsets remain
structural families.

All offsets use the existing readout prefix with an offset suffix; the previous
sum-offset artifact name is removed. Regenerate artifacts. Descriptor fields
are unchanged. Terminal ciphertexts, sampler sharing, output values, and ModDown
placement are unchanged; offset persistence adds the required cache ports for
previously reconstructed offsets.

Validation: aggregate readout artifact reopen passed once (both shared and
indexed sum cases, 3.33 seconds), and independent indexed readout artifact reopen
passed once (10.21 seconds). These fixtures also exercise scalar readout reopen.
The release library unit target compiled warning-free in 28.68 seconds. Formatting
and git diff --check passed. No GPU or integration tests were run. This completes
the offset-cache simplification, not the full generic setup/noise/proof refactor.

## Canonical label topology ownership

RefreshPrfLabelIndex now owns RefreshTranscriptTopology plus its cached mask
prefixes, replacing five separately copied dimension/digit fields. Prefixes stay
cached to preserve direct group lookup. Constructor cardinality validation uses
the topology calculation; checked total cardinality and positive group sizes
justify prefix arithmetic without repeating per-prefix overflow checks.

Setup's PRF output count now uses that same topology calculation instead of
independently adding mask and fresh products. The shared fresh group is counted
once, while masks retain per-slot digit counts. No scheme-dependent polynomial
count is hardcoded into the index.

Updated the ragged label fixture to exercise the serialized topology and compare
its total with the index, then retain all existing label/coordinate checks.
Three targeted CPU label fixtures passed once; the final library unit-target
build was warning-free in 28.15 seconds. Formatting and git diff --check passed.
The serialized label-index representation changed and descriptors embedding it
must be regenerated. No GPU/integration execution was added. Production setup
genericity and the full noise/proof obligations remain incomplete.

## General fixed-RHS operations replace the PRF-specific capability

Replaced ExponentLUTPrfScheme with ExponentLUTFixedRhsScheme in the root encoding
interface. The optional capability separates public weighted RHS combination,
fixed-product preparation, and cached encoding evaluation. Its methods do not
refer to PRF or one-hot semantics. PreparedSparseLwrPrf explicitly connects
combination and product preparation; backend code no longer implements a whole
prepare_prf_selection operation.

BFV's adapter delegates weighted structural family summation to the common
weighted_family_combinations utility, retains material identity/level metadata,
and calls its existing prepare_mul_private_family/evaluate_mul_private_family
operations. Every weighted ciphertext error remains in the sum. The composed
ciphertext stays in the low base before the existing private-product ModDown.
Scalar broadcast and prepared cache ownership are retained. No extra capability
trait was added overall: the application-specific trait was removed.

Validation: the existing two-step composed PRF fixture with an active subrange
and signed labels passed once on CPU in 0.24 seconds. Final release library
unit-target build was warning-free in 27.97 seconds. Formatting and diff checks
passed. No GPU or integration tests were run. BGG's old production cache path
still requires migration to this common interface; readout orchestration and
the remaining full refactor requirements are not complete.

## Common polynomial-row table declaration

Replaced BfvTerminalTable with RowLutTable<C> in the common readout module.
The declaration owns public coefficients, native layout, table identity, and
the selected-row coefficient bound; none depends on BFV ciphertext arithmetic.
BFV terminal samplers accept RowLutTable<2>, retaining two-column static shape.
Updated ordinary terminal, affine readout, refresh target construction, and
their fixtures. No wrapper or capability trait was introduced.

The constructor preserves table-length/layout checks and public coefficient
bound calculation. Backend callers map declaration errors at their boundary.
Sampling, branch equations, coefficient storage, and ModDown ordering remain
unchanged. The table hash domain is now scheme-independent, so table-bound
materials/artifacts must be regenerated; no compatibility alias remains.

The indexed affine readout artifact-reopen fixture passed once on CPU in
7.12 seconds, with a warning-free 28.03-second release unit-target build.
Formatting and git diff --check passed. No GPU/integration tests ran. Backend
readout orchestration, the requested five-file layout, generic production setup,
noise-source models, and operational Lean migration remain incomplete.

## Common weighted affine table construction

Moved construction of w*(f-1/2) and the unweighted max(|f|,|1-f|) digit bound
from BFV sampling to ValueReadout::weighted_centered_table. It produces the
existing RowLutTable<C> and the bound directly, without a new wrapper or trait.
The target modulus and public weight are explicit inputs. BFV supplies its
terminal output modulus, applies its indexed/shared weight bound, and calls the
existing sampler.

Integer operation order, modular representatives, table identity, and fresh-error
scaling are unchanged. Inverse-of-two remains inside public target construction,
before encryption. The new common error variant represents the same inverse
failure previously mapped directly to BfvError::Lut; no validation scan was added.

The existing weighted readout noise-multiplier fixture passed once on CPU in
3.99 seconds. Release compilation was warning-free; formatting and diff checks
passed. No GPU/integration tests ran. This extracts public mathematics only;
complete common readout orchestration and the remaining five-file/setup/noise/
operational-proof requirements are still pending.

## Common row-table sieve sum

Moved the public sum sigma(mono(-index))*row[index] from the BFV terminal
sampler to RowLutTable::sieve_sum. The method uses existing GraphValue, scalar
multiplication, addition, and balanced_sum_family; it adds no carrier or trait.
The backend lowers row constants once and reuses them across branches, then
applies its existing normalization, RHS multiplier, routing, and sampler.

Term ordering, Family::pack, structural balanced reduction, and two-polynomial
row layout are unchanged. No new arithmetic, coefficient checks, or online
public work was introduced. Large-weight/raw-batch terminal relations passed
once on CPU in 1.65 seconds. After removing the obsolete import, the release
library unit-target build was warning-free in 27.77 seconds. Formatting and
diff checks passed. GPU and integration tests were not run. Common orchestration,
five-file consolidation, production setup, and full noise/proof work remain.

## DSL-owned aligned-family traversal

Added Family::zip_map to the existing DSL family type and replaced six repeated
aligned-family loops in BGG readout public preparation, public projection, and
online evaluation. The method preserves the existing symbolic-count equality
check and emits the same single parallel loop with paired member reads. It
introduces no wrapper, carrier, or trait. Callers retain their concrete product
or addition operation and existing error mapping.

The removed code performed no BGG-specific work: tuple bookkeeping, shape
alignment, and structural loop construction belong to the DSL. Input ownership,
member order, composite ports, and loop count are unchanged. No ciphertext
validation, decomposition, extra runtime pass, or host enumeration was added.

The existing imported-readout-family fixture passed once on CPU, including its
check that cached online evaluation emits no decomposition nodes. Release unit
compilation, including the modified DSL and dependent crates, was warning-free
in 33.89 seconds. Formatting and diff checks passed. No GPU/integration tests
ran. Further common traversal extraction and all remaining full-plan obligations
remain pending; this checkpoint is not five-file consolidation completion.

## Reuse aligned-family traversal in both BGG interpreters

Replaced eight repeated paired-family loops in public_key.rs and six in
encoding.rs with the existing DSL Family::zip_map. These cover public PRF
selector weighting/indexing, cached family products, branch alignment, and
family addition. The arithmetic closures, operand order, shape check, loop
cardinality, and cache ownership are retained. No new type, trait, or helper
was added in this checkpoint.

The release library unit target compiled warning-free in 28.17 seconds. The
existing structural public-PRF cache fixture and cached family-LUT encoding
fixture each passed once on CPU. Formatting and git diff --check passed. GPU
and integration tests were not run. These changes remove repeated traversal,
not the remaining BGG-specific production PRF path; the full five-file, generic
setup, noise-source, and operational-proof requirements remain incomplete.

## Remove BGG PRF readout test-adapter indirection

Collapsed the test-only readout batch wrapper, inner wrapper, and core into one
method calling the existing scalar readout operation in one structural loop.
Removed the singleton Vec of families and its redundant empty/alignment checks,
per-member Vec collection, and pop. Collapsed the test artifact producer wrapper
into its benchmark-role closure and removed its unused coefficient-bound input;
actual preimage bounds continue to come from the decomposition schema.

No production evaluator behavior changed. Test helper sigma binding checks,
benchmark roles, scalar readout coverage, and preprocessing artifact checks are
retained. Six focused BGG PRF unit fixtures passed once on CPU. Release unit
compilation was warning-free in 27.76 seconds; formatting and diff checks passed.
No GPU/integration tests ran. Production PRF migration and the full generic
setup/readout, five-file, noise-source, and operational-proof work remain pending.

## DSL-owned structural gather

Added Family::gather using the existing parallel loop and indexed member read,
and replaced nineteen repeated gather blocks in BGG encoding (7), public-key
preparation (6), readout (4), and PRF input slicing (2). The index family remains
structural; its order, repeated-source sharing, and count determine the output
exactly as before. No extra shape/range validation, scan, data transfer, or
carrier was added. Public index-range contracts are unchanged.

Release unit-target compilation, including DSL dependents, was warning-free in
33.53 seconds. Three relevant CPU fixtures each passed once: structural public
PRF cache, cached family-LUT encoding, and imported readout without decomposition.
Formatting and diff checks passed. No GPU or integration tests ran. The common
primitive removes traversal duplication but does not complete production PRF
migration or the remaining five-file, setup, noise-source, and Lean obligations.

## Shared traversal in reductions and refresh setup

Reused Family::gather in four balanced/segmented reduction gathers and two
refresh routing gathers, and Family::zip_map in the setup digit-offset pairing.
No helper, type, or capability was added. Index expressions, public metadata
checks, odd-tail selection, balanced addition order, and structural slot axes
are unchanged. The shared application and backend now use the same DSL
traversal instead of maintaining copies.

Both targeted utility CPU tests passed once: odd-width segmented reduction and
public-refresh slot-axis preservation (0.05 seconds total). The release library
unit-target build was warning-free in 27.28 seconds. Formatting and diff checks
passed. No GPU/integration tests ran. Concrete setup dependencies and the wider
five-file, PRF/readout, noise-source, and operational-proof requirements remain
unfinished; this checkpoint only removes shared traversal duplication.

### Shared BFV raw accumulator representation

Replaced `BfvTerminalPublicAccumulator` and `BfvTerminalEncodingAccumulator`
with `BfvTerminalAccumulator<T>`, deleting the duplicated metadata and combine
implementation without compatibility aliases. Public preparation instantiates
`T = BfvMatrix`; online evaluation instantiates `T = BfvRow`. These remain
statically distinct, and finalized outputs retain separate types. Addition order,
stage identities, context checks, and aggregate-only ModDown placement are unchanged.
This is a representation simplification; backend readout removal and the five-file
backend consolidation remain outstanding.

Validation: `cargo test -r -p mxx-exponent-lut --lib
test_bfv_terminal_rows_large_weights_and_raw_batch_relations` passed once on CPU
(1.65 seconds; warning-free build). The existing fixture checks emitted ModDown
counts and weighted aggregate decryption relations. Formatting and diff checks passed.

### BGG fixed-product support in the shared PRF

BGG now implements `ExponentLUTFixedRhsScheme` using the existing
`SetupRhsFamily` and `Family<Preimage>` types. The shared PRF composes public
label weights over the active selector range before each private product.
Public preparation decomposes the composed GSW family once; online evaluation
uses those cached preimages. No PRF-specific cache structure or new modulus
switch was introduced. Existing family views retain structural batching, and
setup layout provenance survives fixed-RHS composition.

The new CPU fixture `test_bgg_common_prf_composes_fixed_rhs_before_decomposition`
passed once after correcting a structural-count mismatch (warning-free build,
0.67 seconds). It executes two selection steps, three output labels, a nonzero
active-range start with nonzero inactive selectors, and signed exponent labels.
It checks exact zero-error encoding relations and one structural decomposition
body per step, with no block ModDown. This fixture does not certify nonzero-error
moments or replace the existing production BGG setup/artifact driver; migration
of that driver and its noise model remains outstanding.

### BFV readout uses one authoritative public artifact

Removed the scalar readout's duplicate cached terminal public key and its
`.terminal-output` artifact. The readout `.output` owns the finalized matrix;
online finalization uses that same handle and the terminal's existing stage
binding. Removed the duplicate outer family `.output` artifact as well: family
imports reuse the terminal preparation's `.terminal.output`. Cached affine
offsets and indexed weights remain stored, with no repeated public arithmetic
or ModDown. These artifact schema removals intentionally require regeneration,
without legacy imports or duplicate ports.

Validation for readout artifact deduplication: the existing CPU
`test_bfv_aggregate_readout_survives_artifact_reopen` passed once (3.26 seconds),
covering scalar and shared/indexed aggregate readout. The centralized library
build was warning-free (28.64 seconds).

### Covariance-safe Add bound without new source wrappers

`NonnegativeRational::sum_variance_bound` now computes an outward rational bound
on `Vx + Vy + 2 sqrt(Vx Vy)`, capped by the previous `2(Vx + Vy)` bound.
BGG's existing Add transfer consumes this utility. It preserves `4V` for equal
coherent operands, leaves `V` unchanged when adding a deterministic operand,
and improves the coherent `V,9V` bound from `20V` to `16V`. Deterministic error
bounds and the existing opt-in average-case acceptance policy are unchanged.
The three CPU `covariance_tests` passed once. This does not establish sampler
source identities or complete source-aware moment propagation; the existing
H3 branch-summation heuristic remains a separately declared model.

### Five-file backend boundary investigation

The user's current target supersedes the original three-file suggestion:
`encoding.rs`, `public_key.rs`, `mod.rs`, `noise.rs`, and `gpu.rs` for each backend.
Dedicated backend PRF/readout modules are not necessary. Their mathematical
operations already belong to the encoding/public-key implementations; their
application traversal belongs to common Program/readout/refresh code.

| Owner | Responsibility |
| --- | --- |
| `encoding.rs` | Encoding carriers, online-secret sampling, cached online products, row operations and final encoding ModDown where required. |
| `public_key.rs` | Public keys, payload-dependent fixed material, public products, prepared decomposition/branch caches, and public target preparation. |
| `mod.rs` | Scheme configuration, errors, exports, and small trait adapters using the two operation implementations. |
| `noise.rs` | Backend transfer equations, coordinate conventions, and sampling/rounding models. |
| `gpu.rs` | GPU-specific code and fixtures. |

Concrete extraction candidates:

- BGG `compile_select_batch_with_preprocessing` and PRF batch wrappers repeat
  Program binding, active-range traversal, and cache indexing. Move the traversal
  to common Program/PRF machinery while retaining typed decomposition operations
  in public-key preparation. Preserve flat label/helper families, shared switch
  material, and output-indexed public caches.
- BGG readout's target scheduling, shared-helper gathers, identity inclusion,
  branch reduction and finalization order can be common. Its gadget/preimage
  equations remain basic public-key/encoding operations, not a separate readout
  backend. Generic artifact transport owns manifest traversal; BGG owns actual
  preimage schemas and cryptographic bindings.
- BFV ordinary and terminal family adapters reconstruct the same scalar material
  and `PreparedBranch` data around `prepare_lut_raw` / `evaluate_lut_raw`.
  Consolidate this reconstruction using existing types. Preserve shared lifts
  outside indexed bodies, ordinary external-range indexing, and terminal
  aggregate-before-ModDown semantics.
- BFV `impl PreparedSparseLwrPrf<BfvScheme>` in `program_artifacts.rs` loops over
  steps and wraps each as `PreparedGate::FamilyMulPrivate` only for transport.
  Move that traversal to common fixed-product artifact handling rather than
  retaining an application-specific backend exporter.

The new BGG fixed-RHS path is not a drop-in replacement for existing production
PRF batching. Production `sample_label_from_preimage` forms online-secret label
encodings `(sG)K_A - x(sG) + e`; existing `MulPublic` combines those encodings
with the private accumulator. Direct weighted-GSW composition instead evaluates
`c D(sum w C)`. Changing between these paths changes public relations, secret
bindings and error sources. Generic production traversal should first preserve
the existing operation sequence. Any later replacement requires its own
algebra, artifact and noise justification.

Measured BGG production size excluding all `cfg(test)` items is approximately
7,102 lines across `mod`, `encoding`, `public_key`, `prf`, and `readout`, including
comments and blanks. Concatenation cannot meet the approximately 2,000-line
per-file guideline; common extraction and deletion are required. Current BFV
counts excluding inline test modules are 1,877 (`mod`), 1,526 (`readout`), 1,360
(`program_artifacts`), 826 (`program`), and 482 (`family`). These counts are
inventory evidence, not evidence that five-file consolidation is complete.

### Shared fixed-product PRF artifact traversal

Moved PRF export/import and descriptor identity into `noise_refresh/prf.rs`.
`PreparedSparseLwrPrfDescriptor<B,D>` replaces `BfvPrfDescriptor`; it stores the
program identity, one backend binding, and ordered operation descriptors.
Existing `ExponentLUTFixedRhsScheme` now supplies single-operation artifact
transport and its concrete metadata types. No additional capability trait was
introduced. BFV retains its general gate descriptor and fixed-product field
exporter; Program and PRF both call the same exporter without cloning a prepared
operation into a temporary Program gate for export.

BGG supplies fixed-product bindings and typed preimage/public-family artifacts.
Its prepared cache uses the existing pair of preimage family and public Program
value, preserving cached public output on import. The common descriptor and
step traversal are used by both backends; the older production BGG setup path
has not yet been migrated to composed selectors.

### Shared BFV family branch reconstruction

Ordinary and terminal family operations now use `BfvLutMaterialFamily::at` and
`PreparedBranch::from_parts`. Ordinary online evaluation calls the existing raw
LUT operation directly, removing the temporary `BfvPreparedLut` and an unused
output-public-key family read. Shared terminal lifts stay outside the family
body; ordinary external range indexing and terminal local indexing remain
distinct. Ordinary per-member and terminal aggregate-only ModDown schedules
are unchanged. The structural active-range and terminal-family/scalar CPU
fixtures each passed once (0.11 and 0.50 seconds, respectively).

### LUT RHS contract clarified by the user

No compatibility with a post-sampling switch-weighting path is required.
For LUT RHS, the only supported construction is to sample each switch freshly
with plaintext `d_k * sigma_k(rhs)`. For `rhs = X^(Bv)`, the branch sum computes
`L_f(X^(u+Bv))`. The existing BGG helper sampler already implements precisely
this construction; its fresh-error contract is now explicit in documentation.
Fresh switch error is not multiplied by the RHS weight. The actual decomposition
action on inherited errors is still included in the noise transfer.

This supersedes any implication above that the old online-label encoding path
must be preserved indefinitely. Direct composed RHS accumulation followed by
one final digit LUT can preserve PRF semantics. Each transition should not
acquire a nonlinear digit LUT unless the intended PRF itself changes. Migrating
production to the composed path must remove obsolete label-encoding error terms
and account for every participating selector-ciphertext error; it need not
preserve the old artifact layout or old ciphertext/noise values.

Validation: the BGG common PRF cache reopen fixture passed once (0.98 seconds)
after passing the reopened manifest to consumer graph validation. Its producer
has two structural decomposition bodies; its imported consumer has none.
The BFV native-PRF-digits refresh artifact fixture passed once (119.92 seconds),
including PRF export/import, refresh, and a following LUT. Central release
library compilation was warning-free (28.06 seconds). No integration or GPU
tests were run in this intermediate checkpoint; formatting and diff checks passed.

### BFV five-file placement and one inherent Scheme block per file

BFV now contains exactly `mod.rs`, `encoding.rs`, `public_key.rs`, `noise.rs`,
and `gpu.rs`. The former `family`, `program`, `program_artifacts`, and `readout`
modules were removed without compatibility namespaces. Sampling and online
operations live in `encoding.rs`; public preparation and artifact operations
live in `public_key.rs`. Shared relation/cache carriers and capability adapters
remain in `mod.rs`. Tests moved beside their operation implementations, and GPU
fixture imports follow those locations without changing device execution.
Each of the three implementation files has one inherent `impl BfvScheme` block;
trait implementations remain separate.

A read-only comparison against the pre-move snapshot accounted for all 296
named functions/methods: 268 bodies were identical after whitespace normalization;
remaining differences were path/import rewrites. The 20 ModDown calls, 17 centered
extensions, 6 fresh-error-row calls, 64 structural loops, and 31 test declarations
were retained. These comparisons support the structural move; runtime validation
is recorded separately below.

Current line counts excluding complete inline test modules (comments and blanks
included) are 2,200 in `mod.rs`, 1,166 in `encoding.rs`, and 2,539 in `public_key.rs`.
The shared carriers/adapters and remaining Program/artifact orchestration still
need reduction/extraction to satisfy the approximate per-file size guideline.
Five-file placement is not completion of backend-independent orchestration.

### BGG PRF uses existing family carriers

Removed `NativePrfLabelInputs`: label selection now constructs the existing
`BggEncodingFamily` directly. Removed `NativePrfEncodingBatchOutputs`: PRF and
readout outputs now use `Family<Mat>`, whose count is authoritative. Readout uses
its prepared artifact's output-count contract rather than a second count in an
output wrapper. `PublicKeyWireFamily` also derives its count from its matrix
family instead of retaining a duplicate field. Renamed the general BGG prepared
readout cache to `BggPreparedReadout`, removing its misleading PRF-specific name.
Updated CPU/GPU fixture consumers without executing integration or GPU tests.

Validation: warning-free release library build (28.51 seconds), six existing
BGG PRF unit tests (all passed once), BFV all-gate Program artifact reopen
(3.72 seconds), and BFV aggregate readout reopen (4.41 seconds). Formatting and
diff checks passed. The full production BGG composed-selector migration,
remaining shared orchestration, source-aware noise models, and operational Lean
certificates remain outstanding.

### Consolidated backend placement and lowering binding contracts

- Both BGG and BFV now use `encoding.rs`, `gpu.rs`, `mod.rs`, `noise.rs`,
  and `public_key.rs`. BGG PRF/readout declarations and methods were relocated
  without changing operation order, cached decompositions, or structural loops.
  External imports use the actual declarations at the BGG module root.
- Each BFV file has at most one inherent `impl BfvScheme` block. Trait
  implementations remain separate.
- Removed `backend_layout` and `public_attribute_available` from
  `ExponentProgramLowering`, all implementations, and the dependency decorator.
  Matching layouts and public-input plaintext metadata are preparation/binding
  contracts, documented at the lowering interface and runtime bindings.
  Removed the repeated public-attribute query before scalar multiplication too.
- Broadcast is still represented by the existing lowering method. Its removal
  requires common family operand/range handling while retaining the same source
  identity and coherent reduction variance; no replacement trait was added.
- Validation: warning-free CPU library unit-test compilation; six relocated
  BGG PRF tests passed once before the query removal; all fifteen Program unit
  tests passed once after it. Formatting and `git diff --check` passed.
  No integration or GPU execution in this checkpoint. Broader generic production
  setup, source-aware noise, and operational Lean migration remain incomplete.

### Shared scalar/family value carrier

The previous checkpoint made authoritative progress by deleting lowering
validation queries and consolidating backend placement. This checkpoint replaces
six independently implemented scalar/family enums with `program::ProgramValue`:
BGG public RHS, online RHS, noise wires and noise RHS, plus BFV public-key and
encoding Program values. Existing role names remain type aliases, so the
cryptographic payload distinctions stay visible without separate enum
implementations. No new capability or conversion trait was introduced.

BFV operation material now has two operation variants, `Lut` and `MulPrivate`,
each carrying the same common scalar/family representation instead of repeating
four operation/storage variants. Scalar known-plaintext metadata remains beside
its key/encoding in the scalar payload; indexed metadata remains with its family.
The scheme still exposes only its existing `PublicKey` and `Encoding` types.

Broadcast scheduling is not yet migrated: BFV currently builds its structural
broadcast once at family entry and reuses it. Recreating it inside every gate
would regress graph work. The remaining common-reference migration must preserve
that reuse or remove materialization through shared/indexed payload storage.
BGG broadcast source identity and coherent family noise reduction are unchanged.

Validation: warning-free CPU library unit-test compilation (27.58 seconds), one
passing run each of BFV active-range structural selection (0.14 seconds), BFV
all-gate/public-value artifact reopen (2.67 seconds), and BGG composed fixed-RHS
PRF artifact reopen (1.00 second). Formatting and diff checks passed. No GPU or
integration execution was added. Shared production setup, complete noise-source
moments and operational Lean certificate migration remain outstanding.

### Shared family payloads and shared fixed RHS execution

The preceding goal turn made progress by replacing duplicated scalar/family
value enums. This turn connects that representation to actual graph storage
and execution rather than only changing type declarations.

- `ProgramValue<T, Family<T>>` implements structural member access and existing
  DSL `GraphValue`/schema transport. A shared member returns the same graph
  handles. BFV family keys, encodings, public multipliers and known plaintexts
  use this storage: broadcasting no longer constructs a repeated DSL family.
  Shared reductions multiply by the symbolic active count; indexed reductions
  retain the existing balanced sum.
- One generic `FamilySource` now describes wire and RHS references. A family
  RHS can refer to an existing scalar binding with `Broadcast`; dependency
  tracing reuses that binding's original node. `FamilyGate` carries its actual
  `ProgramFamilyId`, avoiding any inference from step numbers to active ranges.
- BGG direct family multiplication accepts shared RHS and prepares one gadget
  decomposition outside the member loop. Its cache uses common scalar/indexed
  storage. Batch output sharing and member sharing remain separate axes.
- BFV direct family multiplication accepts shared or indexed material through
  the existing `ProgramValue` carrier. Shared RHS lift occurs once outside the
  product loop; online evaluation consumes that cached lift. Indexed RHS
  preparation retains the existing active slice and lift schedule.
- BFV operation/public-value descriptors preserve physical shared/indexed
  storage separately from logical family ranges. BGG cache artifact metadata
  likewise preserves member sharing. No export-time broadcast materialization,
  compatibility format, or replacement broadcast capability was introduced.
- BGG shared-RHS noise uses the existing coherent family bound/variance. BFV
  deterministic transfers already include the full fresh RHS error per member
  and apply multiplicity at reduction; no extra family multiplier is applied
  to the per-RHS fresh-error gain. Full BFV moment/source tracking is still
  outstanding.

Validation was limited to affected CPU unit coverage, with each final version
run successfully once: Program tests (15), BGG PRF tests (6), BGG coherent
shared-RHS noise, BFV repeated-source active selection, BGG shared-RHS file
artifact reopen, and BFV all-gate artifact reopen including shared RHS. The
last two ran in 0.41 and 2.73 seconds. The BFV fixture initially referenced a
composite output without its column suffix; the test was fixed to compare both
actual polynomial ports before its successful run. Final library compilation
was warning-free (27.61 seconds); formatting and diff checks passed.

The BGG reopen fixture verifies one producer decomposition, a scalar cache
shared across two consumer outputs, no consumer decomposition, and exact
residuals. BFV checks original handles across two family scopes, root-level
shared block lifts, restored storage, and equality with scalar products. No
GPU or integration tests ran at this checkpoint.

`broadcast_wire` still performs backend metadata conversion and has not yet
been removed from the lowering interface. Remaining common family dispatch,
shared production setup, complete moment/source traces, and operational Lean
certificates are not complete.

### Whole-body family lowering: graph transport foundation

The revised target is one common DSL `parallel` around each complete family
body, with scalar scheme operations, lexical captures, and indexed inputs.
Per-gate range parameters and backend broadcast conversions are not the target.
The implementation has not yet migrated the production traversal to this form.

Existing BFV scalar keys, encodings, private-product caches, linear caches, and
LUT caches now implement DSL `GraphValue`. Their existing parameterized records
also represent schemas, retaining identities and levels without adding a
parallel hierarchy of descriptor wrappers. BGG uses its existing scalar wire
schemas and now transports its existing operation maps and LUT cache records
through the same interface. Scalar BGG caches are keyed by the complete gate
instance, including the family body gate index.

DSL optional fields and ordered maps now support graph transport: absent fields
emit no dummy ports, and map keys remain static schema metadata. The graph test
returns a prepared map from one loop and consumes it from another, including an
active input offset and a captured shared matrix. BFV's focused CPU fixture
returns scalar prepared records from a public loop and indexes them in online
evaluation, comparing results with the ordinary scalar arithmetic. Its shared
two-by-two RHS lift consists of four root-level conversions and no inner-loop
conversions.

BGG scalar private-product preparation can reuse an outer prepared RHS from
its existing cache. Shared cache fields must ultimately remain in common outer
storage rather than being persisted repeatedly as indexed loop outputs. No
additional LUT seed map was introduced; shared switch preparation still needs
to be integrated with the common loop's cache ownership.

The migration must also replace the production BGG public selection batch path,
which bypasses the current common Program traversal. Updating `PreparedProgram`
alone does not cover that path. Current backend family enums, family cache
variants, and `broadcast_wire` are still present; the transport changes are
prerequisites rather than evidence that those interfaces have been removed.

Validation: one successful CPU run each of the DSL prepared-map graph test,
BFV scalar cache crossing test (0.05 seconds), and the extended BGG shared-RHS
fixture including artifact reopen (0.40 seconds). The latter also verifies one
outer RHS decomposition and none inside the scalar body. Final exponent-lut
library compilation was warning-free (28.39 seconds); formatting and diff
checks passed. No GPU or integration tests ran in this step.

### Production BGG selection body uses shared scalar dispatch

`program::lower_body` now owns the SSA traversal of a resolved family body.
The existing Program interpreter and production BGG public batch preparation
both use it. The batch path receives the actual accumulation Program rather
than reconstructing a separate two-gate arithmetic schedule.

Each BGG selection step prepares its setup-fixed selector decompositions once
outside the output axis. One label-major structural loop resolves state and
public member references, executes the complete Program body through scalar
gate operations, and returns its output and local cache. The shared selector
entry is removed from that local cache before it becomes a family output;
only the output-dependent decomposition is persisted with the output axis.
Existing cache names, label-major order, and segmented reduction are retained.
Intermediate output/local-index families, gather chains, and separate arithmetic
loops for the two gates have been removed from this production path.

This is an actual production-path migration, but not completion of common
family lowering: loop construction still resides in the BGG batch adapter,
and the generic Program/BFV paths still use their old family carriers and
`broadcast_wire`. The scalar body dispatcher is shared with symbolic traversal;
whole-loop binding/cache ownership and backend-family removal remain required.

The affected Program tests passed (15). The CPU production PRF unit fixture
passed with its real preprocessing, file-artifact reopen, cached online
accumulation, and readout (54.73 seconds). Noiseless execution verifies the
arithmetic/artifact relation; it is not a replacement for noise-source proofs.
The batch graph fixture checks both scalar products in one scope, with only
the selected-state decomposition in that scope and shared selector cache
remaining outside the output-indexed cache.

### Production BGG online body and common reductions

The production online batch now resolves scalar state/member references and
prepared cache entries directly in one label-major loop per accumulation step.
It calls the same `lower_body` dispatcher as public preparation. Shared cache
entries retain their shared axis; output-dependent entries use their existing
flat label/member index. The initial scalar encoding retains its plaintext
metadata without constructing a broadcast family. The obsolete single-label
compiler and label-gather helper have been removed.

Common scalar and segmented sums now use one balanced reduction implementation.
Each tree level uses one structural loop with direct indexed reads, replacing
the separate left/right index families and gather loops. Odd tail members remain
unchanged, addition order is preserved, and no ModDown moves into the reduction.
The existing width-five graph fixture verifies three loops for three levels;
CPU arithmetic coverage compares both scalar and segmented reductions at widths
one, three, and five against ordinary scalar addition.

Validation: warning-free CPU library compilation (27.89 seconds), one passing
run of the two shared reduction tests, the extended online batch graph test
(two small-RHS products per body scope and zero decompositions), BFV's two-step
active-range/signed-label PRF fixture (0.22 seconds), and the production BGG
noiseless PRF file-artifact round trip (37.72 seconds). The preceding production
run took 54.73 seconds; these single randomly sampled runs are observations,
not a controlled performance benchmark. Formatting and diff checks passed.
No GPU or integration tests were executed.

The batch adapters still construct their outer loops. Moving that construction
and binding/cache ownership to common graph lowering, eliminating legacy
backend family variants and `broadcast_wire`, and completing generic setup,
noise-source moments, and operational Lean certificates remain outstanding.

### Common graph family execution and local cache lifetime

`scheme::prepare_family` and `scheme::evaluate_family` now own the DSL parallel
scope, actual family-body lookup and dispatch, evaluator/cache lifetime, and
operation-error transport. They use only `ExponentLUTEncodingScheme` and the
common Program representation. Their callbacks resolve scalar inputs/material
and project graph fields; they do not implement the Program arithmetic or
traversal. No replacement broadcast trait or whole-protocol backend callback
was introduced.

Both production BGG batch adapters call these common drivers. Shared selector
decomposition remains an outer preparation. Only member-dependent cache fields
leave public execution as explicit family outputs. The first public state is
now captured directly as a scalar, just like the initial online state; a
repeated public-state family is no longer constructed before the first body.
No-step programs materialize the scalar only at the output boundary.

The existing BFV cache-scope fixture now instantiates both common drivers with
an actual Program Add body and scalar scheme operations. Only the body-local
linear cache and output key leave its public loop. Direct RHS preparation,
its four block lifts, and the direct online product remain outer captures,
instead of repeating the direct cache as family output fields. CPU output
comparison still uses the ordinary scalar arithmetic.

Scalar dispatch now receives only its two helper maps, rather than a complete
Program binding object with artificial empty root/family maps. Graph-construction
errors are represented separately from primitive-operation failures so their
context is preserved through the common driver.

This does not yet migrate `PreparedProgram` or every backend family carrier.
The current general traversal still calls `broadcast_wire`; BFV production
family cache ownership and the remaining generic setup/noise/proof obligations
are incomplete. The new common-driver BFV fixture is not evidence that its
entire production refresh path has migrated.

Validation: warning-free CPU library compilation (29.24 seconds); one successful
run each of the 15 Program tests, BGG public and online batch structure fixtures,
the updated BFV common-driver fixture (0.03 seconds), and the production BGG
PRF file-artifact reopen/online/readout fixture (38.06 seconds). Formatting and
diff checks passed. No GPU or integration tests ran in this step.


### Captured RHS preparation and common reference binding

`FamilyProgram::bind_inputs` now owns the traversal and capture/member dispatch
of both wire and RHS declarations. The common traversal and BGG public/online
batch adapters use it. BGG's frozen PRF binding contract still supplies its one
state/member/selector dataset; no extra validation or graph work was added.
This is reference resolution, not a replacement backend broadcast trait.

BFV's existing `BfvPrivateRhs` now records its actual preparation phase in its
type: sampled/composed low-base material has `L = ()`, while prepared material
has its required high-base lift as `L = C`. The DSL represents the absent field
with zero graph ports. No optional runtime readiness flag or new RHS wrapper
was introduced. Scalar RHS preparation can precede the entire Program body;
indexed preparation selects the active external range and emits its low/high
pair together. Private-product preparation consumes that pair without another
lift or selection. Cached artifact fields retain their low/high roles.

The BFV common-driver fixture now executes the actual `MulPrivate` followed by
Add inside both public and online loops. It exports the varying product key and
linear cache, then reconstructs the private cache with captured shared RHS and
input key. It checks four outer polynomial lifts, none inside either body, and
four public plus two online polynomial ModDown operations inside the bodies.
Ordinary scalar execution supplies the expected public and encoding values.

Sixteen existing BGG decomposition/transfer tests and their helpers moved from
refresh simulation into `bgg::noise::tests`. The gain-plan test remains beside
its private application state, without widening visibility or dropping its
assertions. Production simulation and setup are still BGG-dependent.

Remaining work includes general `PreparedProgram` migration, eliminating the
backend Family arithmetic/cache variants and `broadcast_wire`, completing
scheme-independent production setup/refresh, sampler-source moments, and
operational IR-derived Lean certificates. These changes do not establish those
requirements or complete the overall refactor.

Validation: warning-free CPU library build (28.23 seconds); 39 focused Program,
BGG reference/noise, BFV direct-product/common-body and all-gate artifact tests
passed once (2.23 seconds). The artifact fixture uses a nonzero-start active
range. Two additional BFV composed signed-PRF/selector-alignment tests passed
once (0.19 seconds), covering the changed fixed-product preparation caller.
The DSL explicit-family-cache test also passed with a zero-port phase field.
Formatting and diff checks passed. No GPU or integration tests were run.


### Shared public-cache ownership and fixed RHS bindings

Public evaluators now receive separate mutable local and immutable shared
instances of the existing scheme cache. `prepare_family` owns an initially
empty local cache; its binding callback supplies the read-only preparation seed.
BGG direct products look up local then shared decompositions and only insert
new preparation locally. The production projection no longer removes a shared
entry after the fact. The existing hoisting fixture checks that the output
cache has no copied shared entry and that the product uses the captured
preimage. No new cache wrapper or trait was added.

BFV fixed public RHS bindings now carry their actual prepared low/high operand;
fused LUTs can carry only the existing RHS identity because their fresh branches
already include the weighted plaintext. The old helper-material role enum was
replaced by this RHS role enum, and public helper material is now LUT-only. A
fixed operand also supplies its identity to a fused LUT, so callers need not
prepare a redundant ciphertext just to bind a fused table. Both roles retain
their existing identity checks and plaintext-weighted encryption semantics.

This still leaves scalar/family variants in the arithmetic adapters. The
shared/local cache split enables a future common `PreparedProgram` family owner;
it does not constitute that switch. BFV symbolic noise still takes direct
transfer material through its helper binding, so deleting the generic private
gate's material argument also requires moving that noise-model information to
its correct owner. No noise contribution was removed in this step.

Validation: warning-free CPU library build (28.84 seconds), followed by one
successful run of six focused BGG sharing/batch and BFV common-body, composed
PRF, all-gate binding and artifact-reopen tests (2.21 seconds). Formatting and
diff checks passed. No GPU or integration tests ran.

### Common Family execution and scalar primitive adapters

`PreparedProgram` now constructs each public/online Family body through the
common structural driver. Backend primitive associated values are scalar graph
values; the four-gate interpreter no longer exposes broadcast or reduction.
Captures retain their original handles. Body-local preparation leaves the loop
as an explicit `Family<Cache>`; shared preparation is kept separately and is
not copied into each member cache. The obsolete PRF compiler traversal and its
unused binding owner were removed.

DSL Family schemas now record shared versus indexed storage per flattened
field. Captured cache fields remain scalar physical ports, while body-local
fields are loop outputs. Artifact transport preserves those flags, including
standalone BFV descriptors. Packed wire, RHS, and helper families carry the external range they
represent, so active-only preparation resolves `external - represented.start`
without padding inactive slots or repeating lifts.

Symbolic traversal also captures the original dependency node directly. Each
backend interprets the active range in its noise transfer; aggregation retains
shared-source multiplicities through the scheme's noise reduction operation.
The redundant Broadcast dependency node was removed. This change does not
claim completion of the remaining moment/proof obligations.

The DSL library suite passed all 71 tests once. Focused exponent-lut CPU
validation is in progress. Common readout/refresh capability boundaries still
carry scalar/family values and remain follow-up work; scalar primitive migration
alone does not complete the original plan.

`PreparedTerminal` now takes scalar associated public/encoding types directly;
its BFV term operations previously rejected the Family variant anyway. This
removes that invalid branch without changing raw accumulation or finalization.
Production `PreparedRefreshTerminal` family scheduling is a distinct remaining
migration: shared lifts must stay outside the member loop, cached member
intermediates must remain explicit outputs, and all raw terms must combine
before the final switch.

Intermediate evidence: the exponent-lut library compiled warning-free in
29.88 seconds. Of the first 25 focused tests, 21 passed; four exposed the DSL's
rejection of metadata-only RHS families. The zero-port Family fix passed its
new metadata/count/schema round-trip test. Rechecking the affected paths plus
the scalar terminal changes passed five tests; the remaining Program export
failure was narrowed to publishing an empty shared cache as a standalone graph
output. BGG shared decomposition and production structural-cache tests both
passed once (0.36 seconds). Passing tests were not rerun without related changes.

Final validation for this migration: warning-free CPU library rebuild
(29.94 seconds), then six affected BGG/BFV binding, production structural-cache,
nonzero-range PRF, and full Program artifact-reopen tests passed once
(2.50 seconds). The export fix groups cache components in an existing tuple;
empty shared caches contribute metadata and no dummy artifact. The subsequent
range failure was fixed uniformly for wire/helper/RHS bindings, including
selecting the actual active helper members instead of their first entries.
The separate scalar-terminal relation and independent-online-secret artifact
tests had already passed after their API change. Formatting and diff checks
passed. No GPU or integration tests were run during this migration.

### Direct refresh uses common terminal Family scheduling

`PreparedDirectRefreshSlot` now owns scalar-versus-Family scheduling using the
existing `ProgramValue` carrier. Each member calls the same scalar terminal
preparation/evaluation as an ordinary term. Raw member fields are reduced before
cross-term accumulation and the existing single refresh switch. Scalar terms
remain scalar; no singleton Family is introduced for that fast path. Digit
routing supplies local sliced arrays, with matching material cardinalities.

The refresh-only terminal material/prepared associated types and three opaque
family callbacks were removed. Existing terminal records now implement actual
GraphValue schemas. Direct-slot artifact export/import uses those schemas
without backend transport callbacks, preserving cached public targets and
lexically shared fields.

BFV terminal material now retains its centered ciphertext lifts from public
preparation. Scalar and indexed preparation pass these handles into the existing
branch operation; online evaluation consumes the cached prepared branches.
Prepared branch artifact transport persists both high-base lifts rather than
reconstructing them after import. Raw accumulator schemas retain one logical
stage identity for a Family reduction, avoiding per-member metadata growth.
The streaming cross-term reduction now accepts graph and backend errors through
its existing terminal error type without an intermediate vector of raw values.

Standalone BFV Family readout scheduling remains to migrate; it is not used by
the new common direct-refresh slot scheduler. The broader generic setup,
noise-moment, operational-proof, and final workspace/GPU obligations remain open.

The first common-terminal validation passed six focused CPU tests, including
native PRF digit refresh artifact reopen and post-refresh LUT continuation
(110.68 seconds). Reviewing the emitted cache then identified unnecessary
per-member raw public outputs. Preparation now returns the existing prepared
record and raw accumulator separately; the record's raw field uses the existing
zero-port phase representation in the common cache. The scheduler projects and
persists only the cache, and streams raw terms into the reduction. This avoids
retaining the member outputs after their aggregate has been computed. The
updated phase/schema path requires its own focused verification before this
migration is considered validated.

The final cache-phase version rebuilt warning-free (30.73 seconds) and passed
all six focused CPU tests once (107.96 seconds, compared with 110.68 seconds for
the preceding common-terminal version). This includes actual refresh artifact
export/import, native PRF digits, exact fresh-error replacement, and next-LUT
continuation. Existing terminal fixtures now also verify cached ciphertext-lift
handle reuse and absence of centered-lift nodes in imported online evaluation.
Formatting and diff checks passed. No GPU or integration tests were run; the
original final workspace/GPU and operational-certificate requirements remain.

### Standalone Family readout joins common terminal scheduling

The previous step migrated direct refresh; standalone
`prepare_terminal_family`/`evaluate_terminal_family` still had their own loops
and restored branches from material/intermediate fields. Both now call the
same `readout::prepare_terminal_value`/`evaluate_terminal_value` used by direct
refresh. Independent projection executes inside the existing member loop;
aggregate projection executes once after raw reduction. This avoids introducing
a second mapping loop or changing ModDown placement.

The existing BFV Family prepared record now holds a Family of scalar online
caches instead of separate material/intermediate carriers. Its descriptor keeps
mode/range/identity and exact input/cache/public schemas. Export/import transports
those values directly; the old mask/ciphertext/intermediate reconstruction
format is removed. Per-member raw public outputs remain excluded from the cache.

The CPU library build is warning-free (31.12 seconds). Focused standalone
readout/terminal and shared direct-refresh artifact tests are in progress.

All five focused CPU tests passed (106.14 seconds), including shared direct
refresh with artifact reopen and next-LUT continuation. The Family artifact
review additionally removed the separately serialized input matrix: import now
projects that field from the actual member cache and restores its range metadata.
After this final transport change, the library rebuilt warning-free
(30.75 seconds) and the three affected standalone artifact tests passed once
(9.78 seconds). No extra member arithmetic or padding was introduced. Formatting
and diff checks passed; no GPU or integration tests ran. Generic production
setup, full noise-moment/proof migration, and final workspace/GPU verification
remain required by the original objective.

### Refresh digit routing and setup validation ownership

Removed `refresh_public_range` and `refresh_encoding_range` from the direct
refresh capability and their identical BFV implementations. Canonical label
routing now selects both public and encoding digit groups with one generic
structural operation. Fresh digits are still selected once outside the CRT-slot
loop and reused; member handles and the existing parallel slice graph are
unchanged. The required Family input is checked once per preparation/evaluation,
rather than once per group. No new carrier or capability trait was introduced.

Removed setup validation's redundant public scaling, modulus-switch and witness
product graph construction: it compared only matrix types already checked in
the preceding loop, not algebraic target bindings. Removed its anchor lookup
map and a duplicate state-type check. Retained declaration identity, canonical
artifact roles, counts and ring/shape checks. Documentation now describes this
as structural validation rather than claiming it verifies setup equations.
The focused CPU setup test passed once (0.07 seconds), after a warning-free
29.83-second build.

The setup topology audit found that candidate parameters remain the sole shape
authority for BGG's graph-free parameter search, declaration identity and import.
The existing key-derived generic topology constructor should not be introduced
as a competing authority only at production call sites. Full backend parameter
migration remains necessary; this cleanup does not claim generic production
setup is complete.

The common digit-routing change compiled warning-free (29.98 seconds). The
existing CPU native-PRF artifact refresh fixture passed once (103.50 seconds),
including independent online preparation after reload, exact refreshed fresh
error, and continuation through the next LUT. No GPU or integration tests ran.

### Remove the BFV terminal material Family adapter

Deleted `BfvTerminalFamilyMaterial` and its hand-built Family schema/port
conversion. Readout material, terminal preparation and direct refresh now use
`ProgramValue<BfvTerminalMaterial, Family<BfvTerminalMaterial>>` directly.
Indexed sampling returns complete scalar material from the existing single
parallel body instead of splitting masks, ciphertexts and high lifts and then
reassembling them. Hash tags, fresh samplers and centered-lift operations are
unchanged. Binding metadata comes from the existing scalar material schema;
there is no new carrier, trait, graph operation or per-member host expansion.

The indexed terminal artifact fixture now also checks that two member views
retain the original mask bank's graph handles. Existing nonzero-range,
independent/summed output and fresh-online-secret reload checks are retained.

The final material-carrier version compiled warning-free (30.74 seconds).
All three focused CPU tests passed once (6.60 seconds): terminal Family versus
scalar members, indexed terminal artifact reload with new online secrets, and
indexed readout weights after reload. Formatting and diff checks passed.
No GPU or integration tests ran. Full generic production setup, moment/proof
migration and final workspace/GPU validation remain outstanding.

Next setup boundary identified by source audit: `BggPreparedReadout` currently
owns the four typed preimage ports and their metadata, but setup independently
reconstructs them during import and independently derives their manifest types.
Migrate that existing record to `GraphValue`, persist its actual schema in the
existing declaration identity, and use schema-based export/import and manifest
validation. Preserve confidentiality, content hashes, exact family counts and
lazy artifact loading; `artifact_input` alone does not validate the manifest.
This is not implemented yet. Keep graph-free candidate parameters/noise planning
as the current single authority until their backend parameter boundary is
migrated; do not add a competing key-derived shape check at only some call sites.

### BGG readout artifact schema owns transport

`BggPreparedReadout` now implements `GraphValue` using its existing record
parameterized over field types; the same record is its schema. A graph-free
backend constructor derives canonical helper sigmas, three Family shapes, one
scalar preimage shape, and bounds from resolved setup geometry. Setup compares
the actual prepared schema once, exports the record under one prefix, and
reopens it via `GraphValueSchema::artifact_input`. Manifest checks consume its
wire types and retain exact Family counts, public confidentiality and content
hashes. The four-field names record and declaration's duplicate helper-sigma
list are deleted. Schema is deterministic from already identity-bound geometry;
there is no new unbound serialized schema or second setup identity.

The typed coefficient bound retains `Mat::decompose`'s `RoundDiv(base, 2)`
expression, while numeric metadata retains its evaluated integer. Comparing a
constant integer bound with that symbolic expression initially failed schema
equality; the constructor now preserves the actual expression instead of
relaxing the schema check. No polynomial evaluation or extra graph nodes are
needed for this construction or check. Transport remains lazy with four ports.

The graph-free candidate snapshot test passed. After bound-expression alignment,
the actual switched setup graph/import test passed; the library build was
warning-free (29.26 seconds). Existing Lean readout consumers were mechanically
updated to the new port suffixes without changing theorem statements. The
concrete refresh export test exposed a remaining proof migration gap:
`lean/root_private.rs` expects separate gather and decomposition loops, while
`bgg/public_key.rs` already fuses indexed selector access and decomposition.
Its old witness extraction cannot certify the current body; removing node-count
checks or weakening the theorem would not fix it. IR-linked proof migration
remains incomplete, and no successful implementation certificate is claimed.

The focused CPU file-store test passed once (0.85 seconds), after a warning-free
30.21-second build. It uses actual `BggScheme::prepare_readout`, serializes its
schema, exports all four typed ports, closes/reopens `FileArtifactStore`, and
compares every reopened preimage value against the producer runtime output.
The consumer graph contains no `GadgetDecompose`. Formatting and diff checks
passed. No GPU or integration tests ran; the concrete Lean export failure above
remains outstanding and is not counted as a successful proof check.

### Fixed selector transfer lemmas and checked execution-fact access

`lean/ExponentSetup.lean` now owns `sampled_selector_cache` and
`sampled_selector_private_step`. They derive the cache equation, dense action
gain, and private-product error bound from sampled ciphertext support and the
actual decomposition relation. The existing graph-specific wrappers apply these
fixed lemmas instead of generating their algebraic proof bodies. This is an
intermediate migration: the remaining Rust proof wrappers and topology matching
have not yet been removed.

The core exporter exposes named, kernel-checked node/output fact accessors for
record-backed scopes through `ScopeProofShape`. Linear suffix definitions keep
conjunction projections inside ir-core without quadratic generated projection
strings. Selector root proofs consume these accessors rather than reconstructing
conjunction positions. Existing execution propositions and witness ordering are
unchanged. Leaf scopes still need equivalent access before the fused private
cache proof can be migrated without another application-specific proof template.

The core projection fixture passed its Rust test and Lean kernel check, including
both directions of operational equivalence. The actual production selector
preprocessing export passed its targeted Rust test once (0.08 seconds after a
warning-free 30.17-second build); both generated `SelectorPreprocessing` and
`SelectorSetupScopes` modules passed Lean checking. The two fixed lemmas and all
five generated selector/root theorems report only `propext`, `Classical.choice`,
and `Quot.sound`. No admitted lemma or assumed output bound was introduced.

These checks cover selector sampling, source augmentation, and the sampled
private transfer. They do not establish the full refresh implementation claim;
the fused-cache operational bridge, remaining proof-generator removal, generic
production setup, noise-moment migration, and final workspace/GPU validation
remain outstanding. No GPU or integration tests were run for these proof-only
changes.

### Leaf proof views and the actual indexed selector cache

Core fact access now also covers leaf scopes. `runs_iff_body` proves both
directions between each existing execution relation and its named witness/body
view. The legacy leaf existential telescope is reconstructed from recorded
binder spans during the same node emission; no second graph traversal or
application-specific node interpretation is introduced. The existing seeded-key
consumer checks actual loop nodes instead of treating witness-view availability
as a proxy for whether a scope contains loops.

Inspection of the exported production IR clarified the private-cache binding:
DSL normalization already turns selector access into `Zip`/`ZipOffset` loop
inputs. The child performs only gadget decomposition. `root_private.rs` now
uses those bindings and checked root/leaf facts, deleting its separate gather
and index-loop extraction. The obsolete intermediate `private_cache_N` theorem
is replaced by the fixed sampled cache lemma and the directly derived
`private_bucket_N`, which retains the concrete selected index, action gain,
cache equation, and error support. The remaining graph-specific wrappers are
still transitional; this does not complete removal of the Rust proof generator.

The captured-family/decomposition fixture passed its final Rust test and Lean
kernel check without warnings, exercising both directions of the witness-view
conversion. The actual selector/private-cache export passed once (0.11 seconds;
warning-free 39.93-second build). All three generated modules passed Lean:
`SelectorPreprocessing`, `SelectorSetupScopes`, and `SelectorPrivateRoot`.
All seven `private_bucket_N` theorems report only `propext`, `Classical.choice`,
and `Quot.sound`. No runtime graph operations changed.

The full concrete-refresh export was then attempted once. It passed the former
private-cache extraction failure and stopped at `missing balanced readout helper
sum`. The readout proof still expects the old reduction graph topology. This is
the next operational proof migration gap; the full refresh certificate remains
unverified. Both `root_readout_public.rs` and `root_readout_stage.rs` expect two
root gather-loop parents per reduction level. `utils::reduce_family_segments`
now captures one family and performs both indexed reads and add/select within
the same loop. The replacement bridge must derive that segmented pair reduction
from its leaf facts and preserve the odd final member unchanged, without
counting the wrapped right operand or introducing independent-source premises.
No GPU or integration tests were run.


### Shared operational readout reduction proof

Producer and online `normalized_execution` proofs now share one binding routine
in `root_readout_public.rs`; `root_readout_stage.rs` no longer duplicates the
balanced reduction traversal or its generated proof. Each level follows its
single captured family and actual internal indexed reads, using core
`runs_iff_body`, node facts, and the output fact. The migrated producer file no
longer constructs conjunction projections or unfolds numbered constraint
fragments.

`lean/ExponentReadout.lean` provides `segmented_pairwise_sum`, reusing `pairwise_sum` to
prove sum preservation for arbitrary segment widths. Its premise matches the
executed pair-or-carry branch: the odd tail retains only its left member, even
though the graph also reads an in-range wrapped right operand. The generator
now emits one quantified equality per level, rather than unrolling every lane.
The proof retains the same captured sources and introduces no independence
assumptions, extra samplers, or changes to arithmetic or ModDown placement.

The existing scope export fixture is now `test_export_refresh_scope_proofs`.
It exports both actual production graphs and checks the producer's normalized
public expression plus the online normalized execution expression. The final
Rust export passed once (0.15 seconds, warning-free 30.11-second build). The
fixed segmented-sum lemma builds without warnings and reports only `propext`,
`Classical.choice`, and `Quot.sound`. All four final generated modules passed
Lean checking without warnings: `RefreshPreprocessing`, `RootReadoutPublic`,
`RefreshVerification`, and `ReadoutVerificationReduction`. The producer's
`normalized_execution`/`normalized_public` and the online
`normalized_execution` theorems have the same three standard axioms only.
These are the artifacts regenerated by Rust after provisional proof debugging,
not hand-edited proof artifacts. No GPU or integration tests were run.

A full concrete-refresh export progressed beyond the former readout reduction
failure and stopped at `unsupported actual public cache ...body-1-k-rhs`.
`root_program_setup.rs` still expects a separate decomposition-only loop and
reads output port zero. The current whole-Program family body returns the
weighted public key on port zero and its decomposition cache on port one.
The next bridge must follow that actual port and decomposition operand, using
core facts and `gadget_runs_reconstruct`; `root_cache_alignment` and
`root_program_keys` also need their obsolete separate-loop assumptions removed.
This does not yet establish the complete refresh implementation certificate.

### Whole-Program public cache and key proof bindings

The public multiplication cache now binds to its actual exported port in the
whole Program body. `program_cache_binding` follows that port to the gadget
decomposition and its product operands; `parallel_operand` and
`parallel_arguments` resolve the existing captures, indexed lookups, and all
output ports. The initial public state remains a shared scalar, while later
state lookups use their actual Family indices. Public-cache reconstruction uses
core named node/output facts and the fixed `gadget_runs_reconstruct` lemma.

Cache alignment reuses that reconstruction instead of decomposing the same body
again. Producer key recurrence uses one quantified member equation and the same
segmented reduction proof binding used by public and online readout; it no longer
emits a separate proof for every member. The `key_after_N` statements retain the
actual public-label offset and cache relation. No runtime graph operations,
samplers, public preparation, or ModDown placement changed.

The focused production-graph fixture also exports `RootProgramSetup`,
`RootProgramKeys`, and `RootCacheAlignment`. The final CPU library build completed
warning-free in 29.55 seconds and that fixture passed once in 0.14 seconds.
Final regenerated Lean modules all passed without warnings:
`RefreshPreprocessing`, `SelectorSetupScopes`, `RootPrivate`,
`RootProgramSetup`, `RootProgramKeys`, `RootCacheAlignment`,
`RootReadoutPublic`, `RefreshVerification`, and
`ReadoutVerificationReduction`. The axiom reports for public-cache reconstruction,
public-input cache reconstruction, key recurrence, cache alignment, and both
readout reduction theorems contain only `propext`, `Classical.choice`, and
`Quot.sound`. No GPU or integration tests were run for these proof-only changes.

The full exporter advances past its former public-cache binding failure and now
stops at `route input not captured` in `root_refresh_source`. Its old reducer
recognizer expects two gathered parents and mistakes the current single-capture
sum for a product. The following route also includes an identity Zip-prefix
binding before the actual product. The next migration should reuse the same
segmented reduction binding and prove the existing identity slice using the core
loop input/output facts, rather than rebuild separate gather graphs. These
execution bindings still need migration;
this is not a completed refresh certificate or removal of the Rust proof-template
layer. Generic production setup and both complete noise/proof models remain part
of the original unfinished objective.

### Refresh route bindings after common Family lowering

The refresh-source proof no longer recognizes the obsolete two-gather reduction
shape or emits its own per-layer proof implementation. It uses the shared
segmented reduction binding and fixed pairwise-sum lemma. The reduction walker
is also shared with the route-sum consumer, so that both identify the same actual
raw Family.

Route views now resolve identity Zip/ZipOffset prefixes and dynamic indexed
views through one source/index binding. The generated view theorem retains an
existential source-index bound and the exact output equation. Public and online
product statements use their actual input modes, including a prefix of a longer
normalized readout Family. A direct readout capture does not require an invented
gather. Mask label offsets and the shared fresh-label mapping remain unchanged;
source approximation statements still use the original sparse-action gain times
the same readout bound and retain the actual cache link.

Public route multiplication already consumes the exported prepared cache. Its
proof now uses the actual multiplication/output equations and input view
lineage; the obsolete internal-decomposition, scale, and canonicality proof
duplication is removed. The separate canonical-cache theorem remains in
`RootRefreshRouteTargets`. Public route equalities and sums are backend-generic
and no longer take an unused gadget argument. The high-source consumer retains
its gadget only for the reconstruction theorem that actually needs it.

The extended focused CPU export passed after a warning-free 30.24-second build
(0.16 seconds for the test). It exports the actual public route targets, public
cache alignment, and route sums for dependent Lean validation. The full exporter
has advanced past the old reducer, mandatory-gather, and fictitious internal
decomposition failures; it now reports a missing actual private Program family
when binding the first private-cache artifact. These are intermediate execution-proof migrations,
not the completed generic production boundary or final implementation
certificate required by the plan.

Final route regeneration passed the same focused CPU export once after a
warning-free 30.42-second build (test: 0.14 seconds). The generated Backend,
RefreshPreprocessing, RefreshVerification, RootRefreshSource,
RootRefreshRouteTargets, RootRefreshRoutePublic, and RootRefreshRouteSums
modules all kernel-checked without warnings. All 60 route/source/cache theorem
axiom reports contain only `propext`, `Classical.choice`, and `Quot.sound`.
The checked generated files were emitted by the Rust source, with no manual
proof edits. Evidence: `exponent-route-scopes-final-rust.log`,
`exponent-route-scopes-final-lean.log`, and `exponent-route-axioms-final.log`
(ephemeral local logs). No GPU or integration tests ran for these proof-only
changes. This verifies the migrated route fragment, not the final linked claim.

The next Program binding migration must be atomic across `root_program.rs`,
`root_program_context.rs`, `native_program_proof.rs`, and their `lean.rs`
orchestration. The first cache-consuming loop is now the complete scalar
Program body, not an outer lane loop containing individual gate loops. Each
stage must retain its actual root output and resolve its child operands through
the existing parallel operand binding. The flat index is
`lane * members + member`; the lane count is the output count divided by the
private-cache member count. Its lane result uses the same segmented reduction
binding as readout. Existing private/public transfer and native Family
reduction lemmas retain the bound
`members * (privateGain * previousBound + cutoff + publicGain * publicBound)`.
One-hot payload selection does not remove the other members' ciphertext errors.
Changing only the failed loop matcher would leave the native execution,
preprocessing-cache, and sampled-public-input links incorrect.

### Online Program body and segmented reduction bindings

`root_program.rs` now binds the actual root-level whole-Program loop. Its stage
operands identify child wires, resolved through the common parallel binding;
the output identifies the real loop output port. The per-member theorem uses
the root execution witness and the original private/public transfer bound.
`root_program_context.rs` follows the actual ciphertext/message lookup facts
to the sampled public input at their common flattened index. It no longer
introduces a fictitious prefix or local gather position.

The Program reduction exporter now calls the shared segmented reduction proof
instead of expanding every member and separately emitting lookup and sum
lemmas. Its actual source-to-result mapping is passed to the native proof
consumer, removing that consumer's duplicate reducer discovery. Remaining
native proof code still needs its old lane-witness and cache-gather assumptions
migrated before the whole linked claim can be verified.

The final focused CPU export passed once after a warning-free 30.16-second
build (test: 0.16 seconds). The generated RefreshVerification, RootProgram,
RootInputs, EncodingInputScopes, RootProgramInputs, RootProgramContext, and
RootReduction modules kernel-checked without warnings. All 21 stage, public
context, and reduction theorem axiom reports contain only `propext`,
`Classical.choice`, and `Quot.sound`. Evidence: the ephemeral
`exponent-online-program-final-rust.log`, `exponent-online-program-final-lean.log`,
and `exponent-online-program-final-axioms.log` logs. The full export has passed
the former Program and reduction extraction failures and now stops at the
native proof's obsolete private-cache gather assumption. No runtime
cryptographic operations changed, and no GPU or integration tests ran.

### Native Program linkage after whole-body lowering

The native proof emitter now consumes root execution witnesses and the actual
stage-to-reduction mapping. Private sampler and public decomposition facts are
linked directly to their prepared artifacts; obsolete lane witnesses and
invented cache gathers are removed. The native recurrence still counts every
member's private and public ciphertext errors. Readout-to-refresh source links
use the actual typed indexed views, retaining distinct mask labels and shared
fresh labels.

The full concrete-refresh export passed once after a warning-free 29.21-second
CPU build (test: 0.23 seconds). The final regenerated dependency closure through
native Program, readout, refresh sources, and route sums kernel-checked without
warnings. All 42 native/readout/source/sum theorem axiom reports contain only
`propext`, `Classical.choice`, and `Quot.sound`. These files were emitted by Rust
without manual proof edits. Evidence is recorded in the ephemeral
`exponent-native-final-export-rust.log`, `exponent-native-final-lean.log`, and
`exponent-native-final-axioms.log`.

A separate accepted worst-case candidate also exported successfully (5.76
seconds; `exponent-accepted-current-export.log`). Its actual graphs produce
nonempty high-source and finish proofs and a `RefreshCorrectness` theorem
targeting `GeneratedClaim.CorrectnessClaim`. This is generation evidence only;
the accepted candidate's final theorem has not yet been kernel-checked. The
default small fixture rejects the strict rounding condition and cannot stand
in for this final certificate.

Remaining work includes replacing mathematical Rust proof templates with fixed
Lean lemmas, completing generic production setup/cache transport and common
noise reporting, and validating both backends' final production certificates.
No GPU or integration tests ran for these proof-only changes.

### Fixed private/public Program transfer

`ExponentProgram.lean` now owns `program_private_public_transfer`, composing
the private product and public product derivations with their exact state
identities and unchanged error bound. `root_program.rs` supplies its actual
scalar equation from the generated execution facts and applies that fixed
lemma, instead of emitting the transfer's mathematical proof body. The lemma
is generic in the secret dimension and does not assume its output error bound.

Direct Lean checking of the fixed module passed without warnings
(`exponent-fixed-transfer-lean.log`). The regenerated application also
kernel-checked. Whole-cache transport made the public cache index definitionally
equal to the exported index; the native proof now uses that equality directly
instead of retaining a redundant arithmetic tactic. The corrected native,
readout, source, and route-sum closure passed provisionally. A final unedited
Rust export passed after a warning-free 31.73-second build (test: 0.23 seconds);
its full native/readout/source/route-sum closure passed without warnings.
All 43 theorem axiom reports, including the fixed transfer lemma, contain only
`propext`, `Classical.choice`, and `Quot.sound`. Evidence:
`exponent-fixed-transfer-final-lean.log` and
`exponent-fixed-transfer-final-axioms.log`. The checked application files were
emitted from Rust without manual proof edits.

### Whole-cache transport in the production PRF accumulator

Removed `ProgramCacheBatchFamily` and the online per-gate cache reconstruction.
The existing batch owner now stores whole scalar caches separately on the
shared member axis and the local label/member axis. Public preparation retains
the cache returned by the common whole-body driver. Selector decomposition
still executes once per active selector, outside the label loop. The common
online driver's binding closure supplies both complete caches directly.

Setup keeps the same single-Preimage artifact names and physical ports. Import
reconstructs the cache schema with `GraphValue::from_values`, without a wrapping
parallel loop. The existing descriptor records the actual `ProgramFamilyId`;
it is not inferred from a step number. Scalar shared artifacts remain scalar.
BGG owns the remaining concrete artifact-field projection; the common driver
has no backend import. Moving the application batch traversal and full setup
to generic contracts remains incomplete.

The CPU library compiled without warnings (31.02 seconds). Four focused tests
passed once: public shared/local cache structure, distinct label vectors and
monomials, cache-name/schema round-trip, and BFV scalar cache transport across
parallel scopes. The actual noiseless PRF file-artifact round-trip also passed
once (41.66 seconds), covering all eight native inputs and two fresh online
executions after artifact import.
Updated the CPU and GPU fixture callers to the new cache binding API; no GPU
arithmetic, kernel, or synchronization behavior changed. Formatting and diff
checks passed. Evidence: `exponent-whole-cache-public-unit.log`,
`exponent-whole-cache-online-unit.log`, and `exponent-whole-cache-prf-runtime.log`.

The full concrete-refresh exporter also passed once after these changes
(0.24 seconds; `exponent-fixed-transfer-export.log`). Its generated Program
module contains the fixed transfer-lemma application. After removing the now
redundant cache-index tactic, the source-regenerated closure and all 43 axiom
reports passed as recorded in the fixed-transfer checkpoint above.

### Production batch traversal uses common PRF drivers

Moved the existing batch-cache and public-batch result records to `prf.rs` with
generic graph-value parameters. No replacement wrapper or capability trait was
added. Common `prepare_batch` and `evaluate_batch` now own stage traversal,
label/member indices, captured state binding, explicit scalar cache transport,
and calls to the common whole-body Program drivers. Both actual BGG production
entrypoints call these functions; their duplicate label loops are deleted.

BGG retains selector decomposition before each label loop and the concrete
segmented matrix reduction. The narrow reduction callbacks preserve the current
flat reduction topology; no per-label nested reduction loop was introduced.
Online vector/plaintext families are combined with their existing GraphValue
schema and handles, without arithmetic, wrapping loops, or new graph ports.
Setup artifact names, cache family identities, and indexed/shared storage remain
unchanged. The common PRF module contains no BGG/BFV import or backend type.

The CPU library rebuilt without warnings (31.73 seconds). The existing public
cache structure, online label isolation, and artifact schema round-trip tests
passed once. The actual noiseless PRF file-artifact test also passed once over
eight inputs and two imported online executions (39.27 seconds, compared with
41.66 seconds before moving the traversal; this is a narrow observation, not a
statistical performance claim). The concrete-refresh export passed (0.36
seconds). Evidence: `exponent-generic-batch-public-unit.log`,
`exponent-generic-batch-online-unit.log`, and `exponent-generic-batch-export.log`.
Formatting and diff checks passed. No GPU or integration tests ran.

This is an actual production orchestration migration, but the complete setup
and noise simulator still depend on BGG. BFV's segmented reduction metadata and
its production setup/decoder path must be integrated without replacing its
rounding schedule or introducing dummy witness material. Final production
certificates for both backends remain required.

### BFV Program reduction cache keeps only its output key

`PreparedGate::Reduction` now stores the output public key directly. The online
reducer uses its level and key identity and the existing cached public plaintext;
it never read the former input public key. Removed that redundant key from the
cache's graph-value and serialized schemas. Standalone prepared family-reduction
objects retain their input range/binding, which those separate APIs actually use.
No matrix arithmetic, rounding, or public-plaintext preparation changed.

The existing structural selection test now verifies that each reduction cache
has exactly its output key's graph ports and survives schema serialization and
graph-value reconstruction. Its runtime check still compares both public and
encoding outputs with the direct family operations and verifies the error bound.
The test passed once (0.11 seconds) after a warning-free CPU library build
(31.39 seconds); evidence: `exponent-bfv-reduction-cache-unit.log`. Formatting
and diff checks passed. No GPU or integration tests ran.

BFV's production PRF integration must retain the existing composed-selector
path (`PreparedSparseLwrPrf`): weighting and summing low-base RHS ciphertexts
precedes one direct product/ModDown per output and block. Instantiating the old
per-selector batch with BFV and summing independently rounded outputs would
change that schedule. The required shared setup should connect the existing
generic composed PRF and direct refresh owners, while witness-based issuance
remains a distinct capability for BGG.

### Accepted eight-slot refresh proof checkpoint

The existing accepted fixture
`test_data/exponent_lut_lean_accepted/449687101c902f56416599b0115af965033997c597fe381d35bd2c9d4b7ae33b`
now passes the remaining generated dependency closure through
`RefreshCorrectness`. The missing fixed Lean library modules were built once;
checking resumed after already checked graph modules rather than repeating them.
The official checker stack, heartbeat, and recursion limits were retained.

A `Certificate.lean` assembled by the actual ir-core `assemble_certificate` API
also passes the Lean kernel. Its endpoint is
`GeneratedCertificate.correctness : GeneratedClaim.CorrectnessClaim`;
the reported axioms are exactly `propext`, `Classical.choice`, and `Quot.sound`.
This connects all eight high-source bounds and the refresh finish theorem to the
actual linked execution claim. No generated mathematical proof was hand-edited
for this accepted fixture. Certificate assembly remains owned by the existing
`certify` pipeline; it is not duplicated in the graph exporter.

Evidence: `exponent-accepted-current-lean.log`,
`exponent-accepted-resume-lean.log`, and
`exponent-accepted-certificate-lean.log`. An ephemeral
`exponent-accepted-kernel-checkpoint.json` records hashes of the 49 generated Lean
sources and these logs. This is direct kernel-check evidence, not the production
checker's durable `checked-record.json`.

This fixture predates the whole-cache and common batch traversal changes. It
establishes a completed proof-coverage checkpoint, not a final certificate for
the current refactor. Final current production certificates, BFV operational
proof coverage, generic setup/noise integration, and deletion of the obsolete
Rust proof generators remain required.

### BGG decomposition gain construction has a concrete owner

Moved `NoiseGainPlan`, `NoiseGainPlanBuilder`, their decomposition validation and
gain derivation, and the existing gain-plan unit fixture from shared refresh
simulation to `bgg/noise.rs`. These operations depend on balanced gadget target,
automorphism-helper, and normalization decompositions; BFV must not implement
that model to participate in shared refresh accounting. Updated actual setup,
parameter-search, and test-helper consumers directly, without compatibility
reexports or new capability traits.

The stored program/readout gain fields retain their serialized names and values.
The application still reads the cached derived gains, so no decomposition,
sampler, graph operation, or online preparation was added. The existing ordered
gain-plan CPU unit test passed once after a warning-free release library build
(30.06 seconds); evidence: `exponent-bgg-gain-owner-unit.log`. Formatting and diff
checks passed. No GPU or integration tests ran for this ownership-only move.

The shared simulator still consumes concrete BGG Program reports. Moving the
constructor establishes its cryptographic owner; it does not complete the
required common trace or generic setup boundary.

### Runtime owns the reusable Lean certificate checker

Moved the dependency-ordered process/checking implementation and its tests from
`crates/exponent-lut/src/lean/check.rs` to
`crates/runtime/src/lean/check.rs`; deleted the application checker module and
updated certificate callers without a compatibility reexport. Runtime accepts
the Lake package directory and the complete local source roots. The application
retains its six dependency roots and proof theorem identity. Lake's declared
default target is `MxxExponentLut`, preserving the previous static build closure;
the checker itself contains no exponent-lut package name or dependency list.

Source snapshots, fresh object directories, process cleanup, deadlines, final
claim binding, and the closed axiom policy are retained. This is reusable process
execution, not relocation of the Rust mathematical proof generator. No encoding,
PRF, rounding, noise formula, or GPU operation changed.

The runtime library built without warnings (17.77 seconds). All eight narrow
checker tests passed once. The real Lake/package unit fixture also passed once
(11.63 seconds), including kernel compilation of the core certificate and its
axiom audit. That small checker fixture validates the moved infrastructure, not
cryptographic refresh correctness. Evidence: `runtime-lean-check-owner-unit.log`
and `runtime-lean-check-package-unit.log`.

The exponent-lut caller library also rebuilt without warnings (34.08 seconds),
and its five narrow certification-boundary tests passed once (0.01 seconds;
`exponent-runtime-check-caller-unit.log`). The two expensive full-production
certificate tests remained ignored because this move changes checker ownership
and package selection, not generated cryptographic mathematics. Final current
BGG/BFV production certificates remain an outstanding completion requirement.
Formatting and diff checks passed; no GPU or integration tests ran.

### Numeric proof bodies are fixed Lean source

Removed `src/lean/numeric.rs`. The Rust exporter now emits only
`NumericBoundsData.lean`, containing the report's scalar fields and worst-case
acceptance flag. The transported fixed `lean/NumericBounds.lean` proves validity
for every exported slot against `RefreshNumericSlot.Valid` in
`ExponentRefreshBounds.lean`. The former `pre_switch_i`, `decoder_i`,
`switched_bound_i`, `positive_lift_i`, `mask_bound_i`, and accepted-only
`strict_rounding_i` statements are the six conjuncts of `slot_valid`. No consumer
referenced the old per-slot theorem names. Rejected candidates still have
arithmetic checks without asserting strict decoding.

The fixed library compiled. The actual accepted eight-slot exporter passed once
(warning-free release build, 33.68 seconds; export 5.40 seconds). Its freshly generated data and fixed proof kernel-checked successfully
for fixture `80709cadcb537b699ee090d75dfadab6d06397a04d13fc86e5b7d4d598ba8c9e`.
`ExponentGeneratedNumeric.slot_valid` reports only `propext` and `Quot.sound`.
Evidence: `exponent-fixed-numeric-library.log`, `exponent-fixed-numeric-export.log`,
and `exponent-fixed-numeric-kernel.log`. This covers numeric prerequisites, not a
new full operational refresh certificate.

### WE and exponent-lut share one dependency scheduler

Removed `crates/we/src/lean/check.rs` and its module declaration. The WE Diamond
certificate adapter calls `mxx_runtime::lean::check::check_generated_modules` with
its existing runtime Lake package and extra library paths. Both that entrypoint
and `check_candidate` use the same environment resolver, dependency validation,
bounded parallel DAG scheduler, direct Lean invocation, durable log handling,
and process-group cleanup. No application-to-application dependency or old-path
reexport was added.

The shared scheduler preserves WE's maximum of three independent processes.
Dependencies must finish successfully before their consumers start, and the
Certificate starts only after every other module completes. The exponent-lut
entrypoint retains fresh object directories, source/dependency fingerprints,
and ir-core certificate matching. WE retains its existing
`DiamondCertificate.correctness` theorem and exact axiom-report requirement.
This division preserves the two actual certificate contracts while deleting the
duplicate execution implementation. Duplicate runner unit fixtures were removed;
WE's unique real-DAG and infrastructure-timeout fixtures moved to runtime.

Validation: eight runtime runner unit tests passed once, then the real-DAG,
real-package core-certificate, and Lean-timeout unit fixtures each passed once
(3.35 seconds, 10.05 seconds, and 1.04 seconds respectively). Both WE and
exponent-lut CPU library test targets compiled without warnings (35.31 seconds).
Evidence: `runtime-shared-lean-runner-unit.log`, `runtime-shared-lean-dag.log`,
`runtime-shared-lean-certificate.log`, `runtime-shared-lean-timeout.log`, and
`shared-lean-callers-build.log`. Full Diamond/refresh proof closures were not
rerun for this runner consolidation. Formatting and diff checks passed; no GPU
or integration tests ran.

### Native trace permutation proof is fixed Lean source

Removed `src/lean/trace.rs`. The root exporter retains only mechanical native
layout/index and inverse-position data construction in `TraceIndicesData.lean`.
The transported fixed `lean/TraceIndices.lean` establishes both position bounds,
the inverse permutation, and the existing `dimension`, `width_power`, `all_odd`,
`indices_valid`, `index_correspondence`, and `identity_index` theorems. Their
namespace and proposition interfaces remain available to the operational
readout proofs. Rust no longer constructs the permutation or these theorem
bodies as proof strings.

The actual concrete-refresh exporter passed once after a warning-free release
build (30.54 seconds; export 0.22 seconds). Its newly generated data and fixed
trace proof passed the Lean kernel; `TraceIndices.index_correspondence` reports
only `propext`, `Classical.choice`, and `Quot.sound`. Evidence:
`exponent-fixed-trace-export.log` and `exponent-fixed-trace-kernel.log`, fixture
`f523a0614b3e8d344f7b2f7946f2f9f6ecc3bb58e4795db31136d1a439387f00`.
This narrow check preserves the trace-permutation proof coverage; it does not
replace the final current operational certificate for either backend.
Formatting and diff checks passed. No GPU or integration tests ran.

### Direct Program products close over the signed native domain

The composed PRF could not be represented by scalar direct gates: RHS input
creation imposed the squared-width constraint of packed binary LUTs, and scalar
private multiplication multiplied widths without accounting for native ring
closure. Family member products already kept the native width. This prevented
binding an arbitrary native monomial RHS and chaining such direct products.

RHS declarations now require their domain to fit the stride's native ring
extent. Direct product width is the product-domain bound capped at that extent;
builder construction, frozen validation, and captured scalar RHS evaluation in
Family scopes use the same calculation. The squared-width packing restriction
is checked at binary LUT gates, including frozen and Family validation. No
ciphertext operation, preparation cache, sampler, level transition, or ModDown
changed. The closed signed native domain accounts for negacyclic wrap; it does
not assert an injective packing for a binary LUT.

The new Program test chains full-domain products at strides 0, 1, and 3 and
checks that the corresponding oversized binary LUT packing remains rejected.
The existing out-of-domain rejection case now uses RHS width four in a native
width-two ring; width two is a valid native RHS under the revised direct contract.
All 16 Program unit tests passed on the final version, including identity/layout
binding, explicit binary LUTs, public attributes, and structural Family checks.
The release build was warning-free; evidence:
`exponent-native-product-program-final-unit.log`. Formatting and diff checks
passed. No GPU or integration tests ran.

This removes a concrete Program-domain obstacle. Connecting the composed PRF
preparation/evaluation to that frozen direct schedule and binding its actual
noise sources remains the next required step; the full trace/setup migration is
not complete.

### Composed PRF has an authenticated direct-product Program

`PreparedSparseLwrPrf` now owns the frozen direct-product lowering of its logical
selection schedule. It has one full-native `MulPrivate` gate per PBC block,
with explicit RHS inputs and the same initial/final wire domain. Public
preparation and cached online evaluation traverse those gates and bind the
corresponding composed RHS/prepared product. No host loop over label members or
additional cryptographic operation was introduced. RHS weighting/summing still
precedes preparation of each product and BFV ModDown remains after that product.

The descriptor commits both the logical PRF identity and the executed
accumulation identity. Import deterministically reconstructs that lowering and
checks the identity, preserving the distinction between protocol meaning and
actual rounding schedule without serializing a duplicate Program. The read-only
`accumulation()` accessor supplies the same frozen gates to noise/proof clients.

The existing BFV native-digit refresh artifact fixture now traces that actual
Program through the common noise interpreter instead of calling a handwritten
one-step PRF bound recurrence. Each composed RHS binds both monomial label gains;
its fresh error includes both ciphertexts while its one-hot payload l1 bound is
one. The traced output bound feeds the existing aggregate terminal calculation.
This connects the executed PRF schedule to deterministic noise accounting in the
artifact fixture; generic production setup and per-sampler moment/covariance
bindings still remain required.

Validation: the CPU library built without warnings (30.32 seconds). The BFV
native-digit PRF/refresh artifact fixture passed once (104.41 seconds, compared
with the earlier 119.92-second observation). It retains actual file-artifact
reopen, online mask-secret preparation, refresh, and continuation. This narrow
timing comparison is not a statistical performance claim. The BGG composed-RHS
fixture also passed once using the same built library binary. Evidence:
`exponent-composed-prf-trace-unit.log` and `exponent-composed-prf-bgg-unit.log`.
Formatting and diff checks passed. No GPU or integration tests ran.

### Refresh planning consumes BFV Program trace outputs

Following the user's priority change, further Lean migration and proof checking
are deferred until the non-Lean implementation is complete.

`BfvRefreshPlanRequest` no longer copies `input_level`, `data_noise`, or
`prf_noise`. `BfvParameters::plan_refresh` receives the existing `BfvNoiseTrace`
and data/PRF wire identities. It derives the common terminal level, inherited
error bounds, and payload bounds from those outputs before solving the shared
digit fixed point. The trace must use the same selected parameters; this is a
setup binding contract, with no ciphertext inspection or second Program run.
No additional noise record, wrapper, backend dispatch, or compatibility API was
introduced. The fixed-point and terminal error formulas are unchanged.

The existing planner unit fixture now executes the Program noise interpreter
for a composed direct product, including both one-hot ciphertext error gains,
instead of supplying a detached PRF error constant. Its first run correctly
rejected insufficient mask spacing: consuming the newly explicit PRF stage had
left the old two-stage workload at its last modulus. The fixture now declares
all three actual stages (PRF product, refresh, continuation), retaining the same
prime bit size, terminal depth, CRT budget, statistical requirement, and fresh
support requirements. No acceptance inequality was weakened.

The final targeted CPU test passed once after a warning-free release library
build (29.92 seconds); evidence: `exponent-refresh-plan-from-trace-final.log`.
Formatting and diff checks passed. No Lean, GPU, or integration tests ran.
This connects deterministic refresh planning to Program output noise; it does
not yet complete the production setup/artifact migration or source-aware
moment transfers for either backend.

### Planned digits drive the BFV file-artifact refresh path

`SparseLwrPrfProgram::fixed_accumulation` is available before preparing label
families. Setup can therefore trace the exact composed direct-product schedule,
solve mask/fresh digit counts, and only then generate the corresponding labels.
`PreparedSparseLwrPrf` reconstructs and authenticates that same schedule during
preparation and import; exposing its construction adds no online graph work.

Connecting the existing file-artifact fixture exposed an unnecessary restriction
in the preceding planner API: the refreshed data and PRF need not be wires of
one Program. `plan_refresh` now borrows their existing `BfvNoiseValue` records,
which can come from separate traces or an input sampling contract under the same
parameters. The workload request still has no copied input level or bounds.
No synthetic combined Program or extra carrier type is required.

The BFV native-digit artifact fixture now uses this planner before label
creation, replacing its fixed mask support and late handwritten terminal bound
recurrence. It checks that the emitted label count matches the planner's
separate distinct-output count, and that the prepared PRF uses the traced
schedule identity. The existing base-four fresh support, exact fresh-error
oracle, file-store reopen, online secret boundary, and next-LUT continuation are
retained. The fixture now also exercises mask smudging parameter selection
(default 16 statistical bits, overridable through the existing
`MXX_EXPONENT_LUT_REFRESH_MASK_STATISTICAL_SECURITY_BITS` variable).

Validation: the file-artifact CPU fixture passed once (656.20 seconds), including
store reopen, online preparation/evaluation, exact fresh-error replacement, and
next-LUT continuation (`exponent-bfv-planned-refresh-artifact.log`). The release
library compiled without warnings (30.50 seconds), and the revised planner unit
fixture passed once using that binary (`exponent-bfv-planned-refresh-unit.log`).
The prior fixed-two-mask-digit fixture took 104.41 seconds; this run evaluates a
larger planner-selected workload with smudging requirements, so the timings are
not a comparison of equal workloads. The live test used approximately 230 MiB
RSS while writing artifacts. Formatting and diff checks passed. No Lean, GPU,
or integration checks ran. This is a real arithmetic/artifact path connection,
not completion of the generic production setup or covariance work.

### BGG owns refresh decomposition gains and one fresh multiplicity

Moved `RefreshNoiseGainDecompositions`, its gain derivation, and its two
concrete decomposition fixtures from refresh setup into `bgg/noise.rs`. Setup
still owns the topology/count checks and maps the existing mathematical error
at that boundary; the backend does not depend on `RefreshSetupError`. All
workspace references use the concrete owner, without an old-path reexport.

The former per-slot `fresh_multiplicity` vector stored the same ring-coordinate
count in every production constructor. It is now one required scalar. Each CRT
slot retains its own decomposition plans and kappa-dependent gains, while the
shared fresh coordinate multiplicity has one owner. This describes the compact
representation, not an independence or covariance assertion. Removed both serde
default helpers and the empty-vector/mask-multiplicity fallback. Mask and fresh
counts remain distinct, so changing a mask candidate cannot silently change the
fresh representation. Existing explicit and compact route tests retain equal
l1/l2-squared sums and unequal gains for unequal CRT weights.

The CPU release library compiled without warnings (29.78 seconds). Both moved
route tests passed once, and the existing setup fixture confirming that changed
decompositions affect route gains and digit variance also passed once using the
same binary. Evidence: `exponent-bgg-refresh-route-owner.log` and
`exponent-bgg-refresh-route-setup.log`. Formatting and diff checks passed. This
changes setup metadata ownership and serialization, not ciphertext graphs; no
additional runtime, GPU, integration, or Lean checks were run. The concrete BGG
setup orchestration and shared simulation's concrete noise dependencies remain
to be migrated.

### BGG native readout preparation separates payload and online secrets

`sample_native_readout_helper_families` no longer accepts the online secret or
returns online mask vectors. Its existing public carrier now owns only sigma
order, fixed switch ciphertexts, and public mask matrices. Public matrices and
payload-secret switches use independent structural families, preventing a
projection of the public matrices from retaining the payload-switch sampler's
loop dependencies.

`sample_native_readout_masks` performs the actual online preparation from the
public matrix family and online secret alone, using the same BGG encoding
sampler, augmented constant, automorphism order, and error configuration as
before. The production setup driver invokes it at the online-preparation point.
No new data carrier, trait, compatibility method, per-member host loop, or
runtime ciphertext validation was introduced. Deterministic public mask hashes
and switch tags remain unchanged. The phase split changes graph scopes and
removes cross-phase sampler captures; deferred operational proof migration must
use these new graphs rather than the earlier combined-helper checkpoint.

The width-1024 graph fixture now checks both phase boundaries explicitly:
preprocessing has no online-secret input, online masks have no payload-secret
input, and both graphs validate while preserving structural branch execution.
It passed once (0.01 seconds) after a warning-free release build (29.93 seconds).
The existing production preprocessing/selected-decoder-ring graph fixture also
passed once (0.08 seconds) with the same binary. Evidence:
`exponent-bgg-readout-phases-final.log` and
`exponent-bgg-readout-phase-setup.log`. Updated both CPU and GPU fixture call
sites for the removed argument; no GPU, integration, or Lean test was run.
Formatting and diff checks passed. Generic production setup and source-aware
moment integration remain unfinished.

### Native readout switch preparation no longer constructs a LUT

The BGG native readout switch sampler accepts the native width directly. Its
switch plaintext is one: the coefficient map has already been applied by the
readout target. It never consumed table values. Removed the temporary canonical
coefficient conversion and `LutTable` construction from production setup and
both CPU/GPU PRF fixtures. Ordinary LUT sampling still receives its real table
and retains fused weighted fresh-target semantics. Native branch order,
automorphisms, public mask hash tags, and switch plaintexts are unchanged.

Production setup also constructs its BGG public-key compiler parameters once.
The public and encoding interpreters derive from that same value, and one
encoding compiler is retained across preprocessing and online construction.
Removed the separately rebuilt `refresh_compiler_encoding`; no new wrapper or
getter was added to expose backend internals.

Validation: the release CPU library compiled without warnings (30.57 seconds).
The width-1024 phase/structure fixture passed once (0.01 seconds), and the
production selected-decoder-ring graph fixture passed once using the same
binary. Evidence: `exponent-bgg-readout-width-contract.log` and
`exponent-bgg-readout-width-setup.log`. Formatting and diff checks passed. No
GPU, integration, or Lean checks ran. This removes unused setup work and a
misleading table dependency; it does not complete generic setup or moment
tracking.

### Heterogeneous Family variance bounds retain covariance without zero-term inflation

Added the reusable `NonnegativeRational::sum_variance_bounds` operation and
connected it to BGG's existing Family reduction interpreter. It composes the
existing exact-rational outward Cauchy bounds, capped by r times the marginal
variance sum, where r counts only nonzero variances. No independence assumption
is introduced. Zero-variance terms may have nonzero means but cannot contribute
covariance. Shared uniform families retain their compact r-squared transfer;
no member expansion was introduced. Worst-case coefficient bounds, physical
coordinate conventions, and acceptance-policy flags are unchanged.

The prior heterogeneous transfer used the entire family count times the
variance sum, including deterministic members. The new bound is tighter: for
marginals 32 and 68 it is 194 instead of 200; for 32, 41, 50 it is 368 instead
of 369. The tests now assert those exact outward bounds rather than the old
looser transfer. Rational proportional sources retain the attained coherent
bound, including Var(E+2E+3E)=36 Var(E). The actual three-member Program fixture
also verifies that one variable public error with gain three and two constant
errors has variance bound nine, without multiplication by the family size.

Validation: seven targeted variance tests passed once after a warning-free
release build (30.03 seconds), followed by the actual three-member Program
fixture on its final updated expectations (warning-free build 29.87 seconds).
Evidence: `exponent-heterogeneous-covariance-final.log` and
`exponent-heterogeneous-covariance-program-final.log`. Formatting and diff checks
passed. No GPU, integration, or Lean checks ran. These are conservative
unknown-covariance transfers, not complete per-sampler covariance tracing;
BGG/BFV source/moment bindings and generic production setup still remain.

### Additive gadget residuals no longer imply independent errors

`PrimitiveNoiseTransfer::add_residual` now composes the supplied residual's
marginal variance with both fresh and total gate variance through the common
outward Cauchy bound. The residual contract did not establish independence from
the decomposed ciphertext's errors, so simply adding variances could omit
covariance. `ProgramNoiseInputs` documents this distinction. The setup's
conditional model for the residual's own marginal distribution is unchanged;
this change only removes an additional unstated independence assumption at
composition. Coefficient error bounds and exact-gadget paths are unchanged.

The targeted CPU fixture covers an attained correlated case (the residual is
the same random variable as the fresh error, giving four times its variance)
and a possibly nonzero constant residue (increased error bound, unchanged
variance). It passed once after a warning-free release build (30.24 seconds):
`exponent-residual-covariance.log`. Formatting and diff checks passed. No GPU,
integration, or Lean checks ran. Approximate-gadget average estimates can become
larger; previously selected approximate candidates require their final current
model audit. This is a conservative composition correction, not completion of
per-sampler source/moment tracing or the generic setup migration.

### Restore the accepted average-case residual model

This correction supersedes the preceding subsection. The existing
`docs/plans/exponent_lut_average_case_noise_spec.md` declares a heuristic
average-case envelope, and
`docs/plans/exponent-lut-universal-approximate-depth-window-results.md`
explicitly retains residual independence as a modeling assumption. Treating
that assumption as absent was incorrect and changed the accepted model.

`PrimitiveNoiseTransfer::add_residual` again adds residual variance to fresh
and total variance. The input contract and main average-case specification now
state this composition assumption directly. The narrow test checks variance
addition and a nonzero constant residual with unchanged variance. Explicit Add,
repeated-wire covariance, heterogeneous Family bounds, and worst-case error
bounds are unchanged. No runtime validation or new data structures were added.

Validation: the targeted CPU test passed once after a warning-free release
build (29.53 seconds), recorded in `exponent-residual-heuristic-restored.log`.
`cargo +nightly fmt --all` and `git diff --check` passed. No GPU, integration,
or Lean checks ran. Generic production setup and complete backend moment/source
tracing remain unfinished; this correction does not claim their completion.

### Common channel artifact traversal and schema-owned refresh output keys

Moved ordered direct-refresh channel export/import loops from the BFV adapter
into `PreparedDirectRefresh<S>`. The existing per-slot descriptor and transport
remain authoritative; no additional carrier or capability trait was introduced.
BFV retains only its scheme binding, output-key handling, schema declarations,
and CRT recomposition preparation at this boundary.

The BFV refresh descriptor now stores the existing public-key GraphValue schema
instead of copying level and ID into independent fields and manually rebuilding
the key. Export/import both use that key's existing GraphValue implementation.
This retains polynomial block ports and their order while binding the key's
context and matrix schema in the descriptor identity. Old descriptor formats
are intentionally not supported. Public sums, samplers, and ModDown scheduling
are unchanged; no ciphertext inspection or additional runtime arithmetic was
introduced.

The targeted CPU native-digit refresh artifact test is running, recorded in
`exponent-refresh-schema-roundtrip.log`. It covers JSON descriptor reload,
file-artifact import, online refresh, exact fresh-error replacement, and the
following LUT. Compilation initially found one obsolete import, now removed.
Formatting and diff checks passed. This is partial transport extraction, not
completion of generic production setup; no GPU or Lean checks ran.

### Prepared Program artifact ownership moves to the shared interpreter

Removed `BfvPreparedProgramDescriptor` and the specialized BFV
`PreparedProgram` transport implementation. The replacement
`PreparedProgramDescriptor<C, K, B>` lives beside the common interpreter and
uses cache/key schemas plus the setup's existing binding representation.
`PreparedProgram::export` and `import` now own the complete scalar/shared/cache
Family/public-value port ordering. No new trait, backend dispatch, compatibility
alias, or additional descriptor wrapper was introduced. The setup supplies its
resolved scheme binding explicitly; imports must not source that expectation
from an untrusted descriptor. Program, ranges, binding and expected descriptor
identity are checked at import, using the existing Program validation error
boundary. No checks were added to online gates.

The generic descriptor hash domain is `exponent-lut-prepared-program`; old BFV
formats are not retained. Concrete key/cache schemas still bind cryptographic
metadata and preserve shared versus indexed graph fields. The existing BFV
fixture now uses this shared transport and retains all-gate/public-value,
shared-RHS, cache-tampering, range, Program and scheme-binding checks.

Validation: that CPU fixture passed once (3.88 seconds) after a warning-free
release build (31.06 seconds), recorded in `exponent-common-program-artifacts.log`.
Formatting and diff checks passed. The separately running direct-refresh
artifact test is still pending in `exponent-refresh-schema-roundtrip.log`.
This establishes the common Program transport with BFV; BGG's production setup
still needs migration to the common setup/artifact owner. Lean remains deferred.

Follow-up inspection: BGG's `ProgramCacheBindings`/`ProgramLutCache` and
lower-level `BggPublicKeyType` do not yet serialize their schemas, so the new
whole-Program JSON descriptor is not yet instantiated by BGG. Its existing
shared-RHS artifact fixture still reopens the decomposition manually. That is
a concrete remaining migration: provide schema serialization (structured gate
keys must use ordered entries, as in BFV) and route that fixture and production
owner through the shared transport. A generic method definition alone is not
evidence that this BGG route has been migrated.

### BGG instantiates the common prepared-Program transport

Added schema serialization to BGG `ProgramCacheBindings`, `ProgramLutCache`,
and the lower-level `BggPublicKeyType`. Structured gate keys use ordered entry
vectors, matching the existing BFV cache convention; graph handles themselves
are not serialized. The lower-level BGG crate now declares its serde dependency.
No new carrier or artifact trait was introduced.

Migrated the existing BGG shared-family-RHS file-store fixture to export and
JSON-roundtrip the complete common `PreparedProgramDescriptor`, then import
its cache and public keys through `PreparedProgram::import`. Removed the
fixture's manual decomposition/public-key artifact reopening and mutation of
the prepared shared cache. It still exercises two batch uses of the same
imported decomposition, and retains the exact zero-residual oracle, one
producer decomposition, and no consumer decomposition assertions. The trusted
fixture binding includes its modulus, dimension, gadget layout and exponent
layout. This tests transport with both backends, not the yet-unmigrated BGG
production refresh setup.

Validation: the targeted CPU test passed once (0.37 seconds), after a
warning-free release build (34.15 seconds), recorded in
`exponent-bgg-common-program-artifacts.log`. Formatting and diff checks passed.
The preceding long BFV direct-refresh artifact test remains running separately;
no GPU or Lean checks ran.

### Direct-refresh schema roundtrip completed

The previously running BFV native-digit direct-refresh artifact test passed
once in 665.04 seconds, after its warning-free 30.62-second build:
`exponent-refresh-schema-roundtrip.log`. It covered the common channel
export/import and schema-owned output-key change, JSON/file-store reopen,
online refresh, exact fresh-error replacement and the next LUT. Compared with
the prior 656.20-second run this is about 1.3% longer; other narrow builds/tests
ran concurrently, so this is validation timing rather than an isolated
performance measurement. The tested refresh path was not modified by the later
PreparedProgram transport changes, which have their own BFV/BGG CPU fixtures.

### BGG owns production decomposition-cache import

Moved the typed reconstruction of BGG scalar/shared/indexed Preimage caches
from `ImportedRefreshSetup::import` into the existing BGG
`ProgramCacheBatchBindings<ProgramCacheBindings>` implementation. The common
setup now passes canonical artifact names, output count, production identity,
and a single Preimage schema. The adapter uses GraphValue schema import rather
than repeatedly building rings and matrix shapes for every field. The manifest
has already validated field types and scalar/family cardinalities; removed the
second identical manifest family-count lookup from this reconstruction loop.
Existing cache insertion invariants remain intact. No new carrier, trait,
coefficient scan, public decomposition, or host expansion of Family members
was introduced. The setup still has other BGG-specific operations to migrate.

Validation: `switched_preprocessing_artifacts_use_only_selected_decoder_rings`
passed once (0.08 seconds) after a warning-free release build (29.41 seconds),
recorded in `exponent-bgg-cache-import-owner.log`. Its production setup build
passes through `ImportedRefreshSetup::import` and constructs the linked graphs;
it is not an executed full BGG refresh roundtrip. Formatting and diff checks
passed. No GPU or Lean checks ran. No validation jobs remain running.

### Complete ownership of native BGG batch-cache transport

Moved the corresponding decomposition export loop from common setup into the
same existing BGG cache implementation as import. Export uses the existing
GraphValue implementation of `ProgramValue`, removing the duplicate
scalar-versus-Family output branches. Canonical artifact names, field order,
sharing, and the count/output-axis checks are retained. The common setup calls
this adapter for BGG decomposition fields; replacing the remaining concrete
setup owner is still required by the plan.

Deleted the test-only public `compile_program_wrapped` implementation and its
unused optional cache output. Its sole caller always supplied `None`; it now
uses the existing `compile_program_with_preprocessing` and takes the public
values from the common PreparedProgram result. This removes duplicated RHS,
public-family and helper binding conversion plus the duplicated preparation
call, without adding a wrapper or changing the emitted gate arithmetic.

Validation: the production setup graph fixture passed once (0.08 seconds) after
a warning-free release build (30.31 seconds), recorded in
`exponent-bgg-cache-transport-owner.log`. It validates the setup's exported
manifest and imported graph construction. Formatting and diff checks passed.
No additional runtime arithmetic was introduced, and no GPU, integration or
Lean tests ran. No test processes remain active.

### BGG owns the approximate-gadget residual distribution

Moved the shortened-gadget reconstruction bound and marginal variance formula
from common refresh setup to `bgg/noise.rs`, beside the authoritative ordinary
CRT gadget layout. Setup normalizes its public parameters and assigns the
returned existing `ProgramNoiseState` to operation/readout routes. No new noise
carrier, distribution enum, or runtime validation was introduced. Exact gadget
width returns no residual; the augmented ternary dimension-two requirement is
unchanged.

The formula preserves the accepted independent-uniform discarded-residue and
ternary-secret model, including the Var(2e) coordinate convention. Its new
narrow CPU fixture enumerates the complete small independent-residue/ternary
model and compares the exact mean, variance and attained coefficient bound.
It also checks the no-residual full-width case. This validates the declared
heuristic model, not independence of actual ciphertext residues. Shared-wire
covariance and the recently restored additive-residual heuristic are unchanged.

Validation: the targeted model test passed once after a warning-free release
build (29.33 seconds), recorded in `exponent-bgg-residual-model-owner.log`.
Formatting and diff checks passed. No GPU, integration or Lean checks ran.
The common setup still carries BGG-specific route assignment and other
parameters; this formula extraction is not the completed generic setup boundary.

### BGG owns residual placement as well as its marginal model

Moved the remaining ordinary-gadget residual assignment from common setup to
`ExponentLutNoiseSnapshot::apply_approximate_gadget_residual`, implemented beside
the BGG noise transfers. This includes the direct/helper source multiplicity
per operation and the target/readout gain weighting, with zero residuals for
exact target/normalization actions. The public setup now supplies normalized
parameters and invokes that method; it no longer defines these BGG equations.
The method is called once on a newly constructed snapshot, preserving the
existing construction contract and the average-case independence heuristic.
No new wrapper, trait, per-gate validation or execution work was introduced.

The snapshot type itself still resides in common simulation and contains BGG
report/plan types. The audit found that moving only this method cannot make
simulation scheme-independent: its report/error boundary and planner consumers
must migrate together. That larger requirement remains open, along with generic
production setup, BFV moments/source tracing and deferred Lean integration.

Validation: the residual marginal model fixture passed once after the move;
its build reported an obsolete setup import, which was removed. The final
warning-free build (28.93 seconds) and graph-free setup/snapshot fixture passed
in `exponent-bgg-residual-routing-setup.log`. That fixture compares ordinary
and cached-route snapshots plus hard/average report decisions; it is not an
executed ciphertext roundtrip. Formatting and diff checks passed. No GPU,
integration or Lean checks ran; no test process remains active.

### Refresh reports no longer fix the Program result to BGG

Parameterized `PreRefreshNoiseEvaluation`, `ExponentLutNoiseReport`, and
`ExponentLutAverageNoiseReport` by their contained Program result. These existing
report containers now transport an arbitrary result without a BGG default,
backend alias, new trait, or extra wrapper. The scalar bridge transformation
on pre-refresh bounds/variance is likewise independent of that payload.
All existing BGG snapshot, setup and parameter-search consumers explicitly
select `ProgramNoiseReport`. Serialized fields and arithmetic are unchanged.

The four Rust Lean-adapter consumers only received the necessary report type
arguments/imports; no generated proof expression, Lean source, or proof check
was changed. Lean work remains deferred. The common snapshot and error still
reference BGG and must be handled in the next ownership step; this change does
not claim complete scheme-independent simulation or a BFV average-case model.

Validation: the final graph-free snapshot/report fixture passed once after a
warning-free release build (29.07 seconds), recorded in
`exponent-generic-noise-reports-final.log`. It compares the ordinary and cached
snapshot paths and hard/average report decisions. Formatting and diff checks
passed. No GPU, integration or Lean tests ran; no test process remains active.

### Remove concrete backend dependencies from common simulation

Moved the concrete `ExponentLutNoiseSnapshot` and its complete evaluation
implementation from common simulation to `bgg/noise.rs`, renaming it
`BggNoiseSnapshot` without a compatibility alias. Its fields contain BGG gadget
plans, BGG Program input/error reports and doubled-coordinate equations, so
representing it as a generic snapshot was misleading. Its residual-placement
method is merged into the same implementation block. The move preserves all
Program, readout, switched-refresh and average-authority formulas and checks.

Common simulation retains generic report containers, parameter/event records,
exact factorization/tail/rounding helpers and storage accounting. It no longer
imports or names any concrete BGG/BFV type. `NoiseSimulationError::Program`
carries the existing diagnostic text; the conversion from the concrete BGG
error is implemented in the backend. The prior typed nested error source is
no longer exposed through the common error API. No runtime arithmetic or
ciphertext validation was added.

Updated setup, parameter-search and Rust proof-adapter type references to the
concrete backend owner. Proof bodies and Lean source are unchanged and no Lean
check is part of this move. This removes the simulation import dependency but
does not finish generic production setup, BFV moment integration, or reduce
the now larger BGG noise implementation to the final size target. The common
parameter/report semantics still need the final backend-neutral contract audit.

Validation: the existing graph-free snapshot/report fixture passed once after
a warning-free release build (29.04 seconds), recorded in
`exponent-bgg-snapshot-owner.log`. It compares ordinary/cached-route snapshots
and hard/average report decisions. Source search finds no concrete backend
reference in `noise_refresh/simulation.rs` and no old snapshot name or old
snapshot import path in Rust consumers. Formatting and diff checks passed.
No GPU, integration or Lean tests ran; no test process remains active.

### Preserve both norms for BFV composed-RHS fresh errors

The BFV noise audit found that `BfvGateNoiseMaterial` retained only the sum of
public multiplier l1 gains. That value cannot recover coefficient-square gains
needed for moments. Replaced its scalar gain with the existing shared
`RightActionNoiseGain`; no new data type or wrapper was introduced.
`from_one_hot_labels` now consumes those gains, sums l1 bounds for all fresh
errors, sums squared-l2 gains for separately sampled errors, and retains the
maximum l1 gain for the selected payload. A reused error source must have its
weights combined before supplying these gains; this is a construction contract,
not a runtime ciphertext check.

All current worst-case transfers explicitly consume the l1 component and are
unchanged. Existing monomial fixtures and the planned PRF/refresh fixture supply
unit gains. The composed-noise test uses actual coefficient vectors `(2,-1)`,
`(2)`, and `(0)`: error l1 gain 5, coefficient-square gain 9, payload l1 bound 3.
It retains the existing Program worst-case error comparison. This preserves
necessary moment inputs; it does not yet propagate BFV per-wire variances or
means or claim a completed source-covariance model.

Validation: the targeted composed-RHS Program noise test passed once after a
warning-free release build (29.75 seconds), recorded in
`exponent-bfv-composed-rhs-norms.log`. Formatting and diff checks passed.
No GPU, integration or Lean checks ran; no test process remains active.

### Workspace CPU compile audit after public API changes

`cargo test -r --workspace --lib --no-run` completed warning-free in 47.31
seconds: `exponent-workspace-cpu-refactor-build.log`. It compiled all eleven
workspace library test targets, including the BGG serde dependency and the
moved/generic noise APIs. No tests were executed by this command.

Current source inventory confirms five Rust files per backend. Within
`noise_refresh`, concrete backend references remain in `setup.rs`; the other
Rust modules no longer name BGG/BFV. This is an import-boundary observation,
not proof that all remaining parameter/report semantics are backend-neutral.
No removed snapshot or BFV prepared-Program descriptor name remains in Rust
consumers. The separately compiled parameter-search target is being checked
because its updated shared test-support source is not covered by the library
compile. GPU validation and Lean proof work remain deferred.

### Separate parameter-search target exposed and fixed remaining API fallout

The explicit `parameter_search_unit` compile caught a damaged snapshot-producing
return type (`Result<String>` instead of `Result<BggNoiseSnapshot, String>`) in
shared search support. Restored the concrete backend snapshot return type. This
consumer is not compiled by `--workspace --lib`, illustrating why that earlier
success was not sufficient evidence for all updated call sites.

The ordinary non-test library build also exposed imports that had become
test-only after the transport moves. Scoped BFV `PreparedProgram`/`WireId`,
BGG family-wrapper imports, and setup's test route-gain import with `cfg(test)`.
These are ownership/import fixes only; no GPU or mathematical code changed.

`cargo test -r -p mxx-exponent-lut --test parameter_search_unit --no-run` now
passes warning-free (21.60 seconds):
`exponent-parameter-search-refactor-build.log`. It only compiled the executable;
no parameter search or integration test was executed. A final exponent-lut
library-test compile is checking the adjusted test import scopes.

The final exponent-lut library-test build also passed warning-free (29.90
seconds): `exponent-test-import-scope-build.log`. Both test and ordinary library
import configurations are now covered. Formatting and diff checks passed.
No test processes remain active. This compile audit does not complete the
remaining implementation, GPU validation or deferred proof requirements.

### BFV direct-product conditional variance transfer

Added `BfvParameters::private_mul_noise_variance` next to the direct worst-case
transfer, using the same four signed error terms. With physical input variance
V, N coefficients, low modulus q and dropped scale p, the centered-block bound
contributes `V*N*(q-1)^2/(2*p^2)`. Fresh CBD(eta) error contributes
`eta/2 * payload_l2_squared * fresh_error_gain.l2_squared`. The two rounding
terms contribute `(N+3)*(p^2-1)/(18*p^2)`: augmented ternary secret squared norm
has expectation `2N/3+1`, plus one encoding remainder. The matrix term uses a
public coefficient cap and performs no ciphertext scan.

This is explicitly conditional on uncorrelated centered state coefficients,
distinct independent CBD errors and independent centered-uniform remainders,
including independence from the ternary secret and inherited/fresh terms.
These assumptions do not follow from arithmetic correctness. Source reuse
must combine linear actions before applying the transfer. The function reports
physical variance, never BGG's four-times variance. It is not yet connected to
per-wire BFV mean/variance propagation or parameter acceptance; those remain
required work. Existing worst-case paths and accepted parameters are unchanged.

Extended the existing composed-RHS test with exhaustive eta=2 CBD bit outcomes
for independent errors weighted by 2, -1 and 2. This gives an independent
fresh-distribution oracle for the difference between zero-fresh and composed
variance transfers, while retaining its existing worst-case Program test.

Validation: the targeted composed-RHS/variance test passed once after a
warning-free release build (29.41 seconds), recorded in
`exponent-bfv-direct-variance.log`. Formatting and diff checks passed. No GPU,
integration or Lean checks ran; no test process remains active.

### BFV composed convolution correction and explicit LUT moments

The previous direct-product entry's fresh formula was valid for a monomial
payload but not a general composed polynomial. Corrected it to
`eta/2 * min(L1(mu)^2 * sum L2(a_j)^2,
L2(mu)^2 * (sum L1(a_j))^2)`. These are Young convolution upper bounds;
known repeated actions on one error source are not independent coefficients.
For example `(1+X)^2` has squared coefficient norm 6, not 4. The retained
norms yield a bound of 8 in this example, not an exact covariance calculation.
Monomial payloads and a single unweighted fresh error retain their exact
squared-norm gains. The targeted test passed once, warning-free, in
`exponent-bfv-direct-convolution-variance.log` (29.72-second release build).

Added the explicit-LUT physical variance formula alongside the direct formula.
For J branches, input variance V, CBD variance v=eta/2, next modulus q and
scale p, the inherited and mask-helper contribution is
`J*(V+v)*N*(q-1)^2/(2*p^2)`. The fresh term is J times the corrected composed
convolution bound. Rounding contributes
`((J+1)*(2*N+3)+3)*(p^2-1)/(36*p^2)`: J internal public remainders and exactly
one outer public and one outer encoding remainder. This matches mask-helper
sampling in `BfvScheme::prepare_online_masks` and the explicit LUT equation.
It uses the named coefficient, RHS and flat-branch decorrelation heuristics,
plus the centered-uniform independent remainder model. It must not be reused
for raw terminal aggregation or shared-helper invocation sums.

The average-case specification now explains H4's conditional right-action and
fresh-error assumption instead of only listing its name. Updated its obsolete
implicit-identity selector notation to the accepted MulPrivate primitive.
Neither change strengthens the model into a proof or changes acceptance policy.
Both BFV variance functions still need integration into per-wire moments and
planner acceptance; standalone formulas do not complete that requirement.

Validation: the extended composed-RHS test passed once after a warning-free
release build (29.45 seconds), recorded in `exponent-bfv-lut-variance.log`.
It checks the enumerated fresh CBD variance in one, two and four LUT branches,
in addition to direct composition and its worst-case Program trace. This is
not an end-to-end statistical validation of the branch/remainder heuristics.
Formatting and diff checks passed. No GPU, integration or Lean tests ran.

### BFV Program moments and plaintext squared norms

Extended the existing `BfvNoiseValue` with physical `NoiseMoments` and a
plaintext squared-l2 bound; `BfvNoiseRhs` now retains the corresponding
squared norm. This uses the same interpreter, gate identifiers and dependency
DAG as worst-case tracing. Direct MulPrivate and explicit Lut now invoke their
variance transfers from that interpreter, rather than leaving them as isolated
formulas. No additional Program traversal or runtime ciphertext scan was added.

Known public polynomials provide exact l1 and squared-l2 payload norms. Product
payloads use Young bounds; one-hot composition takes the maximum payload
squared norm and keeps all fresh error contributions. Family reduction retains
one selected payload but sums every error. Repeated-wire Add gives 4V and a
three-member captured reduction gives 9V. Public error convolution currently
uses l1 squared to retain arbitrary coefficient covariance; this is a bound,
not an exact covariance calculation. More precise accumulated public actions
remain required. Table collisions similarly retain an l1-based payload cap.

Moment input is optional and explicitly conditional. Missing moments do not
mean zero variance. Add and reduction preserve supplied nonzero means;
nonlinear transfers and public polynomial multiplication currently propagate
moments only for centered inputs. A nonzero-mean input retains its deterministic
bound but does not acquire fabricated centered moments. Full polynomial mean
propagation and average-case planner acceptance remain unfinished; current
parameter acceptance still uses the existing deterministic audit. The
production roundtrip fixture's bound-only inputs remain explicitly bound-only.

Validation: `cargo test -r -p mxx-exponent-lut --lib bfv::noise::tests`
passed both targeted tests once after the final assertions, warning-free
(30.65-second build), in `exponent-bfv-program-moments-final.log`. Assertions
cover composed-RHS moments through MulPrivate, explicit Lut dispatch through
Program, public payload squared norms, one-hot reduction, and repeated-wire
variance. They supplement the existing source-DAG and worst-case checks.
No GPU, integration or Lean tests ran. Full generic setup, source-aware moment
precision, planner integration and the deferred proof migration remain open.

### Nonzero mean enclosures in the existing Program trace

Extended the existing common `NoiseMoments` record with `mean_radius`.
Its rational mean fields now describe an interval center; zero radius still
represents the exact native-digit mean. This represents coefficient means
without allocating polynomial-sized vectors or a covariance matrix. Variance
continues to mean centered variance, not a second moment. Serialization uses
the new record directly; no old-format fallback was introduced.

Common add and repeated-source scaling preserve signed centers and radii.
A linear image uses its row-l1 gain to enclose every output coefficient mean.
This handles negacyclic sign changes: multiplication by 2-X need not preserve
a uniform scalar mean. The generic methods contain no scheme operations.
BFV MulPrivate and Lut now propagate these intervals with their prepared
right-action l1 caps, and MulPublic uses its known polynomial l1 norm.
This supersedes the previous entry's centered-input-only limitation: modeled
nonzero means no longer cause BFV to discard moments. Missing input models
still remain missing.

The variance formulas are conditional on centered deviations about these
means, with the prepared action fixed under the documented H4 heuristic.
Fresh/helper/remainder terms keep their existing centered models; inherited
bias is carried separately. A later average-case acceptance check must include
the mean interval in its deterministic clearance, not treat the stored center
zero of a symmetric interval as a centered error law. Such planner acceptance
is still unfinished. Exact public-action/source covariance precision also
remains open; interval containment is not an exact mean-vector computation.

Validation: the common mean-enclosure test enumerates native base-four digit
pairs for wrapped and nonwrapped 2-X coefficient actions, checks containment
and variance, and verifies repeated-source mean/variance scaling. It passed
once after a warning-free 29.51-second release build in
`exponent-mean-enclosure.log`. The same built binary then passed both BFV
noise Program tests once in `exponent-bfv-nonzero-mean-trace.log`, including
nonzero-mean explicit LUT, public multiplication and Add. No second build,
GPU, integration or Lean checks were needed. Formatting and diff checks passed.

### Mean-aware BFV Program audit and parameter-search policy

`BfvParameters::audit_program` and `search_program` now accept the existing
optional `AverageCaseConfig`. No config, or a config without its explicit
acceptance opt-in, retains worst-case authority. Average acceptance uses the
same per-wire/per-gate moments and plaintext bounds produced by the existing
interpreter. A missing moment model is an error, not zero noise or a hidden
fallback. Successful audit records its authority, event count and corrected
z-squared in `BfvNoiseTrace`; a raw trace has no acceptance authority.

Moved acceptance out of the evaluator's `record` method into the full audit,
so a worst-case failure cannot abort an average-case trace prematurely. The
hard audit still checks input, intermediate family, reduction and output
values. Audited values are deduplicated by their dependency node, and the
union count includes two polynomial components times ring dimension times
each active family cardinality. Captures are not cloned into new input events.
The existing transcript-based tail API shares its exact arithmetic with this
internally derived Program event count; callers do not supply an arbitrary
scalar event multiplier to either audit.

The common `NoiseMoments::fits` compares
`limit > |mean_center| + mean_radius + sqrt(z_squared * variance)` by exact
rational cross multiplication. Equality fails, including a zero-variance
mean interval that exhausts clearance. BFV applies the original modulus/8,
nonterminal scale/2 and terminal payload-range limits in physical units.
The average policy remains conditional on H1/H3/H4, the remainder model and
H6 tail closure. Its union budget covers this Program only; it does not yet
certify a complete refresh protocol or its masking/security obligations.
The refresh digit fixed-point planner is still worst-case.

Validation: both BFV noise Program tests passed once after a warning-free
30.86-second release build (`exponent-bfv-audit-policy.log`). They verify
worst-case default, explicit average acceptance, exact event counts for
captures/families, missing-model rejection and mean-radius rejection at the
strict boundary. A rational-clearance unit test and the seven small common
simulation tests also passed once using that same binary. The latter preserve
the existing transcript-derived tail factor and digit moment behavior.
An initial compile caught a temporary modulus-reference lifetime; it was fixed
before these successful checks. Formatting and diff checks passed. No GPU,
integration or Lean tests ran. Generic production setup, more precise source
covariance, refresh-wide average accounting and deferred proofs remain open.

### Backend-owned public PRF setup and common setup errors

Moved the existing concrete public PRF setup declaration and its matrix/layout
validation from `noise_refresh/setup.rs` to `bgg/public_key.rs`, naming it
`BggPublicPrfSetup`. Updated its request field and construction site directly;
there is no old-name reexport or compatibility alias. The same validation
still runs at the producer boundary; the existing dimension validator is now
crate-visible so the move does not duplicate it or add graph operations.

Replaced the common setup error's four BGG-specific variants (encoding,
encoding sampling, BGG sampling, and exponent-LUT boundary) with one backend
source variant. BGG conversions live in `bgg/mod.rs`; BFV supplies its conversion
in `bfv/mod.rs`. The boxed source retains its concrete error and source chain,
rather than flattening diagnostics into a string. This adds no successful-path
arithmetic, graph nodes, artifact fields or runtime scans.

Validation: the graph-free canonical snapshot test passed once after a
warning-free 29.56-second release build in
`exponent-setup-backend-boundary-final.log`. The same binary passed the selected
ring preprocessing/manifest graph fixture once (0.09 seconds), recorded in
`exponent-setup-backend-boundary-graph.log`. An initial compile identified the
moved validator's private dependency and a now-unused import; both were fixed
before the successful build. Formatting and diff checks passed.

This is a concrete ownership/error-boundary improvement, not completion of the
generic production setup. The common request still names BggPublicPrfSetup,
and its parameters, producer and importer retain other concrete BGG fields.
Both production backends must still be connected to shared orchestration.
No GPU, integration or Lean checks ran in this step.

### Schema-based production import for matrices and typed preimages

Replaced the production importer's manual Ring reconstruction and separate
matrix/preimage/family artifact constructors with existing `MatType`,
`PreimageType` and `FamilyType::artifact_input` transport. A single cache
Preimage schema now serves the setup-public-key family, scalar state scales,
mask/fresh route families and Program cache import. The decoder witnesses keep
their distinct private schema and per-slot switching ring.

The importer still validates the manifest before constructing any handle.
After that validation, it consumes the already-validated exact family counts
instead of repeating count lookup/positivity checks through another validation
closure. Canonical artifact names, production identity, confidentiality,
coefficient bounds, scalar/family storage and symbolic dimensions are preserved.
No new carrier, fallback, sampling operation or ciphertext arithmetic was added.
The parameter-to-schema formulas are still BGG-specific and remain work for
the generic production boundary; this removes duplicated transport, not all
remaining concrete setup ownership.

Validation: the selected-ring production preprocessing/import graph fixture
passed once after a warning-free 30.00-second release build, recorded in
`exponent-setup-schema-import-final.log` (test runtime 0.08 seconds). The initial
compile correctly required the existing MatType schema wrapper around an IR
MatrixType; that was fixed before validation. Formatting and diff checks passed.
No GPU, integration, long runtime refresh or Lean checks were rerun for this
transport-only change.

### Freeze producer state and release construction handles

`RefreshPreprocessingProducer` no longer owns its complete construction request
or `RefreshPreprocessingWires`. It retains resolved parameters, the frozen
BuiltGraph, the public declaration, and the already-existing compact noise
plans. The request's compilers, secret/selector handles, and temporary graph
carriers are released when construction returns. This does not remove private
outputs or sampling nodes from the executable preprocessing graph.

Artifact validation now runs as a private construction step against the live
request/wires before those handles are dropped. The actual readout/witness
schemas are compared with their expected compact schema once at this boundary,
and the noise plans are derived there. The later simulation assembly consumes
these plans directly instead of revisiting the retained construction graph.
Removed the now-unnecessary noise-plan accessor and public repeat-validation
method; no compatibility wrappers were retained.

The producer also reuses one PBC active-cell index across its input family-count
checks, artifact validation and noise-plan derivation, replacing three index
builds. This preserves member ordering and the same validation conditions.
It adds no cryptographic operations or online public preparation. The producer
still uses BGG-specific resolved parameters/plans, so generic production setup
is not yet complete.

Validation: after the final index-reuse change, the production graph/import
fixture passed once, warning-free (30.66-second build, 0.08-second test), in
`exponent-producer-frozen-state-final.log`. The test runtime matches the previous
0.08-second schema-import fixture; these short timings are not a memory or GPU
benchmark. Formatting and diff checks passed. No GPU, integration or Lean tests
ran. No live test processes remain.

### Reuse Program storage for BFV readout values and offsets

Removed the public `BfvReadoutValues` enum and private `BfvReadoutOffsets`
enum. Readout outputs now use `ProgramValue<T, (Family<T>, FamilyRange)>`,
retaining the active range beside the indexed values. Offsets use
`ProgramValue<BfvRow, Family<BfvRow>>` and its existing indexed/capture access.
The scalar accessor is shared on ProgramValue rather than reimplemented by
another backend carrier. No aliases or replacement wrapper structs were added.
All workspace references were updated; neither removed type remains in Rust.

Public versus encoding output records, level metadata, target weights and
prepared terminal phases remain distinct. Scalar/independent-family/summed
terminal behavior and aggregate-before-rounding arithmetic are unchanged.
Artifact import/export retains the same names, schemas and sharing; the change
only consolidates host-side structural carriers.

Before this change, the ordinary non-test library also passed `cargo check -r
-p mxx-exponent-lut` warning-free (9.91 seconds), recorded in
`exponent-normal-library-check.log`, covering the preceding API/error changes.
The readout change compiled warning-free (29.87 seconds), then the aggregate
readout artifact roundtrip passed once (2.79 seconds) in
`exponent-bfv-common-readout-carrier.log`. That fixture covers both shared and
indexed weights. The same binary passed the independent indexed-weight
artifact roundtrip once (6.05 seconds) in
`exponent-bfv-common-readout-indexed.log`. Formatting and diff checks passed.
GPU guidance was reviewed; no GPU code or arithmetic changed and intermediate
validation remained CPU-only. Generic production setup and the remaining noise,
proof and final GPU requirements are still open.

### Common scalar/family carrier for readout descriptors

Removed `BfvReadoutTerminalDescriptor`; the existing prepared readout descriptor
now holds `ProgramValue<BfvPreparedTerminalDescriptor,
BfvPreparedTerminalFamilyDescriptor>`. Export/import uses the common variants.
Scalar and Family variant names and their operation-specific payloads remain
unchanged; no alias, extra metadata record or runtime conversion was introduced.
Together with the preceding value/offset consolidation, three redundant
readout carrier enums have been removed. Prepared terminal arithmetic and
its phase/rounding distinctions remain backend-owned.

Validation: the aggregate readout artifact roundtrip, including shared and
indexed weights, passed once after a warning-free 30.21-second release build
(2.78-second test) in `exponent-readout-descriptor-carrier.log`. The preceding
same fixture took 2.79 seconds; arithmetic and artifact field layout were not
changed. Formatting and diff checks passed. No removed carrier names remain
in workspace Rust. No GPU, integration or Lean tests ran. The full refactor
is still incomplete, particularly generic production setup and full noise/
proof integration.

### BGG fixed-product artifacts use their actual GraphValue schema

Replaced the manually maintained count/matrix/bound fields in
`BggFixedProductDescriptor` with the schema of its existing prepared tuple:
`(Family<Preimage>, Family<BggPublicKeyWire>)`. Export now emits that tuple;
import reconstructs it through GraphValueSchema. Removed the matrix-only
public-key projection and the manual family reconstruction from scheme-ring
rows/columns. Public-key metadata and exact ring/count/bound information now
come from the actual prepared value, as in the common Program transport.

Artifact ports use the common tuple names beneath the operation prefix rather
than the old `.decomposition`/`.public` suffixes. The descriptor authenticates
the new schema; no compatibility fallback is retained. Dropped the descriptor's
unused Debug derive rather than imposing it on the lower-level key schema.
No encryption, decomposition scheduling or ciphertext arithmetic changed.

The production-path investigation confirmed a remaining phase mismatch:
`RefreshParameterSimulationRequest::build_with_mode` currently creates
`refresh-parameter-search-native-monomials` as an online input and samples its
label encodings with s after preprocessing/import. `PreparedSparseLwrPrf`
combines its label weights during fixed-product preparation. Connecting the
production path requires migrating these public label bindings and their
execution inputs to the preparation contract, then replacing the old label-
encoding cache/trace schedule. Replacing the evaluation call alone would not
preserve the public-key relation. This remains unfinished work, not a blocker.

Validation: the common BGG fixed-RHS PRF test passed once after a warning-free
29.77-second release build (0.64-second test), recorded in
`exponent-bgg-fixed-product-schema-final.log`. It covers descriptor JSON,
file artifact export/reopen, multiple output labels and two accumulation steps,
and asserts one structural decomposition body per composed step. An initial
compile exposed the unused Debug bound, removed before the successful check.
Formatting and diff checks passed. No GPU, integration or Lean checks ran.


## Public PRF labels cross the preprocessing artifact boundary

Moved the native label monomial family from an online external input to an
explicit public input of `RefreshPreprocessingRequest`. The producer exports
it as a canonical, content-hashed public artifact. The consumer imports the
family through GraphValueSchema with the same scalar polynomial type and
label-major cardinality as its setup public-key family. Manifest validation
includes the new artifact's role, shape and count; no coefficient scans or
per-member host expansion were introduced. Old artifact manifests are not
accepted through a compatibility path.

Online label encodings still use the online secret s and the cached public-key
preimages, but now consume imported label plaintexts. Updated both real GPU
refresh fixture callers to supply labels to preprocessing and use no external
label input online. The Rust protocol exporter retains the signed-monomial
contract at its new preprocessing input; this is a port adaptation, not Lean
proof migration. This supersedes the earlier online-label isolation description.

This is a dependency migration toward the required composed fixed-RHS path,
not completion of that path: the production accumulator still uses the old
label-encoding arithmetic/cache/noise schedule. Replacing it with
`PreparedSparseLwrPrf` and connecting its actual error terms remain necessary.
The additional label artifact is required by the current intermediate consumer;
it should disappear when the consumer needs only composed-RHS artifacts.

Validation: the existing selected-decoder-ring producer/import graph unit test
passed once after a warning-free release build (29.48 seconds; test 0.08 seconds).
It now also checks that the label family is an external preprocessing input and
is no longer an external online input. Evidence: `exponent-preprocessed-labels.log`.
This checks graph/schema construction, not numeric execution of the new label
artifact transport. No GPU execution, integration test or Lean build ran.


## BGG direct-product tracing consumes the prepared RHS error

The production fixed-RHS migration exposed a trace dependency: the BGG
`MulPrivate` interpreter previously ignored RHS values and always charged one
fresh helper error. A composed selector has all participating ciphertext
errors, even with a one-hot payload, so that interface could not describe the
required executed equation.

Removed `NoiseProgramRhs = ProgramValue<(), usize>`. RHS bindings now reuse
`NoiseProgramWire`, including its existing compact uniform/member family
representation. The direct transfer consumes the bound and doubled-coordinate
variance of the supplied RHS error. Scalar capture and indexed families use
the same map/zip operations as ordinary noise values; no new family enum or
traversal was added. Reductions still preserve the coherent covariance bound
for reused RHS sources. Explicit-table LUTs keep their fresh-branch model;
this does not charge a second external-product error for their fused RHS.

`ProgramNoiseInputs.rhs_errors` supplies composed scalar RHS errors to the
existing Program simulator. Ordinary unspecialized RHS bindings still denote
one fresh ciphertext using the helper sampler model. Composed-error derivation
must include every weighted ciphertext error, and known reused sources must
combine their actions before variance estimation. The direct transfer retains
its existing unit-norm plaintext contract; it does not establish arbitrary
polynomial payload covariance tracking.

Validation: the extended shared-RHS unit fixture passed once, checking both
coherent family reduction and a three-selector composed error through the
ordinary Program traversal. The existing heterogeneous/uniform family visitor
fixture also passed once using the same binary. Build was warning-free
(30.17 seconds); both tests took under 0.01 seconds. Logs:
`exponent-bgg-rhs-noise-final.log` and `exponent-bgg-rhs-family-noise.log`.
An initial compile identified two explicit input literals requiring the new
field; both were updated. Formatting and diff checks passed. No GPU,
integration or Lean execution occurred.

The production accumulator still needs to call the common fixed-RHS route and
supply these errors from its selector preparation. This prerequisite removes
an incorrect one-ciphertext assumption from the interface; it is not evidence
that the production arithmetic/cache/noise migration is complete.


## BGG compact noise-plan ownership leaves application setup

Moved the compact preimage/readout schema implementation, gate gain derivation,
CRT-corrected route scalars, mask decomposition plans and complete route gain
plans from `noise_refresh/setup.rs` into `bgg/noise.rs` (689 lines). These
functions depend on BGG gadget geometry, balanced decomposition, doubled
coordinates and decoder preimage bounds. The shared application should not
implement those formulas. Callers now import backend-owned plan operations;
private arithmetic helpers remain private, and the small schema types are
crate-visible only where the existing setup contract needs them.

This is an ownership move, not a change to the noise equations, accepted
heuristics, prepared graph, artifact schema, structural batching or cache
schedule. It introduces no new type, trait, module or evaluation traversal.
Setup still chooses digits, schedules preprocessing/online work and assembles
application reports. Its concrete BGG parameter/signature dependencies remain
and must be replaced by capabilities in the unfinished production migration.
The compact planner still follows the old accumulator until that migration;
relocation does not certify the new fixed-RHS arithmetic.

Validation: the selected-ring producer/import graph unit fixture passed once
with a warning-free release build (30.13 seconds; execution 0.08 seconds,
unchanged from the preceding recorded run). Formatting and diff checks passed.
Evidence: `exponent-bgg-plan-ownership.log`. No GPU, integration or Lean run.
The raw files remain large (setup 5,366 lines, BGG noise 4,811 including tests),
so this does not establish the final implementation-size requirement. No new
module was introduced merely to distribute these lines.


## Bind full PBC label rows to fixed selection blocks

Added `PrfSelectionInputs::new` to the existing common PRF input type. It
binds a full `[output][selector capacity]` public label family to one active
selector range, producing `[output][active member]` through a structural DSL
loop. The index is `output * capacity + active_start + member`; output and
member axes remain distinct, and no host enumeration or selector ciphertext
inspection is introduced. The existing block record, fixed-RHS capability and
prepared PRF traversal remain unchanged. Each block may use its own ring.

The BGG fixed-PRF fixture now passes full rows through this common binding,
including nonzero inactive selectors and their labels, rather than building
already-filtered label rows by hand. Its two-step CPU round-trip still passes,
including descriptor serialization, file export/reopen and the assertion of
one structural decomposition body per composed step. Build was warning-free
(29.46 seconds), and the test passed once in 0.66 seconds, versus 0.64 seconds
in the previous recorded fixed-PRF run. Evidence:
`exponent-pbc-fixed-label-binding.log`. Formatting and diff checks passed.
No GPU, integration or Lean execution occurred.

This supplies the common PBC indexing needed by production fixed-RHS
preparation. The production producer/importer still has to replace its old
cache descriptors and accumulator together with the corresponding noise
schedule. The fixture is not evidence of that unfinished production migration.


## BGG cache field declarations share their transport owner

Moved and renamed `ProgramCacheFamilyNames` to `BggProgramCacheFamily` in
`bgg/public_key.rs`, together with its canonical-name and validation methods.
The existing BGG cache exporter/importer no longer depends on a concrete
cache-layout type defined by application setup. Updated setup and the two
production PRF fixtures without aliases. Fields and serialized artifact names,
scalar/shared/indexed layouts, cardinalities and graph operations are unchanged.
Backend validation errors now cross the existing typed backend-error boundary.
This changes ownership, not the still-pending fixed-RHS production cache format.

The existing cache-name/descriptor round-trip unit passed once after a
warning-free 31.21-second release build (test under 0.01 seconds):
`exponent-bgg-cache-ownership-final.log`. Initial compilation identified imports
that moved with the implementation; these were corrected, with the remaining
setup-only test import placed under cfg(test). Formatting and diff checks passed.
No GPU, integration or Lean execution ran. Generic production setup and the
composed fixed-RHS cache/noise schedule remain unfinished.


## Route scalar lookup uses one structural loop

Simplified `corrected_route_scalar_family`: compute slot and digit from one
`route_coordinates` call inside one DSL parallel body, then read
`scalars[offsets[slot] + digit]`. Removed separate slot and digit families,
their subsequent zip and gathers, and empty capture placeholders. The compact
slot/digit scalar bank and its offsets remain shared captures. No host loop
over coefficients, component columns or expanded route members was added.
Mask/fresh counts, corrected scalar values and index ordering are unchanged.

The existing selected-ring production/import graph test passed once after a
warning-free release build (30.11 seconds; test 0.09 seconds versus 0.08 in the
preceding run). Evidence: `exponent-route-scalar-loop.log`. This establishes
valid graph/schema construction, not measured GPU performance or a numerical
refresh round-trip. GPU validation remains for the final pass; no integration
or Lean execution ran. Formatting and diff checks passed. This simplification
is separate from the still-unfinished production fixed-RHS cache/noise switch.


## Prepared RHS family errors reach the BGG Program interpreter

Extended the existing `ProgramNoiseInputs` with `rhs_family_errors`, using
`FamilyNoiseState` rather than introducing another scalar/family representation.
Family RHS binding now preserves supplied uniform or heterogeneous prepared
errors instead of replacing every member with one fresh-helper model. Default
ordinary ciphertext bindings retain that model. Added serialization to the
existing family state so snapshots preserve these values. No ciphertext work,
coefficient scans, alternate traversal or new independence assumption was added.

The existing family visitor unit now round-trips heterogeneous RHS errors
through JSON and the actual Program simulator. With other errors zero, bounds
3 and 5 aggregate to 8; variances 1 and 0 aggregate to 1, rather than the
fresh-helper defaults. The extended test passed once after a warning-free
30.05-second release build (execution under 0.01 seconds):
`exponent-bgg-family-rhs-errors.log`. Formatting and diff checks passed. No GPU,
integration or Lean execution ran. Production fixed-RHS preparation still has
to derive and bind these errors; its full cache/arithmetic switch is unfinished.


## Noise input bounds and variances share one state

Replaced `ProgramNoiseInputs.input_bounds`/`input_variances` with
`input_errors: BTreeMap<ProgramInputId, ProgramNoiseState>`. Replaced the four
public-family bound/variance maps with `public_family_errors`, using the
existing `FamilyNoiseState`. This removes map alignment and mixed-representation
reconstruction while retaining compact uniform families, heterogeneous member
states, doubled-coordinate units and missing-input rejection. No new carrier,
trait, arithmetic formula or independence heuristic was added. Old field names
and serialization fallbacks are absent.

Updated production snapshot input construction and all callers. Existing tests
retain their exact values and expected bounds/variances; equal bounds with
heterogeneous variances now use explicit member states. Removed the separate
BGG missing-bound/missing-variance errors and the unused copies of these input
errors in common refresh simulation. The latter still receives Program errors
through its existing backend conversion.

All 23 BGG noise unit tests passed once (29.94-second warning-free release
build; execution under 0.01 seconds): `exponent-noise-input-consolidation.log`.
The ordinary library check also passed warning-free in 5.24 seconds after
moving a previously test-only setup gain import under cfg(test):
`exponent-noise-input-check-final.log`. Formatting and diff checks passed.
No GPU, integration or Lean execution ran. Full production fixed-RHS migration,
generic setup, complete source/moment tracing and final validation remain open.


## Program reports retain the evaluator's actual noise states

Replaced the BGG report's separate `wire_bounds` and `wire_variances` maps with
`wire_states`, retaining the existing `NoiseProgramWire` values returned by
common tracing. Removed the projection loop that copied each scalar into two
maps and discarded non-scalar values. Readout variance and parameter-search
consumers now read the same state's doubled variance. Scalar/family identity,
source dependencies and the existing output-bound summary remain intact.
No new carrier, evaluator, moment formula or heuristic was introduced.

All 23 BGG noise unit tests passed once after a warning-free release build
(29.76 seconds; execution under 0.01 seconds): `exponent-noise-report-states.log`.
The updated parameter-search integration target was compiled only, without
execution, and passed warning-free (21.36 seconds):
`exponent-noise-report-consumer.log`. Formatting and diff checks passed. No GPU
or Lean execution ran. Production fixed-RHS migration and generic setup remain
unfinished; retaining complete returned states does not establish full per-source
covariance or all BFV refresh moments.


## Noise report output is a reference into its wire states

Replaced `ProgramNoiseReport.output_bound` with the declared output `WireId`.
`output_state()` borrows its existing scalar state, so readout, parameter search
and reporting use the same bound/variance pair rather than a copied bound and
a separately selected variance. Simulation validates the declared scalar output
once. The output may precede unrelated later gates. Removed the unused common
refresh missing-wire-bound/variance errors; no arithmetic or heuristic changed.
The old Rust proof adapter only changes its bound accessor; Lean work remains
deferred. No legacy output-bound field or serialization alias remains.

The existing family visitor/output-order unit passed once after a warning-free
release build (30.13 seconds; execution under 0.01 seconds):
`exponent-noise-output-reference.log`. Updated parameter-search callers compiled
without warnings in 21.59 seconds, without executing integration tests:
`exponent-noise-output-consumer.log`. Formatting and diff checks passed. No GPU
or Lean execution ran. The full goal, including production fixed-RHS cache and
noise scheduling, generic setup and remaining proof obligations, is still open.

### BGG refresh parameter ownership

Moved the sampler/gadget/decoder-specific `RefreshSetupParameters` definition,
its parameter and noise-planning methods, and its canonical identity payloads
from `noise_refresh/setup.rs` to `bgg/mod.rs` as `BggRefreshParameters`.
`BggRefreshAnchorSecurityInputs` now belongs to the same backend. No aliases
or format fallbacks were added. Identity payload fields, serialized schema
strings, cutoff formulas, and noise arithmetic are unchanged.

The common `RefreshPrfContract` no longer accepts the concrete setup type:
BGG validates it through `BggRefreshParameters::validate_prf`. Producer,
importer, noise planner, and parameter-search consumers use that method.
The common refresh module has no remaining setup-type reference. Backend
parameter identity computation no longer calls a function owned by setup.

The existing CPU selected-decoder-ring setup/export/import unit fixture
passed once after the identity move (0.07 s test execution). An unused import
warning was then removed without changing behavior. Consumer and library
build results are recorded below. No GPU, Lean, or integration test was run.

This is an ownership checkpoint, not completion of generic production setup:
setup still holds concrete BGG artifacts and its old production PRF cache
schedule. The large backend module still needs extraction of genuinely shared
orchestration; relocation alone does not meet the implementation-size goal.

Validation after import cleanup: `cargo test -r -p mxx-exponent-lut --lib
--no-run` and the `parameter_search_unit` consumer build (`--no-run`) both
completed without warnings. The integration target was compiled only, not
executed. `cargo +nightly fmt --all` and `git diff --check` completed.

### Consolidated BGG parameter-search snapshot construction

Replaced three public snapshot builders on `RefreshParameterSimulationRequest`
with one borrowed `BggRefreshParameters::build_noise_snapshot_from_public_layout`
in `bgg/noise.rs`. Its optional cached gain plan can carry invariant routes;
no cache wrapper type or compatibility entrypoint was added. The search caller
now makes one call instead of duplicating three branches. Removed the unused
standalone gain-only plan constructor. Invariant route planning and mask-route
gain calculation now belong to BGG noise, alongside the parameter methods
that build and evaluate snapshots; each file has one parameter impl block.

The current decoder plan is still derived even when state/fresh routes are
cached, and only mask routes are rebuilt for a mask-digit scan. The caller's
layout-bound active-cell index supplies family counts directly, replacing the
previous redundant index reconstruction. Cache compatibility is a construction
contract, without extra slot-length scans on the cached path. Numeric formulas,
residual independence, and report arithmetic are unchanged. Production graph
construction remains on its existing PRF schedule; this change does not claim
the fixed-RHS production migration is complete.

Validation: the existing `graph_free_snapshot_uses_canonical_setup_without_lowering_graph`
CPU unit test passed once, comparing cached and uncached full WorstCase and
AverageCase reports across mask digit counts (30.18 s build, <0.01 s test).
The `parameter_search_unit` integration consumer compiled without execution
(34.75 s). Both builds were warning-free. Formatting and `git diff --check`
passed. No GPU or Lean execution was needed for this host-only consolidation.

### Preserve native rows through online refresh binding

Removed `BggEncodingWire { plaintext: None }` wrapping from shared decoder bases
and the slot's scaled-state, mask, fresh-error and decoder rows. The common
shared-base preparation now returns its native `Mat` results directly; BGG's
slot binding consumes the same graph handles. Actual input/output encodings
retain their encoding type. The decoder's anchor/target binding record remains,
as do same-modulus anchor identity checks and the shared-secret preparation.
No matrix operations, sampler sites, switching placement or artifact fields
were changed.

Validation: the CPU `online_` unit-test selection passed 10 tests once
(30.07 s build, 2.52 s execution), including one-error-per-anchor and
one-secret/per-ring sharing. The existing grouped low-P slot-order/anchor-binding
unit test passed separately. Compilation was warning-free; formatting and
`git diff --check` passed.

The additional `switched_decoder_numeric_round_trip_with_shared_fresh_error`
run did not complete: despite its CPU backend, it initialized CUDA through
DCRT support and failed with `gpu_context_create failed: no CUDA-capable device
is detected` in the sandbox. This is not numerical round-trip evidence. Retain
that validation for the final GPU-capable run; no arithmetic or test expectation
was changed to bypass it. GPU execution and Lean remain deferred.

### Remove redundant online decoder binding wrappers

Removed the private `RefreshAnchor` and `RefreshDecoderPreimage` structs.
Their sole construction path recorded the freshly constructed anchor/target
handles and immediately compared those same values, without establishing an
additional relation between the decoder row and its target. The immutable
slot now stores the decoder row directly. Its sole private constructor is
called by `bind_imported_wires`, immediately after `prepare_refresh_slot`
constructs the decoder from the actual shared base and imported witness.
The source state, target, B, K and shared decoder-base identity remain in the
slot package; same-modulus base substitution checks remain in manifest binding.
The redundant B*K expression used only to record and compare its own handle
is no longer constructed. No executed row operation or ModDown was changed.

Validation: the existing grouped-low-P test passed once, including original
slot order, two structural switch bodies, and rejection of a substituted shared
anchor (30.44 s warning-free build, <0.01 s test). Constructor visibility and
its contract comment were tightened afterward. Formatting and diff whitespace
checks passed. Numeric refresh remains pending the final GPU-capable run;
production setup genericity and fixed-RHS migration are still incomplete.

### Retain the common prepared slot through BGG evaluation

`RefreshScalarPackage` now owns the existing `RefreshSlotValues<BggScheme>`
returned by `prepare_refresh_slot`, instead of destructuring it into four
backend fields and reconstructing it immediately before evaluation. The
common evaluator accepts borrowed slots and groups references with their
original output indices. BGG passes its package-value iterator directly;
there is no temporary vector of cloned row carriers or repeated level lookup.
The slot value has a fieldwise Clone implementation without requiring the
scheme itself to implement Clone. No new carrier or capability was introduced.

Level grouping still emits one symbolic body per homogeneous level, with
scaled-state + mask + fresh-error before switch and decoder application.
Original slot order is restored after each group. Backend binding metadata
remains separate from the common arithmetic values. This does not complete
the larger production setup/fixed-RHS or noise/proof migration.

Validation: the grouped-low-P CPU unit test passed once, checking slot order,
one switch body per level and rejection of a substituted shared anchor.
The library-test build was warning-free. Formatting and `git diff --check`
passed; no numerical/GPU/Lean tests were rerun for this carrier change.

### Rely on immutable refresh binding for source identity

Removed per-slot copies of the manifest state and A-prime handles. The sole
private slot constructor is called from `bind_imported_wires`, which derives
all scaled states from the one state subsequently owned by the manifest.
A-prime is owned only by the manifest and is never used as an independently
substitutable per-slot argument. Removed the repeated slot/source-handle scan
from `refresh`: private manifest fields and read-only accessors preserve the
binding after construction. Binding still checks slot order, mask/fresh shapes
and shared decoder-base identities. Compiler layout and slot-count compatibility
checks remain at evaluation because the compiler is supplied separately.
No arithmetic, source sharing, schema, or serialized identity input changed.

Validation: the existing grouped-low-P unit test passed once, including
rejection of a substituted shared anchor and preservation of slot order and
symbolic switch count (29.95 s warning-free build, <0.01 s execution).
Formatting and diff whitespace checks passed. This scoped cleanup does not
complete the generic production setup or fixed-RHS migration.

### Production fixed-RHS migration dependency audit

The current production path still uses `ProgramCacheBatchBindings<ProgramCacheBindings>`
in both preprocessing wires and imported setup. `BggProgramCacheFamily::canonical`
determines its artifact names and counts. Replacing evaluation alone would
leave a different public-key equation and an obsolete noise schedule.

The coordinated replacement must cover these concrete owners:

- `RefreshPreprocessingRequest`, `RefreshPreprocessingWires`,
  `ImportedRefreshSetup`, and `RefreshPreprocessingArtifactNames` currently
  carry `setup_public_key_rhs` and `label_plaintexts`. Fixed preparation uses
  the public label family to compose selector GSWs; it must remove the sampled
  per-label public-key/preimage family and its online label-encoding issuance.
  The label family is a preprocessing input, not an online dependency afterward.
- `BggPublicPrfSetup` supplies the initial key, selector family and public layout.
  `SparseLwrPrfProgram::active_block_ranges` already derives the compact global
  active-member capacity and block offsets; reuse it to bind
  `PrfSelectionInputs::new`, whose label indexing is [output][active member].
  Do not substitute padded bucket-width indexing for these compact offsets.
- `PreparedSparseLwrPrf<BggScheme>` provides the actual direct-product Program,
  per-step prepared products, final public output, descriptor export/import,
  and cached online evaluation. Its descriptor, not `BggProgramCacheFamily`,
  must authenticate the new preparation schedule and artifact bindings.
  Update producer name/count validation and import together with this replacement.
- The unchanged native digit readout consumes the new final public output for
  preprocessing and its matching online accumulator. The readout's setup-fixed
  switches and online s-dependent masks still belong to their existing phases.
- `BggNoiseSnapshot` and compact gain generation must use the same
  `fixed_accumulation` Program as the prepared evaluator, rather than the current
  `MulPrivate -> MulPublic -> FamilyAggregation::Add` Program. For each block,
  `C_j = sum_i w_ji C_i`, `F_j = sum_i w_ji F_i`, and
  `e_next = e D_j - mu F_j`, with `D_j = G^-1(C_j)` prepared once.
  Monomial label weights preserve each fresh coefficient bound; every active
  ciphertext error contributes even though the selected payload is one-hot.
  The H4 independent fresh-draw variance sum and reused-helper source identity
  must coexist; old selector-family accumulation gains are not valid evidence
  for this new schedule.

These dependencies were verified in the current source, including the existing
weighted-family combination and artifact import implementations. No production
route was switched during this audit, and no tests were rerun. The next substantive
edit must change the producer, persisted cache, online issuance and actual noise
schedule as one migration rather than retaining both production routes.

### Bind noise snapshots to the supplied execution Program

`BggNoiseSnapshot::from_setup` now accepts the actual `ExponentLutProgram`
and its separate `ValueReadout`, instead of accepting a PRF container and
implicitly selecting its old accumulation field. Gain-plan validation uses
that exact Program, and ring geometry comes from its bound `ProgramLayout`.
The readout's layout must match. The producer currently supplies its existing
accumulation explicitly, so current production behavior and noise values are
unchanged. The fixed-RHS migration can supply the prepared direct schedule
through the same constructor without adding a snapshot variant or backend
switch. Full RHS-error binding and the production cache migration remain to
be connected before switching that caller.

Validation: the existing graph-free snapshot/cache equivalence CPU test passed
once with a warning-free build. It compares complete WorstCase and AverageCase
reports while varying mask digits and reusing invariant plans. Formatting and
diff whitespace checks passed. No GPU or Lean validation was run.

### Execution-Program noise boundary (clarified user requirement)

The final API must pass the encoding evaluator's exact Program and its prepared
noise bindings to the general Program noise evaluator. PRF-specific snapshot
construction is transitional, not the target architecture. The Program
interpreter owns order, families, captures and reductions; backend transfers
own primitive error equations; PRF preparation binds input errors, composed
RHS errors, public weights and shared source identities. All selector errors
in a weighted fixed RHS are included before ordinary MulPrivate propagation.
Readout uses its corresponding shared evaluation path. Refresh-specific code
retains digit scheduling, mask-distribution and decoding-condition decisions.
Do not implement a second PRF recurrence or reconstruct a separate noise-only
accumulation Program to complete the migration.

### Prepared Program noise bindings replace evaluation-time PRF reconstruction

`BggNoiseSnapshot` now retains `ProgramNoiseInputs` supplied by preparation.
Removed the public-family cutoff/variance copies, the separate operation-residual
map, and `setup_average_inputs`, which previously traversed a PRF-shaped Program
each time it was evaluated. Default evaluation passes the retained bindings to
`simulate_program`; approximate-gadget preparation attaches residuals to those
same operation bindings. Scalar/family RHS errors are preserved in the existing
input structure rather than lost during reconstruction of default PRF inputs.

The current production preparation explicitly binds its sampled scalar inputs
and online public-label family errors before constructing the snapshot. This
preserves its old schedule and numbers while establishing the proper boundary
for the later fixed-RHS binding. Original refreshed-state moments remain
separate because refresh also consumes that state outside the PRF evaluation.
No PRF-specific propagation equation or new carrier was introduced. The deferred
Rust Lean adapter's public-input cutoff access now checks the actual family
bindings instead of the removed duplicate scalar; no Lean proofs were changed.

The preceding compact-plan change likewise accepts the actual Program and
its Family counts. The extended CPU fixture verifies that direct-program plans
validate against the direct chain and reject the old Family Program.

Validation: graph-free cache/report equivalence passed once (30.00 s
warning-free build, <0.01 s execution). The parameter-search integration
consumer compiled without execution and without warnings (26.80 s).
Formatting and diff whitespace checks passed. GPU and Lean were not run.
Production fixed-RHS and fully general readout/noise preparation remain open.

### Borrow prepared noise inputs during evaluation

Default WorstCase/AverageCase and pre-refresh evaluation now borrow the retained
`ProgramNoiseInputs`; the override APIs borrow their supplied bindings as well.
Removed full-map/state clones on each normal evaluation. Candidate-specific
mutation remains an explicit caller-owned copy. No traversal or transfer changed.

Validation: the graph-free report/cache equivalence unit test passed once;
the parameter-search consumer compiled without execution. Both builds were
warning-free, and formatting/diff checks passed. Production fixed-RHS,
complete general readout/noise binding, final GPU validation and deferred Lean
work remain unfinished.

### Separate the BGG value-readout transfer from the refresh snapshot

Extracted `evaluate_value_readout_noise` in `bgg/noise.rs`. It consumes the
already evaluated Program output, prepared readout gains/residuals, target
plaintext gain, and borrowed helper-error parameters. It returns the existing
`ProgramNoiseState`; it does not accept a PRF profile, label topology or refresh
plan. The snapshot now only connects `simulate_program` to that transfer and
wraps the result for refresh decisions. The signed-monomial payload contract,
H1/H3/H4 assumptions and additive reconstruction-residual model are unchanged.
The readout transfer is not yet wired through a complete cross-backend common
noise-readout capability; this is removal of the PRF snapshot dependency, not
completion of that capability or production fixed-RHS migration.

The existing graph-free cached/uncached report equivalence test passed once
with a warning-free build. Helper arguments were then changed to direct
references to avoid allocating/cloning a temporary error state; validation of
that signature change uses a targeted library check. No GPU or Lean runs.
The final borrowed-parameter version passed `cargo check -r -p mxx-exponent-lut`
without warnings (5.24 s), with formatting and diff checks clean.

### Explicit prepared RHS noise bindings

Removed the noise simulator's implicit one-fresh-helper fallback for scalar
and family RHS ports. Preparation now binds their error states explicitly,
including the legacy production selector families. The evaluator consumes
those states through the existing shared Program traversal. Composed selector
RHS bindings retain all selector errors; the H4 and reconstruction-residual
independence heuristics are unchanged.

Updated existing unit fixtures to supply their intended RHS distributions,
including the deterministic RHS in the single-variable aggregation case.
Validation: 25 BGG noise unit tests passed and the graph-free setup snapshot
unit test passed. Release compilation emitted no warnings. No GPU or
integration tests were executed. Production fixed-RHS migration and shared
BGG/BFV setup/import are still unfinished; this checkpoint does not complete
that migration.

### Share the caller's PRF Program across execution and candidate planning

`RefreshParameterSimulationRequest` now receives and owns a prepared
`SparseLwrPrfProgram`, replacing its profile-only field. Its executable and
graph-free paths consume that Program. The compact snapshot and invariant
route APIs also accept a borrowed Program; BGG noise code no longer rebuilds
PRF Programs from profiles. Parameter-search context stores the Program once,
uses its identity directly, and passes it to both compact evaluation and
selected-candidate graph construction. Existing CPU/GPU fixtures and Rust
Lean fixture call sites were updated for the constructor contract; no Lean
proof work was performed.

Validation: the graph-free snapshot/report-equivalence unit test passed after
the final constructor change (`exponent-shared-request-program.log`),
and the parameter-search integration target compiled without execution
(`--no-run`, `exponent-shared-request-consumer.log`). Both release builds
were warning-free. Formatting and diff whitespace checks passed. GPU execution
remains deferred to final validation.

This removes schedule reconstruction in the noise entrypoints, but does not
yet switch production from the legacy selection Program to fixed-RHS
composition. That switch still requires coordinated replacement of public
label-key artifacts, accumulation caches, online label encodings, and their
prepared error bindings. Shared BGG/BFV setup/import and deferred Lean work
remain required for full completion.


### Common fixed-product artifact transport

Fixed-product caches now implement the existing DSL GraphValue/schema contract.
The common PRF owner exports and imports those fields directly. Removed the
BGG `BggFixedProductDescriptor`, BFV `GateDescriptor`, the descriptor associated
type, and both backend export/import callbacks from `ExponentLUTFixedRhsScheme`.
The existing `BfvPublicKeyFamily` is generic over its matrix carrier so its
schema retains key/context identities, level, active range, and scalar/shared
storage without a separate transport record.

PRF preparation returns `(public_output, prepared_prf)`. The surrounding
readout/setup owns that public output; the online PRF owner retains operation
caches only. BGG fixed-product caches now contain only decomposition families,
removing intermediate public-key families from retained/exported step caches.
A caller that needs the final key after reopening exports it once through the
ordinary GraphValue interface. BFV retains its existing prepared-operation
fields; this change does not introduce another exported final-key copy.

The common descriptor still authenticates the Program identities, scheme
binding, artifact names and exact cache schemas. The artifact producer identity
and manifest bind imported values. Existing BGG/BFV numeric PRF fixtures were
updated for the returned public key and the common transport; their final GPU
round trips remain outstanding.

Validation: the CPU-only `test_bfv_fixed_product_schema_preserves_family_bindings`
passed, including a serialized schema, shared input, indexed output, nonzero
active-range start, and consumer validation against the producer manifest.
The library test target compiled and release `cargo check` passed without
warnings. Formatting and whitespace checks passed. No GPU, integration, or
Lean tests were executed. Production fixed-RHS setup migration, shared BGG/BFV
setup/import, and the other full-plan requirements remain unfinished.

### Keep online stage composition in its construction owner

Removed the single-use `RefreshOnlineEvaluationRequest` and
`RefreshOnlineEvaluation` records. The production bundle builder now connects
accumulation, imported readout, online-secret decoder preparation and refresh
directly. Its caller only consumed the refreshed encoding; intermediate
accumulation/readout/manifest values no longer travel through an unused result
record. This preserves the existing arithmetic and preprocessing/online
boundary while locating the pending fixed-RHS substitution in one construction
path. Imported setup documentation now refers to that stage rather than the
deleted request type.

Validation: the CPU graph-free snapshot unit test and the structural
`switched_preprocessing_artifacts_use_only_selected_decoder_rings` test passed.
The latter builds the actual benchmark preprocessing and online graphs and
checks public-label placement, decoder bases and public ModSwitch placement.
This is graph-construction evidence, not a numerical round trip. Formatting
and diff whitespace checks passed. No GPU, integration or Lean tests ran.
Production fixed-RHS replacement and generic BGG/BFV setup/import remain
unfinished, along with the remaining full-plan requirements.

### Prepared PRF exposes only its executed Program

`PreparedSparseLwrPrf` no longer retains the full logical selection Program
beside its direct-product execution Program. It retains the logical source's
identity for artifact authentication, the actual execution Program, and the
prepared operation caches. `program()` now returns that execution Program;
the competing `accumulation()` accessor was removed. The descriptor names
these identities `source` and `program`, and import authenticates both.
Existing BFV fixture bindings now compare the actual `program()` with the
Program used for their noise trace.

Validation: all exponent-lut library test targets compiled in release mode
without warnings. Formatting and diff whitespace checks passed. No arithmetic
or artifact matrix fields changed, and no GPU/integration/Lean tests ran.
This eliminates an ambiguous Program accessor; it does not complete the
production fixed-RHS switch or the remaining shared-setup and noise-binding
work.

### BGG public polynomial gains in the Program noise evaluator

`GateNoiseGains.public_plaintext_action` now binds the public polynomial's
integer-lift l1 and l2-squared bounds separately from the private-key
decomposition gain. The shared Program-driven BGG evaluator uses these norms
in both scalar and Family MulPublic transfers:
`e_out = nu_public * e_private + e_public * K_private`.
Native PRF preparation explicitly supplies unit monomial gains. No ciphertext
coefficient scans or cryptographic runtime operations were added.

Extended the existing Program Family test with public gains l1=3 and
l2-squared=5; it exercises the actual evaluator and its correlated reduction,
not only a standalone recurrence. All 25 BGG noise unit tests passed. The
parameter-search target compiled with `--no-run`. Its library build exposed
two stale imports from the earlier online-request deletion; those imports were
moved to test scope, and the final release check was warning-free. Formatting and whitespace checks passed. GPU, integration and
Lean execution remained deferred.

This closes the unit-public-polynomial assumption in MulPublic's transfer.
It does not yet establish general plaintext norm propagation through every
wire/gate, nor complete production fixed-RHS preparation, its selector-source
bindings, shared BGG/BFV setup/import, or deferred certificate work.

### Production BGG refresh uses the common fixed-RHS PRF

The production `RefreshPreprocessingProducer` now composes the public native
label monomials with all active selector GSW ciphertexts through
`PreparedSparseLwrPrf<BggScheme>`. Each block prepares one decomposition family
indexed by output. Public readout consumes the returned final public-key family.
The declaration carries the common PRF descriptor and its exact GraphValue
schemas; import takes the caller's logical Program and authenticates the
prepared direct-product schedule before reopening the cache.

Online construction calls the prepared PRF evaluator. It no longer samples
per-label public keys, exports their preimages or public label artifacts, or
samples label encodings under the online secret. Public labels remain producer
inputs; online evaluation consumes only the composed product caches. The
redundant preprocessing encoding-compiler field and duplicate public selector
wrapper were removed. Public/encoding PRF benchmark roles remain attached to
the corresponding construction phases. The former BGG-specific batch cache
export/import methods were deleted.

Production and graph-free noise planning now target the common fixed
accumulation Program: one MulPrivate per PBC block, with no selection Family
reduction. For a block with m active signed-monomial weights, its RHS binding
has bound m B_chi and doubled variance m V_chi under the existing independent
fresh-draw heuristic. Every selector error participates, including selectors
whose payload is zero. The general Program evaluator applies the product
transfer. Existing custom-gain fixtures now bind the direct execution schedule.

Validation:

- The CPU production graph test passed through preprocessing export/import and
  online refresh construction, preserving two selected-ring public switches
  and decoder groups. It also checks that public label inputs occur only in
  preprocessing and that online evaluation has no gadget decomposition.
- The graph-free snapshot/cache comparison passed, with new assertions that
  the Program identity is the direct execution identity and every RHS binding
  includes the complete active-selector bound and variance.
- The existing decomposition-gain/variance sensitivity test passed.
- The parameter-search consumer compiled with `--no-run`; no integration,
  GPU, or Lean execution was performed.

The first production graph checks exposed concrete Family dimensions retained
as division expressions. The common weighted-combination preparation now
constant-evaluates resolvable dimensions, retaining symbolic dimensions for
nested loops. This changes construction metadata only. The final passing unit
build was warning-free; the subsequent consumer library build still reports
11 dead-code warnings for old PRF APIs/fixtures that have not yet been removed.

This is a production-path switch, not completion of the refactor. Remaining
work includes deleting the obsolete selection/batch APIs and migrating their
fixtures, sharing production setup/import with BFV, completing noise moments
and source/payload propagation, final GPU round trips and workspace checks,
and the separately deferred Lean migration. Numeric production round-trip
correctness for the changed path remains to be verified on GPU.

### Fixed-RHS round-trip fixtures and obsolete sampling removal

The exact native-domain PRF round-trip fixture and the GPU one-output PRF
fixture now prepare/evaluate `PreparedSparseLwrPrf` and serialize its descriptor.
They preserve their independent signed-LWR oracle, real artifact-store reload,
and repeated online sampling. Public monomials enter preprocessing. The
exhaustive fixture creates an isolated preprocessing store per public-label
case; the GPU fixture supplies its selected public-label vector before the
preprocessing execution, leaving no label input in online execution.

Removed `sample_setup_public_key_families` and `sample_label_from_preimage`,
which no longer have production or round-trip callers. Their structural/secret
checks now concern the actual fixed-RHS preparation and the initial encoding's
distinct mask/payload secrets. Also removed the unused
`NativePrfPublicReadoutPlan` wrapper, getters, and constructor API; the common
prepared PRF returns the final public key directly to ordinary readout.

A 49,172-member, two-output CPU graph fixture validates the concrete types and
requires exactly one structural weight multiplication, one gadget decomposition,
and one public external product. The balanced sum legitimately emits logarithmic
addition scopes; the test checks the expensive operation multiplicities rather
than an arbitrary total-node ceiling. This fixture and the distinct-secret
initial-encoding fixture each passed once. The updated CPU unit target compiled
warning-free. The GPU unit target also compiled warning-free with `--no-run`;
that check exposed and fixed the BFV shared PRF fixture module's visibility to
its sibling GPU module. Neither numerical round-trip fixture was executed in
this checkpoint, and there was no integration or Lean execution.

Older select-batch compiler APIs and their standalone fixtures still remain;
removing the two samplers does not establish full removal of that older API.
Production BGG/BFV setup unification, complete noise/source propagation, final
numeric GPU evidence and workspace checks, and deferred Lean work remain open.

### Legacy select-batch API removal

Removed `NativePrfPublicInputs`, `NativePrfOnlineBatchInputs`,
`ProgramCacheBatchBindings`, `PublicProgramBatchPreprocessing`, and
`BggProgramCacheFamily`, together with their constructors, transport projections,
canonical-name machinery, and old public/encoding batch compiler entrypoints.
The shared PRF module no longer contains the duplicated `prepare_batch` and
`evaluate_batch` interpreters. A workspace Rust-source search found no remaining
references to these APIs.

The migrated structural tests use the common fixed-product preparation and
artifact transport. The public-side test retains 17 independent outputs and
requires one output-indexed decomposition family. The encoding-side test
serializes/reloads the common descriptor, validates the imported manifests,
and checks two direct product scopes with one product each and no online
gadget decomposition. The general shared-RHS Program round-trip retains its
residual oracle and repeated evaluations; it now inspects the existing shared
Program cache directly instead of wrapping it in the deleted PRF batch cache.
The former name-schema test described the deleted shared/local artifact format;
common descriptor serialization/import is checked by the replacement fixture
and the migrated round-trip fixtures. The former online-exponent wrapper test
was removed with that online-label boundary; native label arithmetic remains
covered by the signed-LWR round-trip oracle.

Both targeted CPU structural tests passed once. The CPU unit-target build and
parameter-search consumer `--no-run` build were warning-free. Formatting and
whitespace checks passed. No integration, numerical GPU, or Lean tests were
executed in this checkpoint.

Remaining work includes replacing the logical-selection/direct-execution
Program pair with one authoritative Program, generic BGG/BFV production setup,
complete primitive moments/source/payload propagation, final workspace/GPU
verification, and the deferred IR/Lean migration. Removing the old batch API
does not establish completion of those requirements.

### 2026-09-09: authoritative PRF Program and integrated GPU workflows

`SparseLwrPrfProgram` now owns the actual private-product accumulation Program.
Preparation, cached encoding evaluation, and noise tracing consume that Program;
the separate logical-selection/fixed-accumulation Program pair was removed.
The generic structural-selection regression builds its own explicit Family
Program, preserving its original per-member selection oracle.

The two Cargo integration entries are `tests/lwr_prf.rs` and
`tests/noise_refresh.rs`, selected with `MXX_EXPONENT_LUT_TEST_SCHEME=bgg|bfv`.
Each entry runs simulation and its subsequent GPU stages in one invocation.
The PRF entry performs one-output readout and a file-store round trip. The
refresh entry measures preprocessing and online graphs after simulation.
Standalone plan-file handoff and the PRF test's dependency on an unrelated
accepted refresh-search report were removed. Small PRF fixtures check arithmetic
and conditional noise bounds, not lattice-security certification.

The GPU wrappers in these integration files were flattened; a file-level GPU
feature requirement expresses their runtime contract. The scheme selector and
BFV PRF-specific setup are written directly in the integration files. Shared
parameter-search and BFV refresh/runtime/noise fixtures remain in `tests/support`
because library regression tests use them too; duplicating those implementations
would make the integration and regression oracles drift. Four empty legacy test
directories were removed.

Unit ownership was tightened: readout indexing lives in `bgg/mod.rs`, noiseless
PRF evaluation in `bgg/encoding.rs`, decoder/artifact GPU execution in
`bgg/gpu.rs`, and search regressions in `noise_refresh/simulation.rs`. The GPU
integration entries no longer contain separate arithmetic-only accounting tests
or a separate monomial-boundary test; the latter assertions run in the BGG PRF
round trip itself.

BFV fixture execution uses ordinary fresh runtime execution and imports the
actual manifest after reopening the store. This preserves the artifact boundary
without durable-resume journaling. BGG PRF now uses the same execution pattern;
its final verification executes the online graph twice, retaining all numeric
and noise assertions. Benchmark graphs exclude correctness oracles and, for
refresh, the post-refresh continuation gate. Measurement first collects shapes,
measures them, then estimates the same production graphs. `DslContext` can be
cloned to select production-only output boundaries from the same DSL values.

Validation so far: workspace CPU/GPU library builds and the later targeted
builds were warning-free. The authoritative-Program and graph-free noise
snapshot unit checks passed once. BGG readout schema persistence passed once.
BFV PRF passed two fresh GPU runs (about 7.6 seconds each). The BGG combined
flow passed with both final online executions (22.66 seconds including graph
estimation, preprocessing and store reopening; the earlier journal-backed flow
was 123.33 seconds). BFV refresh simulation plus GPU estimate passed in 2.46
seconds. These are workflow checks, not isolated performance benchmarks; other
GPU validation was in progress. Full BFV refresh/store/continuation repeats are
still running and are not claimed successful here.

The broader non-Lean work is not complete: production refresh setup/PBC still
has BGG-specific boundaries, complete generic moments/source/payload propagation
needs its final audit, and the deferred IR/Lean work remains outstanding.

### 2026-09-09: common model selection and BFV terminal moments

The previous turn made implementation progress (integration ownership and
normal artifact execution). Its BFV full-refresh repeat runner remains live;
no completed result is inferred from elapsed time.

BGG decoder-specific parameter projections were moved from application setup
into `BggRefreshParameters` methods in `bgg/public_key.rs` and `bgg/encoding.rs`.
The generic decoder grouping routines and their one-sample-per-modulus behavior
remain unchanged. The structural shared-anchor test passed once.

Both integration files now select the actual acceptance model through
`NoiseModelKind::from_env`, using `worst-case` or `average-case` and defaulting
to average. The duplicate test-only model enum was removed. BFV PRF uses traced
CBD input moments and the new terminal moment transfer; BFV refresh feeds the
selected terminal/decoder envelope into its mask-digit fixed point. Terminal
internal terms are summed before adding the single outer rounding contribution.
The H1/H3/H4 fresh-target decorrelation, remainder and H6 assumptions are
explicitly documented in the average-case specification. Worst support bounds
remain available. No ciphertext scans or runtime model checks were added.

BGG PRF now shares exact tail-envelope arithmetic with BFV, with its physical
variance conversion explicit. The refresh integration accepts either average
evidence or matching worst-case production reports. Noise-only preparation
and search no longer invoke the explicit Lean checker; certificate checking
remains a separate operation (including explicitly requested checkpoint
certification).

Validation: the BFV fixed-point test exercises both models, rejects missing
average moments and verifies that regrouping terminal branches does not add
outer rounding. It passed once. Exact combined-mean/tail rounding passed once.
BFV PRF GPU checks passed once per mode: selected bounds were 7 (average) and
22 (worst) in the small fixture. BFV refresh simulation/estimate selected 15
mask digits with average exposed error 329, versus 16 with worst error 2472.
BGG PRF worst-case passed its two online executions in one integration run.

The worst-case estimate exposed an order-dependent measurement initialization
bug: a full-shape request could allocate synthetic operands before any fleet
column width was selected. The estimator now assigns that request's actual
operand/output width before preparation, without changing production kernels
or synchronizing/serializing execution. A regression makes centered extension
the first request; it passed two GPU runs. An initial version of that regression
retained extra GPU parameter contexts and was corrected to release them before
exclusive memory measurement. BFV worst-case simulation/estimate then passed
with the fix. Workspace CPU/GPU library builds were warning-free; later narrow
CPU validation also passed. Full BFV refresh repetitions and the remaining
production-generic setup, source/moment audit and deferred Lean migration are
still outstanding.

### 2026-09-09: inline integration fixtures and online binding ownership

The previous status turn was a verified wait: session 51237 remained live.
The full BFV refresh process was subsequently verified from the host process
namespace (PID 3631392, active CPU work); a sandbox process listing did not
show it and is not terminal evidence. The runner and original executable were
not restarted or replaced.

`RefreshCompiler::bind_online_outputs` now lives in `bgg/encoding.rs`.
Canonical aggregation still calls the existing generic refresh operations;
BGG decoder generation and affine offset construction belong to the adapter.
Contract/layout checks use the imported declaration before routing. The
transient routing carrier no longer duplicates those unused metadata fields.
The shared-fresh routing CPU regression passed once.

The user requested complete inlining of test support. Both support files were
removed. Search setup and its regressions now live together in
`tests/noise_refresh.rs`, eliminating the concrete BGG test-harness dependency
from generic simulation. Its CPU build and reviewed-tuple regression passed.
Refresh artifact GPU checks moved with their fixtures into the refresh
integration file. The redundant CPU full-runtime fixture was removed under
the user's GPU runtime contract; the exact fresh-error and continuation
assertions remain in the GPU check. PRF keeps only its own smaller fixture,
without unused benchmark graphs or a refresh-residue option. The two small
execution fixtures are local to their different workflows; the full refresh
setup and parameter-search harness are not duplicated. Empty support was
removed and no source file imports a test-support path.

This completes the requested support-directory cleanup, not the full refactor.
Generic production setup and the complete source/moment audit remain pending;
Lean remains deferred. CPU library/integration compilation and GPU compilation
of both relocated integration targets passed warning-free. These build checks
are separate from the still-running full-refresh repetition.

### 2026-09-09: one authoritative parameter binding

The previous turn made progress by removing test-support sources, relocating
consumers, and removing unused PRF benchmark graphs. Production setup still
needs its full generic boundary.

The preprocessing declaration now has six fields instead of 29. It retains
its identity, PBC identity, canonical PRF contract, artifact names, prepared
PRF descriptor and readout identity. The producer/import owner already owns
the authoritative parameters, and the existing setup digest binds them all.
Duplicated dimensions, digit counts, sampler settings, levels and scale
expressions were removed from the declaration. Import still validates the
supplied parameters, recomputes their digest and checks artifact types,
confidentiality and names. Redundant checks against copied parameter fields
were deleted. The PRF contract is serialized directly; its six-field
reconstruction helper was removed. No graph operations, samplers or rounding
positions changed, and the setup identity hash computation is unchanged.

The existing selected-decoder-ring preprocessing/import test passed once after
the final change (0.08 seconds), and the setup identity digit/security-binding
regression passed once. CPU library compilation was warning-free.

At the user's subsequent request, both integration files now have a single
file-level `#![cfg(feature = "gpu")]`; all per-item GPU guards were removed.
Consequently their search regressions also require the GPU feature to compile,
while ordinary library unit tests remain CPU-buildable. This supersedes the
previous checkpoint's description of CPU-only integration regressions.

### 2026-09-09: generic direct-refresh artifact transport

`BfvRefreshDescriptor` and the specialized `PreparedDirectRefresh<BfvScheme>`
export/import implementation were replaced by the generic
`PreparedDirectRefreshDescriptor` and common export/import methods in refresh.
The existing direct-refresh capability now supplies concrete artifact binding,
output-key type, and key-dependent recomposition/schema preparation. It does
not receive a callback implementing the refresh protocol. Identity validation,
output-key transport and ordered slot transport remain visible in the common
application. Two intermediate slot-list export/import wrappers were removed.

BFV retains its key/context binding and polynomial-block schema definitions.
The same output-key, target, offset and term artifacts are emitted under the
same names and confidentiality roles. Recomposition still constructs only its
existing public coefficients; no public terminal sums or ModDown are repeated
on import. The descriptor hash domain is now scheme-independent and includes
the concrete binding; no compatibility alias or old-domain fallback remains.
The real BFV refresh integration fixture uses the generic importer, including
its descriptor serialization round trip. Both integration targets compiled
with GPU support warning-free. Final workspace builds are in progress; no new
runtime success is claimed for this transport move yet.

The user's next instruction removes all `#[ignore]` attributes from the two
integration files, including their retained parameter-search regressions.
README commands no longer use `--ignored`; GPU gating remains file-level.
Existing full-refresh runner 51237 was polled and remained live; it continues
using its original executable, which has not been overwritten by workspace
library builds or integration-only builds.

### 2026-09-09: BGG route-witness ownership

The prior direct-refresh transport checkpoint's workspace builds completed
warning-free: CPU libraries in 34.71 seconds and GPU libraries in 33.69 seconds.
The previous turn therefore made implementation progress, not only a wait.

BGG state-scaling and weighted mask/fresh route decomposition construction now
lives with `BggRefreshParameters` in `bgg/public_key.rs`, in its existing impl
block. The setup producer requests the three prepared witness carriers and
passes them to the existing generic public-route/sum/decoder orchestration.
No new wrapper type or expanded loop over route family members was added.
Corrected digit/CRT scalar construction and BGG route-width calculation moved
with their owner; the decomposition order and shared fresh routing are intact.
CPU library compilation passed warning-free in 31.07 seconds. The selected-ring
preprocessing/import graph regression passed once after this move.

Remaining work includes the concrete BGG production setup container/import and
simulation boundary, final complete source/moment accounting, and the deferred
IR/Lean migration. The new generic direct-refresh transport still needs its
final runtime verification; old running binaries do not establish that result.

After the route move and removal of ignore attributes, both integration GPU
builds passed warning-free (27.19 seconds). Listing ignored tests from each
built executable returned zero tests. No integration workload was launched
by those inventory checks. The integration file still contains the retained
search regressions and diagnostics; making the application search reusable
and reducing that harness to the final minimal workflows remains part of the
broader architecture cleanup, not a completed consequence of deleting support.

### 2026-09-09: canonical PRF cache artifact ordering

The prior turn made implementation progress by moving BGG route preparation
into its owner and verifying the actual setup graph. This turn removes the
per-step `(artifact name, schema)` pair from `PreparedSparseLwrPrfDescriptor`.
A descriptor now stores its prefix once and a vector of schemas. Export,
import and BGG manifest validation use one canonical artifact iterator, with
names derived from the Program step position. Emitted artifact names, shapes,
confidentiality and decomposition counts remain unchanged. The descriptor
identity commits to the new compact representation; no old-format fallback
was added. This changes preparation metadata, not online cryptographic work.

CPU library compilation passed warning-free (30.03 seconds). The existing
BGG common-PRF test passed once with descriptor serialization, actual producer
execution, store reopening and cache import (0.71 seconds). The selected-ring
production setup/import regression passed once (0.09 seconds). GPU integration
compilation is checked separately; the ongoing original full BFV runner does
not validate later transport changes.

### 2026-09-09: common refresh-level grouping and selected round-trip model

Refresh preparation and evaluation now use the same `group_refresh_levels`
helper for canonical level groups. The helper replaces BGG's `anchor_groups`
method without changing sampler sharing or arithmetic order. CPU library
compilation and GPU integration compilation passed warning-free. The shared
online-anchor and selected-decoder-ring preprocessing regressions passed once.
An additional numeric test using `cpu_backend` attempted CUDA initialization
inside the sandbox and failed before producing a round-trip result; this is
not counted as a correctness pass.

The original BFV complete-refresh binary passed its first run in 4186.19 seconds.
Its second run was stopped at the user's request; it is not counted as a pass
or a correctness failure. Those runs use the earlier implementation.

The current complete BFV refresh integration test now shares the benchmark's
noise-model and failure-exponent selection instead of forcing worst case.
GPU integration compilation passed warning-free in 11.15 seconds. A current-code
round trip was launched with `MXX_EXPONENT_LUT_REFRESH_NOISE_MODEL=average-case`
and `MXX_EXPONENT_LUT_REFRESH_SECURITY_BITS=5`. Simulation selected 14 base-4
mask digits for each of five groups, one fresh digit, 568 distinct PRF outputs,
and exposed-error bounds of 133. Runtime completion is still pending.
Integration tests remain unignored, with one GPU feature gate per file.

### 2026-09-09: public PRF material preparation ownership

The previous turn made progress by enabling and launching the requested
average-case, five-bit complete-refresh round trip, and stopping the old second
run. The requested run remains live; no runtime result is inferred from its
completed simulation.

`BggPublicPrfSetup::sample` now owns the public-key sampling, terminal-helper
sampling and normalization gadget construction previously embedded in the
application simulation builder. The existing setup object carries the results;
no additional request/result carrier or protocol callback was introduced.
The caller retains its payload-secret sample, online-secret sample, structural
selector family and artifact orchestration. Sampler tags, polynomial shapes,
helper sharing and preprocessing order are unchanged. The online builder also
reuses its existing scheme for readout instead of constructing it twice.

CPU compilation passed warning-free in 30.57 seconds. The existing production
setup/import graph regression passed once in 0.07 seconds, checking selected
rings, two trapdoor samples, two public switches and absence of label inputs
from online evaluation. GPU-feature compilation is checked without starting
another GPU workload. This is a concrete ownership extraction; the remaining
BGG-specific setup owners and the generic production boundary are not yet
complete.

### 2026-09-09: construction-owned setup artifacts

Removed `RefreshPreprocessingProducer::validate_artifacts`, which recomputed
identities, canonical names, matrix shapes and family counts immediately after
constructing those values from the same validated request. Its duplicated
route-count scan and private family-count helper are also gone. The input
public-key contract, readout/noise schema agreement and external manifest
validation remain: externally supplied identities, roles, shapes and private
artifact confidentiality are still checked at their existing boundary.

The `RefreshPreprocessingWires` carrier no longer retains `masks` or
`scaled_fresh` just for that internal validation. Routing results move into
their structural families; private preimage export iterates the actual
preimages and canonical names. All construction sites still derive counts
from the validated channel schedule. No artifact, sampler or arithmetic node
is removed, and no validation has been transferred into online gate execution.
The public setup builder also reuses its existing scheme across readout and
routing.

CPU compilation passed warning-free in 29.83 seconds. The existing selected-ring
setup/import regression passed once in 0.09 seconds, retaining its checks of
public switch placement, grouped trapdoor sampling and preprocessing-only
labels. GPU-feature compilation is checked separately. The requested BFV
average-case five-bit complete-refresh runtime remains live and is not counted
as a pass yet. The broader generic production boundary and deferred proof
migration remain unfinished.

### 2026-09-09: direct export of construction-owned values

Removed the twelve-field `RefreshPreprocessingWires` temporary carrier and its
single-use `add_setup_outputs` adapter. The preprocessing owner now exports
its existing locals directly, preserving the exact artifact order, names,
confidentiality and handles. Public matrices and route families transfer their
ownership into the DSL outputs. Readout and preimage handles remain available
only for the existing actual-schema noise-plan extraction after freezing.
The prepared PRF's own Program still supplies the noise plan; it is not
reconstructed from another schedule. No new type or callback replaces the
removed carrier, and no online arithmetic or artifact schema changes.

CPU compilation passed warning-free in 29.26 seconds. The production graph
and import regression passed once in 0.08 seconds, preserving grouped sampling,
public-switch counts and preprocessing-only labels. GPU-feature compilation is
checked without an additional runtime run. The five-bit average-case BFV
complete-refresh session is still live; its final result remains pending.
This removes a pass-through concept but does not complete the remaining generic
setup migration.

### 2026-09-09: one authoritative preprocessing input key

Removed `RefreshPreprocessingRequest.state`. It was required to refer to the
same matrix handle as `public_prf.initial`, so it represented neither another
key nor a distinct phase. Public PRF accumulation, refresh sums and state-key
export now all use `public_prf.initial`. This makes their identity structural
and removes the corresponding equality check and duplicate construction clone.
The existing public-PRF input validation still checks key shape and visibility;
import continues to reconstruct its key from the declared artifact.

CPU compilation passed warning-free in 31.38 seconds. The existing production
setup/import graph regression passed once in 0.08 seconds. This common Rust
input cleanup introduces no GPU operations; no extra GPU workload was started.
The requested BFV average-case five-bit complete-refresh session remains live.
The overall generic setup, complete noise-source accounting and deferred Lean
migration remain active work.

### 2026-09-09: one decoder-preparation body per group

`prepare_refresh_decoders` now produces each public target and its private
witness together in one structural `parallel` body. Previously a target family
was produced by one loop and consumed by a second witness loop. Both values
remain explicit outputs, preserving artifact roles and slot order; the group's
anchor and reduced output key are still captured once. The public switch,
scale, subtraction and preimage sampling stay in their original element-wise
order. No scheme switch or opaque protocol callback was introduced.

CPU compilation passed warning-free in 30.40 seconds. The production setup and
import regression passed once in 0.09 seconds, retaining grouped trapdoor and
ModSwitch counts and online/preprocessing input separation. GPU-feature
compilation is checked separately. This changes structural scheduling on the
witness path, so its final GPU runtime check remains required; the running BFV
direct-decoder test does not cover this BGG preparation change. The requested
BFV five-bit average-case complete-refresh session remains live.

### 2026-09-09: workspace compilation checkpoint

After the public-key/input/export cleanup and combined decoder-preparation
body, all workspace library test targets compile warning-free both without
GPU (29.95 seconds) and with GPU (33.03 seconds). These are `--no-run` checks;
no additional runtime workload was started. The requested complete BFV
average-case five-bit round trip remains a live, separately tracked execution.

An explicit search over all Rust files under `noise_refresh` locates the
remaining concrete BGG/BFV symbols only in `setup.rs`. Its preprocessing
request/producer/declaration/import owner, parameter-simulation request/bundle
and concrete test fixtures still need migration. This narrows the remaining
ownership work; absence of such symbols elsewhere does not prove the full
production boundary or complete noise/proof coverage. The goal remains open.

### 2026-09-09: backend-owned deterministic cache schema

`BggRefreshParameters::refresh_cache_schema` now supplies the deterministic
preimage shape and floor(beta/2) bound used by both import and manifest
validation. Removed the two independent reconstructions in application setup.
Manifest validation converts that same DSL schema through ir-core's existing
concretization and artifact-type facilities; scalar caches, PRF families and
mask/fresh route families share the same role/type/count check. No new schema
wrapper or transport trait is introduced, and Gaussian helper bounds remain
separate from deterministic decomposition bounds.

CPU compilation passed warning-free in 30.60 seconds. The existing production
setup/import graph regression passed once. This changes preparation metadata
ownership, not online arithmetic or GPU scheduling; no new GPU workload was
started. The requested BFV five-bit average-case complete-refresh session is
still live and has not produced a final result.

### 2026-09-09: backend-owned prepared-readout schema

The three setup call sites that reconstructed the BGG readout schema now use
`BggRefreshParameters::refresh_readout_schema`. The backend owns its field
layout; application scheduling still supplies the logical PRF output count.
The same schema is used for producer agreement, consumer artifact import and
manifest validation. Existing boundary error mappings remain in place, with
backend errors preserving their source through the application error carrier.
No additional type, arithmetic operation, output or runtime check is added.

CPU compilation passed warning-free in 30.27 seconds after correcting the
backend error conversion. The existing production graph/import regression
passed once. The requested BFV average-case five-bit complete-refresh process
was also inspected outside the sandbox: it was active after 20 minutes 41
seconds, with 20 minutes 5 seconds of accumulated CPU time. Runtime completion
remains pending; no restart or additional GPU workload was performed.

### 2026-09-09: one declaration/manifest boundary validator

Producer finalization and consumer import now both call
`RefreshPreprocessingDeclaration::validate_manifest`. Removed import's duplicate
parameter, contract, identity, canonical-name and manifest checks. The shared
validator returns its resolved decoder preimage bound, which both artifact
validation and typed witness import reuse. It takes an immutable manifest;
validation does not mutate transported data. The consumer still independently
checks the requested Program identity before constructing artifact handles.

All prior identity, type, confidentiality and family-count conditions remain.
No generic container was added merely to hide the remaining concrete setup
owner. CPU compilation passed warning-free in 30.79 seconds and the production
setup/export/import graph regression passed once. The requested BFV five-bit
average-case complete-refresh runtime is still active and remains unverified
until its final result arrives.

### 2026-09-09: resolved setup owns compiler metadata

Removed the `compiler` field from `BggPublicPrfSetup`. Its ring, base and digit
count duplicated the resolved refresh parameters, while its exponent layout
came from the existing PRF profile. The BGG parameter owner now constructs
compiler metadata through `public_key_compiler`; public preparation, import
and online graph construction use that source. The public-key sampler, helper
samplers and normalization gadget are unchanged. Input schema validation now
compares against the resolved setup ring directly. No new carrier type or
serialized metadata was introduced.

CPU compilation passed warning-free in 29.85 seconds. The existing production
setup/import graph regression passed once. This is a metadata ownership change,
with no extra GPU workload. The requested average-case five-bit BFV complete
refresh session is still active. These reductions do not yet establish the
fully generic setup boundary or the deferred proof coverage.

### 2026-09-09: preimage regression uses production noise functions

Removed the test-only `direct_bounded_preimage_plan` implementation from common
setup. It duplicated the production schema extraction and bounded-plan
conversion already owned by BGG noise. Moved its slot shape/bound regression
into the existing `bgg/noise.rs` test module, where it now calls
`compact_preimage_schema` and `bounded_plan_from_schema` directly. The regression
checks both distinct input bounds and the original rectangular shapes and gains.
The artificial mismatch guard belonged only to the deleted test helper; actual
manifest/schema validation is unchanged and remains covered by the setup tests.

CPU compilation passed warning-free in 30.68 seconds and the migrated production
noise regression passed once. No new helper, module, runtime validation or GPU
workload was introduced. The requested BFV five-bit average-case complete-refresh
session remains live, with its final result pending.

### 2026-09-09: route regression uses the production Program plan

Replaced `setup_decomposition_changes_route_gain_and_digit_variance` with
`setup_gadget_changes_route_gain_without_changing_digit_distribution`. Removed
its manually rebuilt gate traversal, synthetic per-gate error records, readout
records and alternate route-plan builder. The replacement obtains both the
Program gain plan and route decompositions from the production parameter owner,
then builds snapshots through the existing production path. It compares gadget
bases four and eight for the same native PRF and digit schedule: route gains
differ while mask and fresh uniform moments agree. This also removes test-only
imports of the manual decomposition/plan builder from common setup.

The new regression passed once in 0.00 seconds. After removing an unused import,
CPU compilation passed warning-free in 30.11 seconds; the unchanged test body
was not rerun. No runtime or GPU behavior was modified. The requested BFV
five-bit average-case complete-refresh execution remains live and pending.

### 2026-09-09: application request owns the PRF schedule and PBC layout

Moved the Program and PBC layout fields from `BggPublicPrfSetup` into the existing
`RefreshPreprocessingRequest`. The BGG object now holds only the initial public
key, terminal helper material and normalization gadget. Its sampler receives
the required readout width instead of owning an application Program and layout.
Program/PBC validation and active-range scheduling belong to the application
request; matrix/schema validation remains with the backend material. Public
preparation, readout, identity derivation and noise-plan extraction all reference
the same request-owned frozen Program.

No fields are duplicated and no new carrier type or protocol callback was
introduced. Sampler inputs/tags, output artifacts and cryptographic operations
are unchanged. CPU compilation passed warning-free in 30.19 seconds and the
production setup/import graph regression passed once. The requested BFV
five-bit average-case complete-refresh execution is still live. The remaining
concrete parameter/material types in setup still prevent claiming a completed
generic production boundary.

### 2026-09-09: one validated PRF profile in refresh contracts

`RefreshPrfContract` now contains the Program identity, output wire and existing
`SparseLwrPrfProfile`, replacing four independently stored dimension/base fields.
Its accessors project the validated profile without repeated validation. Profile
deserialization already validates the layout and digit-base ranges. The compact
contract serialization changes intentionally, without an old-format fallback.
Readout identity remains separate: `SparseLwrPrfProgram::id` identifies its
accumulation Program and does not replace the terminal readout binding.

CPU compilation passed warning-free in 30.25 seconds. The existing setup-identity
regression passed once with added contract serialization/reload and comparisons
against the profile's modulus/base/ring projections. Mask/fresh digit-count and
security bindings remain covered. The requested BFV five-bit average-case
complete-refresh execution is still live and has no final result yet.

### 2026-09-09: shared PBC-to-PRF input binding

Added `SparseLwrPrfProgram::selection_inputs`, a backend-independent binding
operation over the existing fixed-RHS capability. Production refresh setup,
the BGG PRF fixture and the PRF integration workflow now share its canonical
PBC range traversal instead of rebuilding the same input vector three times
in source. It retains the existing block-specific constructor for operands
whose labels or ciphertexts differ by block. Shared selectors are cloned as
handles under a method-local `Clone` bound; no global backend requirement or
new carrier is introduced. Label gathering, source IDs, active ranges and
structural loop extents are unchanged.

CPU compilation passed warning-free in 29.05 seconds after removing obsolete
fixture imports. The production setup/import graph regression passed once in
0.07 seconds. GPU integration compilation passed warning-free in 18.73 seconds;
no extra GPU workload was started. The requested BFV five-bit average-case
complete-refresh execution remains live. Concrete setup parameters/materials
and the deferred proof migration remain unfinished.

### 2026-09-09: validated-profile construction contracts

Removed the setup builder's repeated ring-dimension comparison: its immediately
preceding `validate_prf` call already compares that same profile dimension with
the resolved setup. Public-material validation performs setup-dimension checks
once before decoder-bound resolution. The backend PRF compatibility check no
longer repeats positivity conditions guaranteed by the contract's validated
profile; it still checks the digit base and ring dimension against this setup.
No external artifact checks or cryptographic operations were removed.

CPU compilation passed warning-free in 30.31 seconds. The production setup and
import graph regression passed once. The requested BFV five-bit average-case
complete-refresh session remains live and pending. These contract reductions
do not replace the remaining associated-type migration of the setup owners.

### 2026-09-09: artifact cardinality follows the readout schema

Removed `expected_setup_artifact_count` and its unused helper-count argument.
Manifest validation now counts the six protocol outputs, three outputs per
slot, actual anchor groups, fixed-product cache outputs and actual readout
schema ports. It reuses the same port list and group schedule for subsequent
field validation instead of computing the groups twice. No extra schema
construction or ciphertext work is introduced.

Removed the isolated arithmetic-only count test: its expected constant was
stale (29 versus the helper's computed 27) and it never constructed an export.
The existing production setup/export/import regression exercises the actual
manifest count together with type, role and group checks; it passed once in
0.09 seconds. CPU compilation passed warning-free in 29.54 seconds. The
requested BFV five-bit average-case complete-refresh session remains live and
pending; no additional GPU workload was started.

### 2026-09-09: requested BFV average-case round trip passed

The requested complete BFV refresh run finished successfully: one test passed,
zero failed, in 3689.30 seconds. Its logged average-case configuration has
`failure_exponent = 5` and average acceptance enabled. The planner selected
base-four mask counts `[14, 14, 14, 14, 14]`, one fresh digit, 568 distinct PRF
outputs and exposed-error bounds of 133. The executed fixture reopened the
artifact store, checked exact fresh-error replacement and continued with the
next LUT. The old second worst-case repetition was previously stopped at the
user's request and is not included in this result. No additional repetition
was launched. The tests README now includes the matching parameter overrides.

Current workspace library targets also compile warning-free without GPU
(29.06 seconds) and with GPU (32.32 seconds). These were compilation-only checks.
The successful direct-decoder BFV run does not cover the later BGG witness-body
scheduling change. The generic setup boundary, final source/moment coverage and
deferred IR/Lean migration remain unfinished; this runtime success does not
establish completion of the overall refactor.

### 2026-09-09: setup naming consumes the committed identity

`RefreshPreprocessingArtifactNames::new` now consumes the resolved identity,
slot count and shared decoder-group count. It no longer interprets BGG
parameters or computes the cryptographic commitment. The producer computes
that commitment once, reuses the actual prepared group count, and constructs
its declaration directly. Removed the single-use `make_declaration` helper
and the cloned artifact-name collection; exports borrow the declaration's
names. Manifest validation also computes the expected commitment only once.
The hexadecimal prefix is formatted once per name collection.

Artifact spellings, ordering, confidentiality, parameter binding and graph
operations are unchanged. The production setup/export/import regression
`noise_refresh::setup::tests::switched_preprocessing_artifacts_use_only_selected_decoder_rings`
passed once (0.08 seconds), following warning-free CPU compilation (29.34
seconds). Formatting and `git diff --check` passed. No GPU run was started.
This removes parameter-dependent naming from the shared workflow; the
remaining concrete setup owners still require migration and this checkpoint
does not claim that the generic production boundary is complete.

### 2026-09-09: simulation requests use authoritative decoder parameters

Removed the duplicate `decoder_sigma` argument from
`RefreshParameterSimulationRequest::new` and all eleven workspace call sites.
The decoder sampling parameter remains in the resolved setup and is used by
preparation there. The constructor no longer compares that value against a
second caller-supplied copy. Existing setup validation and positive decode
length validation remain at construction. The private, consuming request
builder no longer repeats setup and public-layout validation already performed
by its constructor. No sampling formula, error distribution, artifact format
or execution graph operation changes.

The production setup/export/import unit regression passed once in 0.09 seconds
after warning-free CPU compilation (29.82 seconds). The updated
`noise_refresh` integration target compiled warning-free with the GPU feature
(24.36 seconds); it was not executed. Formatting and diff whitespace checks
passed. Lean caller changes only update the constructor signature; proof
migration remains deferred. Concrete setup parameter and material owners still
need the associated-type migration required by the full plan.

### 2026-09-09: concrete refresh fixtures belong to BGG

Moved nineteen BGG setup, noise and refresh tests and their numeric fixtures
from `noise_refresh/setup.rs` into the existing `bgg/encoding.rs` test scope.
BGG GPU tests and Lean test fixtures now refer to that backend-owned scope.
The native PRF exponent-domain check moved into the existing `prf.rs` tests.
No new files or production visibility were added; common setup is now 1740
lines, although its remaining concrete production dependencies are unfinished.

Deleted the test-only `exact_family_count` function and assertions testing
only `Option` equality. Real output/route cardinality and overflow assertions
remain, as does the production setup/export/import regression. The latter
passed once after relocation (0.08 seconds; warning-free CPU compilation in
29.27 seconds). Sampler operations, family layouts, artifact exports and the
GPU fixture body are unchanged; GPU runtime repetitions were not resumed.

GPU-feature library tests also compiled warning-free in 31.37 seconds.
Formatting and `git diff --check` passed.

### 2026-09-09: PRF artifact validation follows prepared schemas

The setup manifest validator is generic over the prepared PRF binding and
`GraphValueSchema`. Its PRF checks no longer substitute the BGG route-cache
schema or assume one exported field per step. Readout and PRF fields share
one schema-driven traversal, preserving DSL port names, scalar versus indexed
storage, exact element types, family cardinalities and public hashes. Artifact
cardinality now counts actual schema ports. State-scaling and route-witness
schemas remain supplied by the BGG parameter adapter until those setup owners
are migrated. There are no new carrier types, ciphertext scans or graph nodes.

The production BGG setup/export/import regression passed once after the change
(0.08 seconds), with warning-free CPU compilation in 28.75 seconds. Formatting
and `git diff --check` passed. No GPU runtime was started. This verifies the
existing BGG instantiation; full BFV production setup through the common owner
and the remaining associated-type conversion are still outstanding.

### 2026-09-09: preprocessing declarations parameterize PRF storage

`RefreshPreprocessingDeclaration<B, D>` now carries the existing generic PRF
binding and prepared-value schema instead of fixing its field to BGG's
preimage family. Producer/import signatures project those types from
`ExponentLUTFixedRhsScheme`. No alias, wrapper, runtime tag, additional trait
or serialized field was introduced. The declaration's parameter commitment,
Program/readout identities and artifact names retain their existing meaning.
The current manifest-validation implementation still accepts BGG parameters;
generic declaration storage alone does not complete the production boundary.

The production setup/import regression passed once in 0.08 seconds following
warning-free CPU compilation (29.24 seconds). The GPU-feature `noise_refresh`
integration target compiled warning-free in 24.85 seconds and was not run.
Formatting and diff whitespace checks passed.

### 2026-09-09: backend field declarations and shared manifest matching

Moved witness-setup artifact dimension formulas into the existing
`BggRefreshParameters` implementation in `bgg/public_key.rs`. Its
`refresh_artifact_types` supplies existing artifact types, confidentiality and
family counts for the state/output keys, setup secret, grouped public anchors,
slot targets/witnesses, state scales and route caches. No new schema carrier or
capability trait was introduced.

The lower-level setup manifest matcher no longer accepts BGG parameters or
computes gadget/decoder dimensions. It combines backend field declarations
with PRF/readout schema ports, then applies one type/count/confidentiality/hash
check per field. Actual declared fields determine artifact cardinality. The
existing separately bound selector-family check is retained. Group order,
preimage bounds, private/public roles and exported graph operations are
unchanged. The upper declaration validator and producer/import owners still
need their remaining concrete parameter dependencies removed.

The production setup/export/import regression passed once in 0.08 seconds
following warning-free CPU compilation in 29.05 seconds. Formatting and
`git diff --check` passed. No GPU execution or additional runtime repetition
was started.

### 2026-09-09: backend commitment validation and full unit-test audit

Moved setup parameter/commitment validation from the generic preprocessing
declaration to `BggRefreshParameters::validate_refresh_manifest`. The BGG
adapter also supplies the selector's separately PBC-bound matrix-family
contract. Shared manifest matching now takes ordinary field declarations and
typed PRF/readout ports without any BGG parameter or selector-specific branch.
No additional contract trait was introduced.

At the user's request, paused further structural refactoring for a full
exponent-lut unit-test audit, excluding deferred Lean tests. The first CPU run
finished with 222 passed and five failed (77.84 seconds). Three failures were
obsolete evaluation-time public-attribute/layout rejection tests superseded by
the explicitly accepted construction contracts; those tests and their unused
test-only `compile_program` wrapper were removed. The readout regression's
independent expected expression omitted the already modeled gadget residuals;
it now includes target, trace and normalization residual terms and passes.

The fifth failure exposed a real unsupported nested indexed-family cache.
BGG LUT branches are already prepared as a static vector by the public
compiler. `ProgramLutCache` now retains those fields directly instead of
packing another indexed axis inside the Program family's cache. Evaluation
packs the fields for its existing parallel branch loop; cryptographic
operations and structural iteration over Program members are retained.
The failing family preprocessing/cache-consumption regression now passes.
The cache schema intentionally changes without a compatibility fallback.

The initial GPU-feature suite had 222 passed and 27 failed (77.37 seconds).
Besides the same five CPU failures, all 22 GPU-specific tests encountered the
runtime's explicit shared-pool calibration rejection for concurrent contexts.
The suite was not serialized to hide that failure. After the cache fix, the
isolated GPU Program/LUT regression passed in 1.08 seconds. This supports the
concurrency diagnosis but does not establish an all-passing GPU suite.
Both final workspace library builds passed warning-free: CPU 28.97 seconds,
GPU 35.05 seconds. The final full CPU rerun is tracked in the task log.

The final CPU rerun passed all 224 non-Lean unit tests in 83.03 seconds.
No test processes remain running. The concurrent GPU calibration failures
and the previously recorded full-plan requirements remain unresolved.

### 2026-09-09: GPU calibration isolation audit

The runtime measures the default CUDA pool's high-water mark. Its context
counter and reset guard are process-local (`Runtime.cu`); the fleet rejects
multiple live contexts before measuring a pilot. Removing that guard would
invalidate the measurement, and serializing compute would violate the GPU
parallelism requirements. No CUDA or runtime implementation was changed.

To separate arithmetic failures from test-process resource sharing, the 22
GPU-specific exponent-lut unit tests were run as separate processes with
four concurrent workers. Each invocation uses the already built GPU library
test binary with `--exact <test-name> --nocapture`, from the exponent-lut crate
directory, outside the sandbox. Internal runtime parallelism is unchanged.
This execution mode does not establish support for multiple independently
calibrating runtimes in one process; that restriction remains explicit.

The full-plan setup boundary also remains incomplete: production setup,
import, and simulation owners in `noise_refresh/setup.rs` still name BGG
parameters and values. Passing unit tests must not be presented as evidence
that both schemes already instantiate one generic production setup owner.
Lean migration remains deferred at the user's request.

### 2026-09-09: smaller refresh unit fixture

At the user's request, reduced the shared BFV refresh unit fixture's default
ring dimension from eight to four. The existing environment override remains
available. Base-four digits, two mask digits versus one fresh digit, both row
components, CRT reconstruction, exact error replacement, and the following LUT
are retained. The native input domain is still larger than the digit range.
The digit-input family size and the branch count each halve; digit/branch work
therefore decreases by a factor of four before any modulus-plan differences.

The dimension-four CPU regression passed in 14.02 seconds, compared with the
previous dimension-eight regression within the 83.03-second full CPU suite.
The old GPU invocation was explicitly stopped while switching to the smaller
fixture after 648.96 seconds; its SIGTERM is not an arithmetic failure. The
other 21 isolated GPU unit tests passed. The replacement GPU refresh test is
tracked separately. Both updated workspace library builds passed warning-free:
GPU 31.62 seconds, CPU 30.17 seconds. No GPU arithmetic or calibration code
was changed.

### 2026-09-09: noiseless BFV refresh unit coverage

The user requested noiseless BFV unit tests with substantially smaller CRT
parameters. Replaced the expensive noisy refresh fixture rather than keeping
both. The default ring dimension is four, each prime is ten bits, and the
explicit nested bases have depths four, two, and one. The existing primitive
ring-dimension, CRT-bit and CRT-depth environment overrides remain available.
There is no worst-case parameter search or `max_crt_depth` budget in this
arithmetic fixture. The intermediate noisy depth-eight candidate passed its
CPU check in 6.78 seconds but is superseded, and the earlier dimension-four
noisy GPU run was stopped for the requested replacement.

`BfvParameters::new` now accepts CBD(0). Error construction emits zero directly
for this degenerate distribution, and a zero-noise decoder emits its exact
secret/public product. The production parameter planner still requires positive
eta. Secrets remain randomly sampled; RNS ModDown, rounding residues, decoder
application, extraction and CRT reconstruction still execute normally. This is
noiseless sampling, not removal of rounding or a security-parameter claim.

Both row components must have exactly the prescribed restored fresh error;
the following LUT must obey its deterministic error bound. Base-four digits,
two mask digits versus one fresh digit, and multiple CRT channels remain.
No small equivalent BFV noiseless fixture existed; the BGG noiseless PRF test
checks a different scheme and protocol. Noisy full-refresh coverage remains in
the existing integration test.

The renamed CPU regression passed in 2.56 seconds after warning-free compilation
in 29.47 seconds. Full CPU and replacement GPU validation are tracked below.

The updated full CPU-library binary run passed 223 tests; one BGG decoder
fixture failed in the sandbox because its primitive dependency initialized a
GPU. The same exact test passed outside the sandbox in 11.44 seconds. All 224
non-Lean CPU-library tests therefore have passing observations, with that
single environment-dependent rerun explicitly distinguished from a clean
single-invocation suite result. Updated workspace library compilation was
warning-free for CPU (30.06 seconds) and GPU (36.03 seconds).

The replacement noiseless GPU refresh/continuation regression passed in 205.95
seconds. Together with the 21 previously successful isolated GPU unit tests,
each GPU unit case now has a passing observation; the changed noiseless case
was rerun after its replacement, and the unchanged cases were not repeated.
The original same-process shared-pool calibration failure is not claimed fixed.
No test jobs remain running. The deferred Lean work and previously documented
generic-production-setup gaps remain outside this unit-validation checkpoint.

### 2026-09-09: readout consumes the ordinary Program value types

Removed `ReadoutPublicInputs` and `ReadoutEncodingInputs` from the readout
capability. Its preparation/projection/evaluation methods and the PRF wrappers
now accept the existing `PublicProgramValue<S>` and `EncodingProgramValue<S>`
aliases. These use the encoding capability's `PublicKey` and `Encoding` types;
no replacement carrier or conversion trait was introduced. BFV already uses
exactly these representations.

BGG resolves its matrix/vector fields inside the readout adapter. Production
setup and online evaluation now pass their actual PRF result directly, as do
the PRF unit/integration callers. The family projection remains `Family::field`,
which selects existing graph fields and schema metadata without inserting a
parallel computation, sampler or arithmetic node. The existing BGG batched
readout receives the same underlying matrix handles and retains its artifact
schema and cached normalization. The scalar case packages one existing field
for that backend's batched operation; Program family members are not expanded
into a host loop.

The existing noiseless BGG PRF round trip (24.49 seconds), readout artifact
reopen regression (0.91 seconds), and production switched-setup regression
(0.08 seconds) each passed once. Both workspace library builds were warning-free
(CPU 28.99 seconds, GPU 35.79 seconds). Formatting and `git diff --check` passed.
This removes one concrete-field conversion boundary from shared setup; it does
not complete generic setup parameters, decoder issuance or imported owners.

The changed GPU-enabled `lwr_prf` integration target also compiled warning-free
in 18.25 seconds; it was not executed. No test processes remain running.

### 2026-09-09: import reuses the validated artifact type contract

`ImportedRefreshSetup::import` no longer reconstructs BGG gadget columns,
anchor-group matrix dimensions, per-slot target dimensions, decoder preimage
bounds, or route-cache element schemas. The existing BGG adapter first checks
the manifest against its authoritative parameter-derived field declarations;
shared import then uses those exact validated artifact types and roles to
construct consumer graph handles. A small local concrete-to-symbolic matrix
conversion contains no encoding arithmetic. No new carrier or trait was added.

Witnesses and route caches remain typed preimages, with their original bounded
small-RHS execution path. Imported families retain the validated cardinalities;
slot/group order still follows the committed artifact names. The backend's
validation no longer returns a separately reconstructed decoder bound that
import does not need. Public construction checks and identity/confidentiality
validation remain in place, with no coefficient inspection or online public
preparation added.

The production switched-setup regression passed once after warning-free CPU
workspace compilation in 29.17 seconds. Formatting and `git diff --check`
passed. No GPU execution was added for this metadata-only change. The setup
owner and its request/parameter types still require the broader generic
migration; this checkpoint does not claim that boundary complete.

### 2026-09-09: commonization paused at the user's request

Replaced the concrete `BggPublicPrfSetup` record with `PrfPublicSetup<S>` in the
shared PRF module. It stores the initial public key, readout preparation, and
public readout masks using the existing encoding/readout associated types.
BGG sampling and parameter checks remain in its public-key implementation;
shared preprocessing no longer unpacks BGG helper sigmas, switch ciphertexts
and the normalization gadget. The old record was removed without an alias,
and no additional trait or duplicate record was introduced.

CPU workspace library compilation passed warning-free in 28.76 seconds. The
existing production switched-setup regression passed once in 0.08 seconds;
formatting and `git diff --check` passed. No GPU execution or integration run
was added for this structural change. The generic record is currently used by
the BGG production owner; this does not establish a common BFV production setup.

The user explicitly requested pausing commonization at a stable checkpoint.
No further commonization or Lean work is to resume without user direction.
No test or build jobs remain running. The current worktree is preserved.
Outstanding full-plan work includes generic setup requests, parameters and
imported owners instantiated by both production routes; the remaining noise
source/moment audit; and the deferred IR-derived Lean certificate migration.
The same-process GPU shared-pool calibration restriction is also unchanged.
These are unfinished requirements, not completion claims or technical blockers.

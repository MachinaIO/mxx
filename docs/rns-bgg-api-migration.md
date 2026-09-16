# RNS-BGG API migration notes

## Stage 5-A contract

Program setup now has one structural derivation path:
`derive_program_requirements(program, input_levels, family_metadata, family_inputs)`. It
records declaration-to-wire mappings, operation order and
`ProgramGateInstance`, checked wire levels, RHS/LUT references, helper needs,
and symbolic family capacity/range/body information. Verified external family
levels are supplied separately as `family_inputs`; missing facts fail closed.
It does not store or
sample matrices, secrets, parameters, material, caches, seeds, or nonces.

`prepare_program` validates this contract before sampling. Private
multiplication and LUT transfer from an input at level `L` require level
`L + 1`; Add and public multiplication require equal operand levels and retain
that level. Structural families are inspected as one local body with
`FamilyProgram::bind_inputs`; root and local identifiers are resolved in
separate maps, and members are not enumerated on the host. The
existing scalar-only setup rejection remains deliberate until the family
interpreter migration is completed.

The caller still owns plaintext values, secret values, parameter selection,
seed/nonce domains, storage, and runtime settings. Sampling site labels,
domain strings, tuple ordering, NAND packing, PRF modulus/stride/shared-secret
contracts, and readout scale are unchanged.

Noise acceptance distinguishes the carrier roles. Intermediate program values
retain the strict q/8 clearance needed for adjacent payload spacing. The final
q/2 Boolean carrier is checked with strict error < q/4, matching the runtime
centered residual check; its coefficient/range bound remains a separate
condition. Average-case audits use the same q/4 limit at the last level and
q/8 for intermediate levels.

## Serialization and trust boundaries

Stage 5-A does not change `RnsBggProgramDescriptor` serialization or its
expectation digest. Imported setup still requires an independently retained
`RnsBggProgramExpectation`; metadata on disk is not trusted as its own
expectation. Import validates the program identity, expected input set and
levels, ordered RHS gate set, and prepared artifact identity before creating
lazy artifact inputs.

The Stage 4 terminal descriptor change is intentionally incompatible with old
serialized descriptors: the public `finalize: bool`/caller-selectable option
was replaced by the internal `TerminalFinalization::{Deferred, Finalize}`
enum. The canonical terminal family stage identity includes mode,
finalization, range, cache schema, and public schema. Existing old descriptors
must be regenerated; they are not silently upgraded.

The RNS-BGG parameter manifest keeps its historical
`RnsBggParameterManifest::SCHEMA_VERSION` value of 2. This is not a promise of
backward-compatible deserialization: serialized derived-workload fields such
as `family_operand_bounds` may change, so stale manifests and dependent
checkpoint or identity artifacts are outside the supported input contract and
must be regenerated. This policy is independent of the program descriptor and
terminal descriptor contracts.

## Migration checklist

1. Keep callers passing input levels and caller-owned setup domains; do not
   derive a level from the program step index.
2. Treat `ProgramGateInstance::Gate`, `FamilyGate`, and `FamilyReduction` as
   distinct identities when adding sampling or artifact domains.
3. Use the requirements record for level/RHS/LUT validation before creating
   material. Do not recreate an independent level walker in an API adapter.
4. Preserve scalar-only family rejection in setup until the family preparation
   path consumes symbolic body requirements. The body must remain one graph
   scope rather than a host `Vec` of members.
5. Regenerate Stage 4 terminal descriptors after the finalization enum change.

## Status

The structural requirement record, shared online checks, native sparse audit
boundary, and Stage 5-C documentation/visibility work are complete for the
current scalar and symbolic-family contracts. Future family-preparation work
must update the inventory rows without introducing a second structural
derivation path.

The application facade now consumes the requirements in generic online graph
construction and validates ordered RHS/helper ports before evaluating an
imported `PreparedProgram`. Generic preprocessing emits LUT helper ports in
that same requirement order. Sparse-LWR keeps its native PBC selector and
readout derivation, but validates the accumulation Program's ordered private
gate chain and terminal level through the shared requirements before preparing
the sparse chain and again before online readout. Sparse-specific block
selection, public-vector routing, and digit-scale derivation remain native
contracts rather than being replaced by generic gate metadata.

Prepared-program artifacts use one stage-owned aggregate prefix. They do not
define a program-derived port name for each gate. The requirements record keeps
the ordered gate/helper/cache mapping used to assign aggregate tuple members;
the runtime's `helper.<index>` names are only that stable ordinal within the
aggregate export. Therefore no additional per-gate port abstraction should be
introduced during migration, and no duplicate port naming contract exists
outside the prepared artifact schema.

## Stage 5-C visibility migration

The ordinary application surface remains at the crate root. Applications use
`ExponentLut::{preprocess, encrypt, online_eval}` and the typed artifact,
secret, and output handles re-exported by `mxx-exponent-lut`. They do not need
to name RNS-BGG graph builders, helper samplers, raw rows, or descriptor
storage types.

The RNS-BGG module remains public as an advanced backend surface, but its
implementation children are no longer direct module paths:

| Old path | New path | Reason |
|---|---|---|
| `rns_bgg::api::*` | crate-root facade types or `advanced::*` | API implementation is crate-internal; stable facade re-exports remain available |
| `rns_bgg::encoding::*` | `rns_bgg::*` named advanced types/methods | Encoding implementation is private; public scheme methods remain reachable on `RnsBggScheme` |
| `rns_bgg::public_key::*` | `rns_bgg::*` named advanced types/methods | Key preparation implementation is private; explicit core types remain reachable |
| `rns_bgg::noise::*` | `rns_bgg::*` named noise reports and helpers | Noise implementation is private; only reports, the public-trait evaluator type, and source-identity helpers are re-exported |
| `rns_bgg::program_setup::*` | crate-root facade setup/export types where supported | Setup implementation is internal; descriptor and expectation types remain explicit public contracts |

This migration does not change program-descriptor serialization, sampling
identity domains or labels, noise formulas, level semantics, or supported
artifact contents. Parameter manifests may gain or change derived workload
fields without a backward-compatibility guarantee; stale manifests are
outside the supported input contract and must be regenerated. A private child
module does not make an inherent public method private; methods that remain
public are intentional advanced operations needed by the concrete backend or
public trait signatures. Internal setup records and family index translators
are crate-visible only.

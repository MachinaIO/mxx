# RNS-BGG exponent-LUT symbol inventory

This inventory describes the supported RNS-BGG exponent-LUT boundary after
the compact-material simplification. It is an implementation guide, not a
second API specification; the normative interface is
`docs/plans/exponent-lut-rns-bgg-simplification-spec.md`.

## Module ownership

| Module | Owns | Does not own |
| --- | --- | --- |
| `rns_bgg/mod.rs` | parameters, block aliases, shared matrix operations, lowering contracts | application-specific circuits, persisted decomposition state |
| `rns_bgg/public_key.rs` | public-key carriers, compact RHS/LUT material, public preprocessing | online RHS plaintexts, expanded decomposition artifacts |
| `rns_bgg/encoding.rs` | payload/mask secrets, input encryption, LUT helper generation, online evaluation | public-key setup policy, a second scheme facade |
| `rns_bgg/noise.rs` | level/workload derivation, noise bounds, reports | ciphertext execution, BGG imports |
| `rns_bgg/gpu.rs` | GPU-only tests and support | production API ownership |

The common `noise_refresh` module is scheme agnostic and has no dependency on
BGG or RNS-BGG implementation modules. There is no `bgg/nand_circuit.rs`, no
RNS-BGG legacy path, no replacement executor, and no Lean migration in this
work.

## Common program contracts

| Symbol | Role | Contract |
| --- | --- | --- |
| `ExponentLUTProgramLowering` | scalar gate lowering | `add`, `mul_public`, `mul_private`, and `lut` operate on one scalar wire per gate; helper/material state is owned by the implementation |
| `RhsRef<I>` / `Rhs` | RHS identity | Carries only `RhsId` and an optional family index; it contains no plaintext or matrix |
| `LutId` | LUT identity | Resolves the actual table passed to `lut`; it is not a lookup into a hidden binary-LUT path |
| `ExpLayout` | exponent representation | Explicitly declared by the LUT/program boundary; no width or stride is inferred from `rhs_width` |
| `RnsBggEncoding` | ciphertext carrier | Pairs row, public key, and optional public payload; payload is present only for a public Program wire |
| `RnsBggPublicKey` | public carrier | Pairs the Program wire identity with its public matrix |
| `RnsBggPayloadSecret` | payload secret | Used for private input/RHS sampling and kept separate from the mask secret |
| `RnsBggMaskSecret` | online mask secret | Used only for online masking and never serialized in stage artifacts |

`RnsBggEncoding` is a paired carrier, not a row plus a parallel public-key
registry. Public payload plaintext is available only where the Program marks
the wire public. Online evaluation never receives RHS plaintexts; the only
plaintext used by the online path is the public-LHS encoding value required by
`MulPublic`.

## Material and artifact keys

Only compact matrices and their existing graph/artifact schemas are persisted.
Deterministic gadget decompositions and expanded full-modulus matrices are
recomputed by the consuming operation from the compact matrix. Decomposition
uses the source level's signed radix convention and centered-extends to a
larger evaluation modulus. Reduction is valid only from a larger modulus to a
smaller one; equal moduli are a no-op.

| Material | Persisted representation | Canonical reuse key |
| --- | --- | --- |
| RHS compact matrix | `RnsBggMatrix`/existing matrix graph value | RHS identity and consuming level |
| `A_sigma` mask-alignment matrix | compact public matrix | `(evaluation_level, sigma)` |
| `F_sigma` encrypted helper | compact helper row/family | `(evaluation_level, sigma)` |
| `B_sigma` switching matrix | compact setup-fixed matrix | `(evaluation_level, decomposition_level, LutId, input_layout, sigma)` |

Keys are shared across LUT sites and layouts exactly as shown. A site or scope
is not part of an `A_sigma`, `F_sigma`, or `B_sigma` identity. The explicit LUT
`input_layout` is part of `B_sigma` because it determines the normalized LUT
input contract. Binary-LUT builders are convenience APIs only; their output is
normalized to the same single-LUT representation before material production.

`operations_per_drop` is a parameter-owned scheduling value. It controls
planned modulus-drop cadence and is not encoded as a per-wire or per-artifact
override. Coefficient finalization occurs once after the complete accumulated
expression.

## Stage boundaries

1. Public preprocessing validates the common Program, samples compact public
   matrices and payload-dependent material, and exports only those compact
   values plus ordinary binding metadata.
2. Encryption samples payload/mask secrets as required by the stage contract;
   it does not require RHS plaintext disclosure.
3. Online evaluation resolves `LutId` to the concrete `LutTable`, derives
   deterministic decompositions at the point of use, performs the required
   public/online ModDown operations, and finalizes output once.

All old RNS-BGG descriptors that contain persisted decomposition families,
parallel helper registries, scheme-specific RHS plaintext containers, or
legacy BGG application bindings are outside the supported contract and must
be regenerated or removed. Runtime formats that remain versioned stay at
version 1; no version bump is introduced for this simplification.

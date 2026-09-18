# RNS-BGG exponent-LUT design notes

This document records the current RNS-BGG exponent-LUT boundary. It is an
incompatible simplification, not a compatibility layer. The authoritative
details are in
`docs/plans/exponent-lut-rns-bgg-simplification-spec.md`; this note is kept
short so that it does not become a second API specification.

## Ownership

RNS-BGG is the only exponent-LUT backend. The common program traversal owns
wire knowledge, RHS references, LUT identifiers, and exponent layouts. The
RNS-BGG files own only backend arithmetic:

- `mod.rs` owns parameters, compact block aliases, shared operations, and the
  `ExponentLUTProgramLowering` contracts;
- `public_key.rs` owns public-key carriers and preprocessing;
- `encoding.rs` owns input encryption, LUT helper generation, and online
  evaluation;
- `noise.rs` owns noise contracts and workload accounting;
- `gpu.rs` contains GPU-only tests and support.

There is no RNS-BGG scheme facade, replacement executor, application-specific
NAND backend, Lean change, or legacy adapter in this path. The common
`noise_refresh` code has no dependency on either BGG implementation.

## Carriers and plaintext boundaries

`RnsBggEncoding` carries its `RnsBggPublicKey` together with the row and, only
for a public program wire, its public payload. The payload is not a parallel
registry and is never inferred from matrix shape. Preprocessing receives no
LHS plaintexts. During online evaluation RHS plaintexts are never public; the
only plaintext available there is the public-LHS encoding value used by
`MulPublic`.

RHS operands are references (`RhsRef`), not plaintext or matrix containers.
The same RHS identifier has one consuming level throughout a Program. A
different level requires a different identifier and input binding.

## Compact artifacts and on-demand decomposition

Preprocessing persists compact matrices and public-key carriers only. It does
not persist deterministic gadget decompositions or expanded full-modulus
matrices. The consuming operation recomputes the signed decomposition from the
compact matrix at the point of use, then centered-extends it when the
evaluation modulus is larger. Modulus reduction assumes source modulus is
greater than destination modulus; equal moduli are a no-op.

The public material keys are deliberately shared:

| Material | Key |
| --- | --- |
| `A_sigma` mask-alignment matrix | `(evaluation_level, sigma)` |
| `F_sigma` encrypted helper | `(evaluation_level, sigma)` |
| `B_sigma` setup-fixed switching matrix | `(evaluation_level, decomposition_level, LutId, input_layout, sigma)` |

The lowering boundary passes the resolved `LutTable` for each scalar LUT gate.
Evaluators that need an identifier resolve it against their validated,
Program-owned table context; pointer identity preserves distinct IDs for
equal-valued registrations, and ambiguous external table copies are rejected.
The LUT's explicit `ExpLayout` remains part of the table. A binary-LUT
convenience builder is normalized to the same single-LUT representation before
artifact production; evaluators do not have a separate binary-LUT path.

`operations_per_drop` is part of parameter construction and controls the
planned modulus-drop cadence. Output coefficients are finalized once after
the complete accumulated expression, not once per branch or family member.

## API and compatibility policy

The lowering trait is named `ExponentLUTProgramLowering`. It receives one
scalar wire per gate. `lut` receives the input wire and its resolved
`LutTable`; an evaluator that needs the `LutId` resolves it from its own
validated Program context. Helper and multiplication material belongs to the
lowerer implementation and is accessed through `self`. `Rhs` remains only to
distinguish an RHS input from a wire in `mul_private`; `ProgramLoweringTypes`
and `HelperSet` are not separate contracts.

No version is bumped. Existing versioned runtime formats remain at version 1
where they are still used, while new application descriptors do not add a
version field. Old RNS-BGG descriptors and legacy BGG application paths are
not supported and must be regenerated or removed.

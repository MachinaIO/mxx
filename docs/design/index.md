# Design Documentation Index

Read this file first before reading any document under `docs/design/`.

This index is the entry point for repository design documentation. It explains where design artifacts live and how to find the right document by purpose.

## Role map

Design documents in this repository are organized by function. Each document should satisfy at least one of these roles:

1. Conceptual/theoretical explanation for first-time readers.
2. Target behavior/properties with assumptions and limits.
3. Core technical ideas and trade-off rationale.

## Index

Current registered design documents:

- [gpu_limb_byte_packing.md](./gpu_limb_byte_packing.md)
  - Purpose: Defines the long-lived CUDA limb storage invariant that packs each modulus into its minimum byte width.
  - Roles:
    1. Target behavior/properties with assumptions and limits.
    2. Core technical idea and trade-off rationale for VRAM reduction.
- [execplan_verification_enforcement.md](./execplan_verification_enforcement.md)
  - Purpose: Defines long-lived verification enforcement design using repository-local skills, gate execution, and in-plan ledger evidence.
  - Roles:
    1. Target behavior/properties with assumptions and limits.
    2. Core technical idea and trade-off rationale for enforcement vs. advisory docs.
- [pr_autoloop_builder_reviewer_contract.md](./pr_autoloop_builder_reviewer_contract.md)
  - Purpose: Defines the long-lived autonomous PR iteration contract for builder/reviewer roles, reviewer JSON structured output, and deterministic stop conditions.
  - Roles:
    1. Target behavior/properties with assumptions and limits.
    2. Core technical idea and trade-off rationale for strict contract parsing under shared-account operation.

When adding a design document, place it under `docs/design/` and add it to this index with:

- a short purpose summary,
- its applicable role(s) from the role map above,
- its repository-relative path.

## See also

- [DESIGN.md](../../DESIGN.md)
- [ARCHITECTURE.md](../../ARCHITECTURE.md)
- [PLANS.md](../../PLANS.md)

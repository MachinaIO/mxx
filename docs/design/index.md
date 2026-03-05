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

- [ggh15_arbitrary_polynomial_lookup.md](./ggh15_arbitrary_polynomial_lookup.md)
  - Purpose: Defines how GGH15 public lookup evaluation can support finite-domain non-constant polynomial inputs that remain within the current compact-decomposition contract, without adding per-coefficient preimage families.
  - Roles:
    1. Target behavior/properties with assumptions and limits.
    2. Core technical idea and trade-off rationale for the GGH15 lookup extension.
- [gpu_limb_byte_packing.md](./gpu_limb_byte_packing.md)
  - Purpose: Defines the long-lived CUDA limb storage invariant that packs each modulus into its minimum byte width.
  - Roles:
    1. Target behavior/properties with assumptions and limits.
    2. Core technical idea and trade-off rationale for VRAM reduction.

When adding a design document, place it under `docs/design/` and add it to this index with:

- a short purpose summary,
- its applicable role(s) from the role map above,
- its repository-relative path.

## See also

- [DESIGN.md](../../DESIGN.md)
- [ARCHITECTURE.md](../../ARCHITECTURE.md)

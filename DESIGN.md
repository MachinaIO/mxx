# Design Documentation Meta-Rules

This document defines how design documents must be authored, organized, and maintained in this repository.  
Design documents are long-lived repository contracts. They must remain understandable to first-time readers (human or agent) and must stay aligned with current implementation behavior.

## Purpose of design documents

Each design document must serve at least one of the following roles:

1. Explain the repository's provided capabilities at a theoretical or conceptual level for newcomers.
2. Define ideal target functionality/properties, assumptions, and limits that implementation must satisfy, while leaving implementation detail to architecture documents.
3. Record core technical ideas and trade-off decisions, including why a specific option was chosen and the discussion context behind that choice.

## Design-first precedence rule

Design intent and specification come first.  
Implementation must follow the design/specification defined by design documents.

Architecture documents then describe how those design decisions are concretely realized in code and system structure.

## Longevity and consistency rule

Design documents are expected to remain valid over long periods and must be maintained as long-lived references.  
They must not drift from implementation reality: whenever implementation behavior changes in a way that affects design statements, the design documents must be updated in the same change set.

## Required location and reading order

Concrete design documents must live under `docs/design/`.

`docs/design/index.md` is mandatory and is the table of contents for all design documents.  
Any agent or contributor reading design documentation must read `docs/design/index.md` first.

## Boundary with architecture documentation

Design documentation defines what should be true (goals, expected properties, assumptions, limits, and rationale).  
Architecture documentation defines how those design goals are mapped into implementation structure, interfaces, dependencies, and boundaries.

Design documents must not be replaced by architecture-only detail.  
If a design choice is long-lived or reused across changes, it must be captured in design docs and linked from relevant ExecPlans.

## Update policy

Design documentation updates are mandatory when any of the following changes occur:

- user-visible behavior goals or quality targets change,
- core assumptions or known limits change,
- trade-off decisions are revised,
- a new long-lived technical idea becomes part of the repository baseline.

When in doubt, update design docs in the same pull request to preserve a one-to-one correspondence between design intent and implementation.

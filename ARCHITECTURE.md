# Architecture Documentation Meta-Rules

This document defines how architecture documents must be authored, organized, and maintained in this repository.  
Architecture documents are long-lived repository contracts. They must remain understandable to a reader who has not inspected implementation details yet, and they must always stay aligned with the current implementation.

## Purpose of architecture documents

Architecture documents must explain how design decisions are concretely realized in implementation form. They must describe, in plain and implementation-facing terms:

- repository structure and domain boundaries,
- supported feature set and feature-flag shape,
- external libraries and tools the repository depends on,
- implementation languages and boundary contracts,
- key interfaces and behavioral characteristics that shape how modules are used.

These documents are expected to be long-lived as long as the architecture remains stable.  
Whenever implementation changes invalidate architecture statements, the architecture documents must be updated in the same change set.

## Required location and reading order

All architecture documents must live under `docs/architecture/`.

`docs/architecture/index.md` is the mandatory table of contents and entry point for architecture reading.  
Any agent or contributor reading architecture documents must read `docs/architecture/index.md` first.

## Required directory layout

`docs/architecture/` must include the following top-level directories:

- `scope/`
- `features/`
- `dependencies/`

## Scope documentation rules

`docs/architecture/scope/` is the place where implementation scopes are documented.  
In general, each scope should correspond to:

- one markdown file directly under `docs/architecture/scope/`, and
- concrete implementation files or directories in the codebase.

`docs/architecture/scope/index.md` is mandatory and serves two roles:

1. It is the index for the `scope` directory and must include relative paths to each domain document.
2. It must explicitly document dependencies among domains.

Domain dependency direction must be documented with the following rule:  
if implementation of domain A does not depend on implementation of domain B at all, but implementation of domain B depends on implementation of domain A, then domain A is considered to depend on domain B in the architecture dependency statement.

As a principle, mutual dependency among multiple domains should be avoided.

## Interface vs implementation requirement

When multiple implementations exist (or may exist in the future) for the same interface within a domain, the interface must be explicitly defined as an abstraction (for example: Rust trait, C/C++ header function contract, or abstract class), and concrete implementations are provided separately.

Each domain architecture document must clearly distinguish:

- interface contract (what callers depend on), and
- implementation variants (how the contract is realized).

## Update policy

Architecture documentation updates are mandatory when any of the following changes occur:

- module/domain movement or new layering rules,
- feature-flag topology changes,
- shared infrastructure or cross-domain dependency changes,
- boundary contract changes (for example FFI/CUDA, I/O, storage, build integration),
- supported external dependency/tool changes that affect architectural behavior.

When in doubt, prefer updating architecture docs in the same pull request to preserve one-to-one correspondence between documentation and implementation.

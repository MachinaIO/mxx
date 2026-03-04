# Repository Guidelines

## Repository Purpose
This repository provides implementations for lattice-cryptography operations (polynomail and matrix operations, preimage samplings, BGG+ encodings, and more), written in Rust and CUDA.

## Autonomous PR Loop Rule
- **Hard pledge:** When directly invoked by a human user, you must invoke the `eternal-cycler` skill as the very first action, and you must not proceed to any other work before doing so, unless the user explicitly and unambiguously requests not to use it.
- In that default mode, follow `.agents/skills/eternal-cycler/SKILL.md` as the primary workflow contract for task execution.
- If the user explicitly opts out of `eternal-cycler`, continue with the requested task without applying that skill.


## Meta Rules
The following documents define *meta-rules* for how agents should create, update, and reference documents. Agents must carefully read and understand them.

### Planning and Review Policy
Use the policy documents bundled with eternal-cycler:
- `.agents/skills/eternal-cycler/PLANS.md`
- `.agents/skills/eternal-cycler/REVIEW.md`
When `eternal-cycler` mode is active, treat those files as the source of truth for planning, lifecycle execution, and review behavior.

### Design (DESIGN.md)
Read DESIGN.md when you need to understand why a behavior or API decision exists and what long-lived trade-offs/invariants must be preserved, especially when your change requires a non-obvious decision that should be reusable beyond each PR, e.g.:
- you are choosing between multiple approaches with meaningful trade-offs,
- you introduce a new interface/contract, invariant, or API behavior,
- you add a pattern that future work should follow consistently.
If the decision is long-lived, create/update the relevant design artifact (per DESIGN.md) and link it from your ExecPlan.

### Architecture (ARCHITECTURE.md)
Read ARCHITECTURE.md when you need to understand how repository components are structured, how responsibilities are split, and which boundaries/dependencies must stay intact, especially before making changes that could affect code structure, e.g.:
- moving/adding modules or domains, changing package layout, or layering,
- adding/changing feature flags, shared infrastructure, or cross-domain dependencies,
- introducing new external dependencies,
- touching boundaries (e.g., FFI/CUDA, IO/storage, build integration) that rely on invariants.

## Global Requirements
- All documentation in this repository, along with git commit messages and PRs, must be written in English.
- When documenting file paths, use only paths relative to the repository top directory. Do not write absolute paths in documentation.

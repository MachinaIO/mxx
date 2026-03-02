# Scope: `src/lookup`

## Purpose

Implements public lookup-table evaluation layers (LWE/GGH15/commit evaluators) and lookup abstractions used during circuit execution.

## Implementation mapping

- `src/lookup/mod.rs`
- `src/lookup/poly.rs`
- `src/lookup/lwe_eval.rs`
- `src/lookup/ggh15_eval.rs`
- `src/lookup/commit_eval.rs`

## Interface vs implementation

- Interface: `crate::lookup::PltEvaluator`
- Shared LUT structure: `crate::lookup::PublicLut`
- Concrete evaluator implementations:
  - LWE evaluator
  - GGH15 evaluator
  - commit-related evaluator paths

## Depends on scopes

- `bgg`
- `circuit`
- `matrix`
- `poly`
- `sampler`
- `storage`

## Used by scopes

- `circuit`
- `gadgets`
- `simulator`
- `tests`

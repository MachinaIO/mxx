# Scope: `src/circuit`

## Purpose

Defines circuit structures, gate semantics, evaluable abstractions, serialization, and evaluation flow for polynomial-based circuits.

## Implementation mapping

- `src/circuit/mod.rs`
- `src/circuit/gate.rs`
- `src/circuit/serde.rs`
- `src/circuit/evaluable/mod.rs`
- `src/circuit/evaluable/poly.rs`
- `src/circuit/evaluable/bgg.rs`

## Interface vs implementation

- Interface: `crate::circuit::evaluable::Evaluable`
- Concrete evaluable variants:
  - polynomial evaluable path
  - BGG evaluable path
- Core orchestrator: `PolyCircuit`

## Depends on scopes

- `poly`
- `lookup`

## Used by scopes

- `gadgets`
- `lookup`
- `simulator`
- `tests`

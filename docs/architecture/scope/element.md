# Scope: `src/element`

## Purpose

Defines low-level ring element interfaces and concrete finite-ring element behavior used by higher polynomial layers.

## Implementation mapping

- `src/element/mod.rs`
- `src/element/finite_ring.rs`

## Interface vs implementation

- Interface: `crate::element::PolyElem`
- Concrete implementation: `crate::element::finite_ring::FinRingElem`

## Depends on scopes

- none (foundation scope)

## Used by scopes

- `poly`
- `matrix`

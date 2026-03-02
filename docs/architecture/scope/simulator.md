# Scope: `src/simulator`

## Purpose

Provides simulation and norm-estimation helpers for error growth and lattice-parameter analysis around circuit/lookup workflows.

## Implementation mapping

- `src/simulator/mod.rs`
- `src/simulator/error_norm.rs`
- `src/simulator/poly_norm.rs`
- `src/simulator/poly_matrix_norm.rs`
- `src/simulator/lattice_estimator.rs`

## Interface vs implementation

- Main data model: `SimulatorContext` and norm wrapper types.
- Integrates with circuit and lookup abstractions to estimate behavior rather than execute full cryptographic payloads.

## Depends on scopes

- `circuit`
- `lookup`
- `poly`

## Used by scopes

- `tests`

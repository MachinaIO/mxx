# Scope: `src/gadgets`

## Purpose

Defines reusable gadget-level circuit constructions, including nested RNS helpers and secret inner-product composition.

## Implementation mapping

- `src/gadgets/mod.rs`
- `src/gadgets/secret_ip.rs`
- `src/gadgets/arith/nested_rns.rs`

## Interface vs implementation

- This scope mostly provides concrete gadget builders over existing circuit/poly/lookup interfaces.
- No separate trait hierarchy is introduced here; scope behavior is composed from imported abstractions.

## Depends on scopes

- `circuit`
- `lookup`
- `poly`

## Used by scopes

- `tests`

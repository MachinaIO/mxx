# Scope: `src/storage`

## Purpose

Defines lookup/auxiliary artifact persistence and retrieval formats for batched matrix data and global index metadata.

## Implementation mapping

- `src/storage/mod.rs`
- `src/storage/read.rs`
- `src/storage/write.rs`

## Interface vs implementation

- Read interface: functions like `read_matrix_from_multi_batch` / `read_bytes_from_multi_batch`
- Write/index interface: `BatchLookupBuffer`, `TableIndexEntry`, `GlobalTableIndex`, append/flush helpers
- Concrete format implementation uses split binary batch files plus index metadata.

## Depends on scopes

- `matrix`
- `poly`

## Used by scopes

- `lookup`
- `commit`

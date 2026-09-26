# OpenFHE Rust bridge declarations

`DCRTPoly.h` is the unmodified public declaration header from
`MachinaIO/openfhe-rs`, revision `9c9d81c`, matching the dependency in
`Cargo.lock`. Its BSD-2-Clause license is reproduced in `LICENSE`.

The upstream crate does not export its include directory to dependents. This
small declaration-only copy permits the repository-owned exact-CRT-basis
adapter to return the existing upstream C++ wrapper types without changing
their representation or duplicating their implementation. Keep this header
identical to the pinned upstream header when updating the dependency.

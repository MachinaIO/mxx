# Repository Guidelines

## Repository Purpose
This repository provides implementations for lattice-cryptography operations (polynomial and matrix operations, preimage samplings, BGG+ encodings, and more), written in Rust and CUDA.

## Workspace Architecture
- The authoritative crate map and dependency rules are documented in `docs/architecture.md`.
- The workspace has no root facade crate. The authoritative crate list is the
  workspace member list in `Cargo.toml`.
- Keep dependencies layered as documented in `docs/architecture.md`;
  application crates never depend on one another.
- The reusable gadget layer is `crates/gadgets/`; its circuit-specific gadget module is `circuit_gadgets`.

## Global Requirements
- All documentation in this repository, along with git commit messages and PRs, must be written in English.
- When documenting file paths, use only paths relative to the repository top directory. Do not write absolute paths in documentation.
- Directories named `references` are read-only reference directories for humans and agents. Agents may read them when relevant and must never edit them.
- Integration tests must not be run unless the user has explicitly asked for them in the current task. Prefer targeted unit tests or other narrow validation until such approval is given.
- Rust formatting must use `cargo +nightly fmt --all`.
- Follow the builder/reviewer guidance in `BUILDER.md` and `REVIEWER.md`.

## Codex Workflow
- Before starting any task or reading any other files, read and follow `REVIEWER.md` for explicit review tasks and `BUILDER.md` for all other tasks.
- Builders should clarify unclear requirements, implement scoped changes once requirements are clear, and run the narrowest relevant validation.
- Reviewers should stay read-only, prioritize correctness and regression risk, and ground findings in concrete files, commands, or logs.
- If the work touches CUDA, GPU kernels, GPU wrappers, GPU tests, or GPU-facing performance-sensitive behavior, read `GPU.md` before editing or reviewing.

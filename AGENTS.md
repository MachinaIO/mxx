# Repository Guidelines

## Repository Purpose
This repository provides implementations for lattice-cryptography operations (polynomail and matrix operations, preimage samplings, BGG+ encodings, and more), written in Rust and CUDA.

## Global Requirements
- All documentation in this repository, along with git commit messages and PRs, must be written in English.
- When documenting file paths, use only paths relative to the repository top directory. Do not write absolute paths in documentation.
- Directories named `references` are read-only reference directories for humans and agents. Agents may read them when relevant and must never edit them.
- Integration tests must not be run unless the user has explicitly asked for them in the current session. Prefer targeted unit tests or other narrow validation until such approval is given.
- Rust formatting must use `cargo +nightly fmt --all`.

## Codex Workflow
This repository uses the session-plan workflow defined in `BUILDER.md`, `PLANS.md`, and `REVIEWER.md`.

- Use `plans/active/session-<session_id>.md` for complex work.
- Discover the current session id from the handoff or hook payload.
- Treat `## Plan approval` and `## Phase` in the session plan as the workflow state.
- Follow `BUILDER.md` for normal builder turns and `REVIEWER.md` for explicit review turns.
- Keep the session plan in sync with the work; detailed phase transitions and hook behavior live in the documents above.

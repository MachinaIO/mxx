# Repository Guidelines

## Repository Purpose
This repository provides implementations for lattice-cryptography operations (polynomail and matrix operations, preimage samplings, BGG+ encodings, and more), written in Rust and CUDA.

## Global Requirements
- All documentation in this repository, along with git commit messages and PRs, must be written in English.
- When documenting file paths, use only paths relative to the repository top directory. Do not write absolute paths in documentation.
- Directories named `references` are read-only reference directories for humans and agents. Agents may read them when relevant and must never edit them.
- Integration tests must not be run unless the user has explicitly asked for them in the current session. Prefer targeted unit tests or other narrow validation until such approval is given.
- Rust formatting must use `cargo +nightly fmt --all`.
- Follow a lifecycle defined in `BUILDER.md`.

## Codex Workflow
This repository uses a long-running Codex session workflow governed by `BUILDER.md`, `PLANS.md`, and `REVIEWER.md`.
- Before starting any task or reading any other files, read and follow `REVIEWER.md` for explicit review tasks and `BUILDER.md` for all other tasks.
- Follow `PLANS.md` to create and update a plan document for each session.
- Discover the current session id from the handoff or hook payload.
- Treat `## Plan approval` and `## Phase` in the session plan as the workflow state.
- The builder behaves differently by plan approval status:
- - When `## Plan approval` is `unapproved`, stay in planning: interview the user, revise the session plan, and ask for explicit approval.
- - When `## Plan approval` is `approved` and `## Phase` is `implementation`, execute the approved subtasks, run the most relevant tests after each subtask, and only then check the subtask off.
- - When `## Plan approval` is `approved` and `## Phase` is `review`, address only the concrete follow-up work created by review-phase tests or reviewer feedback, keeping completed historical subtasks intact.
- Keep the session plan in sync with the work; detailed phase transitions and hook behavior live in the documents above.

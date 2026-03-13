# Repository Guidelines

## Repository Purpose
This repository provides implementations for lattice-cryptography operations (polynomail and matrix operations, preimage samplings, BGG+ encodings, and more), written in Rust and CUDA.

## Global Requirements
- All documentation in this repository, along with git commit messages and PRs, must be written in English.
- When documenting file paths, use only paths relative to the repository top directory. Do not write absolute paths in documentation.

## Codex Workflow
This repository uses a long-running Codex session workflow governed by `BUILDER.md` and `PLANS.md`.

Complex work must use a session-specific plan document at `plans/session-<session_id>.md`.

The builder must discover the current session id by reading `.agents/current-session-id`, then open `.agents/session-<session_id>.json` and inspect the current phase before acting.

If the task is an explicit review rather than builder execution, follow `REVIEWER.md`.

The builder behaves differently by phase:
- In `planning`, update the session plan and discuss plan revisions with the user.
- In `implementation`, execute the approved subtasks, run the most relevant tests after each subtask, and only then check the subtask off.

Automated review is performed by a hooks-disabled nested read-only `codex exec`.
The stop hook records when the workflow is waiting for a reply to the current session plan. After plan approval, the stop hook itself transitions the session into implementation, launches hooks-disabled nested builder runs against the current session plan, and keeps cycling through final tests and reviewer checks until the reviewer accepts.

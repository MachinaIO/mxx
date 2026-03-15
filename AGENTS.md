# Repository Guidelines

## Repository Purpose
This repository provides implementations for lattice-cryptography operations (polynomail and matrix operations, preimage samplings, BGG+ encodings, and more), written in Rust and CUDA.

## Global Requirements
- All documentation in this repository, along with git commit messages and PRs, must be written in English.
- When documenting file paths, use only paths relative to the repository top directory. Do not write absolute paths in documentation.
- Directories named `references` are read-only reference directories for humans and agents. Agents may read them when relevant and must never edit them.
- Integration tests must not be run unless the user has explicitly asked for them in the current session. Prefer targeted unit tests or other narrow validation until such approval is given.

## Codex Workflow
This repository uses a long-running Codex session workflow governed by `BUILDER.md` and `PLANS.md`.

Complex work must use a session-specific plan document at `plans/active/session-<session_id>.md`.

The builder must discover the current session id from the explicit session handoff or hook payload, then open `plans/active/session-<session_id>.md` and inspect the `## Plan approval` flag before acting.

If the task is an explicit review rather than builder execution, follow `REVIEWER.md`.
Explicit review sessions do not create `plans/active/session-<session_id>.md`; the session-start hook classifies that intent from the initial user prompt, and the stop hook exits immediately when neither `plans/active/session-<session_id>.md` nor `plans/completed/session-<session_id>.md` exists.

The builder behaves differently by plan approval status:
- When `## Plan approval` is `unapproved`, stay in planning: interview the user, revise the session plan, and ask for explicit approval.
- When `## Plan approval` is `approved`, execute the approved subtasks, run the most relevant tests after each subtask, and only then check the subtask off.

Automated review is performed by a hooks-disabled nested read-only `codex exec`.
During planning, the stop hook does no workflow work beyond allowing the stop.
After the plan is marked `approved`, the stop hook reevaluates the current session plan on each stop. If the active plan is missing but a matching completed plan exists, it first moves that plan back into `plans/active/`. If unchecked implementation work remains, or if final tests / reviewer feedback append new follow-up tasks, it blocks the current turn with an actionable resume message. If all tracked checkboxes are already checked and the final tests plus reviewer both pass, it moves the session plan into `plans/completed/`, moves any session-specific files from `revision_logs/active/` into `revision_logs/completed/`, and then accepts.

# Scope: `automation_orchestration`

## Purpose

Documents repository-local autonomous PR orchestration implemented as fixed scripts under `scripts/`.

## Implementation mapping

- `scripts/run_builder_reviewer_doctor.sh`
- `scripts/run_builder_reviewer_loop.sh`
- `.agents/skills/execplan-event-action-pr-autoloop/SKILL.md`
- `.agents/skills/execplan-event-action-pr-autoloop/scripts/run_event.sh`
- `.agents/skills/execplan-event-index/references/event_skill_map.tsv` (event registration)

## Interface vs implementation

Interface contract:

- `run_builder_reviewer_loop.sh` required/optional arguments:
  - `--task <text>` or `--task-file <path>`
  - optional `--pr-url <url>`
  - bounded controls (`--max-iterations`, `--max-builder-cleanup-retries`, `--max-reviewer-failures`)
  - optional model selectors (`--model-builder`, `--model-reviewer`)
- Reviewer comment contract fields:
  - `AUTO_AGENT: REVIEWER`
  - `AUTO_REVIEW_STATUS: APPROVED|CHANGES_REQUIRED`
  - `AUTO_TARGET_COMMIT: <sha>`
  - `APPROVE` token only for approved output

Implementation details:

- Fixed-script loop orchestrates builder/reviewer codex execution and owns retry bounds.
- PR routing supports two modes:
  - explicit `--pr-url` (OPEN and unmerged only),
  - branch-first (reuse existing open PR or create new PR).
- Comment retrieval uses `gh api graphql` over both issue comments and review bodies.
- Approval requires `APPROVE` token plus `AUTO_TARGET_COMMIT` equality with current loop target commit.
- Lifecycle events no longer run reviewer orchestration.
- `gh` operations are executed via out-of-sandbox command paths following `.agents/skills/execplan-sandbox-escalation/`.

## Depends on scopes

- `root_modules` (indirectly via repository policy tooling references only)
- `tests` (operationally for manual verification scenarios, not compile-time)

## External/tool boundaries

- `codex` CLI for non-interactive builder/reviewer agent execution.
- `gh` CLI for PR metadata and comment operations.
- `git` for branch sync and worktree isolation.
- `jq` for parsing JSON responses.

This scope is operational tooling and does not change Rust/CUDA runtime behavior.

# Scope: `automation_orchestration`

## Purpose

Documents repository-local autonomous orchestration for pull request iteration implemented as a skill with bundled scripts.

## Implementation mapping

- `.agents/skills/pr-autoloop/SKILL.md`
- `.agents/skills/pr-autoloop/agents/openai.yaml`
- `.agents/skills/pr-autoloop/references/comment_contract.md`
- `.agents/skills/pr-autoloop/references/state_schema.md`
- `.agents/skills/pr-autoloop/scripts/doctor.sh`
- `.agents/skills/pr-autoloop/scripts/run_loop.sh`
- `.agents/skills/execplan-event-action-pr-autoloop/SKILL.md`
- `.agents/skills/execplan-event-action-pr-autoloop/agents/openai.yaml`
- `.agents/skills/execplan-event-action-pr-autoloop/scripts/run_event.sh`
- `.agents/skills/execplan-event-index/references/event_skill_map.tsv` (event registration)

## Interface vs implementation

Interface contract:

- `run_loop.sh` CLI arguments:
  - `--goal-file`
  - `--pr-url`
  - `--max-builder-failures`
  - `--max-iterations`
- Reviewer comment contract fields:
  - `AUTO_AGENT: REVIEWER`
  - `AUTO_REVIEW_STATUS: APPROVED|CHANGES_REQUIRED`
  - `AUTO_TARGET_COMMIT: <sha>`

Implementation details:

- Lock and state files under `.agents/skills/pr-autoloop/runtime/`.
- Builder/reviewer isolated git worktrees for branch-safe iteration.
- Event-level validation via `action.pr_autoloop` skill script.

## Depends on scopes

- `root_modules` (indirectly via repository policy tooling references only)
- `tests` (operationally for manual verification scenarios, not compile-time)

## External/tool boundaries

- `codex` CLI for non-interactive agent execution.
- `gh` CLI for PR metadata and comment operations.
- `git` for branch sync and worktree isolation.
- `jq` for parsing JSON responses.

This scope is operational tooling and does not change Rust/CUDA runtime behavior.

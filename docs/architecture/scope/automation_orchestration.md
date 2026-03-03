# Scope: `automation_orchestration`

## Purpose

Documents repository-local autonomous orchestration for pull request iteration implemented as a skill with bundled scripts.

## Implementation mapping

- `.agents/skills/pr-autoloop/SKILL.md`
- `.agents/skills/pr-autoloop/agents/openai.yaml`
- `.agents/skills/pr-autoloop/references/comment_contract.md`
- `.agents/skills/pr-autoloop/references/state_schema.md`
- `.agents/skills/pr-autoloop/scripts/doctor.sh`
- `.agents/skills/pr-autoloop/scripts/reviewer_daemon.sh`
- `.agents/skills/pr-autoloop/scripts/run_loop.sh`
- `.agents/skills/execplan-event-action-pr-autoloop/SKILL.md`
- `.agents/skills/execplan-event-action-pr-autoloop/agents/openai.yaml`
- `.agents/skills/execplan-event-action-pr-autoloop/scripts/run_event.sh`
- `.agents/skills/execplan-event-index/references/event_skill_map.tsv` (event registration)

## Interface vs implementation

Interface contract:

- `run_loop.sh` CLI arguments:
  - `--goal-file`
  - `--pr-url` (existing PR mode) or `--head-branch` (bootstrap mode)
  - `--base-branch` (optional, bootstrap mode guidance)
  - `--max-builder-failures`
  - `--max-iterations`
- Reviewer comment contract fields:
  - `AUTO_AGENT: REVIEWER`
  - `AUTO_REQUEST_ID: <request_id>`
  - `AUTO_REVIEW_STATUS: APPROVED|CHANGES_REQUIRED`
  - `AUTO_TARGET_COMMIT: <sha>`
  - `APPROVE` token when approved

Implementation details:

- Lock and state files under `.agents/skills/pr-autoloop/runtime/` (PR lock when known, branch lock during bootstrap).
- Reviewer daemon inbox/response state under `.agents/skills/pr-autoloop/runtime/reviewer-daemon/`.
- Builder/reviewer isolated git worktrees for branch-safe iteration.
- Automatic PR URL discovery after builder execution in bootstrap mode; discovered URL is passed to reviewer context.
- ExecPlan lifecycle events use daemon messaging: pre-creation starts reviewer daemon if absent, post-completion sends commit metadata and blocks until reviewer response comment URL is returned.
- `gh` API operations are executed via out-of-sandbox command paths following `.agents/skills/execplan-sandbox-escalation/`.
- Reviewer iteration comments are non-blocking with respect to CI runtime; reviewer does not wait for CI completion to post contract output.
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

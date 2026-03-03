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
- `.agents/skills/execplan-event-action-pr-autoloop/SKILL.md`
- `.agents/skills/execplan-event-action-pr-autoloop/agents/openai.yaml`
- `.agents/skills/execplan-event-action-pr-autoloop/scripts/run_event.sh`
- `.agents/skills/execplan-event-index/references/event_skill_map.tsv` (event registration)

## Interface vs implementation

Interface contract:

- `reviewer_daemon.sh` CLI modes and required arguments:
  - `--start [--pr-url <url>] [--head-branch <branch>]`
  - `--request --commit <sha> [--pr-url <url>] [--head-branch <branch>] [--request-id <id>] [--run-id <id>] [--iteration <n>]`
  - `--status`
  - `--stop`
- Reviewer comment contract fields:
  - `AUTO_AGENT: REVIEWER`
  - `AUTO_REQUEST_ID: <request_id>`
  - `AUTO_RUN_ID: <run_id>`
  - `AUTO_ITERATION: <n>`
  - `AUTO_REVIEW_STATUS: APPROVED|CHANGES_REQUIRED`
  - `AUTO_TARGET_COMMIT: <sha>`
  - `APPROVE` token when approved

Implementation details:

- Reviewer daemon inbox/response/runtime state under `.agents/skills/pr-autoloop/runtime/reviewer-daemon/`.
- Builder sends commit-scoped review requests and blocks until daemon response payload is written.
- Daemon can discover PR URL from head branch when request omits explicit PR URL.
- ExecPlan lifecycle events use daemon messaging: pre-creation starts reviewer daemon if absent, post-completion sends commit metadata and blocks until reviewer response comment URL is returned.
- `gh` API operations are executed via out-of-sandbox command paths following `.agents/skills/execplan-sandbox-escalation/`.
- Reviewer iteration comments are non-blocking with respect to CI runtime; reviewer does not wait for CI completion to post contract output.
- Event-level validation via `action.pr_autoloop` skill script, including rejection of removed legacy loop files.

## Depends on scopes

- `root_modules` (indirectly via repository policy tooling references only)
- `tests` (operationally for manual verification scenarios, not compile-time)

## External/tool boundaries

- `codex` CLI for non-interactive agent execution.
- `gh` CLI for PR metadata and comment operations.
- `git` for branch sync and worktree isolation.
- `jq` for parsing JSON responses.

This scope is operational tooling and does not change Rust/CUDA runtime behavior.

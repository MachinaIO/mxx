# Scope: `automation_orchestration`

## Purpose

Documents repository-local autonomous PR orchestration implemented by a static upstream skill install under `.agents/skills/eternal-cycler/` and repository-facing wrappers under `.agents/skills/pr-autoloop/scripts/`.

## Implementation mapping

- `.agents/skills/pr-autoloop/SKILL.md`
- `.agents/skills/pr-autoloop/scripts/run_builder_reviewer_doctor.sh`
- `.agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh`
- `.agents/skills/eternal-cycler/SKILL.md`
- `.agents/skills/eternal-cycler/setup.sh`
- `.agents/skills/eternal-cycler/scripts/run_builder_reviewer_doctor.sh`
- `.agents/skills/eternal-cycler/scripts/run_builder_reviewer_loop.sh`
- `eternal-cycler-out/` (runtime output root for plans/PR tracking artifacts)
- `.agents/skills/execplan-event-index/references/event_skill_map.tsv` (event registration)

## Interface vs implementation

Interface contract:

- `run_builder_reviewer_loop.sh` required/optional arguments:
  - `--task <text>` or `--task-file <path>`
  - optional `--pr-url <url>`
  - if task input is omitted, fail immediately with usage error
  - bounded controls (`--max-iterations`, `--max-builder-cleanup-retries`, `--max-reviewer-failures`)
  - optional model selectors (`--model-builder`, `--model-reviewer`)
- `pr-autoloop` skill caller behavior:
  - when `--pr-url` is omitted and any docs exist in `docs/prs/active`, caller asks user whether to resume one of them,
  - if resume is selected, caller switches to the selected doc's branch before invoking the loop and uses `PR link` for `--pr-url` when present,
  - if selected doc lacks `PR link`, caller still runs on the switched branch without `--pr-url`,
  - if new PR flow is selected (or no active doc exists), caller creates/switches to a task-derived new branch before invoking the loop.
- Reviewer autonomous-loop output fields:
  - `pr_url` (string)
  - `comment_body` (string)
  - `approve_merge` (boolean)
- Builder autonomous-loop output fields:
  - `plan_doc_filename` (string)
  - `result` (`success` | `failed_after_3_retries`)
  - `failure_reason` (string; empty on success)

Implementation details:

- Fixed-script loop orchestrates builder/reviewer codex execution and owns retry bounds.
- Loop output is streamed directly to stdout/stderr; no per-iteration codex log files are persisted.
- `.agents/skills/eternal-cycler/` is installed from upstream as static skill content; runtime plan/PR artifacts are written under `eternal-cycler-out/`.
- PR routing supports two modes:
  - explicit `--pr-url` (head branch must match current local branch),
  - branch-first on current local branch (reuse existing open PR or create new PR).
- Builder output is generated with `codex exec --output-schema` and captured with `--output-last-message`.
- Builder output is parsed as JSON, schema-validated (`plan_doc_filename`, `result`, `failure_reason`), and when builder reports `failed_after_3_retries` the loop script pushes branch state and posts a builder-identity PR comment with failure reason.
- Reviewer output is generated with `codex exec --output-schema` and captured with `--output-last-message`.
- Reviewer output is parsed as JSON, schema-validated (`pr_url`, `comment_body`, `approve_merge`), then posted by loop script with `gh pr comment`.
- Approval requires `approve_merge: true` in validated reviewer JSON output.
- On approval, the loop script updates/moves PR tracking docs (`docs/prs/active` -> `docs/prs/completed`) and records completed tracking metadata including `review state: OPEN`.
- The loop script performs mechanical git finalization for builder output: stage tracked edits, stage non-baseline untracked files, commit when needed, then push to origin.
- Lifecycle events no longer run reviewer orchestration.
- No dedicated `action.pr_autoloop` event exists. Loop behavior validation is covered by regular tooling/script checks and direct `pr-autoloop` runtime operation.
- `execplan.post_completion` does not mark PR ready, does not move PR tracking docs, and does not run git add/commit/push; it is lifecycle validation-only.
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

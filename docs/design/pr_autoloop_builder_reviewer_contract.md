# Design: Fixed-Script Builder/Reviewer Loop Contract

## Purpose

Define a long-lived contract for autonomous PR iteration where a fixed script controls builder/reviewer execution deterministically.

## Problem Statement

Embedding reviewer startup, waiting, and PR creation inside lifecycle verification events made control flow opaque and hard to trust. The requested model requires explicit, operator-invoked script control and strict, machine-readable review decisions.

## Design Goals

1. Deterministic orchestration from a single fixed loop script.
2. Explicit PR selection behavior for both `--pr-url` and branch-first execution.
3. Strict machine-readable review decision criteria from reviewer structured output.
4. Bounded retries to avoid infinite loops.
5. Lifecycle events (`execplan.pre_creation` / `execplan.post_completion`) remain validation-focused and do not run reviewer automation.
6. PR tracking completion transition happens only inside loop approval handling, not in lifecycle verification events.

## Non-Goals

- Replacing review policy in `REVIEW.md`.
- Replacing CI checks.
- Introducing daemon/background process supervision.

## Core Contract

### Roles

- `builder agent`: implements task and feedback, then commits and pushes.
- `reviewer agent`: reviews target commit/newer commits and returns one contract-compliant JSON payload.

### Entrypoint and inputs

Fixed entrypoint:

- `.agents/skills/eternal-cycler/scripts/run_builder_reviewer_loop.sh`

Required input:

- `--task <text>` or `--task-file <path>`
- if both task inputs are omitted, the script exits with usage error (`exit 2`)

Optional input:

- `--pr-url <url>`

Bounded controls:

- `--max-iterations`
- `--max-builder-cleanup-retries`
- `--max-reviewer-failures`

### PR targeting behavior

- The loop script always uses the current local branch as `TARGET_BRANCH`.
- If `--pr-url` is provided, the script validates the PR with `gh pr view --json headRefName,url` and asserts `headRefName == TARGET_BRANCH`.
- If the PR head branch differs from the current local branch, the script exits with input error.
- If `--pr-url` is not provided, the current branch is used. Existing open PR for the branch is reused; otherwise a new PR is created.
- Resume-vs-new selection is not handled inside the loop script. The `eternal-cycler` skill caller handles this selection before invocation:
  - when `--pr-url` is omitted and `docs/prs/active/*.md` has entries, caller asks whether to resume one of those PRs,
  - if resume is selected, caller switches to the selected doc's branch before running the loop, then passes `PR link` as `--pr-url` when present,
  - if selected doc lacks `PR link`, caller still runs on that switched branch without `--pr-url`,
  - if new PR flow is selected (or no active doc exists), caller creates/switches to a task-derived new branch before loop invocation.

### Builder cleanup rule

After each builder run, the script enforces stabilization before proceeding:

- tracked dirty changes must be resolved,
- untracked files not present in baseline must be resolved,
- branch head must be pushed to `origin/<branch>`.

The script performs mechanical finalization by staging tracked changes, staging new untracked files outside the baseline set, committing when needed, and pushing to origin. Builder cleanup prompts are bounded by `--max-builder-cleanup-retries` and focus only on remaining code fixes, not git finalization.

### Builder structured output contract

Builder output in autonomous loop mode requires exactly one JSON object with:

- `plan_doc_filename` (string): repository-relative path of the created/updated plan document.
- `result` (string enum): `success` or `failed_after_3_retries`.
- `failure_reason` (string): empty string when `result=success`; non-empty explanation when `result=failed_after_3_retries`.

The loop script invokes builder with `codex exec --output-schema <schema.json> --output-last-message <file>` so Codex is constrained to this structured response contract.

If builder returns `failed_after_3_retries`, the loop script performs `git push` first and then posts a PR comment containing the failure reason with explicit builder identity (`AUTO_AGENT: BUILDER`).

### Reviewer structured output contract

Reviewer output in autonomous loop mode requires exactly one JSON object with:

- `pr_url` (string): target PR URL.
- `comment_body` (string): review comment text the loop script will post.
- `approve_merge` (boolean): merge approval decision.

Reviewer timing rule:

- reviewer must return JSON without waiting for CI completion.

The reviewer agent must not post GitHub comments directly in autonomous loop mode. The loop script is the only component that posts the review comment using `gh pr comment`.

### JSON validation and approval decision

For each review cycle, the loop script invokes reviewer with `codex exec --output-schema <schema.json> --output-last-message <file>` so Codex is constrained to the structured response contract.

The loop script then validates the returned JSON and checks field types/required values (`pr_url` and `comment_body` non-empty strings, `approve_merge` boolean).

Approval is valid only when `approve_merge` is `true` in the validated JSON payload.

### Stop behavior

- No-op stop: if first builder phase produces no new commit after cleanup.
- Success stop: `approve_merge` is `true` and the script finalizes PR tracking by writing completed metadata (`review state: OPEN`) and moving the tracking file to `docs/prs/completed/`.
- Failure stop: retry bounds reached (`max-iterations` or reviewer/builder cleanup failure bounds).

### Output forwarding contract

- `codex exec` output is not persisted to per-iteration log files.
- Builder/reviewer subprocess output is forwarded to caller stdout/stderr.

### Lifecycle event integration

- `execplan.pre_creation`: captures plan/PR tracking initialization only.
- `execplan.post_completion`: performs lifecycle completion checks only (validation-only, no git mutation).
- Neither lifecycle event starts reviewer automation, performs reviewer wait/approval gating, nor transitions PR tracking documents to completed/ready state.

## Safety and isolation

- Loop execution begins only from tracked-clean working tree.
- Baseline untracked files are excluded from cleanup targets.
- Reviewer output parsing is strict and schema-validated before any PR comment post or approval decision.

# Design: Fixed-Script Builder/Reviewer Loop Contract

## Purpose

Define a long-lived contract for autonomous PR iteration where a fixed script controls builder/reviewer execution deterministically.

## Problem Statement

Embedding reviewer startup, waiting, and PR creation inside lifecycle verification events made control flow opaque and hard to trust. The requested model requires explicit, operator-invoked script control and strict, machine-readable review decisions.

## Design Goals

1. Deterministic orchestration from a single fixed loop script.
2. Explicit PR selection behavior for both `--pr-url` and branch-first execution.
3. Strict machine-readable approval criteria bound to the current target commit.
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
- `reviewer agent`: reviews target commit/newer commits and posts contract-compliant PR comment output.

### Entrypoint and inputs

Fixed entrypoint:

- `.agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh`

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
- Resume-vs-new selection is not handled inside the loop script. The `pr-autoloop` skill caller handles this selection before invocation:
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

### Reviewer comment contract

Reviewer comments require:

- `AUTO_AGENT: REVIEWER`
- `AUTO_REVIEW_STATUS: APPROVED|CHANGES_REQUIRED`
- `AUTO_TARGET_COMMIT: <sha>`
- `APPROVE` token only when approved

Reviewer timing rule:

- reviewer must post without waiting for CI completion.

### Comment collection and approval decision

For each review cycle, the script fetches self-authored PR output after the target commit timestamp from:

- issue comments,
- review bodies,

via `gh api graphql`.

Time filtering rule:

- compare timestamps as epoch seconds (`fromdateiso8601` in jq), not raw ISO string lexical comparison, so timezone-offset differences cannot hide valid newer comments.

Approval is valid only when both are true in at least one collected comment:

1. `APPROVE` token is present,
2. `AUTO_TARGET_COMMIT` equals current target commit.

### Stop behavior

- No-op stop: if first builder phase produces no new commit after cleanup.
- Success stop: approval condition above is met and the script finalizes PR tracking by writing completed metadata (`review state: OPEN`) and moving the tracking file to `docs/prs/completed/`.
- Failure stop: retry bounds reached (`max-iterations` or reviewer/builder cleanup failure bounds).

### Output forwarding contract

- `codex exec` output is not persisted to per-iteration log files.
- Builder/reviewer subprocess output is forwarded directly to caller stdout/stderr.

### Lifecycle event integration

- `execplan.pre_creation`: captures plan/PR tracking initialization only.
- `execplan.post_completion`: performs lifecycle completion checks only (validation-only, no git mutation).
- Neither lifecycle event starts reviewer automation, performs reviewer wait/approval gating, nor transitions PR tracking documents to completed/ready state.

## Safety and isolation

- Loop execution begins only from tracked-clean working tree.
- Baseline untracked files are excluded from cleanup targets.
- Approval parsing is strict and commit-bound to prevent stale comment acceptance.

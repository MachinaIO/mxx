# Design: Builder/Reviewer Autonomous PR Loop Contract

## Purpose

Define a long-lived contract for autonomous PR iteration where a builder agent and reviewer agent run in a deterministic loop using machine-readable comment tags.

## Problem Statement

Manual coordination between implementation and review steps causes inconsistent stop conditions and ambiguous ownership when both roles use a shared GitHub identity.

## Design Goals

1. Deterministic loop control with explicit stop/continue states.
2. Explicit role attribution in PR comments even under shared GitHub account usage.
3. Contract-first parsing: reviewer status must be machine readable.
4. Bounded failure behavior to prevent runaway execution.

## Non-Goals

- Replacing repository review policy in `REVIEW.md`.
- Replacing CI status checks.
- Defining host-level process supervision (`systemd`) as a mandatory requirement.

## Core Contract

### Roles

- `builder agent`: implements feedback, commits, and pushes.
- `reviewer agent`: performs independent review and posts one contract-compliant PR comment per iteration.

### Comment tags

All autonomous-loop comments include:

- `[AUTO_LOOP]`
- `AUTO_RUN_ID: <id>`
- `AUTO_ITERATION: <n>`
- `AUTO_AGENT: BUILDER|REVIEWER`

Reviewer comments additionally require:

- `AUTO_REVIEW_STATUS: APPROVED|CHANGES_REQUIRED`
- `AUTO_TARGET_COMMIT: <sha>`
- `AUTO_REQUEST_ID: <request_id>` when driven by daemon request/response flow
- `APPROVE` token when and only when status is approved

Reviewer timing rule in autonomous-loop mode:

- Reviewer posts a contract-compliant comment for each iteration without waiting for CI completion.
- If CI is still running, reviewer still emits `APPROVED` or `CHANGES_REQUIRED` based on current evidence.

### Startup modes

- Existing PR mode: operator provides `--pr-url`; loop binds to that PR immediately.
- Bootstrap mode: operator provides `--head-branch` (and optional `--base-branch`) without `--pr-url`; builder creates or reuses a PR, then loop auto-discovers the PR URL and passes it to reviewer.

### Stop conditions

- `APPROVED`: successful termination.
- `CHANGES_REQUIRED`: continue with next builder iteration.
- Missing/invalid reviewer contract tags: fail-stop (`FAILED_CONTRACT`).
- Builder consecutive failure count reaches configured threshold: fail-stop (`FAILED_LIMIT`).

### Lifecycle event integration

- `execplan.pre_creation`: ensure reviewer daemon process is running (start in background when absent).
- `execplan.post_completion`: send latest commit metadata to reviewer daemon, wait for response, fetch returned review-comment URL, and pass only when `APPROVE` token is present.
- Reviewer daemon exits after emitting an approved (`APPROVE`) review comment; otherwise it stays active awaiting new commit messages.

### Safety and isolation

- One lock per PR prevents concurrent loops for the same PR.
- Before PR discovery in bootstrap mode, loop uses one lock per head branch and keeps the lock for the run lifetime.
- Runtime state is persisted in skill-local files for auditability and restart analysis.

## Trade-offs

- Strict contract parsing increases failures for malformed comments but avoids ambiguous behavior.
- Shared account operation remains possible, but correctness depends on role-tag discipline.

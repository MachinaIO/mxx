# Design: Builder-Started Reviewer Daemon Contract

## Purpose

Define a long-lived contract for autonomous PR iteration where the builder starts and communicates with a long-running reviewer daemon using machine-readable comment tags.

## Problem Statement

Manual coordination between implementation and review steps causes inconsistent handoff timing and ambiguous ownership when both roles use a shared GitHub identity.

## Design Goals

1. Deterministic builder-to-reviewer handoff with explicit machine-readable request/response state.
2. Explicit role attribution in PR comments even under shared GitHub account usage.
3. Contract-first parsing: reviewer output status and target commit must be machine readable.
4. Deterministic lifecycle gating for `execplan.pre_creation` and `execplan.post_completion`.

## Non-Goals

- Replacing repository review policy in `REVIEW.md`.
- Replacing CI status checks.
- Defining host-level process supervision (`systemd`) as a mandatory requirement.

## Core Contract

### Roles

- `builder agent`: implements feedback, commits, and pushes.
- `reviewer agent`: performs independent review and posts one contract-compliant PR comment per iteration.

### Reviewer comment tags

Reviewer comments require:

- `AUTO_REVIEW_STATUS: APPROVED|CHANGES_REQUIRED`
- `AUTO_TARGET_COMMIT: <sha>`
- `AUTO_REQUEST_ID: <request_id>`
- `AUTO_RUN_ID: <run_id>`
- `AUTO_ITERATION: <n>`
- `AUTO_AGENT: REVIEWER`
- `APPROVE` token when and only when status is approved

Reviewer timing rule:

- Reviewer posts a contract-compliant comment for each iteration without waiting for CI completion.
- If CI is still running, reviewer still emits `APPROVED` or `CHANGES_REQUIRED` based on current evidence.

### Daemon request/response flow

1. Builder (or lifecycle event) starts reviewer daemon with `reviewer_daemon.sh --start`.
2. Builder sends one message for each review cycle with `reviewer_daemon.sh --request --commit <sha>`.
3. Reviewer daemon posts exactly one contract-compliant PR comment and writes response JSON containing `comment_url` and parsed status fields.
4. Caller consumes response JSON as deterministic control input.

Startup modes:

- Existing PR mode: operator provides `--pr-url`; daemon binds to that PR immediately.
- Branch-first mode: operator provides `--head-branch`; daemon discovers PR URL by head branch when needed.

### Stop behavior

- `APPROVED` with `APPROVE` token: lifecycle post-completion verification can pass; daemon exits.
- `CHANGES_REQUIRED`: daemon remains active and waits for next request.
- Missing/invalid reviewer contract tags, commit mismatch, or missing comment URL: hard failure.

### Lifecycle event integration

- `execplan.pre_creation`: ensure reviewer daemon process is running (start in background when absent).
- `execplan.post_completion`: send latest commit metadata to reviewer daemon, wait for response, fetch returned review-comment URL, and pass only when `APPROVE` token is present.
- Reviewer daemon exits after emitting an approved (`APPROVE`) review comment; otherwise it stays active awaiting new commit messages.

### Safety and isolation

- Runtime state is persisted in skill-local daemon files for auditability and restart analysis.
- Contract parsing is strict: prose-only reviewer outputs are invalid.

## Trade-offs

- Strict contract parsing increases failures for malformed comments but prevents ambiguous lifecycle decisions.
- Daemon process management adds operational complexity, but gives deterministic synchronization and reply semantics between builder and reviewer.

# Update CPU Behavior Verification Event Policy

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan is maintained under `PLANS.md` and follows that file’s rules.

ExecPlan start context:
- Branch at start: `feat/harness_enginnering`
- Commit at start: `2c202a2`
- PR tracking document: `docs/prs/active/pr_feat_harness_enginnering.md`

Repository-document context used for this plan: `PLANS.md`, `VERIFICATION.md`, `docs/verification/index.md`, `docs/verification/execplan_pre_creation.md`, and `docs/verification/cpu_behavior_changes.md`.

## Purpose / Big Picture

After this change, the CPU behavior verification event will reflect local-development policy more accurately: no mandatory clippy runs in this event, scope-targeted unit testing by default, full `cargo test -r --lib` only for completed feature batches or foundational changes, no forced CI-only test flags locally, no automatic integration-test runs under `tests/` unless explicitly requested by a human operator, and explicit human reporting requirement for failed tests after ExecPlan completion.

## Progress

- [x] (2026-03-02 02:49Z) Completed pre-ExecPlan checks (`git branch --show-current`, `git status --short`, `git log --oneline --decorate --max-count=20`) and PR-context fallback check (`gh pr status` unavailable locally).
- [x] (2026-03-02 02:49Z) Reused PR tracking document path `docs/prs/active/pr_feat_harness_enginnering.md` for this plan.
- [x] (2026-03-02 02:52Z) Updated `docs/verification/cpu_behavior_changes.md` according to requested rules.
- [x] (2026-03-02 02:52Z) Ran docs-only verification checks and recorded outcomes (`git diff --name-only --`, `rg -n \"TODO|TBD|FIXME\" ...`, and content-keyword checks).
- [x] (2026-03-02 02:53Z) Moved this plan to `docs/plans/completed/`.

## Surprises & Discoveries

- Observation: GitHub CLI is unavailable in this environment.
  Evidence: `gh pr status` returned `/bin/bash: gh: command not found`.

## Decision Log

- Decision: Keep CPU-event verification focused on unit-test strategy and explicitly separate CI-only flags from local execution guidance.
  Rationale: The user requested local policy refinement and removal of CI-only defaults from this event.
  Date/Author: 2026-03-02 / Codex

## Outcomes & Retrospective

The CPU behavior verification event is now aligned with local testing workflow. It no longer mandates clippy calls, now uses scope-targeted unit testing as the default strategy, conditionally escalates to full `cargo test -r --lib` for completed feature batches or foundational changes, and explicitly separates CI-only test flags from local execution. It also now forbids automatic integration-test execution under `tests/` unless explicitly requested by a human operator and requires a clear human-facing post-ExecPlan failure report when tests fail.

## Design/Architecture/Verification Document Summary

Design documents:

- Referenced: none.
- Created/modified: none.

Architecture documents:

- Referenced: none.
- Created/modified: none.

Verification documents:

- Referenced: `VERIFICATION.md`, `docs/verification/index.md`, `docs/verification/cpu_behavior_changes.md`.
- Modified:
  - `docs/verification/cpu_behavior_changes.md`

## Context and Orientation

Current CPU behavior event documentation mirrors CI command shape more than local execution policy and includes clippy/test flags that the user now wants excluded from local default testing guidance.

## Plan of Work

Rewrite `docs/verification/cpu_behavior_changes.md` to remove mandatory clippy commands and CI-only test flags, define scope-based unit-test execution policy, add the conditional full unit-test sweep trigger, explicitly prohibit automatic `tests/` integration-test execution unless operator-requested, and add reporting requirements for failures at ExecPlan completion.

## Concrete Steps

Run from repository root (`.`):

    apply_patch << 'PATCH'
    ...
    PATCH
    sed -n '1,260p' docs/verification/cpu_behavior_changes.md
    git diff --name-only --
    rg -n "clippy|RUST_TEST_NOCAPTURE|--test-threads=1|tests/|cargo test -r --lib|failed tests|human" docs/verification/cpu_behavior_changes.md -S

## Validation and Acceptance

Acceptance criteria:

1. `docs/verification/cpu_behavior_changes.md` no longer requires clippy commands.
2. Document instructs scope-targeted tests when edited scope is clear.
3. Document requires full `cargo test -r --lib` for completed feature batches or foundational shared-scope changes.
4. Document explicitly states CI-only flags (`RUST_TEST_NOCAPTURE=1`, `--test-threads=1`) are not required for local runs.
5. Document explicitly forbids running `tests/` integration tests unless requested by a human operator.
6. Document requires clear human-facing failure reporting after ExecPlan completion.

## Idempotence and Recovery

This is a documentation-only update and can be reapplied or reverted safely.

## Artifacts and Notes

Expected modified file:

    docs/verification/cpu_behavior_changes.md

## Interfaces and Dependencies

No code interfaces or runtime behavior changes.

Revision note (2026-03-02, Codex): Initial plan created for CPU behavior verification policy update.
Revision note (2026-03-02, Codex): Updated progress/outcomes after rewriting the CPU behavior event rules and validating docs-only checks.
Revision note (2026-03-02, Codex): Marked completion and moved this plan from `active` to `completed`.

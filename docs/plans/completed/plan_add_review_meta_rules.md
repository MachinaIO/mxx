# Add Review Policy Meta-Document

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` are updated as work proceeds.

This plan follows `PLANS.md`.

ExecPlan start context:
- Branch at start: `feat/harness_enginnering`
- Commit at start: `36f9bbe`
- PR tracking document: `docs/prs/active/pr_feat_harness_enginnering.md`

Repository-document context used for this plan: `PLANS.md`, `AGENTS.md`, `ARCHITECTURE.md`, `DESIGN.md`, `docs/verification/index.md`, `docs/verification/docs_only_changes.md`, and `docs/verification/execplan_post_completion.md`.

## Purpose / Big Picture

After this change, the repository will have a root-level `REVIEW.md` that defines strict reviewer-mode behavior for PR review tasks. The document will require independent and skeptical review, concrete validation checks, and mandatory GitHub PR comment reporting in English.

## Progress

- [x] (2026-03-02 04:24Z) Completed pre-ExecPlan checks (`git branch --show-current`, `git status --short`, `git log --oneline --decorate --max-count=20`, `gh pr status`, `gh pr view 56 ...`) and confirmed branch/PR alignment.
- [x] (2026-03-02 04:24Z) Added `REVIEW.md` in architecture-style meta-policy tone with all requested reviewer rules.
- [x] (2026-03-02 04:24Z) Ran docs-only verification checks and recorded results (`git status --short`, `rg -n "TODO|TBD|FIXME" docs PLANS.md ARCHITECTURE.md VERIFICATION.md DESIGN.md REVIEW.md`).
- [x] (2026-03-02 04:25Z) Updated this completed plan and ran post-ExecPlan validation by checking linked PR tracking metadata and current PR state (`gh pr view 56 ...`).
- [x] (2026-03-02 04:25Z) Persisted final completed state with commit/push.

## Surprises & Discoveries

- None so far.

## Decision Log

- Decision: Keep `REVIEW.md` at repository root and align section style with `ARCHITECTURE.md`, `DESIGN.md`, and `VERIFICATION.md`.
  Rationale: This is a repository-wide reviewer meta-rule and should be discoverable at the same level as other governance documents.
  Date/Author: 2026-03-02 / Codex

## Outcomes & Retrospective

Completed:
- Added root-level `REVIEW.md` that defines independent reviewer-mode behavior and strict PR review checks.
- Captured all requested checks, review cycle behavior, and reviewer-mode restrictions in English.

Pending:
- None.

## Design/Architecture/Verification Document Summary

Design documents:
- Referenced: `DESIGN.md`.
- Created/modified:
  - `REVIEW.md` (new reviewer policy meta-document)

Architecture documents:
- Referenced: `ARCHITECTURE.md` (tone and policy style baseline).
- Created/modified: none.

Verification documents:
- Referenced: `docs/verification/docs_only_changes.md`, `docs/verification/execplan_post_completion.md`.
- Created/modified: none.

Post-ExecPlan validation result:
- Linked PR doc: `docs/prs/active/pr_feat_harness_enginnering.md`
- PR state checked: `OPEN`, `isDraft=true` (PR #56)
- Readiness decision: not ready for review yet; keep PR tracking doc in `docs/prs/active/`.

## Context and Orientation

The repository currently has policy documents for plans, architecture, design, and verification, but no explicit reviewer-mode policy document. This task adds `REVIEW.md` so agents can switch to strict, independent reviewer behavior when asked to review PRs.

## Plan of Work

Create `REVIEW.md` with operational reviewer rules from the request, keeping documentation language in English and matching existing policy-document style. Then run docs-only verification checks, complete and move this plan, run post-ExecPlan validation, and commit/push final state.

## Concrete Steps

Run from repository root (`.`):

    apply_patch ... (create REVIEW.md)
    git status --short
    rg -n "TODO|TBD|FIXME" docs PLANS.md ARCHITECTURE.md VERIFICATION.md DESIGN.md REVIEW.md

## Validation and Acceptance

Acceptance criteria:

1. `REVIEW.md` exists at repository root.
2. It explicitly instructs reviewer independence and skeptical review posture.
3. It includes all requested review checks (CI, test quality/static analysis, impacted unit tests, redundancy/fallback cleanup checks, benchmark delta checks, and anomaly checks).
4. It defines the PR-identification/review cycle and asks for user clarification only when confidence is very low.
5. It requires posting review results as an English GitHub PR comment.
6. It explicitly forbids commit/push during reviewer-mode execution.

## Idempotence and Recovery

This is documentation-only work. Re-running edits and checks is safe.

## Artifacts and Notes

Expected touched files:

    REVIEW.md
    docs/plans/completed/plan_add_review_meta_rules.md

## Interfaces and Dependencies

No code interface or runtime behavior changes.

Revision note (2026-03-02, Codex): Initial plan created for adding `REVIEW.md`.
Revision note (2026-03-02, Codex): Updated progress and outcomes after creating `REVIEW.md` and running docs-only checks.
Revision note (2026-03-02, Codex): Added post-ExecPlan validation result and PR readiness decision.

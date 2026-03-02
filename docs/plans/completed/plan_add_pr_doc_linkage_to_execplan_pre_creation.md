# Add PR Documentation Linkage Rules to Pre-ExecPlan Verification Event

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan is maintained under `PLANS.md` and follows that file’s rules.

Repository-document context used for this plan: `PLANS.md`, `VERIFICATION.md`, `docs/verification/index.md`, and `docs/verification/execplan_pre_creation.md`.

## Purpose / Big Picture

After this change, the pre-ExecPlan verification event will also enforce PR tracking: when a new branch is created, agents must create a draft PR, create/update a PR metadata document under `docs/prs/active`, and carry that PR document path into future ExecPlans. When reusing an existing branch/PR, agents must still add the corresponding PR document path to the ExecPlan. In all cases, future ExecPlans must also record the branch name and git commit hash at the moment the ExecPlan starts.

## Progress

- [x] (2026-03-02 02:31Z) Completed pre-ExecPlan checks (`git branch --show-current`, `git status --short`, `git log --oneline --decorate --max-count=20`, and `gh pr status` fallback due to missing CLI).
- [x] (2026-03-02 02:36Z) Updated `docs/verification/execplan_pre_creation.md` with PR creation/documentation linkage steps.
- [x] (2026-03-02 02:36Z) Added explicit requirement to record start-time branch name and git commit hash in future ExecPlans.
- [x] (2026-03-02 02:37Z) Ran docs-only validation checks and recorded outcomes (`git diff --name-only --`, `rg -n "TODO|TBD|FIXME" ...`, and content/keyword checks on the updated document).
- [x] (2026-03-02 02:38Z) Moved this plan to `docs/plans/completed/`.

## Surprises & Discoveries

- Observation: GitHub CLI is unavailable locally.
  Evidence: `gh pr status` returned `/bin/bash: gh: command not found`.
- Observation: The repository already had many staged/modified files unrelated to this doc update, so `git diff --name-only --` output is not scoped to this single task.
  Evidence: `git diff --name-only --` listed multiple pre-existing documentation changes.

## Decision Log

- Decision: Keep PR creation guidance executable with conditional CLI instructions and fallback notes when `gh` is unavailable.
  Rationale: The event requires PR metadata handling even in environments without GitHub CLI.
  Date/Author: 2026-03-02 / Codex

## Outcomes & Retrospective

The pre-ExecPlan verification event now requires PR lifecycle tracking in addition to branch alignment checks. It enforces draft PR creation for newly created branches, mandatory PR metadata tracking under `docs/prs/active`, and mandatory PR tracking-file linkage in future ExecPlans for both new and reused branch/PR flows.

It also now requires future ExecPlans to record the branch name and git commit hash at the moment the ExecPlan starts.

## Design/Architecture/Verification Document Summary

Design documents:

- Referenced: none.
- Created/modified: none.

Architecture documents:

- Referenced: none.
- Created/modified: none.

Verification documents:

- Referenced: `VERIFICATION.md`, `docs/verification/index.md`, `docs/verification/execplan_pre_creation.md`.
- Modified:
  - `docs/verification/execplan_pre_creation.md`

## Context and Orientation

The current pre-ExecPlan event document covers branch/PR alignment and branch switching, but it does not require creating and linking PR metadata documents under `docs/prs/active` or carrying those paths into ExecPlans. This change adds those missing obligations.

## Plan of Work

Append explicit steps to the pre-ExecPlan event for: draft PR creation on new branches, required PR metadata fields and storage under `docs/prs/active`, mandatory insertion of PR document relative paths into future ExecPlans for both new and reused branch/PR flows, and start-time branch/commit recording in those ExecPlans.

## Concrete Steps

Run from repository root (`.`):

    apply_patch << 'PATCH'
    ...
    PATCH
    sed -n '1,320p' docs/verification/execplan_pre_creation.md
    git diff --name-only --
    rg -n "docs/prs/active|draft PR|ExecPlan" docs/verification/execplan_pre_creation.md -S

## Validation and Acceptance

Acceptance criteria:

1. `docs/verification/execplan_pre_creation.md` requires draft PR creation when a new branch is created.
2. The document requires adding PR metadata (link, creation date, branch name, creation commit, content) under `docs/prs/active`.
3. The document requires adding the PR doc relative path into future ExecPlans for both new and reused branch/PR scenarios.
4. The document requires recording start-time branch name and git commit hash in future ExecPlans.
5. Docs-only checks are run and recorded in this plan.

## Idempotence and Recovery

This is a documentation-only edit and can be safely reapplied or reverted.

## Artifacts and Notes

Expected modified file:

    docs/verification/execplan_pre_creation.md

## Interfaces and Dependencies

No code interfaces or runtime behavior are changed.

Revision note (2026-03-02, Codex): Initial plan created to add PR linkage rules to pre-ExecPlan verification.
Revision note (2026-03-02, Codex): Updated progress/outcomes after adding PR-tracking and start-time branch/commit requirements to the pre-ExecPlan event.
Revision note (2026-03-02, Codex): Marked completion and moved this plan from `active` to `completed`.

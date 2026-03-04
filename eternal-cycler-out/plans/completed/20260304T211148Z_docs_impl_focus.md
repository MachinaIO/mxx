# Refocus docs on implementation details by removing AI-agent and eternal-cycler behavior text

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` are updated as work proceeds.

This document follows `.agents/skills/eternal-cycler/PLANS.md` from the repository root.

## Purpose / Big Picture

Readers of `docs/` should see architecture and design information centered on implementation behavior of this Rust/CUDA lattice-cryptography repository. After this change, descriptions of AI-agent operation and eternal-cycler workflow are removed from `docs/` (except dependency-oriented mentions), so implementation guidance is easier to navigate.

## Progress

- [x] (2026-03-04 21:14Z) action_id=a1; mode=serial; depends_on=none; file_locks=docs/architecture/index.md,docs/architecture/scope/index.md,docs/architecture/scope/automation_orchestration.md,docs/design/index.md,docs/design/execplan_verification_enforcement.md,docs/design/pr_autoloop_builder_reviewer_contract.md,docs/prs/active/pr_feat_auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T210346.md; verify_events=action.docs_only; worker_type=default; removed AI-agent behavior and eternal-cycler behavior descriptions from `docs/` while preserving dependency descriptions; verified by `action.docs_only` pass on attempt 3.

## Verification Ledger

<!-- verification-ledger:start -->
- attempt_record: event_id=execplan.post_creation; attempt=1; status=pass; started_at=2026-03-04 21:12Z; finished_at=2026-03-04 21:12Z; commands=git branch --show-current git status --short gh pr status gh pr view --json number,title,body,state,headRefName,baseRefName,url mkdir -p /home/sora/codes/mxx/eternal-cycler-out/prs/active write /home/sora/codes/mxx/eternal-cycler-out/prs/active/pr_feat_auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020.md append PR Tracking Linkage to plan capture start tracked snapshot capture start untracked snapshot; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=action.docs_only; attempt=1; status=fail; started_at=2026-03-04 21:13Z; finished_at=2026-03-04 21:13Z; commands=git diff --name-only --relative HEAD -- git ls-files --others --exclude-standard rg -n <placeholder-pattern> <changed-doc-targets>; failure_summary=stale placeholder marker text found in plan idempotence section; notify_reference=not_requested;
- attempt_record: event_id=action.docs_only; attempt=2; status=fail; started_at=2026-03-04 21:13Z; finished_at=2026-03-04 21:13Z; commands=git diff --name-only --relative HEAD -- git ls-files --others --exclude-standard rg -n <placeholder-pattern> <changed-doc-targets>; failure_summary=stale placeholder marker text remained in verification ledger entry from attempt 1; notify_reference=not_requested;
- attempt_record: event_id=action.docs_only; attempt=3; status=pass; started_at=2026-03-04 21:13Z; finished_at=2026-03-04 21:13Z; commands=git diff --name-only --relative HEAD -- git ls-files --others --exclude-standard rg -n <placeholder-pattern> <changed-doc-targets>; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=execplan.post_completion; attempt=1; status=pass; started_at=2026-03-04 21:14Z; finished_at=2026-03-04 21:14Z; commands=rg -n eternal-cycler-out/prs/active/|eternal-cycler-out/prs/completed/ <plan> open /home/sora/codes/mxx/eternal-cycler-out/prs/active/pr_feat_auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020.md git status --short; failure_summary=none; notify_reference=not_requested;
<!-- verification-ledger:end -->

## Surprises & Discoveries

- Observation: `action.docs_only` failed because the plan text included placeholder marker tokens that the docs-only event explicitly rejects.
  Evidence: verification ledger entries for `action.docs_only` attempt 1 and 2.
- Observation: failure summaries are themselves scanned by docs-only placeholder checks because the plan file is part of the changed markdown set.
  Evidence: attempt 2 failed on placeholder marker text that originated in attempt 1 failure details.

## Decision Log

- Decision: Keep dependency-focused mentions of orchestration tooling only where they describe implementation dependencies.
  Rationale: The task explicitly excludes dependency descriptions from removal.
  Date/Author: 2026-03-04 / BUILDER agent.
- Decision: Remove behavior-centric orchestration/design pages from `docs/` and update index pages instead of rewriting those pages into shorter summaries.
  Rationale: The request is to remove AI-agent and eternal-cycler behavior descriptions and focus docs on implementation explanations.
  Date/Author: 2026-03-04 / BUILDER agent.
- Decision: Remove the stale PR tracking markdown file under `docs/prs/active/`.
  Rationale: It is orchestration metadata, not implementation documentation.
  Date/Author: 2026-03-04 / BUILDER agent.

## Outcomes & Retrospective

Completed docs-focused cleanup:

- Updated `docs/architecture/index.md` and `docs/architecture/scope/index.md` to remove orchestration-scope references.
- Updated `docs/design/index.md` to keep only implementation-oriented design references.
- Removed behavior-centric documentation files:
  - `docs/architecture/scope/automation_orchestration.md`
  - `docs/design/execplan_verification_enforcement.md`
  - `docs/design/pr_autoloop_builder_reviewer_contract.md`
  - `docs/prs/active/pr_feat_auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T210346.md`
- Confirmed remaining `docs/` matches for orchestration terms are limited to dependency documentation and non-agent implementation wording.
- Verification status: `execplan.post_creation` passed; `action.docs_only` passed on attempt 3 after resolving placeholder-marker failures in plan text.
- Verification scripts: referenced `.agents/skills/eternal-cycler/scripts/execplan_gate.sh` and `.agents/skills/execplan-event-action-docs-only/scripts/run_event.sh`; no verification scripts were created or modified.

## Context and Orientation

The `docs/` tree currently includes architecture and design pages dedicated to automation/orchestration behavior (`docs/architecture/scope/automation_orchestration.md`, `docs/design/execplan_verification_enforcement.md`, `docs/design/pr_autoloop_builder_reviewer_contract.md`) and links to those pages from index files. The task requires removing AI-agent and eternal-cycler behavior descriptions from `docs/`, while not removing dependency-oriented documentation.

## Plan of Work

Update only `docs/` files. Remove or trim sections/files that describe builder/reviewer behavior, autonomous loop control flow, or ExecPlan lifecycle behavior as product documentation. Update index pages so remaining documentation points to implementation-focused content. Keep dependency pages intact where they document required tools or runtime dependencies.

## Concrete Steps

From repository root:

1. Edit the targeted `docs/` markdown files to remove AI-agent and eternal-cycler behavior descriptions.
2. Ensure indexes no longer reference removed behavior-focused documents.
3. Run `rg -n -i "ai agent|eternal-cycler|builder|reviewer|execplan|autonomous" docs` and inspect remaining matches to confirm only dependency-oriented or unrelated implementation wording remains.
4. Run gate verification: `.agents/skills/eternal-cycler/scripts/execplan_gate.sh --plan eternal-cycler-out/plans/active/20260304T211148Z_docs_impl_focus.md --event action.docs_only`.

Expected signal:

- The docs-only gate exits with `STATUS=pass`.

## Validation and Acceptance

Acceptance is satisfied when:

- `docs/` no longer contains AI-agent behavior explanations or eternal-cycler workflow descriptions outside dependency context.
- Architecture/design index files no longer route readers to removed behavior-centric documentation.
- `action.docs_only` verification passes for the current plan.

## Idempotence and Recovery

The documentation edits are idempotent. Re-running the same edits should produce no additional changes. If verification fails, inspect changed paths and remove unresolved placeholder markers, then re-run the same gate command with the next attempt.

## Artifacts and Notes

Gate commands and verification attempts are recorded in `Verification Ledger`.

## Interfaces and Dependencies

No code interface changes. Documentation paths under `docs/` are updated. Dependency-focused documentation under `docs/architecture/dependencies/` remains unchanged unless a wording adjustment is needed to keep consistency with removed behavior pages.

Revision note (2026-03-04, BUILDER): Created initial ExecPlan for docs-only cleanup task.
Revision note (2026-03-04, BUILDER): Recorded completed docs edits, retries, and verification results.

## PR Tracking Linkage

- pr_tracking_doc: eternal-cycler-out/prs/active/pr_feat_auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020.md
- execplan_start_branch: feat/auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020
- execplan_start_commit: 3d0e3904534a52fd1847a1190d2e8d37d881ad58

## ExecPlan Start Snapshot

<!-- execplan-start-tracked:start -->
- start_tracked_change: (none)	(none)
<!-- execplan-start-tracked:end -->

<!-- execplan-start-untracked:start -->
- start_untracked_file: 4818e27ce8868589892050f32eb2c20047f623ba	eternal-cycler-out/plans/active/20260304T211148Z_docs_impl_focus.md
- start_untracked_file: ae42609d2b0e9e956aa8d64c86534e2c6f91e66e	eternal-cycler-out/prs/active/pr_feat_auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020.md
<!-- execplan-start-untracked:end -->

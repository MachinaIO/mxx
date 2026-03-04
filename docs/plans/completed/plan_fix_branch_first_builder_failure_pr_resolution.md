# Fix Branch-First Builder Failure PR Resolution

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan follows `PLANS.md` and executes the lifecycle and verification policy defined there.

Repository-document context used for this plan: `AGENTS.md`, `PLANS.md`, `DESIGN.md`, `docs/design/index.md`, `docs/design/pr_autoloop_builder_reviewer_contract.md`, `ARCHITECTURE.md`, `docs/architecture/index.md`, `docs/architecture/scope/automation_orchestration.md`, `.agents/skills/execplan-event-index/SKILL.md`, `.agents/skills/execplan-event-index/references/event_skill_map.tsv`, `.agents/skills/execplan-event-action-tooling/SKILL.md`, `.agents/skills/execplan-event-action-tooling/scripts/run_event.sh`, `.agents/skills/eternal-cycler/assets/default-verification/execplan-event-action-tooling/scripts/run_event.sh`, `scripts/execplan_gate.sh`, and `scripts/execplan_notify.sh`.

Design and architecture guidance impact: the change is a bug fix in existing shell-script behavior and verification coverage for branch-first failure handling. It does not introduce a new long-lived interface/contract beyond the existing builder JSON contract already documented in `docs/design/pr_autoloop_builder_reviewer_contract.md`, and it does not change repository/module boundaries described in `docs/architecture/scope/automation_orchestration.md`. Therefore design/architecture documents are referenced and left unchanged.

## Purpose / Big Picture

After this change, the branch-first failure path in `.agents/skills/eternal-cycler/scripts/run_builder_reviewer_loop.sh` will always be able to resolve or create the target PR and post the required builder failure report comment, even when `PR_URL` is initially empty and builder output is `failed_after_3_retries`. A regression guard in tooling verification will fail fast if this call path drops required PR metadata arguments again.

## Milestones

The implementation has one milestone: update the failure-path argument plumbing and defensive defaults in `resolve_or_create_pr_for_branch`, then add a tooling verification guard that asserts the branch-first failure call still forwards PR title/body metadata. Success is demonstrated by syntax checks plus passing `action.tooling` verification through `scripts/execplan_gate.sh` with this plan recorded in `Verification Ledger`.

## Progress

- [x] (2026-03-04 15:32Z) action_id=a0; mode=serial; depends_on=none; file_locks=docs/plans/active/plan_fix_branch_first_builder_failure_pr_resolution.md; verify_events=none; worker_type=default; read policy/design/architecture/verification context and created this active ExecPlan.
- [x] (2026-03-04 18:19Z) action_id=a1; mode=serial; depends_on=a0; file_locks=.agents/skills/eternal-cycler/scripts/run_builder_reviewer_loop.sh,.agents/skills/execplan-event-action-tooling/scripts/run_event.sh,.agents/skills/eternal-cycler/assets/default-verification/execplan-event-action-tooling/scripts/run_event.sh,docs/plans/active/plan_fix_branch_first_builder_failure_pr_resolution.md; verify_events=action.tooling; worker_type=default; forwarded `pr_title`/`pr_body` in builder-failure PR resolution path, added optional-safe defaults in `resolve_or_create_pr_for_branch`, and added tooling-event regression guards for this branch-first failure flow.
- [x] (2026-03-04 18:19Z) action_id=a2; mode=serial; depends_on=a1; file_locks=docs/plans/active/plan_fix_branch_first_builder_failure_pr_resolution.md,docs/plans/completed/plan_fix_branch_first_builder_failure_pr_resolution.md; verify_events=none; worker_type=default; finalized plan sections and moved this plan to completed status before lifecycle post-completion verification.

## Surprises & Discoveries

- Both `action.tooling` scripts were syntax-only checks before this change, so the branch-first failure-path argument mismatch could pass verification until runtime.

## Decision Log

- Decision: use `action.tooling` as the only action verification event because the change touches shell orchestration and verification scripts, not runtime CPU/GPU behavior.
  Rationale: event mapping in `.agents/skills/execplan-event-index/references/event_skill_map.tsv` assigns script/tooling changes to `action.tooling`.
- Decision: keep `resolve_or_create_pr_for_branch` backward-safe by treating `pr_title`/`pr_body` as optional inputs with defensive defaults, while also explicitly forwarding builder-provided `pr_title`/`pr_body` from the failure-report path.
  Rationale: this eliminates `set -u` positional-parameter crashes in the branch-first failure path and protects against future one-argument call regressions.

## Outcomes & Retrospective

Implementation and `action.tooling` verification passed. The loop failure path now forwards PR metadata and has optional-safe argument defaults, and tooling verification now guards this branch-first failure path. Final lifecycle state is complete after `execplan.post_completion` pass is recorded.

## Verification Ledger

<!-- verification-ledger:start -->
- attempt_record: event_id=execplan.pre_creation; attempt=1; status=pass; started_at=2026-03-04 18:17Z; finished_at=2026-03-04 18:17Z; commands=git branch --show-current git status --short git log --oneline --decorate --max-count=20 gh pr status gh pr view --json number,title,body,state,headRefName,baseRefName,url mkdir -p docs/prs/active capture execplan start tracked snapshot capture execplan start untracked snapshot write/update docs/prs/active/pr_feat_auto-https-github-com-sorasuegami-eternal-cyc-20260304151714.md update plan linkage metadata; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=action.tooling; attempt=1; status=pass; started_at=2026-03-04 18:18Z; finished_at=2026-03-04 18:18Z; commands=bash -n scripts/*.sh .agents/skills/execplan-event-*/scripts/*.sh, rg -F 'PR_URL="$(resolve_or_create_pr_for_branch "$TARGET_BRANCH" "$pr_title" "$pr_body")"' .agents/skills/eternal-cycler/scripts/run_builder_reviewer_loop.sh, rg -F 'local pr_title="${2:-}"' .agents/skills/eternal-cycler/scripts/run_builder_reviewer_loop.sh, rg -F 'local pr_body="${3:-}"' .agents/skills/eternal-cycler/scripts/run_builder_reviewer_loop.sh; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=execplan.post_completion; attempt=1; status=pass; started_at=2026-03-04 18:19Z; finished_at=2026-03-04 18:19Z; commands=rg -n docs/prs/active/|docs/prs/completed/ <plan> open docs/prs/active/pr_feat_auto-https-github-com-sorasuegami-eternal-cyc-20260304151714.md git status --short; failure_summary=none; notify_reference=not_requested;
<!-- verification-ledger:end -->

## PR Tracking Linkage

- pr_tracking_doc: docs/prs/active/pr_feat_auto-https-github-com-sorasuegami-eternal-cyc-20260304151714.md
- execplan_start_branch: feat/auto-https-github-com-sorasuegami-eternal-cyc-20260304151714
- execplan_start_commit: e00b456c7beb3f9a6d10a37b1378c6ec651c5b92

## ExecPlan Start Snapshot

<!-- execplan-start-tracked:start -->
- start_tracked_change: 559d56b0fb2728dfb7ed2cf34b0ec87ecd7b54d3	docs/prs/active/pr_feat_auto-https-github-com-sorasuegami-eternal-cyc-20260304151714.md
<!-- execplan-start-tracked:end -->

<!-- execplan-start-untracked:start -->
- start_untracked_file: 9fe954a1dd2aeda7c99bc15a3837293d3028ea0f	docs/plans/active/plan_fix_branch_first_builder_failure_pr_resolution.md
<!-- execplan-start-untracked:end -->

# Install eternal-cycler and Complete PR #63 Tracking

This ExecPlan is a living document. The sections `Progress`, `Verification Ledger`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan follows `PLANS.md` and `AGENTS.md`.

Repository-document context used for this plan: `AGENTS.md`, `PLANS.md`, `DESIGN.md`, `ARCHITECTURE.md`, `docs/design/index.md`, `docs/architecture/index.md`, `docs/architecture/scope/index.md`, `docs/architecture/scope/automation_orchestration.md`, `.agents/skills/execplan-event-index/SKILL.md`, `.agents/skills/execplan-event-index/references/event_skill_map.tsv`, `.agents/skills/execplan-event-pre-creation/SKILL.md`, `.agents/skills/execplan-event-action-tooling/SKILL.md`, `.agents/skills/execplan-event-post-completion/SKILL.md`, `.agents/skills/execplan-sandbox-escalation/SKILL.md`, `.agents/skills/execplan-sandbox-escalation/references/allowed_command_prefixes.md`, `scripts/execplan_gate.sh`, and `scripts/execplan_notify.sh`.

ExecPlan start context:
- Branch at start: `feat/auto-https-github-com-sorasuegami-eternal-cyc-20260304151714`
- Commit at start: `7a25cfbcaab4d59de449e823901930a38e4deaeb`
- Lifecycle-linked PR tracking doc: `docs/prs/completed/pr_feat_pr-autoloop-skill.md`
- Branch tracking doc created by pre-creation event: `docs/prs/active/pr_feat_auto-https-github-com-sorasuegami-eternal-cyc-20260304151714.md`

## Purpose / Big Picture

After this change, this repository will have the upstream `eternal-cycler` skill installed exactly in `.agents/skills/eternal-cycler/` and initialized via upstream setup so verification skills/output directories are present. In the same change, PR #63 tracking docs will be consistent with merged state: `docs/prs/active/pr_feat_pr-autoloop-skill.md` removed from active and `docs/prs/completed/pr_feat_pr-autoloop-skill.md` updated with merged metadata.

## Progress

- [x] (2026-03-04 15:22Z) action_id=a0; mode=serial; depends_on=none; file_locks=docs/plans/active/plan_install_eternal_cycler_and_complete_pr63_tracking.md,docs/prs/active/pr_feat_auto-https-github-com-sorasuegami-eternal-cyc-20260304151714.md; verify_events=action.tooling; worker_type=default; initialize the ExecPlan lifecycle linkage for this branch plan document.
- [x] (2026-03-04 15:23Z) action_id=a1; mode=serial; depends_on=a0; file_locks=.agents/skills/eternal-cycler,.agents/skills/execplan-event-index,.agents/skills/execplan-event-pre-creation,.agents/skills/execplan-event-post-completion,.agents/skills/execplan-event-action-docs-only,.agents/skills/execplan-event-action-tooling,.agents/skills/execplan-sandbox-escalation,eternal-cycler-out; verify_events=action.tooling; worker_type=default; install upstream eternal-cycler subtree and run its setup procedure.
- [x] (2026-03-04 15:24Z) action_id=a2; mode=serial; depends_on=a1; file_locks=docs/prs/active/pr_feat_pr-autoloop-skill.md,docs/prs/completed/pr_feat_pr-autoloop-skill.md; verify_events=action.tooling; worker_type=default; update PR #63 tracking docs to merged-completed state and remove the stale active tracking file.
- [x] (2026-03-04 15:25Z) action_id=a3; mode=serial; depends_on=a2; file_locks=docs/plans/active/plan_install_eternal_cycler_and_complete_pr63_tracking.md,docs/plans/completed/plan_install_eternal_cycler_and_complete_pr63_tracking.md; verify_events=action.tooling; worker_type=default; finalize outcomes/ledger and move this plan to completed before post-completion gate.

## Verification Ledger

<!-- verification-ledger:start -->
- attempt_record: event_id=execplan.pre_creation; attempt=1; status=pass; started_at=2026-03-04 15:21Z; finished_at=2026-03-04 15:21Z; commands=git branch --show-current git status --short git log --oneline --decorate --max-count=20 gh pr status gh pr view --json number,title,body,state,headRefName,baseRefName,url mkdir -p docs/prs/active capture execplan start tracked snapshot capture execplan start untracked snapshot write/update docs/prs/active/pr_feat_auto-https-github-com-sorasuegami-eternal-cyc-20260304151714.md update plan linkage metadata; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=action.tooling; attempt=1; status=pass; started_at=2026-03-04 15:21Z; finished_at=2026-03-04 15:21Z; commands=bash -n scripts/*.sh .agents/skills/execplan-event-*/scripts/*.sh; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=action.tooling; attempt=2; status=pass; started_at=2026-03-04 15:23Z; finished_at=2026-03-04 15:23Z; commands=bash -n scripts/*.sh .agents/skills/execplan-event-*/scripts/*.sh; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=action.tooling; attempt=3; status=pass; started_at=2026-03-04 15:24Z; finished_at=2026-03-04 15:24Z; commands=bash -n scripts/*.sh .agents/skills/execplan-event-*/scripts/*.sh; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=execplan.post_completion; attempt=1; status=pass; started_at=2026-03-04 15:25Z; finished_at=2026-03-04 15:25Z; commands=rg -n docs/prs/active/|docs/prs/completed/ <plan> open docs/prs/completed/pr_feat_pr-autoloop-skill.md git status --short; failure_summary=none; notify_reference=not_requested;
<!-- verification-ledger:end -->

## Surprises & Discoveries

- Observation: `git subtree add --squash` created local commits immediately, which conflicts with this loop’s requirement to leave edits in the worktree/index.
  Evidence: post-subtree `git log --oneline --max-count=3` showed merge/squash commits, then `git reset --mixed HEAD~2` restored unstaged edits.
- Observation: upstream `setup.sh` intentionally skipped existing verification skill directories under `.agents/skills/` and only created `eternal-cycler-out/` directories.
  Evidence: setup output lines `SKIP execplan-event-* (already present ...; not overwritten)`.

## Decision Log

- Decision: Install via upstream `git subtree add --prefix .agents/skills/eternal-cycler ... --squash` and then run `bash .agents/skills/eternal-cycler/setup.sh` without local reinterpretation.
  Rationale: The user requested installation according to upstream README, so reproducing exact upstream installation flow minimizes drift.
  Date/Author: 2026-03-04 / Codex
- Decision: After subtree import, reset local commits to keep all requested edits as uncommitted worktree changes.
  Rationale: Builder-loop contract in this task requires leaving edits for the loop script to stage/commit/push.
  Date/Author: 2026-03-04 / Codex

## Outcomes & Retrospective

Completed so far:

- Installed upstream `eternal-cycler` skill under `.agents/skills/eternal-cycler/` using the README-prescribed subtree command.
- Ran upstream setup script and created local runtime output directories under `eternal-cycler-out/`.
- Updated PR #63 tracking documentation to merged state and removed stale active tracking file.
- Updated `docs/architecture/scope/automation_orchestration.md` so architecture documentation reflects the newly installed static skill path and runtime output root.

Remaining:

- None.

## Context and Orientation

The repository already contains local PR/autoloop tooling under `.agents/skills/pr-autoloop/` and verification event skills under `.agents/skills/execplan-event-*`. Upstream `eternal-cycler` is installed as a static skill directory under `.agents/skills/eternal-cycler/` and its setup script is expected to create `eternal-cycler-out/` dynamic output directories while not overwriting existing event skill directories. PR #63 (`feat/pr-autoloop-skill`) has merged, so its active tracking file should no longer remain in `docs/prs/active/`.

## Plan of Work

Apply upstream installation commands as documented, then reconcile PR #63 tracking metadata to merged-completed state. Run required action verification gate(s), update this plan with actual evidence, move it to completed, and run lifecycle post-completion verification.

## Concrete Steps

Run from repository root (`.`):

    scripts/execplan_gate.sh --plan docs/plans/active/plan_install_eternal_cycler_and_complete_pr63_tracking.md --event execplan.pre_creation
    git subtree add --prefix .agents/skills/eternal-cycler https://github.com/SoraSuegami/eternal-cycler.git main --squash
    bash .agents/skills/eternal-cycler/setup.sh
    gh pr view 63 --json number,url,state,mergedAt,isDraft,headRefName,baseRefName,title
    rm docs/prs/active/pr_feat_pr-autoloop-skill.md
    # edit docs/prs/completed/pr_feat_pr-autoloop-skill.md with merged metadata
    scripts/execplan_gate.sh --plan docs/plans/active/plan_install_eternal_cycler_and_complete_pr63_tracking.md --event action.tooling
    mv docs/plans/active/plan_install_eternal_cycler_and_complete_pr63_tracking.md docs/plans/completed/plan_install_eternal_cycler_and_complete_pr63_tracking.md
    scripts/execplan_gate.sh --plan docs/plans/completed/plan_install_eternal_cycler_and_complete_pr63_tracking.md --event execplan.post_completion

## Validation and Acceptance

Acceptance requires all of the following:

1. `.agents/skills/eternal-cycler/` exists with upstream content and setup has been run.
2. `eternal-cycler-out/plans/{active,completed,tech-debt}` and `eternal-cycler-out/prs/{active,completed}` exist.
3. `docs/prs/active/pr_feat_pr-autoloop-skill.md` does not exist.
4. `docs/prs/completed/pr_feat_pr-autoloop-skill.md` reflects merged metadata for PR #63.
5. Required gates pass: `execplan.pre_creation` (with plan), `action.tooling`, and `execplan.post_completion`.

## Idempotence and Recovery

`setup.sh` is designed to skip overwriting existing verification skill directories, so rerunning is safe. If subtree add fails due to pre-existing prefix content, remove partial directory state and rerun the exact command. If any gate fails, record failure in `Verification Ledger`, remediate, and rerun within the three-attempt bound.

## Design/Architecture/Verification Document Summary

Design documents:

- Referenced: `docs/design/index.md`.
- Created/modified: none expected for this installation-oriented change.

Architecture documents:

- Referenced: `docs/architecture/index.md`, `docs/architecture/scope/index.md`, `docs/architecture/scope/automation_orchestration.md`.
- Created/modified: `docs/architecture/scope/automation_orchestration.md` updated to record `.agents/skills/eternal-cycler/` static skill mapping and `eternal-cycler-out/` runtime output path.

Verification skill/scripts:

- Referenced: `.agents/skills/execplan-event-index/SKILL.md`, `.agents/skills/execplan-event-index/references/event_skill_map.tsv`, `.agents/skills/execplan-event-pre-creation/SKILL.md`, `.agents/skills/execplan-event-action-tooling/SKILL.md`, `.agents/skills/execplan-event-post-completion/SKILL.md`, `scripts/execplan_gate.sh`.
- Created/modified: none expected.

## Plan Revision Notes

- 2026-03-04 15:21Z: Initial plan created for upstream eternal-cycler installation and PR #63 tracking completion.
- 2026-03-04 15:23Z: Completed subtree install + setup and recorded `action.tooling` gate pass attempt 2.
- 2026-03-04 15:24Z: Updated PR #63 merged tracking metadata, removed stale active tracking doc, and recorded `action.tooling` gate pass attempt 3.
- 2026-03-04 15:25Z: Finalized plan state in active location and marked action `a3` complete before moving to completed.
- execplan_start_branch: feat/auto-https-github-com-sorasuegami-eternal-cyc-20260304151714
- execplan_start_commit: 7a25cfbcaab4d59de449e823901930a38e4deaeb

## ExecPlan Start Snapshot

<!-- execplan-start-tracked:start -->
- start_tracked_change: (none)	(none)
<!-- execplan-start-tracked:end -->

<!-- execplan-start-untracked:start -->
- start_untracked_file: 0bbca70032c3e1d299f09bdf9ab4920804328cc2	docs/plans/active/plan_install_eternal_cycler_and_complete_pr63_tracking.md
- start_untracked_file: 76942c52a89720591c87e91b66c9321bb00de009	docs/prs/active/pr_feat_auto-https-github-com-sorasuegami-eternal-cyc-20260304151714.md
<!-- execplan-start-untracked:end -->

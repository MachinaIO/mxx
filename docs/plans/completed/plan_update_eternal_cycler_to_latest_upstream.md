# Update eternal-cycler to Latest Upstream and Re-run Setup

This ExecPlan is a living document. The sections `Progress`, `Verification Ledger`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan follows `PLANS.md` and `AGENTS.md`.

Repository-document context used for this plan: `AGENTS.md`, `PLANS.md`, `DESIGN.md`, `ARCHITECTURE.md`, `docs/design/index.md`, `docs/architecture/index.md`, `docs/architecture/scope/index.md`, `docs/architecture/scope/automation_orchestration.md`, `.agents/skills/execplan-event-index/SKILL.md`, `.agents/skills/execplan-event-index/references/event_skill_map.tsv`, `.agents/skills/execplan-event-pre-creation/SKILL.md`, `.agents/skills/execplan-event-action-tooling/SKILL.md`, `.agents/skills/execplan-event-post-completion/SKILL.md`, `.agents/skills/execplan-sandbox-escalation/SKILL.md`, `.agents/skills/execplan-sandbox-escalation/references/allowed_command_prefixes.md`, `scripts/execplan_gate.sh`, and `scripts/execplan_notify.sh`.

ExecPlan start context:
- Branch at start: `feat/auto-https-github-com-sorasuegami-eternal-cyc-20260304151714`
- Commit at start: `01016769f79db3343d69eb859eb79924ca655bee`
- Lifecycle-linked PR tracking doc: `docs/prs/active/pr_feat_auto-https-github-com-sorasuegami-eternal-cyc-20260304151714.md`

## Purpose / Big Picture

After this change, `.agents/skills/eternal-cycler/` will match the latest upstream `main` from `https://github.com/SoraSuegami/eternal-cycler.git`, and repository-local setup outputs/verification skills will be refreshed via the upstream `setup.sh` flow. The repository documentation and PR-tracking metadata will remain aligned with this branch work, and ExecPlan lifecycle evidence will be complete in the plan ledger.

## Progress

- [x] (2026-03-04 18:06Z) action_id=a0; mode=serial; depends_on=none; file_locks=docs/plans/active/plan_update_eternal_cycler_to_latest_upstream.md,docs/prs/active/pr_feat_auto-https-github-com-sorasuegami-eternal-cyc-20260304151714.md; verify_events=action.tooling; worker_type=default; initialize this plan lifecycle linkage and capture start snapshots with pre-creation gate on the plan path.
- [x] (2026-03-04 18:08Z) action_id=a1; mode=serial; depends_on=a0; file_locks=.agents/skills/eternal-cycler,.agents/skills/execplan-event-index,.agents/skills/execplan-event-pre-creation,.agents/skills/execplan-event-post-completion,.agents/skills/execplan-event-action-docs-only,.agents/skills/execplan-event-action-tooling,.agents/skills/execplan-event-action-cpu-behavior,.agents/skills/execplan-event-action-gpu-behavior,.agents/skills/execplan-sandbox-escalation,eternal-cycler-out; verify_events=action.tooling; worker_type=default; pull latest upstream eternal-cycler subtree and re-run upstream setup.
- [x] (2026-03-04 18:10Z) action_id=a2; mode=serial; depends_on=a1; file_locks=docs/architecture/scope/automation_orchestration.md,docs/prs/active/pr_feat_auto-https-github-com-sorasuegami-eternal-cyc-20260304151714.md,docs/plans/active/plan_update_eternal_cycler_to_latest_upstream.md; verify_events=action.tooling; worker_type=default; update repository docs and PR tracking metadata as needed to match upstream-synced orchestration behavior.
- [x] (2026-03-04 18:10Z) action_id=a3; mode=serial; depends_on=a2; file_locks=docs/plans/active/plan_update_eternal_cycler_to_latest_upstream.md,docs/plans/completed/plan_update_eternal_cycler_to_latest_upstream.md; verify_events=action.tooling; worker_type=default; finalize outcomes and ledger, then move this plan to completed before post-completion validation.

## Verification Ledger

<!-- verification-ledger:start -->
- attempt_record: event_id=execplan.pre_creation; attempt=1; status=pass; started_at=2026-03-04 18:06Z; finished_at=2026-03-04 18:06Z; commands=git branch --show-current git status --short git log --oneline --decorate --max-count=20 gh pr status gh pr view --json number,title,body,state,headRefName,baseRefName,url mkdir -p docs/prs/active capture execplan start tracked snapshot capture execplan start untracked snapshot write/update docs/prs/active/pr_feat_auto-https-github-com-sorasuegami-eternal-cyc-20260304151714.md update plan linkage metadata; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=action.tooling; attempt=1; status=pass; started_at=2026-03-04 18:06Z; finished_at=2026-03-04 18:06Z; commands=bash -n scripts/*.sh .agents/skills/execplan-event-*/scripts/*.sh; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=action.tooling; attempt=2; status=pass; started_at=2026-03-04 18:08Z; finished_at=2026-03-04 18:08Z; commands=bash -n scripts/*.sh .agents/skills/execplan-event-*/scripts/*.sh; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=action.tooling; attempt=3; status=pass; started_at=2026-03-04 18:09Z; finished_at=2026-03-04 18:09Z; commands=bash -n scripts/*.sh .agents/skills/execplan-event-*/scripts/*.sh; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=execplan.post_completion; attempt=1; status=pass; started_at=2026-03-04 18:10Z; finished_at=2026-03-04 18:10Z; commands=rg -n docs/prs/active/|docs/prs/completed/ <plan> open docs/prs/active/pr_feat_auto-https-github-com-sorasuegami-eternal-cyc-20260304151714.md git status --short; failure_summary=none; notify_reference=not_requested;
<!-- verification-ledger:end -->

## Surprises & Discoveries

- Observation: `git subtree pull --prefix .agents/skills/eternal-cycler ... --squash` initially failed because the worktree was already modified by pre-creation lifecycle artifacts.
  Evidence: command output `fatal: working tree has modifications.  Cannot add.`
- Observation: after stashing a clean tree, `git subtree pull` still failed because historical installation commits did not preserve subtree metadata markers.
  Evidence: command output `fatal: can't squash-merge: '.agents/skills/eternal-cycler' was never added.`
- Observation: the latest upstream update in this cycle changed only `.agents/skills/eternal-cycler/scripts/run_builder_reviewer_loop.sh` in tracked content.
  Evidence: `git diff --stat` reported one subtree file changed plus lifecycle/doc updates.

## Decision Log

- Decision: Use the upstream README update flow exactly (`git subtree pull ... --squash` then `bash .agents/skills/eternal-cycler/setup.sh`) and preserve changes as local worktree/index edits for loop-owned commit/push.
  Rationale: The task explicitly requires README flow compliance and loop-controlled staging/commit/push.
  Date/Author: 2026-03-04 / Codex
- Decision: When subtree pull could not proceed due missing subtree metadata, apply a compatibility fallback: clean-stash current edits, temporarily remove subtree directory, run `git subtree add --prefix ... --squash` to latest upstream, run setup, then `git reset --mixed` back to the starting commit and restore the stash.
  Rationale: This keeps the final result as uncommitted worktree edits while still using upstream subtree mechanics and preserving unrelated baseline files.
  Date/Author: 2026-03-04 / Codex

## Outcomes & Retrospective

Completed so far:

- Lifecycle pre-creation and action tooling verification executed and recorded in-plan.
- Synced `.agents/skills/eternal-cycler/` to latest upstream content and re-ran upstream `setup.sh`.
- Updated `docs/architecture/scope/automation_orchestration.md` to reflect diverging builder payload contracts between local wrapper and upstream eternal-cycler script.
- Kept branch PR tracking metadata current in `docs/prs/active/pr_feat_auto-https-github-com-sorasuegami-eternal-cyc-20260304151714.md`.

Remaining:

- None.

## Context and Orientation

This repository uses a static upstream skill install under `.agents/skills/eternal-cycler/` and local runtime output under `eternal-cycler-out/`. The verification gate resolves event execution through `.agents/skills/execplan-event-index/references/event_skill_map.tsv`, with action-level checks executed via `.agents/skills/execplan-event-*/scripts/run_event.sh`. This change primarily touches automation/orchestration skill content and associated repository docs.

## Plan of Work

1. Initialize plan-bound pre-creation evidence and snapshots.
2. Pull latest upstream subtree into `.agents/skills/eternal-cycler/` and run setup.
3. Inspect deltas and update architecture/PR tracking docs only where behavior or interface mapping changed.
4. Run action verification gates after each action, finalize this plan, move it to completed, and run post-completion gate.

## Concrete Steps

Run from repository root (`.`):

    scripts/execplan_gate.sh --plan docs/plans/active/plan_update_eternal_cycler_to_latest_upstream.md --event execplan.pre_creation
    git subtree pull --prefix .agents/skills/eternal-cycler https://github.com/SoraSuegami/eternal-cycler.git main --squash
    # fallback when subtree metadata is absent:
    git stash push -u -m "tmp-builder-ec-update"
    git rm -r .agents/skills/eternal-cycler && git commit -m "tmp: remove eternal-cycler before subtree re-add"
    git subtree add --prefix .agents/skills/eternal-cycler https://github.com/SoraSuegami/eternal-cycler.git main --squash
    bash .agents/skills/eternal-cycler/setup.sh
    git reset --mixed <execplan_start_commit>
    git stash pop
    scripts/execplan_gate.sh --plan docs/plans/active/plan_update_eternal_cycler_to_latest_upstream.md --event action.tooling
    # update docs/architecture/scope/automation_orchestration.md and docs/prs/active/pr_feat_auto-https-github-com-sorasuegami-eternal-cyc-20260304151714.md when required by observed deltas
    scripts/execplan_gate.sh --plan docs/plans/active/plan_update_eternal_cycler_to_latest_upstream.md --event action.tooling
    mv docs/plans/active/plan_update_eternal_cycler_to_latest_upstream.md docs/plans/completed/plan_update_eternal_cycler_to_latest_upstream.md
    scripts/execplan_gate.sh --plan docs/plans/completed/plan_update_eternal_cycler_to_latest_upstream.md --event execplan.post_completion

## Validation and Acceptance

Acceptance requires all of the following:

1. `.agents/skills/eternal-cycler/` is updated to latest upstream `main` via subtree command flow (`pull` where possible, `add` fallback when subtree metadata is missing).
2. Upstream setup has been re-run successfully.
3. Repository docs and PR tracking metadata remain consistent with current orchestration behavior.
4. Plan ledger contains pass entries for `execplan.pre_creation`, required action event(s), and `execplan.post_completion`.

## Idempotence and Recovery

`setup.sh` can be rerun safely and skips overwriting existing verification skills where designed. If subtree pull fails because the tree is dirty or metadata is missing, use a temporary stash and subtree re-add fallback, then restore edits and reset back to the original commit so final results remain as worktree changes. If any verification gate fails, remediate and rerun within the three-attempt limit.

## Design/Architecture/Verification Document Summary

Design documents:

- Referenced: `docs/design/index.md`.
- Created/modified: none.

Architecture documents:

- Referenced: `docs/architecture/index.md`, `docs/architecture/scope/index.md`, `docs/architecture/scope/automation_orchestration.md`.
- Created/modified: `docs/architecture/scope/automation_orchestration.md` updated to document the upstream eternal-cycler builder payload extension (`pr_title`, `pr_body`) while keeping local wrapper contract notes accurate.

Verification skill/scripts:

- Referenced: `.agents/skills/execplan-event-index/SKILL.md`, `.agents/skills/execplan-event-index/references/event_skill_map.tsv`, `.agents/skills/execplan-event-pre-creation/SKILL.md`, `.agents/skills/execplan-event-action-tooling/SKILL.md`, `.agents/skills/execplan-event-post-completion/SKILL.md`, `scripts/execplan_gate.sh`, and `scripts/execplan_notify.sh`.
- Created/modified: none expected.

## Plan Revision Notes

- 2026-03-04 18:06Z: Initial plan created for upstream eternal-cycler update and setup refresh on current feature branch.
- 2026-03-04 18:08Z: Completed subtree synchronization via metadata-safe fallback and reran upstream setup.
- 2026-03-04 18:10Z: Updated architecture scope documentation for upstream builder payload extension and finalized active-plan action statuses.
- 2026-03-04 18:10Z: Moved plan to completed and recorded `execplan.post_completion` pass in the verification ledger.
- execplan_start_branch: feat/auto-https-github-com-sorasuegami-eternal-cyc-20260304151714
- execplan_start_commit: 01016769f79db3343d69eb859eb79924ca655bee

## ExecPlan Start Snapshot

<!-- execplan-start-tracked:start -->
- start_tracked_change: 4bf74a937b1ad6f4f618f7335f3331014e980337	docs/prs/active/pr_feat_auto-https-github-com-sorasuegami-eternal-cyc-20260304151714.md
<!-- execplan-start-tracked:end -->

<!-- execplan-start-untracked:start -->
- start_untracked_file: f0c32840a44560fbddb98e966d96a1a1be49e479	docs/plans/active/plan_update_eternal_cycler_to_latest_upstream.md
<!-- execplan-start-untracked:end -->

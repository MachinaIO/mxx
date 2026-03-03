# Align pr-autoloop Skill to Builder-Started Reviewer Daemon Architecture

This ExecPlan is a living document. The sections `Progress`, `Verification Ledger`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan follows `PLANS.md`.

ExecPlan start context:
- Branch at start: `feat/pr-autoloop-skill`
- Commit at start: `5b56dc8ea9ab4c17800eef290d220f05c75d6f77`
- Target PR: `https://github.com/MachinaIO/mxx/pull/63`

Repository-document context used for this plan: `PLANS.md`, `AGENTS.md`, `DESIGN.md`, `ARCHITECTURE.md`, `REVIEW.md`, `docs/design/index.md`, `docs/architecture/index.md`, `docs/architecture/scope/index.md`, `.agents/skills/execplan-event-index/SKILL.md`, `.agents/skills/execplan-event-index/references/event_skill_map.tsv`, `.agents/skills/execplan-sandbox-escalation/SKILL.md`, `.agents/skills/execplan-sandbox-escalation/references/allowed_command_prefixes.md`, `scripts/execplan_gate.sh`, and `scripts/execplan_notify.sh`.

## Purpose / Big Picture

After this change, the `pr-autoloop` skill surface and validation contracts are consistent with the latest architecture where the builder-side lifecycle starts and communicates with a long-running reviewer daemon. Any old loop-era interfaces and documents that conflict with this daemon-centric model are removed without compatibility shims.

## Progress

- [x] (2026-03-03 22:14Z) action_id=a0; mode=serial; depends_on=none; file_locks=docs/plans/active/plan_pr_autoloop_daemon_alignment_cleanup.md,docs/prs/active/pr_feat_pr-autoloop-skill.md; verify_events=execplan.pre_creation; worker_type=default; initialized this plan and bound it to PR tracking metadata via pre-creation gate.
- [x] (2026-03-03 22:23Z) action_id=a1; mode=serial; depends_on=a0; file_locks=.agents/skills/pr-autoloop/SKILL.md,.agents/skills/pr-autoloop/agents/openai.yaml,.agents/skills/pr-autoloop/references/comment_contract.md,.agents/skills/pr-autoloop/references/state_schema.md,.agents/skills/pr-autoloop/scripts/run_loop.sh,.agents/skills/pr-autoloop/scripts/reviewer_daemon.sh,.agents/skills/execplan-event-action-pr-autoloop/scripts/run_event.sh; verify_events=action.pr_autoloop,action.tooling; worker_type=default; removed `run_loop.sh`, converted contracts/verification to daemon-only semantics, and removed tracked runtime artifacts.
- [x] (2026-03-03 22:23Z) action_id=a2; mode=serial; depends_on=a1; file_locks=docs/design/pr_autoloop_builder_reviewer_contract.md,docs/architecture/scope/automation_orchestration.md,REVIEW.md,docs/plans/active/plan_pr_autoloop_daemon_alignment_cleanup.md; verify_events=action.tooling; worker_type=default; updated design/architecture/review policy text to daemon-first reviewer request/response terminology.
- [x] (2026-03-03 22:23Z) action_id=a3; mode=serial; depends_on=a2; file_locks=docs/plans/active/plan_pr_autoloop_daemon_alignment_cleanup.md; verify_events=action.pr_autoloop,action.tooling; worker_type=default; re-ran action-level gates on the full modified set and reached passing state for both events.
- [x] (2026-03-03 22:46Z) action_id=a4; mode=serial; depends_on=a3; file_locks=docs/plans/active/plan_pr_autoloop_daemon_alignment_cleanup.md,docs/plans/completed/plan_pr_autoloop_daemon_alignment_cleanup.md,docs/prs/active/pr_feat_pr-autoloop-skill.md,docs/prs/completed/pr_feat_pr-autoloop-skill.md,.agents/skills/execplan-event-post-completion/SKILL.md,.agents/skills/execplan-event-post-completion/scripts/run_event.sh,.agents/skills/execplan-event-action-pr-autoloop/scripts/run_event.sh,.agents/skills/pr-autoloop/runtime/.gitignore; verify_events=action.tooling; worker_type=default; completed remediation for reviewer findings (active/completed consistency, blocker idempotence, runtime artifact de-tracking) and prepared final post-completion verification.

## Verification Ledger

<!-- verification-ledger:start -->
- attempt_record: event_id=execplan.pre_creation; attempt=1; status=pass; started_at=2026-03-03 22:14Z; finished_at=2026-03-03 22:14Z; commands=git branch --show-current git status --short git log --oneline --decorate --max-count=20 gh pr status gh pr view --json number,title,body,state,headRefName,baseRefName,url mkdir -p docs/prs/active .agents/skills/pr-autoloop/scripts/reviewer_daemon.sh --start --runtime-dir .agents/skills/pr-autoloop/runtime --head-branch feat/pr-autoloop-skill --pr-url https://github.com/MachinaIO/mxx/pull/63 capture execplan start tracked snapshot capture execplan start untracked snapshot write/update docs/prs/active/pr_feat_pr-autoloop-skill.md update plan linkage metadata; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=action.tooling; attempt=1; status=pass; started_at=2026-03-03 22:23Z; finished_at=2026-03-03 22:23Z; commands=bash -n scripts/*.sh .agents/skills/execplan-event-*/scripts/*.sh; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=action.pr_autoloop; attempt=1; status=fail; started_at=2026-03-03 22:23Z; finished_at=2026-03-03 22:23Z; commands=bash -n .agents/skills/pr-autoloop/scripts/doctor.sh bash -n .agents/skills/pr-autoloop/scripts/reviewer_daemon.sh rg -n AUTO_AGENT: REVIEWER|AUTO_REQUEST_ID|AUTO_RUN_ID|AUTO_ITERATION|AUTO_REVIEW_STATUS|AUTO_TARGET_COMMIT|APPROVE .agents/skills/pr-autoloop/references/comment_contract.md rg -n -- --start|--request|--status|--stop|--commit|--pr-url|--head-branch|--request-id|--run-id|--iteration|--runtime-dir|--wait-timeout-sec .agents/skills/pr-autoloop/scripts/reviewer_daemon.sh rg -n reviewer.pid|state.json|inbox/<request_id>.json|responses/<request_id>.json|WAITING|RUNNING|APPROVED .agents/skills/pr-autoloop/references/state_schema.md rg -n CI|do not wait .agents/skills/pr-autoloop/references/comment_contract.md; failure_summary=comment contract missing non-blocking CI timing rule; notify_reference=not_requested;
- attempt_record: event_id=action.pr_autoloop; attempt=2; status=pass; started_at=2026-03-03 22:23Z; finished_at=2026-03-03 22:23Z; commands=bash -n .agents/skills/pr-autoloop/scripts/doctor.sh bash -n .agents/skills/pr-autoloop/scripts/reviewer_daemon.sh rg -n AUTO_AGENT: REVIEWER|AUTO_REQUEST_ID|AUTO_RUN_ID|AUTO_ITERATION|AUTO_REVIEW_STATUS|AUTO_TARGET_COMMIT|APPROVE .agents/skills/pr-autoloop/references/comment_contract.md rg -n -- --start|--request|--status|--stop|--commit|--pr-url|--head-branch|--request-id|--run-id|--iteration|--runtime-dir|--wait-timeout-sec .agents/skills/pr-autoloop/scripts/reviewer_daemon.sh rg -n reviewer.pid|state.json|inbox/<request_id>.json|responses/<request_id>.json|WAITING|RUNNING|APPROVED .agents/skills/pr-autoloop/references/state_schema.md rg -n CI|do not wait .agents/skills/pr-autoloop/references/comment_contract.md; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=action.tooling; attempt=2; status=pass; started_at=2026-03-03 22:23Z; finished_at=2026-03-03 22:23Z; commands=bash -n scripts/*.sh .agents/skills/execplan-event-*/scripts/*.sh; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=action.tooling; attempt=3; status=pass; started_at=2026-03-03 22:23Z; finished_at=2026-03-03 22:23Z; commands=bash -n scripts/*.sh .agents/skills/execplan-event-*/scripts/*.sh; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=action.pr_autoloop; attempt=3; status=pass; started_at=2026-03-03 22:23Z; finished_at=2026-03-03 22:23Z; commands=bash -n .agents/skills/pr-autoloop/scripts/doctor.sh bash -n .agents/skills/pr-autoloop/scripts/reviewer_daemon.sh rg -n AUTO_AGENT: REVIEWER|AUTO_REQUEST_ID|AUTO_RUN_ID|AUTO_ITERATION|AUTO_REVIEW_STATUS|AUTO_TARGET_COMMIT|APPROVE .agents/skills/pr-autoloop/references/comment_contract.md rg -n -- --start|--request|--status|--stop|--commit|--pr-url|--head-branch|--request-id|--run-id|--iteration|--runtime-dir|--wait-timeout-sec .agents/skills/pr-autoloop/scripts/reviewer_daemon.sh rg -n reviewer.pid|state.json|inbox/<request_id>.json|responses/<request_id>.json|WAITING|RUNNING|APPROVED .agents/skills/pr-autoloop/references/state_schema.md rg -n CI|do not wait .agents/skills/pr-autoloop/references/comment_contract.md; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=execplan.post_completion; attempt=1; status=fail; started_at=2026-03-03 22:24Z; finished_at=2026-03-03 22:24Z; commands=rg -n docs/prs/active/|docs/prs/completed/ <plan> open docs/prs/active/pr_feat_pr-autoloop-skill.md .agents/skills/pr-autoloop/scripts/reviewer_daemon.sh --request --pr-url https://github.com/MachinaIO/mxx/pull/63 --commit 5b56dc89d2a340d9e86131d0a6c6fd0bdbdbb6c7 --run-id execplan-post-completion --iteration 0 --request-id post-completion-20260303T222442Z-2851894 rollback plan docs/plans/completed/plan_pr_autoloop_daemon_alignment_cleanup.md -> docs/plans/active/plan_pr_autoloop_daemon_alignment_cleanup.md; failure_summary=failed while waiting for reviewer daemon response for commit 5b56dc89d2a340d9e86131d0a6c6fd0bdbdbb6c7; notify_reference=not_requested;
- attempt_record: event_id=execplan.post_completion; attempt=2; status=fail; started_at=2026-03-03 22:25Z; finished_at=2026-03-03 22:30Z; commands=skill event runner execplan.post_completion; failure_summary=event runner failed; notify_reference=not_requested;
- attempt_record: event_id=execplan.post_completion; attempt=3; status=escalated; started_at=2026-03-03 22:32Z; finished_at=2026-03-03 22:35Z; commands=skill event runner execplan.post_completion; failure_summary=event runner failed , retry bound exceeded (attempt=3); notify_reference=not_requested;
- attempt_record: event_id=execplan.post_completion; attempt=3; status=escalated; started_at=2026-03-03 22:38Z; finished_at=2026-03-03 22:38Z; commands=rg -n docs/prs/active/|docs/prs/completed/ <plan> open docs/prs/active/pr_feat_pr-autoloop-skill.md git status --short git add changed plan files; failure_summary=failed to stage target files: .agents/skills/pr-autoloop/runtime/reviewer-daemon/logs/daemon.log,.agents/skills/pr-autoloop/runtime/reviewer-daemon/state.json , retry bound exceeded (attempt=3); notify_reference=not_requested;
- attempt_record: event_id=execplan.post_completion; attempt=3; status=escalated; started_at=2026-03-03 22:38Z; finished_at=2026-03-03 22:42Z; commands=rg -n docs/prs/active/|docs/prs/completed/ <plan> open docs/prs/active/pr_feat_pr-autoloop-skill.md git status --short git add changed plan files git commit -m <finalize-message> git push origin feat/pr-autoloop-skill .agents/skills/pr-autoloop/scripts/reviewer_daemon.sh --request --pr-url https://github.com/MachinaIO/mxx/pull/63 --commit 7f66b5f26e61e8a455620b6db7397033cc4ed10c --run-id execplan-post-completion --iteration 0 --request-id post-completion-20260303T223900Z-2862415 gh pr view https://github.com/MachinaIO/mxx/pull/63 --json comments; failure_summary=reviewer did not approve pushed commit 7f66b5f26e61e8a455620b6db7397033cc4ed10c, comment_url=https://github.com/MachinaIO/mxx/pull/63#issuecomment-3994012167, comment_excerpt=Review findings (changes required): 1. High: Completed-plan state is internally inconsistent and blocks lifecycle closure. - `docs/plans/completed/plan_pr_autoloop_daemon_alignment_cleanup.md:24` keeps action `a4` unchecked. - `docs/plans/completed/plan_pr_autoloop_daemon_alignment_cleanup.md:36-39` records only `fail/escalated` for `execplan.post_completion` with no pass attempt. - The file is st , retry bound exceeded (attempt=3); notify_reference=not_requested;
- attempt_record: event_id=execplan.post_completion; attempt=3; status=escalated; started_at=2026-03-03 22:45Z; finished_at=2026-03-03 22:45Z; commands=rg -n docs/prs/active/|docs/prs/completed/ <plan> open docs/prs/active/pr_feat_pr-autoloop-skill.md upsert blockers sections git status --short git add changed plan files; failure_summary=failed to stage target files: .agents/skills/pr-autoloop/runtime/reviewer-daemon/logs/daemon.log,.agents/skills/pr-autoloop/runtime/reviewer-daemon/state.json , retry bound exceeded (attempt=3); notify_reference=not_requested;
<!-- verification-ledger:end -->

## Surprises & Discoveries

- Observation: Runtime reviewer-daemon artifacts under `.agents/skills/pr-autoloop/runtime/` are currently tracked by git.
  Evidence: `git ls-files .agents/skills/pr-autoloop/runtime` returned daemon logs, prompts, and response files.
- Observation: `action.pr_autoloop` validation failed once due strict string check for "do not wait" in reviewer timing contract.
  Evidence: gate failure summary: `comment contract missing non-blocking CI timing rule`.
- Observation: `execplan.post_completion` reviewed commit `5b56dc8...` (pre-finalization commit) because review request happened before add/commit/push in event ordering.
  Evidence: reviewer daemon request command in failed ledger entry referenced `--commit 5b56dc89d2a340d9e86131d0a6c6fd0bdbdbb6c7` before final staging/push.
- Observation: Reviewer returned `CHANGES_REQUIRED` for commit `7f66b5f...` because the plan stayed under `docs/plans/completed/` while unfinished and blocker sections were duplicated.
  Evidence: PR comment `https://github.com/MachinaIO/mxx/pull/63#issuecomment-3994012167`.

## Decision Log

- Decision: No compatibility layer will be kept for obsolete `run_loop.sh`-driven behavior if it conflicts with daemon-first architecture.
  Rationale: User explicitly requested destructive cleanup of old-spec files/wording when misaligned.
  Date/Author: 2026-03-04 / Codex
- Decision: Replace `action.docs_only` verification references in this plan with `action.tooling`.
  Rationale: This plan intentionally includes non-doc skill/script deletions, while `action.docs_only` validates the entire working tree and would be structurally inapplicable.
  Date/Author: 2026-03-04 / Codex
- Decision: Reorder `execplan.post_completion` event flow to add/commit/push first, then request reviewer daemon on the pushed HEAD commit.
  Rationale: Prevent stale commit review and satisfy deterministic contract that reviewer evaluates the exact pushed result of lifecycle finalization.
  Date/Author: 2026-03-04 / Codex
- Decision: Enforce runtime artifact de-tracking and idempotent blocker-section updates in lifecycle scripts.
  Rationale: Prevent repeated post-completion churn and avoid mutable runtime logs/state being versioned.
  Date/Author: 2026-03-04 / Codex

## Outcomes & Retrospective

Completed so far:

- Plan initialized and pre-creation verification passed.
- Daemon-only skill/script/doc cleanup completed and validated through action-level gates.

Remaining:

- None (waiting for lifecycle post-completion event execution result).

## Context and Orientation

The repository currently has mixed contracts: lifecycle event scripts are daemon-driven, but `pr-autoloop` skill docs and validation still enforce older loop-era interfaces centered on `run_loop.sh`. This mismatch creates policy ambiguity. The fix is to make one authoritative architecture: builder-side lifecycle operations with reviewer daemon messaging and strict machine-readable reviewer comments.

## Plan of Work

First, update `pr-autoloop` skill-facing files so they describe only daemon-aligned invocation and state contracts; remove obsolete `run_loop.sh` artifacts if they are no longer part of that architecture. Then adjust `action.pr_autoloop` event validation to verify only daemon-centric requirements. Next, revise design/architecture/review docs to eliminate outdated loop-era statements. Finally, run syntax and event gates, then complete lifecycle post-completion.

## Concrete Steps

Run from repository root (`.`):

    scripts/execplan_gate.sh --event execplan.pre_creation
    scripts/execplan_gate.sh --plan docs/plans/active/plan_pr_autoloop_daemon_alignment_cleanup.md --event execplan.pre_creation

    # Implement daemon-only alignment
    # Remove obsolete loop-era files/sections as needed

    scripts/execplan_gate.sh --plan docs/plans/active/plan_pr_autoloop_daemon_alignment_cleanup.md --event action.tooling
    scripts/execplan_gate.sh --plan docs/plans/active/plan_pr_autoloop_daemon_alignment_cleanup.md --event action.pr_autoloop
    mv docs/plans/active/plan_pr_autoloop_daemon_alignment_cleanup.md docs/plans/completed/plan_pr_autoloop_daemon_alignment_cleanup.md
    scripts/execplan_gate.sh --plan docs/plans/completed/plan_pr_autoloop_daemon_alignment_cleanup.md --event execplan.post_completion
    # If prior ledger reached escalated for post-completion during this same lifecycle, rerun with explicit bounded override after fixing root cause:
    scripts/execplan_gate.sh --plan docs/plans/completed/plan_pr_autoloop_daemon_alignment_cleanup.md --event execplan.post_completion --attempt 3

## Validation and Acceptance

Acceptance requires all of the following:

1. `pr-autoloop` skill no longer exposes conflicting loop-era interface/contracts.
2. If loop-era files are incompatible, they are deleted rather than preserved.
3. `action.pr_autoloop` verification script enforces daemon-only contracts.
4. Design/architecture/review docs are consistent with daemon-first behavior.
5. Required gates pass with recorded evidence in `Verification Ledger`.

## Idempotence and Recovery

If post-completion fails, the event script must roll plan/PR tracking docs back to active state and this plan resumes from the first unfinished action.
- execplan_start_branch: feat/pr-autoloop-skill
- execplan_start_commit: 5b56dc89d2a340d9e86131d0a6c6fd0bdbdbb6c7

## ExecPlan Start Snapshot

<!-- execplan-start-tracked:start -->
- start_tracked_change: 726fd67a33e5b242b2a9dc93a06413fe2cde7635	.agents/skills/pr-autoloop/runtime/reviewer-daemon/logs/daemon.log
- start_tracked_change: 6de3fa178e0cb258a97a1100204ce81f34cc2162	.agents/skills/pr-autoloop/runtime/reviewer-daemon/state.json
<!-- execplan-start-tracked:end -->

<!-- execplan-start-untracked:start -->
- start_untracked_file: 7e5bf7de84f0d1ce808f72d892ccc1518686bf79	.agents/skills/pr-autoloop/runtime/reviewer-daemon/reviewer.pid
- start_untracked_file: 407abee9eb43fc216ff4cc95507c75984d925ff2	docs/plans/active/plan_pr_autoloop_daemon_alignment_cleanup.md
- start_untracked_file: 147519f01292492cf742fbf0fead826a7a2c3870	docs/prs/active/pr_feat_pr-autoloop-skill.md
<!-- execplan-start-untracked:end -->

## Plan Revision Notes

- 2026-03-03 22:23Z: Converted verification targets to daemon-only contracts, removed `action.docs_only` from this plan because non-doc skill/script cleanup is intentionally in scope, and updated reviewer timing wording to satisfy strict gate validation.
- 2026-03-03 22:36Z: Added post-completion remediation because reviewer request happened before final push. Updated event script/skill to push first, then request reviewer for pushed commit; resumed from active plan `a4` after escalation.
- 2026-03-03 22:43Z: Applied reviewer-requested remediation: returned plan to active state, deduplicated blocker sections, removed tracked runtime daemon outputs, and added runtime de-tracking validation.
- 2026-03-03 22:46Z: Marked `a4` complete after remediation implementation and set plan to ready for final lifecycle `execplan.post_completion` execution.


## Post-Completion Blockers

- remaining blockers not provided

# Fix ExecPlan command path logging

This ExecPlan is a living document. The sections `Progress`, `Verification Ledger`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be updated as work proceeds.

This document follows `.agents/skills/eternal-cycler/PLANS.md` from the repository root.

## Purpose / Big Picture

This change removes absolute filesystem paths from ExecPlan documentation that is supposed to stay repository-relative. After the change, the completed plan `eternal-cycler-out/plans/completed/20260305T220916Z_fix_review_feedback_for_lookup_design_and_execplan_tooling.md` will record repo-relative `commands=` entries, and future lifecycle or tooling gate attempts will log repo-relative paths as well. A reviewer should be able to inspect the repaired completed plan, run the relevant gate scripts again, and see that ledger command text references only repository-relative paths.

## Progress

- [x] (2026-03-06 00:13Z) action_id=a1; mode=serial; depends_on=none; file_locks=.agents/skills/eternal-cycler/scripts/execplan_gate.sh,.agents/skills/execplan-event-post-creation/scripts/run_event.sh,.agents/skills/execplan-event-post-completion/scripts/run_event.sh,.agents/skills/execplan-event-resume/scripts/run_event.sh,.agents/skills/execplan-event-action-tooling/scripts/run_event.sh,eternal-cycler-out/plans/completed/20260305T220916Z_fix_review_feedback_for_lookup_design_and_execplan_tooling.md; verify_events=action.tooling; worker_type=default; normalized repo-relative command logging for ExecPlan verification, repaired the blocked completed plan ledger, and validated direct event output for the affected scripts.

## Verification Ledger

<!-- verification-ledger:start -->
- attempt_record: event_id=execplan.post_creation; attempt=1; status=pass; started_at=2026-03-06 00:07Z; finished_at=2026-03-06 00:07Z; commands=git branch --show-current git status --short gh pr status mkdir -p eternal-cycler-out/prs/active gh pr list --state open --head feat/auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020 --json url,title,state,headRefName,baseRefName,updatedAt --limit 20 write eternal-cycler-out/prs/active/pr_feat_auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020.md append PR Tracking Linkage to plan capture start tracked snapshot capture start untracked snapshot; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=action.tooling; attempt=1; status=pass; started_at=2026-03-06 00:11Z; finished_at=2026-03-06 00:11Z; commands=bash -n .agents/skills/eternal-cycler/scripts/*.sh .agents/skills/execplan-event-*/scripts/*.sh; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=execplan.post_completion; attempt=1; status=pass; started_at=2026-03-06 00:14Z; finished_at=2026-03-06 00:14Z; commands=rg -n eternal-cycler-out/prs/active/|eternal-cycler-out/prs/completed/ <plan> open eternal-cycler-out/prs/active/pr_feat_auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020.md gh pr view https://github.com/MachinaIO/mxx/pull/68 --json url,state git status --short; failure_summary=none; notify_reference=not_requested;
<!-- verification-ledger:end -->

## Surprises & Discoveries

- Observation: The original `action.tooling` script only ran correctly through `execplan_gate.sh` because its direct-invocation fallback resolved `ETERNAL_CYCLER_ROOT` to the repository root instead of `.agents/skills/eternal-cycler`.
  Evidence: A direct validation run initially tried `scripts/*.sh` and failed before the fallback path was corrected.
- Observation: Normalizing at the gate boundary was still necessary even after updating event-local `COMMANDS=` builders, because this ExecPlan's own `execplan.post_creation` attempt was recorded before the script fixes landed.
  Evidence: The new plan's first ledger entry initially contained an absolute repository-root prefix and required a manual rewrite to preserve an accurate but policy-compliant attempt record.

## Decision Log

- Decision: Start a new ExecPlan on the existing review branch instead of editing the prior completed plan without lifecycle tracking.
  Rationale: The reviewer feedback is a new objective on the same PR, so the fix itself must be tracked through a fresh ExecPlan lifecycle.
  Date/Author: 2026-03-06 / BUILDER agent.
- Decision: Normalize repo-root prefixes in `execplan_gate.sh` and also emit repo-relative command strings directly from the affected event runners.
  Rationale: The gate-level normalization protects all future ledger writes, while event-local normalization keeps direct `run_event.sh` validation output compliant and readable.
  Date/Author: 2026-03-06 / BUILDER agent.
- Decision: Fix the `action.tooling` direct-invocation fallback while touching its command logging.
  Rationale: The validation workflow in this ExecPlan exercises event scripts directly, so the fallback needed to resolve the actual eternal-cycler installation path instead of relying solely on gate-provided environment variables.
  Date/Author: 2026-03-06 / BUILDER agent.

## Outcomes & Retrospective

Completed the reviewer-requested compliance fix on the existing PR branch. The gate now strips repository-root prefixes from ledger-facing text before appending `commands=`, `failure_summary=`, or `notify_reference`, and the affected event runners now emit repo-relative `COMMANDS=` strings on their own. The blocked completed plan `eternal-cycler-out/plans/completed/20260305T220916Z_fix_review_feedback_for_lookup_design_and_execplan_tooling.md` was rewritten so its `execplan.post_creation`, `action.tooling`, and `execplan.post_completion` entries no longer contain absolute filesystem paths.

Verification status: `action.tooling` passed on attempt 1 for this ExecPlan. Additional direct validation confirmed that `execplan-event-action-tooling`, `execplan-event-post-creation`, and `execplan-event-post-completion` now emit repo-relative `COMMANDS=` output without absolute repository-root prefixes. A repository search for the prior absolute prefix in the repaired completed plan and this plan returned no matches.

Verification scripts modified:

- `.agents/skills/eternal-cycler/scripts/execplan_gate.sh` to normalize repository-root prefixes before writing ledger fields.
- `.agents/skills/execplan-event-post-creation/scripts/run_event.sh` to log tracking-document paths as repo-relative.
- `.agents/skills/execplan-event-resume/scripts/run_event.sh` to log tracking-document paths as repo-relative during resume flows.
- `.agents/skills/execplan-event-post-completion/scripts/run_event.sh` to log opened, fallback, and rollback paths as repo-relative.
- `.agents/skills/execplan-event-action-tooling/scripts/run_event.sh` to emit repo-relative syntax-check commands and to fix direct-invocation fallback resolution.

Verification scripts referenced and left unchanged:

- `.agents/skills/execplan-event-pre-creation/scripts/run_event.sh` was not part of the path-leak regression, so its existing behavior remained unchanged.
- `.agents/skills/execplan-event-index/references/event_skill_map.tsv` already mapped the required events correctly and needed no update.

## Context and Orientation

`eternal-cycler-out/plans/completed/20260305T220916Z_fix_review_feedback_for_lookup_design_and_execplan_tooling.md` is the latest completed ExecPlan on this branch and was the review target for this cycle. At the start of this work, its Verification Ledger contained absolute filesystem paths in `commands=` for `execplan.post_creation`, `action.tooling`, and `execplan.post_completion`. Repository policy in `.agents/skills/eternal-cycler/PLANS.md` requires documentation to use repository-relative paths only, so the completed plan needed a ledger repair even though the underlying tooling behavior was otherwise accepted.

The relevant automation lives in `.agents/skills/eternal-cycler/scripts/execplan_gate.sh` and the repository-local event runners under `.agents/skills/execplan-event-*/scripts/run_event.sh`. `execplan_gate.sh` captures `COMMANDS=` output from each event and writes it into the plan ledger. Some event scripts currently build those command strings with `${REPO_ROOT}` or other absolute path prefixes, which then leak into plan documentation. This task needs both a repair of the existing completed plan and a durable normalization path so future plans do not repeat the same policy violation.

## Plan of Work

Inspect how the gate and affected event scripts construct `COMMANDS=` text, then update that logging so any repository path written into documentation is normalized to a repository-relative path before the ledger entry is appended. Repair the reviewer-blocked completed plan by rewriting the affected ledger entries to the same repo-relative form. After the code change, run the tooling verification gate and targeted script checks that demonstrate the new logging stays repository-relative.

## Concrete Steps

From the repository root:

1. Update `.agents/skills/eternal-cycler/scripts/execplan_gate.sh` and any affected event runners so ledger-facing command text is normalized to repository-relative paths before it is written into an ExecPlan.
2. Rewrite the absolute-path `commands=` values in `eternal-cycler-out/plans/completed/20260305T220916Z_fix_review_feedback_for_lookup_design_and_execplan_tooling.md` so the completed plan complies with documentation policy.
3. Run `.agents/skills/eternal-cycler/scripts/execplan_gate.sh --plan eternal-cycler-out/plans/active/20260306T000653Z_fix_execplan_command_path_logging.md --event action.tooling` after the edits.
4. Run targeted local validation that exercises the relevant event scripts and confirms their emitted command text is repository-relative.

## Validation and Acceptance

Acceptance requires three observable results. First, the completed plan `eternal-cycler-out/plans/completed/20260305T220916Z_fix_review_feedback_for_lookup_design_and_execplan_tooling.md` must no longer contain absolute filesystem paths in its Verification Ledger. Second, the affected gate and event scripts must emit repo-relative paths in their `COMMANDS=` output when run from this repository. Third, `action.tooling` must pass for the modified shell scripts.

## Idempotence and Recovery

The ledger rewrite is idempotent because the completed plan should converge to one repo-relative representation. Re-running the command-normalization changes should keep emitting the same repo-relative text. If validation fails, inspect the script emitting the remaining absolute path, tighten the normalization logic, and retry the failed gate attempt until the retry bound is reached.

## Artifacts and Notes

All verification evidence will be recorded directly in this plan's `Verification Ledger`. No external scratch documentation is needed.

## Interfaces and Dependencies

This task changes repository-local shell tooling under `.agents/skills/` and one completed ExecPlan under `eternal-cycler-out/plans/completed/`. It does not change Rust or CUDA runtime behavior. GitHub CLI access may still be used by lifecycle events, but the documented command text must remain repository-relative regardless of whether the live command resolved an absolute working-tree path internally.

Revision note (2026-03-06, BUILDER): Created this ExecPlan after `execplan.pre_creation` passed on branch `feat/auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020`.
Revision note (2026-03-06, BUILDER): Completed the repo-relative command-logging fix, repaired the blocked completed plan ledger, and validated the affected event scripts before finalization.

## PR Tracking Linkage

- pr_tracking_doc: eternal-cycler-out/prs/active/pr_feat_auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020.md

- execplan_start_branch: feat/auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020

- execplan_start_commit: 564683f6fe75b9b35acde98370c87eddb0ac0dee

## ExecPlan Start Snapshot

<!-- execplan-start-tracked:start -->
- start_tracked_change: 4bb797135f9259a030a596ff9e47925acd1b5b5d	eternal-cycler-out/prs/active/pr_feat_auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020.md
<!-- execplan-start-tracked:end -->

<!-- execplan-start-untracked:start -->
- start_untracked_file: b85d431792d10a9472b9032713394f4f9ffd49e4	eternal-cycler-out/plans/active/20260306T000653Z_fix_execplan_command_path_logging.md
<!-- execplan-start-untracked:end -->

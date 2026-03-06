# Fix reviewer feedback for lookup design and ExecPlan tooling

This ExecPlan is a living document. The sections `Progress`, `Verification Ledger`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be updated as work proceeds.

This document follows `.agents/skills/eternal-cycler/PLANS.md` from the repository root.

## Purpose / Big Picture

This change fixes four review blockers on branch `feat/auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020`. After the change, `docs/design/ggh15_arbitrary_polynomial_lookup.md` will match the current `PublicLut<P>` contract and its actual GGH15 consumers, the same design document will stop overstating the `usize`-bounded selector example, `execplan.post_creation` and `execplan.resume` will track the correct branch PR instead of stale merged PR metadata, and `execplan.pre_creation` will reject invalid start states instead of always passing. A reader should be able to inspect the updated design doc, run the ExecPlan gates on a feature branch, and observe that invalid environments fail while PR linkage stays aligned with the live branch PR.

## Progress

- [x] (2026-03-05 22:22Z) action_id=a1; mode=serial; depends_on=none; file_locks=docs/design/ggh15_arbitrary_polynomial_lookup.md,.agents/skills/execplan-event-pre-creation/scripts/run_event.sh,.agents/skills/execplan-event-post-creation/scripts/run_event.sh,.agents/skills/execplan-event-resume/scripts/run_event.sh,.agents/skills/execplan-event-post-completion/scripts/run_event.sh,.agents/skills/eternal-cycler/scripts/run_builder_reviewer_loop.sh,eternal-cycler-out/prs/active/pr_feat_auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020.md; verify_events=action.tooling; worker_type=default; corrected the GGH15 design document, enforced pre-creation environment validation, and made ExecPlan PR tracking follow the live branch PR without stale linkage.

## Verification Ledger

<!-- verification-ledger:start -->
- attempt_record: event_id=execplan.post_creation; attempt=1; status=pass; started_at=2026-03-05 22:10Z; finished_at=2026-03-05 22:10Z; commands=git branch --show-current git status --short gh pr status gh pr view --json number,title,body,state,headRefName,baseRefName,url mkdir -p eternal-cycler-out/prs/active write eternal-cycler-out/prs/active/pr_feat_auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020.md capture start tracked snapshot capture start untracked snapshot; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=action.tooling; attempt=1; status=pass; started_at=2026-03-05 22:20Z; finished_at=2026-03-05 22:20Z; commands=bash -n .agents/skills/eternal-cycler/scripts/*.sh .agents/skills/execplan-event-*/scripts/*.sh; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=execplan.post_completion; attempt=1; status=pass; started_at=2026-03-05 22:23Z; finished_at=2026-03-05 22:23Z; commands=rg -n eternal-cycler-out/prs/active/|eternal-cycler-out/prs/completed/ <plan> open eternal-cycler-out/prs/active/pr_feat_auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020.md gh pr view https://github.com/MachinaIO/mxx/pull/68 --json url,state git status --short; failure_summary=none; notify_reference=not_requested;
<!-- verification-ledger:end -->

## Surprises & Discoveries

- Observation: The current branch already has an open PR (`#68`), so running `execplan.post_creation` now can overwrite the stale active PR tracking doc without any manual recovery logic.
  Evidence: `gh pr view --json url,state,headRefName,baseRefName,title,number` on branch `feat/auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020` returns state `OPEN` and URL `https://github.com/MachinaIO/mxx/pull/68`.
- Observation: `execplan.post_creation` suppresses `## PR Tracking Linkage` if the tracking-doc path appears anywhere in the plan, which let the current plan miss `pr_tracking_doc:` because the same path was listed in `file_locks`.
  Evidence: the first `execplan.post_creation` run appended `execplan_start_branch` and `execplan_start_commit` but no `pr_tracking_doc:` field, because the action metadata already mentioned `eternal-cycler-out/prs/active/pr_feat_auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020.md`.
- Observation: The new `execplan.pre_creation` logic behaves as intended in clean and invalid-start clones when executed from the modified script in this working tree.
  Evidence: a clean feature-branch clone returned `STATUS=pass`, while clone variants with a tracked edit and with branch `main` returned `STATUS=fail` with the expected failure summaries.

## Decision Log

- Decision: Split this work into one docs-only action and one tooling action.
  Rationale: That was the initial decomposition before the tooling fix grew to include `execplan.post_completion` and the `pr_tracking_doc:` linkage repair discovered during plan creation.
  Date/Author: 2026-03-05 / BUILDER agent.
- Decision: Keep the current `PublicLut<P>` output contract in the design document rather than proposing a silent narrowing to `P::Elem`.
  Rationale: `src/lookup/mod.rs` and `src/lookup/ggh15_eval.rs` already return and consume full output polynomials, so the design must preserve that baseline unless it also specifies a migration path.
  Date/Author: 2026-03-05 / BUILDER agent.
- Decision: Collapse the implementation into one tooling-verified action that covers both the design-doc corrections and the ExecPlan lifecycle fixes.
  Rationale: once the active plan itself exposed a `post_creation` linkage bug and `post_completion` needed a matching update, the work stopped being cleanly separable into a docs-only checkpoint followed by tooling edits.
  Date/Author: 2026-03-05 / BUILDER agent.

## Outcomes & Retrospective

Completed the reviewer-feedback fixes in one tooling-verified action. The design document now preserves the existing polynomial-output LUT contract and scopes the selector example to `usize`-representable domains. `execplan.pre_creation` now rejects `main`, tracked dirt, and unmerged paths. `execplan.post_creation` and `execplan.resume` now prefer the current branch's open PR instead of a stale associated PR, `execplan.post_creation` always records `pr_tracking_doc:` explicitly, `run_builder_reviewer_loop.sh` refreshes the active PR tracking doc after resolving or creating the live PR, and `execplan.post_completion` now distinguishes "no PR yet" from "stale PR link" by checking for a current open PR on the tracked branch.

Verification status: `execplan.post_creation` passed on attempt 1 and `action.tooling` passed on attempt 1. Additional manual validation used temporary clones to confirm `execplan.pre_creation` passes on a clean feature branch and fails on tracked dirt and on `main`.

Verification scripts referenced and modified:

- `.agents/skills/execplan-event-pre-creation/scripts/run_event.sh` to restore the promised environment gate.
- `.agents/skills/execplan-event-post-creation/scripts/run_event.sh` to resolve PR metadata by branch and always write `pr_tracking_doc:`.
- `.agents/skills/execplan-event-resume/scripts/run_event.sh` to refresh PR tracking by live branch PR while preserving creation metadata.
- `.agents/skills/execplan-event-post-completion/scripts/run_event.sh` to treat missing PR links as pending only when the tracked branch has no open PR, and to reject stale open-PR mismatches.
- `.agents/skills/eternal-cycler/scripts/run_builder_reviewer_loop.sh` to sync the active PR tracking doc after PR resolution or creation.

Verification scripts referenced and left unchanged:

- `.agents/skills/eternal-cycler/scripts/execplan_gate.sh` remained the lifecycle orchestrator; no policy change was needed there.
- `.agents/skills/execplan-event-action-tooling/scripts/run_event.sh` remained the correct syntax-only verification hook for this mixed docs/tooling change.

## Context and Orientation

`docs/design/ggh15_arbitrary_polynomial_lookup.md` is a long-lived design artifact created in the previous builder cycle. The review found two mismatches between that document and the code under `src/lookup/`: first, `PublicLut<P>` still returns a full polynomial output `P`, and GGH15 lookup evaluation already consumes polynomial outputs; second, the document presents `to_const_int()` plus `from_usize_to_lsb()` as a general example-domain invariant even though both helpers are bounded by `usize` width.

The remaining findings concern the ExecPlan lifecycle under `.agents/skills/`. `execplan.post_creation` and `execplan.resume` both currently call bare `gh pr view`, which can bind a plan to stale PR metadata if the branch has no open PR at post-creation time and the loop later creates a new one. `run_builder_reviewer_loop.sh` resolves or creates the live PR only after the builder returns. `execplan.post_completion` currently assumes the PR link is always already resolvable, so it cannot distinguish "no PR exists yet" from "tracking doc is stale." `execplan.pre_creation` is supposed to validate the starting environment before a plan is created, but its current script only records state and always returns `STATUS=pass`.

## Plan of Work

Correct `docs/design/ggh15_arbitrary_polynomial_lookup.md` so that row outputs remain full polynomials and the example-selector discussion makes its `usize` bound explicit. Then update the lifecycle scripts so branch PR discovery is keyed to the current branch and prefers an open PR, `execplan.post_creation` always records `pr_tracking_doc:` explicitly, `execplan.post_completion` tolerates a pending PR link only when no open PR exists yet, the loop refreshes the active PR tracking document after resolving or creating the actual PR, and the pre-creation event fails on invalid start states such as `main` or tracked worktree changes. After the combined action passes `action.tooling`, update the living sections, move the plan to `eternal-cycler-out/plans/completed/`, run `execplan.post_completion`, and leave the code edits in the worktree.

## Concrete Steps

From the repository root:

1. Update `docs/design/ggh15_arbitrary_polynomial_lookup.md` so the data model and algebra descriptions preserve `PublicLut<P>` polynomial outputs, and rewrite the selector example to state the `usize`-width precondition plus the need for a wider selector API when that precondition fails.
2. Update `.agents/skills/execplan-event-pre-creation/scripts/run_event.sh`, `.agents/skills/execplan-event-post-creation/scripts/run_event.sh`, `.agents/skills/execplan-event-resume/scripts/run_event.sh`, `.agents/skills/execplan-event-post-completion/scripts/run_event.sh`, and `.agents/skills/eternal-cycler/scripts/run_builder_reviewer_loop.sh` so the environment gate blocks invalid states and PR tracking follows the live branch PR throughout the loop.
3. Refresh this plan's `## PR Tracking Linkage` block and confirm `eternal-cycler-out/prs/active/pr_feat_auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020.md` points at the current branch PR metadata.
4. Run `.agents/skills/eternal-cycler/scripts/execplan_gate.sh --plan eternal-cycler-out/plans/active/20260305T220916Z_fix_review_feedback_for_lookup_design_and_execplan_tooling.md --event action.tooling`.

## Validation and Acceptance

Acceptance requires four observable results. The design document must state that lookup outputs remain `P` values and must no longer claim the `to_const_int()` / `from_usize_to_lsb()` example works beyond `usize`-representable domains. The pre-creation gate must fail on `main` and on tracked worktree dirt while still passing on a clean feature-branch clone. The active PR tracking document for this branch must point at the live branch PR instead of merged PR `#65`, and the plan itself must record `pr_tracking_doc:` explicitly. The tooling verification gate must pass with `bash -n` over the modified scripts.

## Idempotence and Recovery

The documentation and shell-script edits are idempotent. Re-running the docs action should only refine wording in the same design document. Re-running the tooling action should rewrite the same lifecycle scripts and PR tracking doc deterministically from the current branch state. If `action.tooling` fails, inspect the changed shell scripts and fix syntax or branch-PR discovery logic before retrying with the next gate attempt.

## Artifacts and Notes

All gate attempts will be recorded directly in `## Verification Ledger`. No external scratch files are required.

## Interfaces and Dependencies

This task updates one long-lived design document under `docs/design/`, four repository-local ExecPlan event scripts under `.agents/skills/execplan-event-*/scripts/`, the main loop orchestrator `.agents/skills/eternal-cycler/scripts/run_builder_reviewer_loop.sh`, and the active PR tracking document under `eternal-cycler-out/prs/active/`. The relevant runtime code context is `src/lookup/mod.rs` and `src/lookup/ggh15_eval.rs`; no Rust implementation behavior changes are expected in this task.

Revision note (2026-03-05, BUILDER): Created the ExecPlan for the reviewer-feedback fixes after `execplan.pre_creation` passed on the clean feature branch.
Revision note (2026-03-05, BUILDER): Collapsed the work into a single tooling-verified action after discovering that the active plan itself exposed a `post_creation` linkage bug.
Revision note (2026-03-05, BUILDER): Completed the design/tooling corrections, validated the new `execplan.pre_creation` behavior in temporary clones, and updated the lifecycle notes before finalizing the plan.

## PR Tracking Linkage

- pr_tracking_doc: eternal-cycler-out/prs/active/pr_feat_auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020.md
- execplan_start_branch: feat/auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020
- execplan_start_commit: 910515b9a0f97fd46168a31ccb4285c629ca5786

## ExecPlan Start Snapshot

<!-- execplan-start-tracked:start -->
- start_tracked_change: 457c73a78279b2bdd4bd04bbe48a23d939e39914	eternal-cycler-out/prs/active/pr_feat_auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020.md
<!-- execplan-start-tracked:end -->

<!-- execplan-start-untracked:start -->
- start_untracked_file: dc4d7851a924cd2d4507f01c5bf57be2aed8a578	eternal-cycler-out/plans/active/20260305T220916Z_fix_review_feedback_for_lookup_design_and_execplan_tooling.md
<!-- execplan-start-untracked:end -->

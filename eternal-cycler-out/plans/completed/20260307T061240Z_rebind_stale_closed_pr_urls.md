# Rebind Stale Closed PR URLs To The Live Branch PR

This ExecPlan is a living document. The sections `Progress`, `Verification Ledger`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document must be maintained in accordance with `.agents/skills/eternal-cycler/PLANS.md`.

## Purpose / Big Picture

The autonomous builder/reviewer loop currently accepts a `--pr-url` value even when that pull request is already closed. When that happens, the reviewer keeps posting feedback to a stale PR even if the branch already has a newer open successor PR. After this change, the loop must treat a closed PR URL as stale, rebind itself to the branch's current open PR when one exists, and create a replacement PR when none exists. A human can observe the fix by running the loop with a closed PR URL for the current branch and seeing the reviewer target switch to the live open PR before review comments are posted.

## Progress

- [x] (2026-03-07 06:15Z) action_id=a1; mode=serial; depends_on=none; file_locks=.agents/skills/eternal-cycler/scripts/run_builder_reviewer_loop.sh; verify_events=action.tooling; worker_type=default; patched the loop startup path so it reads the supplied PR state, rejects stale `CLOSED` or `MERGED` `--pr-url` values, and falls back to the existing branch-based open-PR resolver before reviewer iteration 1.
- [x] (2026-03-07 06:18Z) action_id=a2; mode=serial; depends_on=a1; file_locks=eternal-cycler-out/plans/active/20260307T061240Z_rebind_stale_closed_pr_urls.md; verify_events=action.tooling; worker_type=default; ran focused stale-PR smoke checks, recorded the `#68` closed to `#69` open evidence, and finalized this plan for handoff into `execplan.post_completion`.

## Surprises & Discoveries

- Observation: The branch already has no active ExecPlan, but the live PR tracking document under `eternal-cycler-out/prs/active/` points to open PR `#69`.
  Evidence: `ls eternal-cycler-out/plans/active` returned no plan files, and `eternal-cycler-out/prs/active/pr_feat_auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020.md` records `PR link: https://github.com/MachinaIO/mxx/pull/69`.
- Observation: GitHub still reports PR `#68` as `CLOSED` on the same branch while PR `#69` is `OPEN`, and the sourced loop helper path now resolves that stale input back to PR `#69`.
  Evidence: `gh pr view` returned state `CLOSED` for `https://github.com/MachinaIO/mxx/pull/68` and state `OPEN` for `https://github.com/MachinaIO/mxx/pull/69`; the focused shell smoke test printed `https://github.com/MachinaIO/mxx/pull/69`.

## Decision Log

- Decision: Fix the stale-target behavior inside `.agents/skills/eternal-cycler/scripts/run_builder_reviewer_loop.sh` rather than relying on operators to omit `--pr-url` for closed PRs.
  Rationale: The reviewer feedback shows that stale PR URLs can still enter the loop. Repairing the loop makes the autonomous path self-healing and keeps the reviewer target aligned with the branch head.
  Date/Author: 2026-03-07 / Codex
- Decision: Reuse the loop's existing empty-`PR_URL` recovery path instead of adding a second branch-to-PR resolution path.
  Rationale: `resolve_or_create_pr_for_branch` already finds the latest open PR for the branch or creates one when needed. Clearing stale closed URLs is enough to rebind the loop to the live PR without duplicating logic.
  Date/Author: 2026-03-07 / Codex

## Outcomes & Retrospective

Completed the stale-review-target repair on branch `feat/auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020`. `.agents/skills/eternal-cycler/scripts/run_builder_reviewer_loop.sh` now requests `state` when a `--pr-url` is supplied, preserves the existing branch-match guard, and clears stale `CLOSED` or `MERGED` PR URLs so the already-existing `resolve_or_create_pr_for_branch` path rebinds the loop to the branch's live open PR before the first reviewer cycle.

Focused verification matched the reported regression exactly. `gh pr view` confirmed that PR `#68` is `CLOSED` while PR `#69` is `OPEN` on the same branch, and the sourced loop-resolution smoke test printed `https://github.com/MachinaIO/mxx/pull/69` when seeded with the stale `#68` URL. `action.tooling` passed on all three allowed attempts while the plan and final evidence were being updated.

Verification assets were referenced but not modified. The existing lifecycle gates under `.agents/skills/eternal-cycler/scripts/execplan_gate.sh` and the existing tooling verifier under `.agents/skills/execplan-event-action-tooling/scripts/run_event.sh` remained unchanged because the bug lived only in loop startup PR selection, and syntax coverage plus the GitHub-backed smoke test were sufficient to prove the fix.

## Context and Orientation

`.agents/skills/eternal-cycler/scripts/run_builder_reviewer_loop.sh` is the shell entrypoint that runs the builder/reviewer loop. It accepts an optional `--pr-url` and records the resolved target in the `PR_URL` shell variable. The current script validates that the supplied PR belongs to the current branch, but it does not reject or replace a closed PR. Later review iterations use that same `PR_URL` both in the reviewer prompt and when posting GitHub comments, so a stale closed PR can remain the review target for the entire run.

The branch tracking metadata lives in `eternal-cycler-out/prs/active/pr_<branch>.md`. That document already records the branch's live open PR. The reviewer feedback for this task explicitly states that PR `#68` is closed while the requested head commit now belongs to successor PR `#69`, so the loop must prefer the open successor before review begins.

## Plan of Work

First, inspect the current `--pr-url` handling in `.agents/skills/eternal-cycler/scripts/run_builder_reviewer_loop.sh` and extend the existing `gh pr view` call so it also reads the PR `state`. If the provided PR is still open, keep the existing behavior. If the provided PR is merged or closed, log that it is stale and clear the target so the loop reuses its existing branch-based PR resolver before reviewer iteration 1. If no open PR exists, let the existing PR creation path create one from the builder-returned title and body.

Second, add a focused smoke check that proves the rebinding logic works without needing to run a full one-hour loop. The smoke check can source only the loop's function definitions, execute the exact stale-URL branch-resolution logic against the known closed URL `https://github.com/MachinaIO/mxx/pull/68`, and confirm that it canonicalizes to `https://github.com/MachinaIO/mxx/pull/69` on branch `feat/auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020`. Run the existing tooling gate as the required action verification and record the manual smoke-test evidence in this plan.

## Concrete Steps

1. Run `.agents/skills/eternal-cycler/scripts/execplan_gate.sh --plan eternal-cycler-out/plans/active/20260307T061240Z_rebind_stale_closed_pr_urls.md --event execplan.post_creation` immediately after writing this plan and record the exact result in the `Verification Ledger`.
2. Edit `.agents/skills/eternal-cycler/scripts/run_builder_reviewer_loop.sh` so a supplied `--pr-url` is canonicalized to an open PR for the current branch before the reviewer prompt is constructed.
3. Run a focused shell smoke test that proves the closed PR URL `#68` is rebound to open PR `#69` for the current branch.
4. Run `.agents/skills/eternal-cycler/scripts/execplan_gate.sh --plan eternal-cycler-out/plans/active/20260307T061240Z_rebind_stale_closed_pr_urls.md --event action.tooling` after the implementation work and again after final plan updates, then record both attempts in the `Verification Ledger`.
5. Mark both actions complete, summarize the implementation in `Outcomes & Retrospective`, move the plan to `eternal-cycler-out/plans/completed/`, and run `.agents/skills/eternal-cycler/scripts/execplan_gate.sh --plan eternal-cycler-out/plans/completed/20260307T061240Z_rebind_stale_closed_pr_urls.md --event execplan.post_completion`.

## Acceptance Checks

Run these commands from the repository root and confirm the listed outcomes:

    .agents/skills/eternal-cycler/scripts/execplan_gate.sh --plan eternal-cycler-out/plans/active/20260307T061240Z_rebind_stale_closed_pr_urls.md --event action.tooling
    Outcome: `STATUS=pass`.

    bash -lc 'set -euo pipefail
    source <(sed -n "1,683p" .agents/skills/eternal-cycler/scripts/run_builder_reviewer_loop.sh)
    TARGET_BRANCH="feat/auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020"
    PR_URL="https://github.com/MachinaIO/mxx/pull/68"
    pr_info_json="$(gh pr view "$PR_URL" --json url,headRefName,number,state)"
    pr_head_branch="$(jq -r ".headRefName // \"\"" <<< "$pr_info_json")"
    pr_state="$(jq -r ".state // \"\"" <<< "$pr_info_json")"
    PR_URL="$(jq -r ".url // \"\"" <<< "$pr_info_json")"
    if [[ "$pr_head_branch" != "$TARGET_BRANCH" ]]; then
      exit 1
    fi
    if [[ "$pr_state" != "OPEN" ]]; then
      PR_URL=""
    fi
    if [[ -z "$PR_URL" ]]; then
      PR_URL="$(resolve_or_create_pr_for_branch "$TARGET_BRANCH" "smoke" "smoke")"
    fi
    printf "%s\n" "$PR_URL"'
    Outcome: prints `https://github.com/MachinaIO/mxx/pull/69`.

    gh pr view https://github.com/MachinaIO/mxx/pull/69 --json url,state,headRefName
    Outcome: returns state `OPEN` with head branch `feat/auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020`.

## Verification Ledger

<!-- verification-ledger:start -->
- attempt_record: event_id=execplan.post_creation; attempt=1; status=pass; started_at=2026-03-07 06:13Z; finished_at=2026-03-07 06:13Z; commands=git branch --show-current git status --short gh pr status mkdir -p eternal-cycler-out/prs/active gh pr list --state open --head feat/auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020 --json url,updatedAt --limit 20 gh pr view https://github.com/MachinaIO/mxx/pull/69 --json url,title,state,headRefName,baseRefName,createdAt,headRefOid,commits write eternal-cycler-out/prs/active/pr_feat_auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020.md append PR Tracking Linkage to plan capture start tracked snapshot capture start untracked snapshot; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=action.tooling; attempt=1; status=pass; started_at=2026-03-07 06:15Z; finished_at=2026-03-07 06:15Z; commands=bash -n .agents/skills/eternal-cycler/scripts/*.sh .agents/skills/execplan-event-*/scripts/*.sh; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=action.tooling; attempt=2; status=pass; started_at=2026-03-07 06:17Z; finished_at=2026-03-07 06:17Z; commands=bash -n .agents/skills/eternal-cycler/scripts/*.sh .agents/skills/execplan-event-*/scripts/*.sh; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=action.tooling; attempt=3; status=pass; started_at=2026-03-07 06:18Z; finished_at=2026-03-07 06:18Z; commands=bash -n .agents/skills/eternal-cycler/scripts/*.sh .agents/skills/execplan-event-*/scripts/*.sh; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=execplan.post_completion; attempt=1; status=pass; started_at=2026-03-07 06:19Z; finished_at=2026-03-07 06:19Z; commands=rg -n eternal-cycler-out/prs/active/|eternal-cycler-out/prs/completed/ <plan> open eternal-cycler-out/prs/active/pr_feat_auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020.md gh pr view https://github.com/MachinaIO/mxx/pull/69 --json url,state git status --short; failure_summary=none; notify_reference=not_requested;
<!-- verification-ledger:end -->

## PR Tracking Linkage

- pr_tracking_doc: eternal-cycler-out/prs/active/pr_feat_auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020.md

- execplan_start_branch: feat/auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020

- execplan_start_commit: 52bd2d1d46818fbf4e68779a505e72bc0f897b4e

## ExecPlan Start Snapshot

<!-- execplan-start-tracked:start -->
- start_tracked_change: da58adbe49e758e11de1697d61e8ec1f4de9651a	eternal-cycler-out/prs/active/pr_feat_auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020.md
<!-- execplan-start-tracked:end -->

<!-- execplan-start-untracked:start -->
- start_untracked_file: 37e9a84ccf7342eeff2a2b85e83355cb881f1dd7	eternal-cycler-out/plans/active/20260307T061240Z_rebind_stale_closed_pr_urls.md
<!-- execplan-start-untracked:end -->

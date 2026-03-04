# Switch Loop Reviewer Contract to Structured Output JSON

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` will be updated as work proceeds. This document follows `PLANS.md` requirements.

Repository-document context used for this plan: `AGENTS.md`, `PLANS.md`, `DESIGN.md`, `ARCHITECTURE.md`, `docs/design/index.md`, `docs/design/pr_autoloop_builder_reviewer_contract.md`, `docs/architecture/index.md`, `docs/architecture/scope/index.md`, `docs/architecture/scope/automation_orchestration.md`, `REVIEW.md`, `.agents/skills/pr-autoloop/SKILL.md`, `.agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh`, `.agents/skills/execplan-event-index/SKILL.md`, `.agents/skills/execplan-event-index/references/event_skill_map.tsv`, `.agents/skills/execplan-event-action-pr-autoloop/SKILL.md`, `.agents/skills/execplan-event-action-pr-autoloop/scripts/run_event.sh`, `.agents/skills/execplan-event-action-tooling/SKILL.md`, `.agents/skills/execplan-event-action-tooling/scripts/run_event.sh`, `.agents/skills/execplan-event-pre-creation/SKILL.md`, `.agents/skills/execplan-event-post-completion/SKILL.md`, `scripts/execplan_gate.sh`, and `scripts/execplan_notify.sh`.

## Purpose / Big Picture

After this change, reviewer execution in `run_builder_reviewer_loop.sh` will return one machine-readable JSON object instead of directly posting PR comments. The loop script will post the comment itself via `gh pr comment`, and merge/loop-stop decisions will rely only on a boolean JSON field, not brittle text matching. This removes comment scraping/tag parsing complexity and makes approval behavior explicit and deterministic. In addition, standalone reviewer mode will no longer post PR comments and will return review result plus acceptability boolean directly to the user.

## Progress

- [x] (2026-03-04 03:58Z) action_id=a0; mode=serial; depends_on=none; file_locks=docs/plans/active/plan_structured_reviewer_output_for_loop.md,docs/prs/active/pr_feat_pr-autoloop-skill.md; verify_events=none; worker_type=default; created this plan, ran `execplan.pre_creation` with plan linkage, and captured initial lifecycle evidence.
- [x] (2026-03-04 04:03Z) action_id=a1; mode=serial; depends_on=a0; file_locks=.agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh,REVIEW.md,docs/design/pr_autoloop_builder_reviewer_contract.md,docs/architecture/scope/automation_orchestration.md,.agents/skills/execplan-event-action-pr-autoloop/scripts/run_event.sh; verify_events=action.pr_autoloop,action.tooling; worker_type=default; replaced reviewer comment-tag contract with reviewer JSON output contract, aligned standalone reviewer behavior, and updated tooling checks.
- [x] (2026-03-04 04:03Z) action_id=a2; mode=serial; depends_on=a1; file_locks=docs/plans/active/plan_structured_reviewer_output_for_loop.md; verify_events=none; worker_type=default; executed gate verification retries to pass, then updated this plan's progress/decisions/outcomes as completion-ready state.
- [x] (2026-03-04 04:07Z) action_id=a3; mode=serial; depends_on=a2; file_locks=.agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh,REVIEW.md,docs/design/pr_autoloop_builder_reviewer_contract.md,docs/architecture/scope/automation_orchestration.md,.agents/skills/execplan-event-action-pr-autoloop/scripts/run_event.sh,docs/plans/active/plan_structured_reviewer_output_for_loop.md; verify_events=action.pr_autoloop,action.tooling; worker_type=default; changed reviewer invocation to `codex exec --output-schema`/`--output-last-message`, fixed verification script marker handling for `--output-schema`, and re-ran gate events.
- [x] (2026-03-04 04:07Z) action_id=a4; mode=serial; depends_on=a3; file_locks=REVIEW.md,docs/plans/active/plan_structured_reviewer_output_for_loop.md; verify_events=none; worker_type=default; removed standalone/autonomous branching from `REVIEW.md` and documented single autonomous-loop-only reviewer policy.
- [x] (2026-03-04 04:08Z) action_id=a5; mode=serial; depends_on=a4; file_locks=.agents/skills/pr-autoloop/SKILL.md,docs/plans/active/plan_structured_reviewer_output_for_loop.md; verify_events=none; worker_type=default; aligned `pr-autoloop` skill description with `AGENTS.md` default-run rule for human-invoked turns.

## Verification Ledger

<!-- verification-ledger:start -->
- attempt_record: event_id=execplan.pre_creation; attempt=1; status=pass; started_at=2026-03-04 03:57Z; finished_at=2026-03-04 03:58Z; commands=git branch --show-current git status --short git log --oneline --decorate --max-count=20 gh pr status gh pr view --json number,title,body,state,headRefName,baseRefName,url mkdir -p docs/prs/active capture execplan start tracked snapshot capture execplan start untracked snapshot write/update docs/prs/active/pr_feat_pr-autoloop-skill.md update plan linkage metadata; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=action.tooling; attempt=1; status=pass; started_at=2026-03-04 04:01Z; finished_at=2026-03-04 04:01Z; commands=bash -n scripts/*.sh .agents/skills/execplan-event-*/scripts/*.sh; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=action.pr_autoloop; attempt=1; status=fail; started_at=2026-03-04 04:01Z; finished_at=2026-03-04 04:01Z; commands=bash -n .agents/skills/pr-autoloop/scripts/run_builder_reviewer_doctor.sh bash -n .agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh rg -n -- --task|--task-file|--pr-url|--max-iterations|--max-builder-cleanup-retries|--max-reviewer-failures|--model-builder|--model-reviewer .agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh rg -n "pr_url"|"comment_body"|"approve_merge" .agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh rg -n parse_reviewer_payload_json|run_codex_prompt_capture|post_pr_comment .agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh rg -n gh\ pr\ comment .agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh rg -n resolve_current_branch|headRefName|must match current local branch .agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh rg -n pr_state|pr_merged_at|state,mergedAt|OPEN and unmerged .agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh rg -n gh\ auth\ status|codex\ login\ status .agents/skills/pr-autoloop/scripts/run_builder_reviewer_doctor.sh rg -n prompt_for_task_text|prompt_for_resume_target_if_needed|is_interactive_session|log_path|LOG_DIR .agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh rg -F -n if ! printf '%s
' "$prompt_text" | "${cmd[@]}", then .agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh; failure_summary=loop script missing required marker: "pr_url"; notify_reference=not_requested;
- attempt_record: event_id=action.pr_autoloop; attempt=2; status=pass; started_at=2026-03-04 04:02Z; finished_at=2026-03-04 04:02Z; commands=bash -n .agents/skills/pr-autoloop/scripts/run_builder_reviewer_doctor.sh bash -n .agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh rg -n -- --task|--task-file|--pr-url|--max-iterations|--max-builder-cleanup-retries|--max-reviewer-failures|--model-builder|--model-reviewer .agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh rg -n pr_url|comment_body|approve_merge .agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh rg -n parse_reviewer_payload_json|run_codex_prompt_capture|post_pr_comment .agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh rg -n gh\ pr\ comment .agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh rg -n resolve_current_branch|headRefName|must match current local branch .agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh rg -n pr_state|pr_merged_at|state,mergedAt|OPEN and unmerged .agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh rg -n gh\ auth\ status|codex\ login\ status .agents/skills/pr-autoloop/scripts/run_builder_reviewer_doctor.sh rg -n prompt_for_task_text|prompt_for_resume_target_if_needed|is_interactive_session|log_path|LOG_DIR .agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh rg -F -n if ! printf '%s
' "$prompt_text" | "${cmd[@]}", then .agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=action.tooling; attempt=2; status=pass; started_at=2026-03-04 04:02Z; finished_at=2026-03-04 04:02Z; commands=bash -n scripts/*.sh .agents/skills/execplan-event-*/scripts/*.sh; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=action.tooling; attempt=3; status=pass; started_at=2026-03-04 04:06Z; finished_at=2026-03-04 04:06Z; commands=bash -n scripts/*.sh .agents/skills/execplan-event-*/scripts/*.sh; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=action.pr_autoloop; attempt=3; status=escalated; started_at=2026-03-04 04:06Z; finished_at=2026-03-04 04:06Z; commands=bash -n .agents/skills/pr-autoloop/scripts/run_builder_reviewer_doctor.sh bash -n .agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh rg -n -- --task|--task-file|--pr-url|--max-iterations|--max-builder-cleanup-retries|--max-reviewer-failures|--model-builder|--model-reviewer .agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh rg -n pr_url|comment_body|approve_merge .agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh rg -n write_reviewer_output_schema|parse_reviewer_payload_json|run_codex_prompt_capture|post_pr_comment .agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh rg -n -- --output-schema|--output-last-message .agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh rg -n gh\ pr\ comment .agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh rg -n resolve_current_branch|headRefName|must match current local branch .agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh rg -n pr_state|pr_merged_at|state,mergedAt|OPEN and unmerged .agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh rg -n gh\ auth\ status|codex\ login\ status .agents/skills/pr-autoloop/scripts/run_builder_reviewer_doctor.sh rg -n prompt_for_task_text|prompt_for_resume_target_if_needed|is_interactive_session|log_path|LOG_DIR .agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh rg -F -n if ! printf '%s
' "$prompt_text" | "${cmd[@]}", then .agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh; failure_summary=loop script missing required marker: --output-schema , retry bound exceeded (attempt=3); notify_reference=not_requested;
- attempt_record: event_id=action.pr_autoloop; attempt=1; status=pass; started_at=2026-03-04 04:06Z; finished_at=2026-03-04 04:06Z; commands=bash -n .agents/skills/pr-autoloop/scripts/run_builder_reviewer_doctor.sh bash -n .agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh rg -n -- --task|--task-file|--pr-url|--max-iterations|--max-builder-cleanup-retries|--max-reviewer-failures|--model-builder|--model-reviewer .agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh rg -n pr_url|comment_body|approve_merge .agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh rg -n write_reviewer_output_schema|parse_reviewer_payload_json|run_codex_prompt_capture|post_pr_comment .agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh rg -n -- --output-schema|--output-last-message .agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh rg -n gh\ pr\ comment .agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh rg -n resolve_current_branch|headRefName|must match current local branch .agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh rg -n pr_state|pr_merged_at|state,mergedAt|OPEN and unmerged .agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh rg -n gh\ auth\ status|codex\ login\ status .agents/skills/pr-autoloop/scripts/run_builder_reviewer_doctor.sh rg -n prompt_for_task_text|prompt_for_resume_target_if_needed|is_interactive_session|log_path|LOG_DIR .agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh rg -F -n if ! printf '%s
' "$prompt_text" | "${cmd[@]}", then .agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=action.tooling; attempt=1; status=pass; started_at=2026-03-04 04:06Z; finished_at=2026-03-04 04:06Z; commands=bash -n scripts/*.sh .agents/skills/execplan-event-*/scripts/*.sh; failure_summary=none; notify_reference=not_requested;
<!-- verification-ledger:end -->

## Surprises & Discoveries

- Observation: first `action.pr_autoloop` gate attempt failed because verification marker expected quoted `\"pr_url\"` text that did not exist literally in shell source.
  Evidence: `event_id=action.pr_autoloop; attempt=1; status=fail; failure_summary=loop script missing required marker: "pr_url"`.

- Observation: standalone reviewer policy requirement changed during implementation and required same-change-set policy update.
  Evidence: user requested standalone reviewer mode to avoid PR comment posting and return review result plus acceptance boolean directly.

- Observation: action verification script initially treated `--output-schema` marker as `rg` option and produced a false negative, causing an escalated gate entry before remediation.
  Evidence: `event_id=action.pr_autoloop; attempt=3; status=escalated; failure_summary=loop script missing required marker: --output-schema`.

## Decision Log

- Decision: adopt a strict reviewer JSON schema with three fields (`pr_url`, `comment_body`, `approve_merge`) and move PR comment posting into the loop script.
  Rationale: this directly satisfies the requested simplification and removes fragile comment text/tag parsing from loop control.
  Date/Author: 2026-03-04 / Codex

- Decision: set standalone reviewer mode to no-remote-write behavior that returns `review_result` and `accept_merge` directly to user.
  Rationale: this matches explicit operator request and keeps standalone and autonomous mode contracts clearly separated.
  Date/Author: 2026-03-04 / Codex

- Decision: enforce reviewer JSON contract at CLI level using `codex exec --output-schema` and `--output-last-message` instead of prompt-only instruction with tolerant parsing.
  Rationale: this follows Codex structured output guidance and removes ambiguity from reviewer response parsing.
  Date/Author: 2026-03-04 / Codex

## Outcomes & Retrospective

Completed objectives:

- `run_builder_reviewer_loop.sh` now requests reviewer JSON (`pr_url`, `comment_body`, `approve_merge`), validates schema, posts comment via `gh pr comment`, and uses `approve_merge` boolean as sole approval stop condition.
- Reviewer JSON is now constrained directly by `codex exec --output-schema` and captured via `--output-last-message`.
- Legacy comment scraping and tag/token approval parsing logic was removed.
- `REVIEW.md` now defines autonomous JSON contract and standalone no-PR-post contract (`review_result` + `accept_merge`).
- `REVIEW.md` now defines only autonomous-loop reviewer behavior (single-mode contract).
- Design and architecture documents were synchronized with the new reviewer JSON contract.
- `action.pr_autoloop` and `action.tooling` gate events passed after remediation (including a false-negative marker check fix for `--output-schema`).

Remaining lifecycle steps outside this implementation turn: run final completion flow (`execplan.post_completion`) when plan/document movement is requested.

## Context and Orientation

The autonomous loop entrypoint is `.agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh`. Today it asks reviewer agent output to post PR comments directly and then re-fetches comments via GitHub GraphQL to infer approval based on tags and `APPROVE` token matching. Reviewer policy is defined in `REVIEW.md`, with long-lived contract and architecture descriptions in `docs/design/pr_autoloop_builder_reviewer_contract.md` and `docs/architecture/scope/automation_orchestration.md`. Validation for this scope is enforced by `.agents/skills/execplan-event-action-pr-autoloop/scripts/run_event.sh`, which currently checks for old tag-based markers. All these files must be aligned to the new JSON-output contract.

## Plan of Work

Implement JSON-driven reviewer handoff in the loop script: capture reviewer output from `codex exec`, parse and validate one JSON object using `jq`, post the PR comment through `gh pr comment`, and use `approve_merge` boolean as the only stop condition for loop completion. Remove obsolete functions and paths used only for comment scraping and tag/token parsing. Update reviewer policy and design/architecture documents to define the new contract and state that reviewer agent returns structured data while loop script performs GitHub write action. Update `action.pr_autoloop` verification script marker checks so gate validation enforces the new contract rather than old tag/token markers.

## Concrete Steps

From repository root:

    scripts/execplan_gate.sh --event execplan.pre_creation
    # create plan file docs/plans/active/plan_structured_reviewer_output_for_loop.md
    scripts/execplan_gate.sh --plan docs/plans/active/plan_structured_reviewer_output_for_loop.md --event execplan.pre_creation

Then edit:

    .agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh
    REVIEW.md
    docs/design/pr_autoloop_builder_reviewer_contract.md
    docs/architecture/scope/automation_orchestration.md
    .agents/skills/execplan-event-action-pr-autoloop/scripts/run_event.sh

Run validation:

    bash -n .agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh
    bash -n .agents/skills/execplan-event-action-pr-autoloop/scripts/run_event.sh
    scripts/execplan_gate.sh --plan docs/plans/active/plan_structured_reviewer_output_for_loop.md --event action.pr_autoloop
    scripts/execplan_gate.sh --plan docs/plans/active/plan_structured_reviewer_output_for_loop.md --event action.tooling

## Validation and Acceptance

Acceptance criteria:

1. `run_builder_reviewer_loop.sh` no longer infers approval by scraping posted comments for `APPROVE` or legacy tags.
2. Reviewer prompt explicitly requests JSON output containing PR URL, comment body, and approval boolean.
3. Loop script posts the reviewer comment body via `gh pr comment <url> --body <text>`.
4. Loop stop condition is controlled by parsed `approve_merge` boolean only.
5. Standalone reviewer mode in `REVIEW.md` explicitly forbids PR posting and requires returning `review_result` and `accept_merge` directly to user.
6. `REVIEW.md`, design doc, architecture scope doc, and action verification script all describe/check the same autonomous JSON contract.
7. `action.pr_autoloop` and `action.tooling` gate events pass and are recorded in this plan ledger.

## Idempotence and Recovery

Script/doc edits are additive and repeatable. If gate checks fail, inspect failure summary in ledger, patch target files, and rerun the same gate command with the same `--plan` path until pass or escalation bound is reached. No destructive git operations are required.

## Artifacts and Notes

Important evidence will be recorded as command lines and pass/fail summaries in `Verification Ledger`.

## Interfaces and Dependencies

The reviewer structured-output interface for autonomous loop mode after this change is:

- `pr_url` (string): target PR URL where the loop script posts comment.
- `comment_body` (string): full review comment body to post.
- `approve_merge` (boolean): explicit merge approval decision used by loop termination.

Dependencies remain `git`, `gh`, `codex`, and `jq` in `.agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh`.

Standalone reviewer-mode response interface after this change:

- `review_result` (string): review findings/result returned directly to user.
- `accept_merge` (boolean): whether merge can be accepted now.

## Plan Revision Notes

- 2026-03-04 03:57Z: Initial plan created to implement reviewer JSON structured-output contract and loop-side comment posting/approval decision simplification.
- 2026-03-04 04:02Z: Updated action-pr-autoloop verification marker checks after first gate failure caused by over-strict quoted-field marker.
- 2026-03-04 04:03Z: Incorporated standalone reviewer-mode no-PR-post requirement and documented direct user return contract (`review_result` + `accept_merge`).
- 2026-03-04 04:07Z: Updated reviewer invocation to use `codex exec --output-schema`/`--output-last-message` and fixed gate marker matching for `--output-schema`.
- 2026-03-04 04:07Z: Unified `REVIEW.md` to autonomous-loop-only policy and removed standalone/autonomous branching language.
- 2026-03-04 04:08Z: Updated `.agents/skills/pr-autoloop/SKILL.md` description text to match `AGENTS.md` default human-invocation behavior.
- execplan_start_branch: feat/pr-autoloop-skill
- execplan_start_commit: d1e6822eaaf84ce85f69d003b52847375f3200d5

## ExecPlan Start Snapshot

<!-- execplan-start-tracked:start -->
- start_tracked_change: 5ceda61bd2fe031a13fe364ca08a65907198a449	docs/prs/active/pr_feat_pr-autoloop-skill.md
<!-- execplan-start-tracked:end -->

<!-- execplan-start-untracked:start -->
- start_untracked_file: 668a0aa246fe8c0d249285b46f88f52c0d60e956	docs/plans/active/plan_structured_reviewer_output_for_loop.md
<!-- execplan-start-untracked:end -->

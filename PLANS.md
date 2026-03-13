# PLANS.md

This repository uses a self-contained plan style for long-running Codex work. Each session plan must be detailed enough that a new contributor, starting only from the current working tree and the single session plan file, can understand the goal, continue the work, and validate the result.

The session plan is also the workflow state. The `## Plan approval` section is the single machine-readable source of truth for whether the session is still in planning or has entered implementation.

## General Plan Rules
- Every session plan must be fully self-contained. Do not rely on external articles, prior chat context, or undocumented repository knowledge.
- Every session plan is a living document. Update it as progress is made, discoveries occur, constraints change, or decisions are finalized.
- Write for a novice to this repository. Define non-obvious terms in plain language, restate assumptions explicitly, and name the relevant repository-relative paths, modules, and commands.
- Begin from purpose and observable outcome. The plan must explain why the work matters, what user-visible or maintainer-visible behavior should exist after the change, and how that behavior can be observed.
- Do not outsource important decisions to the reader. If there is ambiguity, resolve it in the plan and record the rationale.
- Do not point the reader to external docs for required context. If knowledge is needed to execute the plan, include that knowledge in the plan itself in your own words.
- Prefer additive, testable, low-risk changes. If a task has meaningful uncertainty, use a prototype or proof-of-concept milestone to validate feasibility before committing to a larger implementation.
- Milestones should be independently verifiable. Each milestone should say what new capability exists after it, what to run, and what success looks like.
- Validation is mandatory. State the exact commands to run, the working directory when relevant, and the concrete behavior or output that demonstrates success.
- Plans should be safe to resume. Record retry paths, rollback-safe habits, or other recovery notes whenever a step can fail partway through.
- Capture meaningful evidence concisely. When tests, logs, or command outputs matter, summarize the proof inside the plan so a future contributor can see what actually validated the work.

## How To Use Session Plans
- When authoring a session plan, start from the required template in this file and fill it with repository-specific detail rather than placeholders.
- Explicit review-only sessions are outside this session-plan workflow: the session-start hook classifies them from the initial user prompt, does not create `plans/session-<session_id>.md`, and the stop hook later exits without workflow coordination when that plan file is absent.
- While `## Plan approval` is `unapproved`, the builder stays in planning, revises the plan with the user, and does not implement code yet.
- When the user explicitly approves the plan, the builder updates `## Plan approval` to `approved` and starts implementation from the same plan.
- Workflow helpers derive the active session strictly from each hook payload; do not rely on repository-global pointer files.
- When implementing from a session plan, proceed to the next milestone or unchecked subtask without asking the user for "next steps" unless a real blocker remains.
- Keep the plan synchronized with reality at every stop point. If a partially completed task needs to be split into "done" and "remaining" work, update the plan immediately.
- Record key design changes in the plan when they happen, not after the fact.
- If you discover unexpected behavior, performance tradeoffs, test failures, or implementation constraints, capture them in the existing log sections so the next contributor can resume without re-discovering them.
- This repository keeps a machine-checkable session format. Preserve the required headings, approval flag, and checkbox sections exactly so the workflow helpers can parse them reliably.
- This repository's existing session plan template is intentionally narrower than the generic plan skeleton. Keep that template intact, and record extra discoveries, rationale, and retrospective notes inside the existing `Decision log` and `Progress log` unless the task explicitly calls for additional prose sections.

## Working Rules
- Narrative sections such as Goal, Constraints, Repo facts / assumptions, and Acceptance criteria should be written in plain prose first. Use bullets only when they make the plan clearer.
- After each subtask, run the most relevant unit tests immediately so bugs are caught with the smallest possible change scope.
- Each subtask must be small enough that implementation plus debugging fits within one context window.
- If multiple subtasks do not touch the same files and do not require shared context, the builder should actively parallelize them with sub agents.
- Session-specific plans are living documents and must be updated throughout the task.
- Completed subtasks must remain preserved as historical record. If tests fail or review finds problems later, add NEW follow-up subtasks instead of rewriting prior completed work.
- Completed checkboxes must never be rewritten back into unchecked boxes.
- While `## Plan approval` is `unapproved`, the stop hook does no workflow coordination beyond allowing the stop.
- Once `## Plan approval` is `approved`, the stop hook first reevaluates the current session plan. If unchecked implementation work remains, it launches hooks-disabled nested builder runs to continue implementation. If every tracked checkbox in the required subtask sections is already checked, it runs the final tests and reviewer checks directly. Only failed final tests or non-accepting reviewer results append new follow-up tasks and trigger another nested builder pass.
- Acceptance criteria must describe observable behavior, not only internal code structure.
- Per-subtask and final validation entries must name the exact command that was run and enough result detail to distinguish success from failure.
- When work spans multiple files or subsystems, the plan should briefly orient the reader by naming the affected paths and how they fit together.
- If a step is risky or can leave the repo in a partial state, document how to retry safely without losing previous progress.

## Session Plan Template
Use this exact section structure for `plans/session-<session_id>.md` so the workflow helpers can parse it mechanically.

```md
# Session Plan: <session_id>

## Plan approval
unapproved

## Goal
Describe the concrete user-visible outcome for this session.

## Constraints
- Record all important technical constraints here.

## Repo facts / assumptions
- Record the current repo facts and assumptions here.

## Acceptance criteria
- Record the concrete acceptance checks here.

## Ordered subtasks
- [ ] First approved implementation subtask.
- [ ] Second approved implementation subtask.

## Follow-up subtasks (append-only)
- [x] No follow-up subtasks have been added yet.

## Per-subtask validation
- Record the most relevant validation command and result immediately after each completed subtask.

## Final validation
- Record whether the stop hook needed any nested-builder passes, then record the final test gate and repeated review gate here.

## Decision log
- Append important decisions with timestamps.

## Progress log
- Append progress updates with timestamps.
```

## Subtask Rules
- `## Plan approval` must contain exactly one machine-readable value: `approved` or `unapproved`.
- Both `## Ordered subtasks` and `## Follow-up subtasks (append-only)` must use markdown checkboxes.
- If tests fail or review finds problems after some tasks were completed, add NEW unchecked items under `## Follow-up subtasks (append-only)`.
- Do not rewrite or delete completed historical subtasks.
- Do not remove prior reviewer or validation obligations from the plan. Add new unchecked follow-up items instead.
- Assume the stop hook may invoke the builder and reviewer multiple times in one completion cycle after the plan is approved; keep the plan accurate enough that either nested run can resume from the file alone.

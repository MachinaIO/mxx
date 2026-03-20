# Reviewer Rules

This document applies to every Codex reviewer run in this repository, including the hooks-disabled nested read-only reviewer used by `scripts/stop_hook.sh` and any explicit review request from the user.

## Role

The reviewer evaluates the current builder result in read-only mode.
The reviewer does not edit tracked files, does not update plan approval or checklist state, and does not manage workflow transitions directly.
The stop hook may invoke the review-phase reviewer multiple times in one outer completion loop; each review must be independent and based only on current observable evidence.

At the start of every review, reset reviewer posture completely.
Review the current work as if it were authored by another party, and do not trust builder claims, summaries, or plan updates without checking the scoped evidence.

## Default Inputs

Unless the handoff explicitly narrows the scope further, the reviewer should read the minimum set of current-session inputs needed to make a decision:

- the explicit session id from the current review handoff or hook payload
- `plans/active/session-<session_id>.md` when that file exists for the current session
- the latest user message for the current session, or the transcript source needed to recover it when planning review is explicitly requested
- the relevant changed files for the scoped task
- the validation commands or outputs named in the session plan when they are needed to verify a claim

Additional repository-specific inputs:

- read `GPU.md` whenever the reviewed change touches CUDA, GPU kernels, GPU wrappers, GPU tests, or GPU-facing performance-sensitive behavior

Do not load raw transcripts, temporary hook logs, or unrelated historical artifacts by default.
Only inspect them when the current evidence is insufficient to verify a concrete claim.

## Core Obligations

1. Review only the current scoped work for the current session.
2. If `plans/active/session-<session_id>.md` exists, read the current workflow state there by inspecting `## Plan approval` and `## Phase`. Otherwise treat the session as an explicit review request scoped directly by the user prompt.
3. Review against the session plan, not against unstated preferences.
4. Return English feedback.
5. Base the decision on evidence that can be inspected now.
6. In planning, only perform review if planning review was explicitly requested. In implementation, review only when the user explicitly requested review. In review, review the repository state against the current session plan.
7. Prefer the smallest correction that preserves correctness.
8. Treat append-only follow-up-subtask rules and completed-checkbox preservation as hard workflow requirements.

## Phase-Specific Scope

### `planning`

Planning corresponds to `## Plan approval` being `unapproved` and `## Phase` being `planning`.

In planning, the reviewer is not performing a general quality review of future implementation.
The reviewer is evaluating the current plan and, when explicitly requested, whether the latest user message clearly approves that current plan or instead asks for further revisions.

Review the latest user message against the current session plan only when that message is part of the review scope.
Do not require implementation yet.

The reviewer should verify that:

- the current session plan exists and is concrete enough to be the object being approved,
- the latest user message either clearly approves the current session plan or requests additional revisions,
- the decision is based on the user's actual message rather than on reviewer preference about how the plan could be improved.

### `implementation`

Implementation corresponds to `## Plan approval` being `approved` and `## Phase` being `implementation`.

The stop hook does not launch the nested reviewer in implementation. Review this phase only when the user explicitly requests review before the builder transitions the session to `review`.

When implementation is explicitly reviewed, review the current repository state against the current session plan.

The reviewer should verify that:

- the implementation satisfies the session plan's Goal, Constraints, Repo facts / assumptions, and Acceptance criteria,
- completed subtasks appear genuinely complete for their claimed scope,
- claimed validation is relevant and sufficient for the completed work,
- any later failures are represented as NEW unchecked follow-up subtasks rather than by rewriting prior completed work,
- the code remains within the approved scope instead of drifting into unrelated redesign.

### `review`

Review corresponds to `## Plan approval` being `approved` and `## Phase` being `review`.

In review, inspect the current repository state against the approved session plan plus any review-phase follow-up subtasks appended after final tests or earlier reviewer passes.

The reviewer should verify that:

- the implementation satisfies the session plan's Goal, Constraints, Repo facts / assumptions, and Acceptance criteria,
- completed subtasks and completed historical follow-up items remain preserved,
- open follow-up subtasks capture the remaining concrete obligations created by review-phase tests or earlier reviewer feedback,
- claimed validation is relevant and sufficient for the current reviewed state,
- the code remains within the approved scope instead of drifting into unrelated redesign.

## What To Review

- correctness against the current session plan,
- adherence to the current phase scope,
- in planning, only when explicitly requested, whether the latest user message approves the current session plan or requests further changes,
- in implementation, only explicit user-requested review before the builder transitions to `review`,
- in review, the current repository state plus any review-phase follow-up subtasks,
- completeness and quality of the stated validation,
- whether the claimed completion is observable from code and checks rather than inferred from intent,
- whether the required session plan sections remain coherent after the builder change,
- whether completed subtasks remain preserved as historical record,
- whether new obligations are captured as NEW unchecked items under `## Follow-up subtasks (append-only)` when appropriate,
- whether the work introduces unrelated scope expansion, dead fallback logic, or unnecessary redesign,
- whether the resulting code follows KISS by minimizing cognitive load and keeping responsibility boundaries rational,
- whether GPU-related changes satisfy `GPU.md` when GPU behavior is in scope.

## Mandatory Review Checks

Before returning a decision, verify all of the following that apply to the current scope:

1. In planning, the decision is grounded in the latest user message and the current session plan, not in reviewer-authored new plan requirements.
2. In implementation and review, the session plan still matches the current repository state.
3. The required session plan headings, approval flag, and checkbox sections remain intact and machine-checkable when they are relevant to the reviewed scope.
4. Claimed completion checks are appropriate for the scope and are not superficial restatements of intent.
5. The reviewed work did not silently broaden scope beyond the session plan.
6. Completed checklist items were not rewritten back into unchecked items.
7. If review finds remaining work, that remaining work can be described as concrete follow-up subtasks.
8. For GPU changes, the implementation and test strategy respect `GPU.md`, including repeated testing for nondeterministic GPU issues when relevant.

## What Not To Do

- do not perform implementation work,
- do not edit repository files,
- do not invent or mutate repository-global session pointers,
- do not rewrite the session plan directly,
- do not request unrelated redesigns,
- do not block on style preferences that are not tied to correctness, scope, or maintainability,
- do not invent requirements outside the current session plan, `AGENTS.md`, `PLANS.md`, `BUILDER.md`, and `GPU.md` when relevant,
- do not rely on builder intent when observable evidence is missing.

## Review Standard

Review against the current contract, not against personal taste.

- Prefer observable failures over speculative concerns.
- Call out the exact claim, file, plan section, validation step, or GPU principle that fails the contract.
- If the work is acceptable, do not ask for extra polish outside the approved scope.
- If multiple outcomes are possible, choose the narrowest result that preserves correctness.
- Evaluate simplicity by readability, local reasoning cost, and scope separation, not by maximizing abstraction.
- Do not reward unrequested compatibility shims, speculative fallback paths, or ornamental indirection.

## Reviewer Decision Contract

When a structured reviewer result is required, use exactly this result space:

- `accept`
- `revision`

### `planning`

- `accept`
  - the latest user message approves the current session plan and does not request further changes.
- `revision`
  - the latest user message requests additional changes, or does not clearly approve the current session plan.

### `implementation`

- `accept`
  - the explicitly requested implementation review satisfies the current session plan and the validation is sufficient for the claimed completion.
- `revision`
  - the explicitly requested implementation review still has concrete deficiencies that must be addressed before the builder should transition into `review`.

### `review`

- `accept`
  - the review-phase implementation satisfies the current session plan and the validation is sufficient for final acceptance.
- `revision`
  - the review-phase implementation, validation, or workflow bookkeeping still has concrete deficiencies that must be addressed before final acceptance.

## Feedback Quality

Feedback should be:

- specific,
- phase-scoped,
- actionable,
- tied to the session plan, the latest user message, or other named evidence,
- narrow enough that the builder can convert it into concrete follow-up subtasks without guessing,
- focused on what must change for acceptance, not on optional improvements.

If the work is acceptable, say so plainly.
If it is not, identify the smallest correction that would make the review pass.

## Workflow Implications

Reviewer outcomes do not update workflow state directly.
The stop hook and builder handle the next step.

Important invariants:

- reviewer feedback should map cleanly to NEW unchecked follow-up subtasks when more work is required,
- reviewer rejection must not imply rewriting or erasing already completed subtasks,
- reviewer approval should mean the current session can finish without additional hidden conditions.

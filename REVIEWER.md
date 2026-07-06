# Reviewer Rules

This document applies to every explicit Codex review request in this repository.

## Role

The reviewer evaluates the current builder result in read-only mode.
The reviewer does not edit tracked files and does not manage workflow transitions directly.

At the start of every review, reset reviewer posture completely.
Review the current work as if it were authored by another party, and do not trust builder claims or summaries without checking the scoped evidence.

## Default Inputs

Unless the user narrows the scope further, read the minimum set of inputs needed to make a decision:

- the latest user request that defines the review scope
- the relevant changed files for the scoped task
- the validation commands or outputs needed to verify a claim
- `GPU.md` whenever the reviewed change touches CUDA, GPU kernels, GPU wrappers, GPU tests, or GPU-facing performance-sensitive behavior

Do not load raw transcripts, temporary logs, or unrelated historical artifacts by default.
Only inspect them when the current evidence is insufficient to verify a concrete claim.

## Core Obligations

1. Review only the current scoped work.
2. Return English feedback.
3. Base the decision on evidence that can be inspected now.
4. Prefer the smallest correction that preserves correctness.
5. Treat user-stated constraints and repository guidance in `AGENTS.md`, `BUILDER.md`, `REVIEWER.md`, and `GPU.md` when relevant as the review contract.

## What To Review

- correctness against the user's request,
- completeness and quality of the stated validation,
- whether the claimed completion is observable from code and checks rather than inferred from intent,
- whether the work introduces unrelated scope expansion, dead fallback logic, or unnecessary redesign,
- whether the resulting code follows KISS by minimizing cognitive load and keeping responsibility boundaries rational,
- whether GPU-related changes satisfy `GPU.md` when GPU behavior is in scope.

## Mandatory Review Checks

Before returning a decision, verify all of the following that apply to the current scope:

1. The reviewed work satisfies the latest user request and any explicitly stated constraints.
2. Claimed completion checks are appropriate for the scope and are not superficial restatements of intent.
3. The reviewed work did not silently broaden scope beyond the request.
4. For GPU changes, the implementation and test strategy respect `GPU.md`, including repeated testing for nondeterministic GPU issues when relevant.

## What Not To Do

- do not perform implementation work,
- do not edit repository files,
- do not request unrelated redesigns,
- do not block on style preferences that are not tied to correctness, scope, or maintainability,
- do not invent requirements outside the user's request, `AGENTS.md`, `BUILDER.md`, `REVIEWER.md`, and `GPU.md` when relevant,
- do not rely on builder intent when observable evidence is missing.

## Review Standard

Review against the current contract, not against personal taste.

- Prefer observable failures over speculative concerns.
- Call out the exact claim, file, validation step, or GPU principle that fails the contract.
- If the work is acceptable, do not ask for extra polish outside the approved scope.
- If multiple outcomes are possible, choose the narrowest result that preserves correctness.
- Evaluate simplicity by readability, local reasoning cost, and scope separation, not by maximizing abstraction.
- Do not reward unrequested compatibility shims, speculative fallback paths, or ornamental indirection.

## Reviewer Decision Contract

When a structured reviewer result is required, use exactly this result space:

- `accept`
- `revision`

Use `accept` when the reviewed work satisfies the request and the validation is sufficient for the claimed completion.
Use `revision` when concrete deficiencies remain.

## Feedback Quality

Feedback should be:

- specific,
- actionable,
- tied to the user request or other named evidence,
- narrow enough that the builder can convert it into concrete follow-up work without guessing,
- focused on what must change for acceptance, not on optional improvements.

If the work is acceptable, say so plainly.
If it is not, identify the smallest correction that would make the review pass.

# Verification Documentation Meta-Rules

This document defines how verification policy and verification runbooks must be authored, organized, and maintained in this repository.

Verification documents are operational contracts for agents and contributors. They must describe concrete, executable validation actions and must remain aligned with current repository behavior, CI policy, and testing constraints.

## Purpose of verification documents

Verification documents define how to prove that a change is correct, safe, and ready to merge for a specific event class (for example: docs-only updates, CPU behavior changes, GPU/CUDA changes, or benchmark-impacting work).

Each verification document must let a contributor answer all of the following without external context:

- what to verify,
- which commands to run,
- where to run them,
- what success and failure look like,
- what minimum evidence must be recorded in a plan/PR.

## Required location and reading order

All verification documents must live under `docs/verification/`.

`docs/verification/index.md` is mandatory and is the entry point for verification guidance.

Any agent or contributor performing verification must read `docs/verification/index.md` first before choosing or executing event-specific verification documents.

## Event documents and filename policy

`docs/verification/` stores markdown files that correspond to verification events.

A verification event is a change class that requires a distinct verification workflow (for example, GPU/CUDA boundary changes versus docs-only changes).

Each event document filename must describe the event clearly and unambiguously.

Examples of acceptable naming:

- `docs_only_changes.md`
- `cpu_behavior_changes.md`
- `gpu_behavior_changes.md`

Avoid opaque names such as `check1.md` or `misc.md`.

## Required role of docs/verification/index.md

`docs/verification/index.md` must function as a decision index for verification selection.

It must explain, for each major change case:

- which verification event document to read,
- why that event document applies,
- what baseline verification depth is expected.

If multiple event documents apply to one change, the index must state that all relevant documents must be executed.

## Actionability requirement for every verification document

Every document under `docs/verification/` must define concrete actions an agent can execute directly.

At minimum, each document must specify:

- prerequisites and environment assumptions,
- exact commands,
- working directory,
- expected success criteria,
- failure triage notes,
- required evidence to record in ExecPlans/PRs.

Narrative-only guidance is not sufficient.

## Sandbox approval and command reuse rule

If a verification command must run outside the sandbox and requires manual operator approval, reuse the same command form already documented in the corresponding verification document whenever practical. This reduces repeated approval overhead and keeps verification execution predictable.

If major execution-efficiency gains require changing the command, do not keep using a slower command only to avoid approval churn. Request approval for the new command and explain the efficiency reason clearly.

At the same time, keep approval requests to the minimum necessary set, considering that an operator may be away for extended periods.

## Update policy

Verification documents must be updated in the same change when verification policy or required checks change, including cases such as:

- CI workflow command changes,
- new required test category,
- changed pass/fail criteria,
- new hardware/runtime constraints,
- updated retry/repetition requirements for flaky or probabilistic paths.

Routine per-PR test outputs must not be written into policy files. They belong in ExecPlans and PR descriptions.

## Relationship to ExecPlans and PRs

ExecPlans must reference `VERIFICATION.md` and the specific event documents used from `docs/verification/`.

Each ExecPlan must record both:

- commands planned to run,
- commands actually run and their outcomes.

PR descriptions must include executed verification commands, notable environment limits, and any deviations from expected verification coverage.

## Maintenance rule

Verification documentation is long-lived and must track the implementation and CI truth sources.

If a verification statement is not backed by concrete repository evidence (workflow config, policy file, test target, or script), do not document it as verification policy.

# Merge Verification Policy into PLANS and Split ExecPlan Event Skills

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan is maintained under `PLANS.md` and follows that file's rules.

Repository-document context used for this plan: `PLANS.md`, `AGENTS.md`, `REVIEW.md`, `DESIGN.md`, `ARCHITECTURE.md`, `.agents/skills/`, `scripts/`, and `docs/design/execplan_verification_enforcement.md`.

## Purpose / Big Picture

After this change, verification policy is integrated in `PLANS.md` (no separate verification-policy authority file), event execution is modularized to one skill per event, and event scripts are modularized under each event skill directory. The gate now resolves event handlers through index mapping, so event additions/removals are data changes rather than monolithic runner rewrites.

## Progress

- [x] (2026-03-03 16:25Z) Created this ExecPlan and captured baseline policy/skill/script context.
- [x] (2026-03-03 16:08Z) Integrated verification policy into `PLANS.md` and removed separate `VERIFICATION.md` policy authority.
- [x] (2026-03-03 16:13Z) Split verification skill into index skill plus per-event skills under `.agents/skills/execplan-event-*/`.
- [x] (2026-03-03 16:16Z) Replaced monolithic event runner dispatch with event-index mapping in `scripts/execplan_gate.sh`.
- [x] (2026-03-03 16:17Z) Confirmed repository-local skills under `.agents/skills/` are discovered by Codex without installer scripts.
- [x] (2026-03-03 16:18Z) Added new `action.tooling` event skill to demonstrate map-driven event extensibility and support script/tooling changes.
- [x] (2026-03-03 16:19Z) Updated policy/review/design docs to reflect modular event-skill architecture and integrated policy location.
- [x] (2026-03-03 16:22Z) Removed `docs/verification/` legacy runbooks entirely (no compatibility retention).
- [x] (2026-03-03 16:19Z) Ran syntax/install/gate validation with fake `gh` harness and recorded outcomes.

## Surprises & Discoveries

- Observation: validating gate behavior requires a deterministic `gh` stub because event notifications are mandatory in gate flow.
  Evidence: local verification used `<tmp_dir>/mxx_fakebin/gh` to emulate `gh pr view` and `gh pr comment`.
- Observation: script/tooling-heavy changes are not semantically covered by `action.docs_only`.
  Evidence: docs-only event intentionally failed when non-doc paths existed, which motivated adding `action.tooling`.

## Decision Log

- Decision: Remove fixed event dispatch from gate and move dispatch to event-index mapping (`event_skill_map.tsv`).
  Rationale: user requested future event add/remove flexibility.
  Date/Author: 2026-03-03 / Codex
- Decision: Keep lifecycle edges (`execplan.pre_creation`, `execplan.post_completion`) mandatory in index map.
  Rationale: user explicitly allowed assuming those two are mandatory.
  Date/Author: 2026-03-03 / Codex
- Decision: Integrate verification policy into `PLANS.md` and remove standalone policy authority in `VERIFICATION.md`.
  Rationale: user requested direct integration and no compatibility constraints.
  Date/Author: 2026-03-03 / Codex

## Outcomes & Retrospective

The requested architecture is implemented:

- verification policy is integrated in `PLANS.md`,
- event handling is modularized into one skill per event,
- event-to-skill mapping is centralized in index skill,
- gate dispatch is map-driven and lifecycle-safe,
- `docs/verification/` legacy runbooks were removed (not frozen),
- tooling event was added without changing policy shape, demonstrating extensibility.

No runtime product code was changed; this is policy/automation architecture work.

## Design/Architecture/Verification Summary

Design documents:

- Referenced: `DESIGN.md`, `docs/design/index.md`
- Modified: `docs/design/execplan_verification_enforcement.md`

Architecture documents:

- Referenced: `ARCHITECTURE.md`, `docs/architecture/index.md`
- Unchanged for this iteration.

Verification policy and execution artifacts:

- Policy integrated/modified: `PLANS.md`
- Removed: `VERIFICATION.md`
- Modified: `AGENTS.md`, `REVIEW.md`
- Created/modified skills:
  - `.agents/skills/execplan-event-index/*`
  - `.agents/skills/execplan-event-pre-creation/*`
  - `.agents/skills/execplan-event-post-completion/*`
  - `.agents/skills/execplan-event-action-docs-only/*`
  - `.agents/skills/execplan-event-action-cpu-behavior/*`
  - `.agents/skills/execplan-event-action-gpu-behavior/*`
  - `.agents/skills/execplan-event-action-tooling/*`
- Modified scripts:
  - `scripts/execplan_gate.sh`
  - `scripts/execplan_notify.sh` (interface unchanged; retained)

## Context and Orientation

The previous model mixed a standalone verification-policy document plus one monolithic verification skill. The new model uses one policy location (`PLANS.md`) and per-event skills referenced through index mapping so event evolution is localized to map + event skill files.

## Plan of Work

Policy and agent-facing docs were updated first to establish the new contract. Then event skills were split, index mapping was introduced, and the gate script was refactored to resolve event scripts from mapping. Installer behavior was generalized to install all local skills. Finally, validation commands and gate functional scenarios were executed.

## Concrete Steps

Run from repository root (`.`):

    bash -n scripts/execplan_notify.sh scripts/execplan_gate.sh .agents/skills/execplan-event-*/scripts/run_event.sh

    # gate functional checks with fake gh
    PATH="<tmp_dir>/mxx_fakebin:$PATH" scripts/execplan_gate.sh --plan <tmp_dir>/mxx_modular_gate_tooling.md --event execplan.pre_creation
    PATH="<tmp_dir>/mxx_fakebin:$PATH" scripts/execplan_gate.sh --plan <tmp_dir>/mxx_modular_gate_tooling.md --event action.tooling
    PATH="<tmp_dir>/mxx_fakebin:$PATH" scripts/execplan_gate.sh --plan <tmp_dir>/mxx_modular_gate_tooling.md --event execplan.post_completion

## Validation and Acceptance

Acceptance criteria and outcomes:

1. Verification policy lives in `PLANS.md`: satisfied.
2. Standalone `VERIFICATION.md` policy authority removed: satisfied.
3. Event skills are one-per-event with scripts inside each skill directory: satisfied.
4. Index skill maps event IDs to event scripts: satisfied.
5. Gate dispatch is mapping-based and not fixed-case monolithic: satisfied.
6. Repository-local `.agents/skills/` auto-discovery model is preserved: satisfied.
7. Event extensibility demonstrated by adding `action.tooling`: satisfied.
8. Gate lifecycle checks (pre/action/post), escalation, and conflict checks still work: satisfied.

## Verification Ledger

<!-- verification-ledger:start -->
- attempt_record: event_id=execplan.pre_creation; attempt=1; status=pass; started_at=2026-03-03 16:19Z; finished_at=2026-03-03 16:19Z; commands=policy and modular skill split implementation prechecks; failure_summary=none; notify_reference=not_run;
- attempt_record: event_id=action.tooling; attempt=1; status=pass; started_at=2026-03-03 16:19Z; finished_at=2026-03-03 16:19Z; commands=bash -n scripts/execplan_notify.sh scripts/execplan_gate.sh .agents/skills/execplan-event-*/scripts/run_event.sh; failure_summary=none; notify_reference=not_run;
- attempt_record: event_id=execplan.post_completion; attempt=1; status=pass; started_at=2026-03-03 16:19Z; finished_at=2026-03-03 16:19Z; commands=gate functional scenarios with fake gh harness for pre/tooling/post and conflict/escalation paths; failure_summary=none; notify_reference=not_run;
<!-- verification-ledger:end -->

## Idempotence and Recovery

All edits are policy/automation/documentation changes and can be safely retried by reapplying file writes and rerunning syntax checks. Functional gate tests run in `<tmp_dir>` and do not mutate tracked repository state outside modified files.

## Artifacts and Notes

Key validation outcomes:

- `bash -n ...` succeeded for gate/notify and all event scripts.
- Gate functional checks with fake `gh` succeeded for pre/tooling/post lifecycle flow.
- Conflict and escalation behaviors were verified with dedicated temporary plans.

## Interfaces and Dependencies

Updated operational interfaces:

- `scripts/execplan_gate.sh --plan <plan_md> --event <event_id> [--attempt <n>]`
- `.agents/skills/execplan-event-index/references/event_skill_map.tsv` (event registry)
- `.agents/skills/execplan-event-*/scripts/run_event.sh` (event-specific execution)

Revision note (2026-03-03, Codex): Initial plan created for policy integration and per-event skill modularization.
Revision note (2026-03-03, Codex): Completed implementation, validation, and documentation updates for modular event skill architecture.

---
name: execplan-event-pre-creation
description: Event skill for execplan.pre_creation verification. Implements the pre-creation workflow for main ExecPlan setup (branch/PR alignment and PR tracking linkage).
---

# Event Skill: execplan.pre_creation

Executes the "before main ExecPlan creation" workflow:

- capture branch/status/log context,
- query PR context when `gh` is available,
- enforce branch-switch rules (`main` or scope-misaligned work must switch),
- create/update PR tracking metadata under `eternal-cycler-out/prs/active/`,
- ensure the ExecPlan contains PR tracking path and start branch/commit linkage,
- capture a start snapshot of tracked/untracked working-tree deltas (hash + path) in the plan when `--plan` is provided.

Dynamic controls:

- `EXECPLAN_SCOPE_ALIGNMENT=aligned|not_aligned|auto`
- `EXECPLAN_NEW_BRANCH=<type/scope>` when branch switch is required
- `EXECPLAN_PR_TRACKING_PATH=eternal-cycler-out/prs/active/<file>.md`
- `EXECPLAN_MANUAL_PR_URL=<url>` when PR metadata must be injected explicitly

Execution policy:

- This event must be executed out-of-sandbox.
- Run through gate as: `scripts/execplan_gate.sh --event execplan.pre_creation` (or `--plan <plan_md> --event execplan.pre_creation`) with out-of-sandbox execution.
- Do not run this event inside sandbox because stable `gh` access is required.

## Script

- `scripts/run_event.sh [--plan <plan_md>]`

---
name: execplan-event-pre-creation
description: Event skill for execplan.pre_creation verification. Lightweight environment check before the ExecPlan document is created.
---

# Event Skill: execplan.pre_creation

Executes a lightweight environment check before the ExecPlan document is created:

- capture branch/status/log context,
- query PR context when `gh` is available.

This event takes **no arguments** — `--plan` is not accepted because the plan file does not exist yet. Branch management is the caller's responsibility.

After creating the plan document, run `execplan.post_creation` (which requires `--plan`) to record the start snapshot and PR tracking linkage.

Execution policy:

- This event must be executed out-of-sandbox.
- Run through gate as: `scripts/execplan_gate.sh --event execplan.pre_creation` (no `--plan`).

## Script

- `scripts/run_event.sh`

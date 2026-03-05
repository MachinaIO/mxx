---
name: execplan-event-resume
description: Event skill for execplan.resume verification. Run when resuming an existing ExecPlan to validate branch consistency and record a resume record in the plan.
---

# Event Skill: execplan.resume

Executes the "resume existing plan" workflow:

- read `execplan_start_branch` from the plan and verify that the current branch matches,
- refresh the PR tracking document at `eternal-cycler-out/prs/active/pr_<branch>.md`,
- append an `## ExecPlan Resume Record` section to the plan (idempotent per resume commit).

Requires `--plan`. Branch management is the caller's responsibility.

Execution policy:

- This event must be executed out-of-sandbox.
- Run through gate as: `scripts/execplan_gate.sh --plan <plan_md> --event execplan.resume`.

## Script

- `scripts/run_event.sh --plan <plan_md>`

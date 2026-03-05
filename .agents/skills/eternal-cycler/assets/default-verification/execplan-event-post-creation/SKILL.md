---
name: execplan-event-post-creation
description: Event skill for execplan.post_creation verification. Run immediately after the ExecPlan document is created to record the start snapshot and PR tracking linkage.
---

# Event Skill: execplan.post_creation

Executes the "after plan document creation" workflow:

- append `execplan_start_branch` and `execplan_start_commit` markers to the plan (idempotent),
- append an `## ExecPlan Start Snapshot` section with tracked and untracked file snapshots (idempotent),
- create or overwrite the PR tracking document at `eternal-cycler-out/prs/active/pr_<branch>.md`,
- append a PR Tracking Linkage section to the plan (idempotent).

Requires `--plan`. Branch management is the caller's responsibility.

Execution policy:

- This event must be executed out-of-sandbox.
- Run through gate as: `scripts/execplan_gate.sh --plan <plan_md> --event execplan.post_creation`.

## Script

- `scripts/run_event.sh --plan <plan_md>`

---
name: execplan-event-pre-creation
description: Event skill for execplan.pre_creation verification. Implements the pre-creation workflow for main ExecPlan setup (branch/PR alignment and PR tracking linkage).
---

# Event Skill: execplan.pre_creation

Executes the "before main ExecPlan creation" workflow:

- capture branch/status/log context,
- query PR context when `gh` is available,
- enforce branch-switch rules (`main` or scope-misaligned work must switch),
- create/update PR tracking metadata under `docs/prs/active/`,
- ensure the ExecPlan contains PR tracking path and start branch/commit linkage,
- capture a start snapshot of untracked files (hash + path) in the plan when `--plan` is provided.

Dynamic controls:

- `EXECPLAN_SCOPE_ALIGNMENT=aligned|not_aligned|auto`
- `EXECPLAN_NEW_BRANCH=<type/scope>` when branch switch is required
- `EXECPLAN_PR_TRACKING_PATH=docs/prs/active/<file>.md`
- `EXECPLAN_MANUAL_PR_URL=<url>` when `gh` is unavailable and a new branch requires manual draft PR creation

## Script

- `scripts/run_event.sh [--plan <plan_md>]`

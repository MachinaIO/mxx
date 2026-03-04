---
name: execplan-event-post-completion
description: Event skill for execplan.post_completion verification. Use after plan document finalization to enforce completion checks and final persistence.
---

# Event Skill: execplan.post_completion

Executes the "after main ExecPlan completion" workflow:

- validate ledger completion prerequisites,
- resolve linked PR tracking document from the plan and verify metadata,
- ensure no unresolved progress actions or unresolved latest verification events remain,
- stage/commit/push only files changed by the plan in this lifecycle,
- apply tracked/untracked baseline rule: keep pre-existing unchanged tracked/untracked edits unstaged, but stage files newly changed during this plan (including a new target plan document) and pre-existing files modified during this plan,
- if validation fails before staging, roll back the plan document to active path and return to action revision flow.

Dynamic controls:

- `EXECPLAN_FINAL_COMMIT_MESSAGE="<message>"`

Execution policy:

- This event must be executed out-of-sandbox.
- Run through gate as: `scripts/execplan_gate.sh --plan <completed_plan_md> --event execplan.post_completion` with out-of-sandbox execution.
- Do not run this event inside sandbox because stable `gh` access is required.

## Script

- `scripts/run_event.sh --plan <plan_md>`

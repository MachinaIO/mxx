---
name: execplan-event-post-completion
description: Event skill for execplan.post_completion verification. Use after plan document finalization to enforce validation-only completion checks.
---

# Event Skill: execplan.post_completion

Executes the "after main ExecPlan completion" workflow:

- validate ledger completion prerequisites,
- resolve linked PR tracking document from the plan and verify metadata,
- ensure no unresolved progress actions or unresolved latest verification events remain,
- verify start snapshot markers are present in the plan,
- require explicit rollback provenance before accepting an active-path retry,
- do not run `git add`, `git commit`, or `git push` in this event,
- if validation fails, roll back the plan document to active path and return to action revision flow.

Execution policy:

- This event must be executed out-of-sandbox.
- Run through gate as: `scripts/execplan_gate.sh --plan <completed_plan_md> --event execplan.post_completion` with out-of-sandbox execution.
- Do not run this event inside sandbox because stable `gh` access is required.
- When validation fails on a completed plan, the runner writes rollback provenance into the moved active plan plus a sidecar receipt under `eternal-cycler-out/plans/active/.post-completion-rollbacks/`. Active-path retries are valid only when that provenance still matches the current retry file.

## Script

- `scripts/run_event.sh --plan <plan_md>`

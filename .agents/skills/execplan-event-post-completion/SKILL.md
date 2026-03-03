---
name: execplan-event-post-completion
description: Event skill for execplan.post_completion verification. Use after all action-level events pass to enforce PR-readiness transition and completion evidence checks.
---

# Event Skill: execplan.post_completion

Executes the "after main ExecPlan completion" workflow:

- validate ledger completion prerequisites,
- resolve linked PR tracking document from the plan and verify metadata,
- determine ready/not-ready state,
- if ready, run `gh pr ready` and move tracking doc to `docs/prs/completed/`,
- if not ready, keep active state and record blockers,
- never auto-run `git add/commit/push`; persistence remains an explicit lifecycle step outside this verification event.

Dynamic controls:

- `EXECPLAN_PR_READY=ready|not_ready|auto`
- `EXECPLAN_BLOCKERS="<text>"` when not ready
- `EXECPLAN_PR_READY_CONFIRMED=1` after manual web-UI readiness fallback when automation is unavailable

## Script

- `scripts/run_event.sh --plan <plan_md>`

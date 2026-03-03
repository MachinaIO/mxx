---
name: execplan-event-post-completion
description: Event skill for execplan.post_completion verification. Use after plan document finalization to enforce completion checks, PR-tracking state transition, and final persistence.
---

# Event Skill: execplan.post_completion

Executes the "after main ExecPlan completion" workflow:

- validate ledger completion prerequisites,
- resolve linked PR tracking document from the plan and verify metadata,
- send latest commit metadata to reviewer daemon and wait for reviewer response,
- fetch reviewer comment by returned URL and require `APPROVE` token,
- determine ready/not-ready state,
- if ready, move tracking doc to `docs/prs/completed/`,
- if not ready, keep active state and record blockers,
- stage/commit/push only files changed by the plan in this lifecycle,
- apply tracked/untracked baseline rule: keep pre-existing unchanged tracked/untracked edits unstaged, but stage files newly changed during this plan (including a new target plan document) and pre-existing files modified during this plan,
- run `gh pr ready` when the PR is complete,
- if validation fails before staging, roll back plan/pr tracking docs to active paths and return to action revision flow.

Dynamic controls:

- `EXECPLAN_PR_READY=ready|not_ready|auto`
- `EXECPLAN_BLOCKERS="<text>"` when not ready
- `EXECPLAN_PR_READY_CONFIRMED=1` after manual web-UI readiness fallback when automation is unavailable
- `EXECPLAN_FINAL_COMMIT_MESSAGE="<message>"`

Execution policy:

- This event runs `gh` operations (including reviewer-comment fetch by URL); execute out-of-sandbox according to `.agents/skills/execplan-sandbox-escalation/SKILL.md`.

## Script

- `scripts/run_event.sh --plan <plan_md>`

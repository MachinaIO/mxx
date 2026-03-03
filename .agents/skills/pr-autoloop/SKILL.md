---
name: pr-autoloop
description: Run a deterministic pull-request automation loop that alternates a builder agent and reviewer agent until reviewer approval or failure thresholds. Use when you need autonomous ExecPlan implementation/review iteration with machine-readable PR comment contracts.
---

# PR Autonomous Loop

Use this skill to run autonomous PR iteration with strict role/comment contracts.

## Inputs

Provide:

- goal file path (`--goal-file`)
- target PR URL (`--pr-url`) when already created, or head branch (`--head-branch`) when PR is not created yet
- optional failure/iteration bounds (`--max-builder-failures`, `--max-iterations`)

## Workflow

1. When the user asks to start the loop, execute commands directly (do not only print instructions).
2. Run `scripts/doctor.sh` to validate local prerequisites and auth state.
3. Run `scripts/run_loop.sh` to execute the builder/reviewer cycle.
4. Review runtime artifacts under `runtime/runs/<run_id>/` for logs and state transitions.

### Start command patterns

Existing PR:

- `./.agents/skills/pr-autoloop/scripts/doctor.sh --pr-url <pr_url>`
- `./.agents/skills/pr-autoloop/scripts/run_loop.sh --goal-file <goal_file> --pr-url <pr_url> [options]`

No PR yet (builder bootstrap mode):

- `./.agents/skills/pr-autoloop/scripts/doctor.sh --head-branch <branch>`
- `./.agents/skills/pr-autoloop/scripts/run_loop.sh --goal-file <goal_file> --head-branch <branch> [--base-branch <base>] [options]`

In bootstrap mode, if builder creates a new PR, `run_loop.sh` auto-detects that PR URL and passes it to reviewer context in the same iteration.

## References

Read these files before changing contracts or parser logic:

- `references/comment_contract.md`
- `references/state_schema.md`

## Safety Constraints

- Keep role tags explicit (`AUTO_AGENT: BUILDER` and `AUTO_AGENT: REVIEWER`).
- Treat missing reviewer contract tags as hard failures; do not infer status from prose.
- Enforce lock ownership per PR to prevent concurrent autonomous loops.

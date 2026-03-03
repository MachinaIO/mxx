---
name: pr-autoloop
description: Run a deterministic pull-request automation loop that alternates a builder agent and reviewer agent until reviewer approval or failure thresholds. Use when you need autonomous ExecPlan implementation/review iteration with machine-readable PR comment contracts.
---

# PR Autonomous Loop

Use this skill to run autonomous PR iteration with strict role/comment contracts.

## Inputs

Provide:

- goal file path (`--goal-file`)
- target PR URL (`--pr-url`)
- optional failure/iteration bounds (`--max-builder-failures`, `--max-iterations`)

## Workflow

1. Run `scripts/doctor.sh` to validate local prerequisites and auth state.
2. Run `scripts/run_loop.sh` to execute the builder/reviewer cycle.
3. Review runtime artifacts under `runtime/runs/<run_id>/` for logs and state transitions.

## References

Read these files before changing contracts or parser logic:

- `references/comment_contract.md`
- `references/state_schema.md`

## Safety Constraints

- Keep role tags explicit (`AUTO_AGENT: BUILDER` and `AUTO_AGENT: REVIEWER`).
- Treat missing reviewer contract tags as hard failures; do not infer status from prose.
- Enforce lock ownership per PR to prevent concurrent autonomous loops.

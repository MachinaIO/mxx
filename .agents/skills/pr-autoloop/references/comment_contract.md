# PR Auto Loop Comment Contract

This contract is mandatory for machine parsing by `.agents/skills/pr-autoloop/scripts/run_loop.sh`.

## Common envelope

Every autonomous-loop comment must include:

- `[AUTO_LOOP]`
- `AUTO_RUN_ID: <run_id>`
- `AUTO_ITERATION: <n>`
- `AUTO_AGENT: BUILDER` or `AUTO_AGENT: REVIEWER`

## Builder comment contract

Builder comments are optional but recommended for traceability.

When posted, include:

- `AUTO_AGENT: BUILDER`
- `AUTO_TARGET_COMMIT: <head_sha_after_push>`
- `AUTO_RESULT: PUSHED|NO_CHANGE|FAILED`

Bootstrap note:

- When loop starts without `--pr-url`, builder must create/reuse a PR for the configured head branch.
- `run_loop.sh` discovers that PR URL and uses it for reviewer execution and comment parsing.

## Reviewer comment contract (required)

Reviewer comments are mandatory and must include all fields below:

- `AUTO_AGENT: REVIEWER`
- `AUTO_REVIEW_STATUS: APPROVED|CHANGES_REQUIRED`
- `AUTO_TARGET_COMMIT: <sha>`

Reviewer timing:

- In autonomous-loop mode, reviewer posts this contract comment even when CI checks are still running.

`AUTO_REVIEW_STATUS` values are strict:

- `APPROVED`: loop exits successfully.
- `CHANGES_REQUIRED`: loop continues and passes reviewer feedback to next builder iteration.

Any missing or invalid reviewer field is treated as a hard failure.

# PR Reviewer Daemon Comment Contract

This contract is mandatory for machine parsing by:

- `.agents/skills/pr-autoloop/scripts/reviewer_daemon.sh`
- `.agents/skills/execplan-event-post-completion/scripts/run_event.sh`

## Reviewer comment contract (required)

Each daemon-driven reviewer output must be one PR comment containing all tags below exactly once:

- `AUTO_AGENT: REVIEWER`
- `AUTO_REQUEST_ID: <request_id>`
- `AUTO_RUN_ID: <run_id>`
- `AUTO_ITERATION: <n>`
- `AUTO_REVIEW_STATUS: APPROVED|CHANGES_REQUIRED`
- `AUTO_TARGET_COMMIT: <sha>`

Reviewer timing rule:

- Reviewer must publish the contract-compliant comment immediately, even if CI is still `pending` or `in_progress`.
- Reviewer must not wait for CI completion before posting the contract output ("do not wait" rule).

`AUTO_REVIEW_STATUS` values are strict:

- `APPROVED`: builder lifecycle can terminate successfully.
- `CHANGES_REQUIRED`: builder must continue with a follow-up implementation cycle.

Approval token rule:

- If and only if status is `APPROVED`, include one separate line containing exactly `APPROVE`.
- If status is `CHANGES_REQUIRED`, do not include `APPROVE`.

Validation behavior:

- Missing/invalid tags, unknown status values, or `AUTO_TARGET_COMMIT` mismatch are hard failures.
- `execplan.post_completion` fetches the comment body from returned `comment_url` and requires the `APPROVE` token.

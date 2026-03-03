# PR Reviewer Daemon State Schema

Runtime state is persisted under:

- `.agents/skills/pr-autoloop/runtime/reviewer-daemon/`

## `state.json` fields

`scripts/reviewer_daemon.sh` writes daemon process state in:

- `runtime/reviewer-daemon/state.json`

Required fields:

- `pr_url` (string): target PR URL when known.
- `head_branch` (string): target head branch when PR URL is not yet known.
- `status` (string): `WAITING`, `RUNNING`, or `APPROVED`.
- `pid` (string): active daemon process id.
- `workdir` (string): repository working directory used by daemon.
- `updated_at` (string): UTC timestamp (`YYYY-MM-DDTHH:MM:SSZ`).

## Request envelope fields

Builder sends requests by writing:

- `runtime/reviewer-daemon/inbox/<request_id>.json`

Required fields:

- `request_id` (string): unique message id.
- `pr_url` (string): optional explicit target PR URL.
- `head_branch` (string): optional head branch fallback for PR discovery.
- `target_commit` (string): commit hash to review.
- `run_id` (string): lifecycle/run identifier.
- `iteration` (string): iteration label.
- `requested_at` (string): UTC timestamp.

## Response envelope fields

Reviewer daemon returns:

- `runtime/reviewer-daemon/responses/<request_id>.json`

Required fields:

- `request_id` (string): request id echoed back.
- `success` (boolean): request handling result.
- `error` (string): non-empty when `success=false`.
- `comment_url` (string): URL of reviewer PR comment when available.
- `review_status` (string): parsed `AUTO_REVIEW_STATUS` value.
- `approved_token_found` (boolean): whether `APPROVE` token was detected.
- `target_commit` (string): parsed `AUTO_TARGET_COMMIT` value.
- `responded_at` (string): UTC timestamp.

## Runtime layout

- `runtime/reviewer-daemon/reviewer.pid`: daemon pid file.
- `runtime/reviewer-daemon/state.json`: daemon runtime state.
- `runtime/reviewer-daemon/inbox/<request_id>.json`: incoming review requests.
- `runtime/reviewer-daemon/responses/<request_id>.json`: response payloads.
- `runtime/reviewer-daemon/requests/reviewer_prompt_<request_id>.md`: generated reviewer prompt snapshots.
- `runtime/reviewer-daemon/logs/daemon.log`: daemon lifecycle log.
- `runtime/reviewer-daemon/logs/reviewer_request_<request_id>.log`: per-request codex execution log.

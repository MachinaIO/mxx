# PR Auto Loop State Schema

Runtime state is persisted at:

- `.agents/skills/pr-autoloop/runtime/runs/<run_id>/state.json`

## Required fields

- `run_id` (string): unique loop execution id.
- `pr_url` (string): target PR URL.
- `pr_number` (string): PR number extracted from URL/API.
- `pr_branch` (string): PR head branch.
- `iteration` (number): current iteration (1-based).
- `max_iterations` (number): configured loop bound.
- `max_builder_failures` (number): configured consecutive builder failure bound.
- `consecutive_builder_failures` (number): current failure counter.
- `last_builder_commit` (string): last commit hash reviewed by reviewer.
- `last_reviewer_status` (string): `APPROVED`, `CHANGES_REQUIRED`, or empty.
- `status` (string): `RUNNING`, `APPROVED`, `FAILED`, `FAILED_LIMIT`, or `FAILED_CONTRACT`.
- `updated_at` (string): UTC timestamp (`YYYY-MM-DDTHH:MM:SSZ`).

## Runtime layout

- `runtime/locks/pr-<number>.lock`: per-PR lock file.
- `runtime/runs/<run_id>/logs/`: builder/reviewer command logs.
- `runtime/runs/<run_id>/feedback/`: reviewer feedback snapshots for subsequent builder iterations.

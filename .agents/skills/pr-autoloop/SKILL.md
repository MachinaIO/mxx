---
name: pr-autoloop
description: Run daemon-based builder/reviewer PR automation where the builder starts a reviewer daemon and exchanges machine-readable review requests/responses.
---

# PR Reviewer Daemon Automation

Use this skill to run builder/reviewer coordination through `scripts/reviewer_daemon.sh`.

## Inputs

Provide:

- target PR URL (`--pr-url`) when known, or head branch (`--head-branch`) when PR URL must be discovered
- target commit hash (`--commit`) for each reviewer request
- optional request metadata (`--request-id`, `--run-id`, `--iteration`)

## Workflow

1. When the user asks to start the automation, execute commands directly (do not only print instructions).
2. Because this workflow relies on `gh` commands, run daemon-start/request commands outside sandbox, following `.agents/skills/execplan-sandbox-escalation/SKILL.md`.
3. Run `scripts/doctor.sh` to validate local prerequisites and auth state.
4. Start or attach to reviewer daemon with `scripts/reviewer_daemon.sh --start`.
5. After each builder commit push, call `scripts/reviewer_daemon.sh --request --commit <sha>` and wait for response JSON.
6. Use response JSON (`comment_url`, `review_status`, `approved_token_found`) as the only machine-readable reviewer result.
7. Reviewer must post a contract-compliant comment immediately even when CI is still running.

### Start command patterns

Existing PR (known URL):

- `./.agents/skills/pr-autoloop/scripts/doctor.sh --pr-url <pr_url>`
- `./.agents/skills/pr-autoloop/scripts/reviewer_daemon.sh --start --pr-url <pr_url> --head-branch <branch>`

Branch-first mode (PR URL unknown at start):

- `./.agents/skills/pr-autoloop/scripts/doctor.sh --head-branch <branch>`
- `./.agents/skills/pr-autoloop/scripts/reviewer_daemon.sh --start --head-branch <branch>`

Per-commit review request:

- `./.agents/skills/pr-autoloop/scripts/reviewer_daemon.sh --request --commit <sha> [--pr-url <pr_url>] [--head-branch <branch>] [--run-id <id>] [--iteration <n>]`

## References

Read these files before changing contracts or parser logic:

- `references/comment_contract.md`
- `references/state_schema.md`

## Safety Constraints

- Keep reviewer role tags explicit (`AUTO_AGENT: REVIEWER`).
- Treat missing reviewer contract tags as hard failures; do not infer status from prose.
- Treat missing daemon response `comment_url` as a hard failure.
- When reviewer approval is reached, include `APPROVE` in the reviewer comment so lifecycle gate logic can terminate deterministically.

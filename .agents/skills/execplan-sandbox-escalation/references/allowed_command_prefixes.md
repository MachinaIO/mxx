# Allowed Out-of-Sandbox Command Prefixes

This allowlist is used by the `execplan-sandbox-escalation` skill.

Before requesting a new out-of-sandbox command approval, verify whether one of these prefixes can safely satisfy the task.

## Pre-approved prefixes

- `git status --short`
- `git branch --show-current`
- `git add -A`
- `git commit -m`
- `git push`
- `gh pr view`
- `gh pr checks`
- `gh pr comment`
- `gh pr create`
- `gh pr ready`
- `gh pr status`
- `gh pr edit`
- `mv docs/prs/active/`
- `mkdir -p docs/prs/active`

## Entry requirements for new prefixes

When adding a new prefix, include:

- exact prefix pattern,
- why existing prefixes were insufficient,
- why the new prefix is the safest reusable generalization.

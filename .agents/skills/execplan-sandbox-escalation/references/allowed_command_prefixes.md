# Allowed Out-of-Sandbox Command Prefixes

This allowlist is used by the `execplan-sandbox-escalation` skill.

Before requesting a new out-of-sandbox command approval, verify whether one of these prefixes can safely satisfy the task.

## Pre-approved prefixes

Policy note: `gh` prefixed commands are expected to run out-of-sandbox by default for stable GitHub API access.

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
- `gh api graphql`
- `mv docs/prs/active/`
- `mkdir -p docs/prs/active`
- `scripts/execplan_gate.sh --event execplan.pre_creation`
- `scripts/execplan_gate.sh --plan`
- `.agents/skills/pr-autoloop/scripts/run_builder_reviewer_doctor.sh`
- `.agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh`

## Deprecated prefixes (cleanup candidates)

- `scripts/run_builder_reviewer_doctor.sh` (obsolete after pr-autoloop skill path consolidation)
- `scripts/run_builder_reviewer_loop.sh` (obsolete after pr-autoloop skill path consolidation)

## Entry requirements for new prefixes

When adding a new prefix, include:

- exact prefix pattern,
- why existing prefixes were insufficient,
- why the new prefix is the safest reusable generalization.

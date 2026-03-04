# Allowed Out-of-Sandbox Command Prefixes

This allowlist is used by the `execplan-sandbox-escalation` skill.

Before requesting a new out-of-sandbox command approval, verify whether one of these prefixes can safely satisfy the task.

## Pre-approved prefixes

Policy note: `gh` prefixed commands are expected to run out-of-sandbox by default for stable GitHub API access.

Path note: prefixes beginning with `scripts/` are relative to the eternal-cycler installation root (injected into your prompt as `Path context`). Prefixes beginning with `eternal-cycler-out/` or `.agents/` are relative to the consuming repository root.

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
- `mv eternal-cycler-out/prs/active/`
- `mkdir -p eternal-cycler-out/prs/active`
- `scripts/execplan_gate.sh --event execplan.pre_creation`
- `scripts/execplan_gate.sh --plan`
- `scripts/run_builder_reviewer_doctor.sh`
- `scripts/run_builder_reviewer_loop.sh`

## Entry requirements for new prefixes

When adding a new prefix, include:

- exact prefix pattern,
- why existing prefixes were insufficient,
- why the new prefix is the safest reusable generalization.

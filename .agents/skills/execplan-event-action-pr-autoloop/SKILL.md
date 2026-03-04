---
name: execplan-event-action-pr-autoloop
description: Event skill for action.pr_autoloop verification. Use for fixed-script builder/reviewer autonomous loop contract enforcement checks.
---

# Event Skill: action.pr_autoloop

Runs deterministic validation for fixed loop scripts under `.agents/skills/pr-autoloop/scripts/` and enforces the non-interactive, direct-output loop contract.

## Script

- `scripts/run_event.sh --plan <plan_md>`

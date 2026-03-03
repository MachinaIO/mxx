---
name: execplan-event-action-docs-only
description: Event skill for action.docs_only verification. Use when an action modifies only docs/policy paths.
---

# Event Skill: action.docs_only

Validates docs-only action scope and policy hygiene:

- ensure changed files are documentation-only paths,
- scan for stale policy placeholders (`TODO|TBD|FIXME`) in documentation/policy files.

## Script

- `scripts/run_event.sh --plan <plan_md>`

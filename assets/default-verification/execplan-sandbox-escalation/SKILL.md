---
name: execplan-sandbox-escalation
description: Policy skill for out-of-sandbox command execution during ExecPlan actions and verification. Use this before requesting any new sandbox escalation.
---

# ExecPlan Sandbox Escalation

This skill defines the required workflow for out-of-sandbox command execution during ExecPlan action execution and validation.
This is a mandatory skill: do not execute any out-of-sandbox command before applying this skill workflow.

## When to use

Use this skill whenever a command cannot run inside sandbox constraints and out-of-sandbox execution is required.

## Workflow

1. Read `references/allowed_command_prefixes.md`.
2. Check whether the required operation can be implemented with an existing allowed prefix.
3. For any `gh` command family operation, prefer out-of-sandbox execution by default, even when a sandbox attempt might intermittently work.
4. Always execute `execplan.pre_creation` and `execplan.post_completion` gate commands out-of-sandbox. These lifecycle events are mandatory out-of-sandbox operations because they require stable GitHub API access.
5. If yes, run the existing allowed command path and continue.
6. If no, request human operator approval for the new out-of-sandbox command.
7. After approval, add the narrowest safely generalized prefix to `references/allowed_command_prefixes.md`.
8. Record the command usage and result in the current ExecPlan `Verification Ledger`. If you add a new prefix entry, also record rationale in the `Decision Log`.

## Safety constraints

- Prefer least-privilege prefixes over broad prefixes.
- Do not request a broad prefix when a narrower reusable prefix can satisfy the same task.
- Keep the allowlist maintainable by documenting why existing prefixes were insufficient.

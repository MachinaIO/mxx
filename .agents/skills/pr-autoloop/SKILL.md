---
name: pr-autoloop
description: When directly invoked by a human user, run this skill by default unless the user explicitly says not to.
---

# Skill: pr-autoloop

When directly invoked by a human user, run this skill by default unless the user explicitly says not to run it.

## Runtime scripts

- `.agents/skills/pr-autoloop/scripts/run_builder_reviewer_doctor.sh`
- `.agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh`

## Required behavior

1. Treat `run_builder_reviewer_loop.sh` as non-interactive.
2. Never invoke the loop script without `--task` or `--task-file`.
3. If task input is missing, stop and ask the user to provide task text or a task file path. Do not create a fallback task.
4. If `--pr-url` is already provided by the user, pass it through unchanged.
5. If `--pr-url` is not provided, inspect `docs/prs/active/*.md` regardless of current local branch:
   - If one or more active docs exist, ask whether to resume an existing tracked PR or create a new PR flow.
   - If multiple active docs exist and user chooses resume, show a numbered list and ask which doc to use.
6. Resume flow rule:
   - Read `- branch name:` from the selected doc (fallback: title `# PR Tracking: <branch>`).
   - Switch to that branch before invoking the loop:
     - if local branch exists: `git switch <branch>`;
     - else if `origin/<branch>` exists: `git switch -c <branch> --track origin/<branch>`;
     - else: stop and ask the user for guidance.
   - Read `- PR link:` from the selected doc.
   - If `PR link` is non-empty and not `(not available locally)`, pass `--pr-url <that_link>`.
   - If `PR link` is missing, run without `--pr-url` on the switched branch.
7. New PR flow rule (when user selects new PR flow, or when no active PR doc exists):
   - Create and switch to a new branch before invoking the loop.
   - Branch naming must be task-derived and deterministic:
     - `task_seed`: first non-empty line from task text or task file.
     - `task_slug`: lowercase, replace non `[a-z0-9]` with `-`, collapse repeated `-`, trim leading/trailing `-`, default `task`.
     - `branch_name`: `feat/auto-${task_slug:0:40}-$(date -u +%Y%m%d%H%M%S)`.
     - If name already exists locally or on origin, append `-1`, `-2`, ... until unique.
   - Run `git switch -c <branch_name>` and then run doctor+loop on that branch.
8. Run doctor before loop using the same resolved target (`--pr-url` if available, otherwise `--head-branch <current_branch_after_switch>`).
9. Forward loop output directly to caller stdout/stderr. Do not add tee/log-file capture.

## Suggested invocation template

If PR URL is resolved:

    git switch <resume-branch-from-doc>
    .agents/skills/pr-autoloop/scripts/run_builder_reviewer_doctor.sh --pr-url <url>
    .agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh --task-file <task.md> --pr-url <url>

If PR URL is not resolved:

    git switch -c <task-derived-branch>
    .agents/skills/pr-autoloop/scripts/run_builder_reviewer_doctor.sh --head-branch <current_branch_after_switch>
    .agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh --task-file <task.md>

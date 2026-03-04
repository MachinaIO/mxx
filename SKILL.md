---
name: eternal-cycler
description: A skill for automating the loop of interactions between the implementation builder agent and the implementation reviewer agent.
---

# Skill: eternal-cycler

A skill for automating the loop of interactions between the implementation builder agent and the implementation reviewer agent.

## Runtime scripts

All paths are relative to this SKILL.md file's location (the eternal-cycler installation root).

- `scripts/run_builder_reviewer_doctor.sh`
- `scripts/run_builder_reviewer_loop.sh`

## Required behavior

### Phase 0: Plan selection (always runs first)

1. Before resolving task or PR, inspect `eternal-cycler-out/plans/active/` for existing plan documents (`.md` files).
2. If one or more active plan documents exist:
   - Ask the user: resume an existing plan, or start a new one?
   - If multiple plans exist, show a numbered list (filename + first heading of each) and ask which to resume, or "new" to start fresh.
3. If the user chooses to resume a plan, follow the **Resume plan flow** below.
4. If the user chooses new (or no active plans exist), follow the **New plan flow** below.

### Resume plan flow

5. Read the selected plan document.
6. Ask the user if there are any modifications to the plan (updated Progress actions, new verify_events, operator feedback). If modifications are provided, apply them to the plan document before continuing.
7. Determine the branch:
   - Read `execplan_start_branch:` from the plan.
   - Switch to that branch:
     - if local branch exists: `git switch <branch>`;
     - else if `origin/<branch>` exists: `git switch -c <branch> --track origin/<branch>`;
     - else: stop and ask the user for guidance.
8. Determine the PR:
   - Read `pr_tracking_doc:` from the plan → open that tracking document → read `- PR link:`.
   - If `PR link` is non-empty and not `(not available locally)`:
     - Check PR state: `gh pr view <url> --json state --jq '.state'`
     - If state is `OPEN`: pass `--pr-url <that_link>` to the loop.
     - If state is `MERGED` or `CLOSED`: do not pass `--pr-url` (a new PR will be created on the current branch by the loop).
   - If `PR link` is missing or `pr_tracking_doc` is absent from the plan: do not pass `--pr-url`.
9. Compose the task for the builder agent:
    - Prepend the following to whatever task text the user provided:
      ```
      You are resuming the ExecPlan at <plan_md>. Read that document in full before taking any action.
      Do NOT create a new ExecPlan. Do NOT modify any other plan document in eternal-cycler-out/plans/.
      ```
    - If the user provided task text or a task file, append it after the above preamble.
    - If the user provided no task, use the following default task body (after the preamble):
      ```
      Execute all incomplete actions listed in the Progress section of the plan (unchecked checkboxes).
      Follow the ExecPlan lifecycle from PLANS.md, starting at step 3 (execute actions).
      ```
    - Pass the composed task as `--task <text>` (or write it to a temp file and use `--task-file`).
10. Run doctor before loop using the same resolved target (`--pr-url` if determined in step 8, otherwise `--head-branch <current_branch_after_switch>`).
11. Invoke the loop. Forward loop output directly to caller stdout/stderr.

### New plan flow

12. Resolve task input: if task text or task file is not provided, stop and ask. Do not create a fallback task. Never invoke the loop script without `--task` or `--task-file`.
13. If `--pr-url` is already provided by the user, pass it through unchanged and skip steps 14–16.
14. Inspect `eternal-cycler-out/prs/active/*.md` regardless of current local branch:
    - If one or more active docs exist, ask whether to resume an existing tracked PR or create a new PR flow.
    - If multiple active docs exist and user chooses resume, show a numbered list and ask which doc to use.
15. Resume PR flow rule (user chose to resume an existing tracked PR):
    - Read `- branch name:` from the selected doc (fallback: title `# PR Tracking: <branch>`).
    - Switch to that branch (same rules as step 7).
    - Read `- PR link:` from the selected doc.
    - If `PR link` is non-empty and not `(not available locally)`, pass `--pr-url <that_link>`.
    - If `PR link` is missing, run without `--pr-url` on the switched branch.
16. New PR flow rule (user chose new PR flow, or no active PR doc exists):
    - Create and switch to a new branch before invoking the loop.
    - Branch naming must be task-derived and deterministic:
      - `task_seed`: first non-empty line from task text or task file.
      - `task_slug`: lowercase, replace non `[a-z0-9]` with `-`, collapse repeated `-`, trim leading/trailing `-`, default `task`.
      - `branch_name`: `feat/auto-${task_slug:0:40}-$(date -u +%Y%m%dT%H%M%S)`.
      - If name already exists locally or on origin, append `-1`, `-2`, ... until unique.
    - Run `git switch -c <branch_name>`.
17. Compose the task for the builder agent:
    - Prepend the following to the user's task text:
      ```
      You are starting a new ExecPlan. Create a new plan document in eternal-cycler-out/plans/active/.
      Do NOT modify or resume any existing plan document in eternal-cycler-out/plans/.
      Run execplan.post_creation gate immediately after writing the new plan. (See PLANS.md lifecycle step 2.)
      ```
    - Append the user's task text after the above preamble. Pass as `--task <text>` or `--task-file`.
18. Run doctor before loop using the same resolved target (`--pr-url` if available, otherwise `--head-branch <current_branch_after_switch>`).
19. Invoke the loop. Forward loop output directly to caller stdout/stderr.

### Always

20. Treat `run_builder_reviewer_loop.sh` as non-interactive.
21. Each builder and reviewer agent invocation enforces a minimum runtime of 1 hour (3600 seconds). The loop script enforces this automatically by sleeping for the remaining time after the agent exits. Do not pass `--min-agent-duration 0` or any value below 3600 unless the user explicitly requests it.

## Suggested invocation template

Resolve the eternal-cycler installation path first (the directory containing this SKILL.md).
Then invoke using that path as the prefix.

Resume existing plan (PR still open):

    SKILL_ROOT=<path-to-eternal-cycler>
    git switch <execplan_start_branch>
    $SKILL_ROOT/scripts/execplan_gate.sh --plan <plan_md> --event execplan.resume
    $SKILL_ROOT/scripts/run_builder_reviewer_doctor.sh --pr-url <url>
    $SKILL_ROOT/scripts/run_builder_reviewer_loop.sh --task-file <task.md> --pr-url <url>

Resume existing plan (PR merged/closed — new PR will be created):

    SKILL_ROOT=<path-to-eternal-cycler>
    git switch <execplan_start_branch>
    $SKILL_ROOT/scripts/execplan_gate.sh --plan <plan_md> --event execplan.resume
    $SKILL_ROOT/scripts/run_builder_reviewer_doctor.sh --head-branch <execplan_start_branch>
    $SKILL_ROOT/scripts/run_builder_reviewer_loop.sh --task-file <task.md>

New plan (PR URL provided or resolved from active PR docs):

    SKILL_ROOT=<path-to-eternal-cycler>
    $SKILL_ROOT/scripts/execplan_gate.sh --event execplan.pre_creation
    git switch <resume-branch-from-doc>
    $SKILL_ROOT/scripts/run_builder_reviewer_doctor.sh --pr-url <url>
    $SKILL_ROOT/scripts/run_builder_reviewer_loop.sh --task-file <task.md> --pr-url <url>

New plan (no PR URL — builder creates ExecPlan + runs post_creation gate):

    SKILL_ROOT=<path-to-eternal-cycler>
    $SKILL_ROOT/scripts/execplan_gate.sh --event execplan.pre_creation
    git switch -c <task-derived-branch>
    $SKILL_ROOT/scripts/run_builder_reviewer_doctor.sh --head-branch <current_branch_after_switch>
    $SKILL_ROOT/scripts/run_builder_reviewer_loop.sh --task-file <task.md>

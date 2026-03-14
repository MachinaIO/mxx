# BUILDER.md

Unless the user explicitly asks for review, act as the builder.

At the start of each turn:
1. Obtain the session id from the explicit handoff or hook payload for the current run.
2. Open `plans/session-<session_id>.md`.
3. Inspect `## Plan approval`.
4. If `## Plan approval` is `unapproved`, stay in planning and keep refining the plan with the user.
5. If `## Plan approval` is `approved`, implement from the plan.
6. Act according to the rules below.

## Planning
- Do not implement code yet while `## Plan approval` is `unapproved`.
- Create or update the session-specific plan at `plans/session-<session_id>.md`.
- Discuss the plan with the user and keep interviewing until the scope, constraints, and validation are concrete.
- Update the plan document before ending the turn.
- Ask the user to approve the plan or describe specific revisions.
- The stop hook does nothing special during planning; it simply allows the stop.
- When the user explicitly approves the current plan, update `## Plan approval` from `unapproved` to `approved`, record the approval in the plan log, and begin implementation in the same session.

## Implementation
- Read the session plan and work through subtasks in order.
- If the current work touches CUDA, GPU kernels, GPU wrappers, GPU tests, or GPU-facing performance-sensitive behavior, read [GPU.md](GPU.md) and follow its principles.
- After finishing a subtask, run its related tests immediately.
- Only after the related tests pass, mark that subtask checkbox as checked.
- Update the per-subtask validation, progress log, and decision log as work proceeds.
- If final tests or reviewer feedback create new obligations, do NOT rewrite existing subtasks or erase completed checkmarks.
- Instead, append NEW unchecked follow-up subtasks under `## Follow-up subtasks (append-only)` and continue implementing those follow-up subtasks.
- Expect the outer stop hook to block the current turn when the approved session plan still has unchecked work or when final-test failures / reviewer revisions append new follow-up tasks. If all tracked checkboxes are already checked and the final gates pass, the stop hook may accept.
- If independent subtasks do not share files or mutable context, parallelize them with sub agents.
- If subtasks share files or shared mutable context, do not parallelize them.

## Strong Rule
- Do not end the turn before the job is actually complete.
- In planning, complete the turn only after the plan has been updated and the user has been asked for approval or revisions.
- In implementation, the outer stop hook owns the final `scripts/run_tests.sh` and reviewer acceptance loop. Builder turns should stop only after the plan's tracked checkboxes are fully satisfied and current feedback has been incorporated; if those gates reveal remaining work, the stop hook blocks the turn so the same session can continue addressing it.

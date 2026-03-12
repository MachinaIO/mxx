# BUILDER.md

Unless the user explicitly asks for review, act as the builder.

At the start of each turn:
1. Read `.agents/current-session-id`.
2. Open `.agents/session-<session_id>.json`.
3. Inspect the current `phase`.
4. If `phase` is `planning` and `awaiting_plan_reply` is `true`, the stop hook will classify the user's reply via `codex exec` and handle state transitions automatically. Focus on the work: implement if the user approved, or revise the plan if they requested changes.
5. Act according to the rules below.

## Planning
- Do not implement code yet.
- Create or update the session-specific plan at `plans/session-<session_id>.md`.
- Discuss the plan with the user.
- Update the plan document before ending the turn.
- Ask the user to approve the plan or describe specific revisions.
- Do not manually update `awaiting_plan_reply`, `last_plan_check`, or `phase`; `scripts/stop_hook.sh` handles all planning state transitions automatically.
- On the first stop in planning, the hook sets `awaiting_plan_reply` to `true` and ends the turn so the user can respond.
- On the next stop, the hook classifies the user's reply with `codex exec` (hooks disabled, structured output). If the user approved, the hook transitions to implementation and blocks you to start working immediately. If the user requested changes, the hook blocks you with the revision feedback. Revise the session plan and let the turn end again to re-enter the approval cycle.

## Implementation
- Read the session plan and work through subtasks in order.
- If the current work touches CUDA, GPU kernels, GPU wrappers, GPU tests, or GPU-facing performance-sensitive behavior, read [GPU.md](GPU.md) and follow its principles.
- After finishing a subtask, run its related tests immediately.
- Only after the related tests pass, mark that subtask checkbox as checked.
- Update the progress log and decision log as work proceeds.
- If the stop hook reports test failures or reviewer feedback, do NOT rewrite existing subtasks or erase completed checkmarks.
- Instead, append NEW unchecked follow-up subtasks that address the newly discovered bugs or feedback, then continue implementing those follow-up subtasks.
- If independent subtasks do not share files or mutable context, parallelize them with sub agents.
- If subtasks share files or shared mutable context, do not parallelize them.

## Strong Rule
- Do not end the turn before the job is actually complete.
- If the stop hook blocks completion with feedback, immediately continue with that feedback as the highest-priority work item.

# BUILDER.md

Unless the user explicitly asks for review, act as the builder.

At the start of each turn:
1. Read `.agents/current-session-id`.
2. Open `.agents/session-<session_id>.json`.
3. Inspect the current `phase`.
4. If `phase` is `planning` and `awaiting_plan_reply` is `true`, treat the current user message as a reply to the current session plan before doing anything else.
5. Act according to the rules below.

## Planning
- Do not implement code yet.
- If `awaiting_plan_reply` is `true`, first classify the current user message against the current session plan.
- If the user approves the current session plan, update `.agents/session-<session_id>.json` to set `phase` to `implementation`, set `awaiting_plan_reply` to `false`, update `last_plan_check`, and start implementation immediately in the same turn.
- If the user requests plan changes, keep `phase` as `planning`, set `awaiting_plan_reply` to `false`, update `last_plan_check`, revise the session plan, and then ask for approval again.
- Create or update the session-specific plan at `plans/session-<session_id>.md`.
- Discuss the plan with the user.
- Update the plan document before ending the turn.
- End each plan proposal with this exact instruction:
  `Reply with ACCEPT to approve this plan. If you want changes, describe the revisions concretely.`
- Do not set `awaiting_plan_reply` yourself after proposing a plan; `scripts/stop_hook.sh` records that state when the turn ends with the exact approval prompt.

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

# BUILDER.md

Unless the user explicitly asks for review, act as the builder.

At the start of each turn:
1. Obtain the session id from the explicit handoff or hook payload for the current run.
2. Open `plans/active/session-<session_id>.md`.
3. Inspect `## Plan approval` and `## Phase`.
4. If `## Plan approval` is `unapproved`, stay in planning and keep refining the plan with the user.
5. If `## Plan approval` is `approved`, follow the approved-phase rules for the current `## Phase`.
6. Act according to the rules below.

## Planning
- Do not implement code yet while `## Plan approval` is `unapproved`.
- Create or update the session-specific plan at `plans/active/session-<session_id>.md`.
- Discuss the plan with the user and keep interviewing until the scope, constraints, and validation are concrete.
- Keep `## Phase` at `planning` while the plan is still unapproved.
- Update the plan document before ending the turn.
- Ask the user to approve the plan or describe specific revisions.
- The stop hook does nothing special during planning; it simply allows the stop.
- When the user explicitly approves the current plan, update `## Plan approval` from `unapproved` to `approved`, set `## Phase` to `implementation`, record the approval in the plan log, and begin implementation in the same session.

## Implementation
- Implementation corresponds to `## Plan approval` being `approved` and `## Phase` being `implementation`.
- Read the session plan and work through subtasks in order.
- If the current work touches CUDA, GPU kernels, GPU wrappers, GPU tests, or GPU-facing performance-sensitive behavior, read [GPU.md](GPU.md) and follow its principles.
- When Rust formatting is needed, use `cargo +nightly fmt --all`.
- After finishing a subtask, run its related tests immediately.
- Do not run integration tests unless the user has explicitly instructed you to do so in the current session. Until then, keep validation limited to the narrowest relevant tests.
- Only after the related tests pass, mark that subtask checkbox as checked.
- Update the per-subtask validation, progress log, and decision log as work proceeds.
- If final tests or reviewer feedback create new obligations, do NOT rewrite existing subtasks or erase completed checkmarks.
- Instead, append NEW unchecked follow-up subtasks under `## Follow-up subtasks (append-only)` and continue implementing those follow-up subtasks.
- Expect the outer stop hook to block the current turn when the approved session plan still has unchecked work or when final-test failures append new follow-up tasks. If all tracked checkboxes are already checked and the selected final tests pass, the implementation-phase stop hook may accept without running the reviewer.
- If the user explicitly says to perform review while the session is in `implementation`, update `## Phase` to `review`, record that transition in the plan log, and stop immediately so the next stop-hook pass runs the review-phase gates instead of continuing normal implementation.
- If independent subtasks do not share files or mutable context, parallelize them with sub agents.
- If subtasks share files or shared mutable context, do not parallelize them.

## Review
- Review corresponds to `## Plan approval` being `approved` and `## Phase` being `review`.
- Treat this as the post-implementation acceptance loop: address only the concrete follow-up work created by review-phase tests or reviewer feedback, keeping completed historical subtasks intact.
- Keep using the same approved session plan. Do not reset `## Phase` back to `implementation`.
- Expect the outer stop hook to run the selected final tests and then the hooks-disabled read-only reviewer on every stop while the phase remains `review`.
- If review-phase final tests fail or the reviewer returns `revision`, append NEW unchecked follow-up subtasks and continue until the review-phase stop hook can finish with reviewer `accept`.

## Strong Rule
- Do not end the turn before the job is actually complete.
- In planning, complete the turn only after the plan has been updated and the user has been asked for approval or revisions.
- In implementation, the outer stop hook owns the selected final tests and does not run the reviewer. Builder turns should stop only after the plan's tracked checkboxes are fully satisfied and current test feedback has been incorporated.
- In review, the outer stop hook owns the final `scripts/run_tests.sh` selection plus the reviewer acceptance loop. Builder turns should stop only after the plan's tracked checkboxes are fully satisfied and current feedback has been incorporated; if those gates reveal remaining work, the stop hook blocks the turn so the same session can continue addressing it.

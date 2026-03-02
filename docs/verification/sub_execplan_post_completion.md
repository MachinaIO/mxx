# Event: After Sub ExecPlan Completion

Use this document after all actions in a `sub ExecPlan` are complete.

## Preconditions

- Working directory: repository root (`.`).
- The target sub ExecPlan document is updated to completed state.
- The sub ExecPlan document includes a repository-relative link to its parent main ExecPlan.

## Required actions

1. Open the completed sub ExecPlan and review completion evidence.

    Confirm at least:

    - `Progress` reflects all sub-plan actions complete,
    - `Outcomes & Retrospective` summarizes delivered scope,
    - recorded validation outcomes for this sub plan are present.

2. Apply the parent main plan's validation strategy for this sub plan.

    Follow whichever strategy the parent main plan declared for this sub-plan scope:

    - run per-action validation adjacent to sub-plan actions, or
    - mark this sub plan as completed and pending for a later aggregated validation step in the parent main plan.

3. Update the parent main ExecPlan with this sub-plan result.

    In the parent plan, update at least:

    - `Progress` status for the mapped parent action,
    - links/references to the completed sub-plan path,
    - any required `Surprises & Discoveries` or `Decision Log` notes caused by sub-plan execution.

4. Move the completed sub ExecPlan document from `docs/plans/active/` to `docs/plans/completed/`.

5. End sub-agent execution by returning control to the parent main-plan scope.

    The sub agent must not ask the user for next steps. Handoff is complete once parent-plan updates are recorded.

## Success criteria

- Sub-plan completion evidence is recorded in the sub plan.
- Parent validation strategy is followed (immediate or aggregated).
- Parent main-plan `Progress` reflects sub-plan completion status.
- Completed sub plan is moved to `docs/plans/completed/`.
- No user interaction is requested during sub-to-main handoff.

## Failure triage

- If parent-plan linkage is missing in the sub plan, add it and rerun this event.
- If parent `Progress` cannot be updated due to missing mapped scope, update parent scope mapping first, then rerun.
- If the plan move fails because of path mismatch, locate the correct sub-plan filename and update references before retry.

## Evidence to record

- Sub ExecPlan path used for this event.
- Parent main ExecPlan path used for handoff.
- Whether validation was immediate or aggregated in parent scope.
- Parent `Progress` update entry for this sub plan.
- File move command/result for moving the sub plan to `docs/plans/completed/`.

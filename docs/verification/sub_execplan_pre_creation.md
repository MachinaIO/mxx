# Event: Before Sub ExecPlan Creation

Use this document before creating a new `sub ExecPlan` file.

## Preconditions

- Working directory: repository root (`.`).
- A parent `main ExecPlan` exists under `docs/plans/active/`.
- Do not change implementation files during this event. Metadata edits required for sub-plan setup are allowed.
- This event runs before adding a new sub-plan file under `docs/plans/active/`.

## Required actions

1. Open the parent main ExecPlan and identify the exact action or milestone that will be decomposed.

    Example check:

        rg -n "## Progress|sub ExecPlan|parallel|sequential" docs/plans/active/<main_plan_file>.md -S

2. Record decomposition scope for the upcoming sub plan.

    Capture at least:

    - parent main-plan path,
    - mapped parent action/milestone,
    - expected deliverable of this sub plan.

3. Confirm execution topology and agent assignment from the parent main plan.

    Confirm at least:

    - whether this sub plan is parallel or sequential relative to sibling sub plans,
    - required predecessor/successor ordering when sequential,
    - assigned sub-agent type when multiple sub-agent types are defined.

4. Verify that the sub-plan scope fits within one sub-agent context window.

    If it does not fit, split it into smaller sibling sub ExecPlans before authoring implementation steps.

5. Create the new sub-plan file under `docs/plans/active/` and include required linkage metadata in that plan.

    The sub plan must include:

    - a repository-relative link to the parent main ExecPlan,
    - explicit mapped scope within the main ExecPlan,
    - assigned sub-agent type when relevant,
    - a completion handoff note that results will be reported back to the parent main plan (not to the user).

## Success criteria

- Parent main-plan mapping is explicit before sub-plan creation.
- Parallel/sequential placement and ordering are explicit.
- Sub-agent type assignment is explicit when relevant.
- Sub-plan scope is bounded for a single sub-agent context window.
- The newly created sub-plan document includes required parent linkage and mapped-scope metadata.

## Failure triage

- If parent scope mapping is ambiguous, update the parent main plan first, then rerun this event.
- If dependencies among sibling sub plans are unclear, mark the uncertainty in the parent `Decision Log` and resolve execution order before creating the sub plan.
- If scope is too large, split into additional sub plans and update parent `Progress` topology.

## Evidence to record

- Parent main ExecPlan path.
- Parent action/milestone mapped to this sub plan.
- Parallel/sequential classification and ordering notes.
- Assigned sub-agent type (if applicable).
- New sub-plan file path under `docs/plans/active/`.

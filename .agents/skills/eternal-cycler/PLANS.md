# Codex Execution Plans (ExecPlans)

## Global Requirements

- All documentation, git commit messages, and PRs must be written in English.
- When documenting file paths, use only paths relative to the repository top directory. Do not write absolute paths in documentation.

An execution plan ("ExecPlan") is a design document that a coding agent follows to deliver a working feature or system change. Treat the reader as a complete beginner: they have only the current working tree and the single ExecPlan file. There is no memory of prior plans and no external context.

## Behavior when working with ExecPlans

- **Read PLANS.md in full** before authoring or implementing. Start a new plan from the skeleton below.
- **Proceed autonomously.** Do not prompt the user for "next steps." Resolve ambiguities in the plan itself; do not pause for confirmation unless the escalation bound is reached.
- **Mark progress immediately.** When a `Progress` action finishes, mark it `[x]` before starting the next action. Commit frequently.
- **One plan per objective.** Merge sub-agent outcomes into the same ExecPlan; do not create separate sub-plan lifecycle documents.
- **Record decisions.** All design choices go into the `Decision Log` with rationale. It must always be possible to restart from only the ExecPlan.
- **Research deeply.** Use prototyping milestones for risky or unknown requirements; read library source code; embed all required knowledge in the plan itself rather than linking to external docs.

## Requirements

Every ExecPlan must satisfy these non-negotiable requirements:

* **Self-contained and novice-enabling.** In its current form it contains all knowledge needed for a complete novice to implement the feature end-to-end without prior knowledge of this repository.
* **Living document.** Revise it as progress is made, as discoveries occur, and as decisions are finalized. Each revision must remain fully self-contained.
* **Demonstrably working.** The plan must produce observable behavior, not merely code changes that "meet a definition."
* **Plain language.** Define every term of art immediately. Do not say "as defined previously" or "see the architecture doc."

State purpose and intent first: explain why the work matters, what someone can do after the change that they could not do before, and how to observe it working. Then provide the exact steps to achieve that outcome.

The executing agent can list files, read files, search, run the project, and run tests. It has no prior context. Repeat any assumption you rely on. If this plan builds on a prior checked-in ExecPlan, incorporate it by reference; otherwise include all relevant context.

## Formatting

Each ExecPlan is one single fenced code block labeled `md` (triple backticks). Do not nest additional triple-backtick fences inside; show commands, transcripts, diffs, or code as indented blocks. When writing an ExecPlan to a standalone `.md` file, omit the triple backticks.

Write in plain prose. Prefer sentences over lists. Checklists are permitted only in `Progress`, where they are mandatory. Narrative sections must be prose-first.

## Guidelines

**Resolve ambiguity in the plan, not in execution.** Do not outsource key decisions to the reader. When ambiguity exists, choose a path and explain why. Err toward over-explaining user-visible effects and under-specifying implementation details.

**Anchor with observable outcomes.** Phrase acceptance as behavior a human can verify ("after starting the server, navigating to http://localhost:8080/health returns HTTP 200 with body OK"), not internal attributes. For internal changes, show how to observe the effect through tests or a scenario.

**Specify repository context explicitly.** Name files by full repository-relative paths. Name functions and modules precisely. Show the working directory and exact command line for every command. When outcomes depend on environment, state the assumptions.

**Be idempotent and safe.** Write steps that can be run multiple times without drift. For risky or destructive steps, provide a retry path or rollback.

**Validation is mandatory.** Include test commands, expected outputs, and a short end-to-end scenario. State exact test commands and how to interpret results. Show how to prove the change is effective beyond compilation.

**Capture evidence.** Include terminal output, short diffs, or logs as indented examples inside the plan fence.

## Milestones

Milestones are narrative, not bureaucracy. Each milestone introduces its scope in one paragraph — goal, work, result, proof — and must be independently verifiable and incrementally implement the overall goal. Never abbreviate a milestone for the sake of brevity; do not omit details that could be crucial for a future implementation.

Milestones and Progress serve different roles: milestones tell the story, Progress tracks granular work. Both must exist.

## Prototyping milestones and parallel implementations

Include explicit prototyping milestones when they de-risk a larger change (feasibility probes, toy implementations, spikes). Keep prototypes additive and testable. Label scope as "prototyping," describe how to run and observe results, and state the criteria for promoting or discarding the prototype.

Prefer additive changes followed by subtractions that keep tests passing. Parallel implementations are acceptable during migration — describe how to validate both paths and how to retire one safely.

## Living plans and design decisions

ExecPlans must contain and maintain these sections (all are mandatory):

* `Progress` — checkbox list, always reflecting the current state
* `Verification Ledger` — all gate attempt records
* `Surprises & Discoveries` — unexpected behaviors, bugs, or insights with short evidence
* `Decision Log` — every decision with rationale, date, and author
* `Outcomes & Retrospective` — summary at major milestones and at completion

If you change course mid-implementation, document why in `Decision Log` and reflect the implications in `Progress`.

## Plan file locations

* Active plans → `eternal-cycler-out/plans/active/`
* Completed plans → `eternal-cycler-out/plans/completed/`
* Technical debt → `eternal-cycler-out/plans/tech-debt/`

Files in `tech-debt/` must include explicit repository-relative markdown links to the related plans (active or completed) that introduced, mitigated, or depend on that debt.

## Action-Level Parallel Execution

One lifecycle, one ExecPlan per objective. For large work, decompose into action-level units inside that one plan; do not create separate sub-plan lifecycle documents.

The `Progress` section must document execution topology for each delegated action:

* `action_id`, `mode` (`serial` or `parallel`), `depends_on`, `file_locks`, `verify_events`, `worker_type`

An action is parallelizable only when all `depends_on` actions are complete and `file_locks` for concurrent actions are disjoint. Delegate only `mode=parallel` actions to sub agents. The parent ExecPlan owner is responsible for conflict resolution and merging outcomes.

## Verification Policy

Verification is enforced by the gate script and event-local skills.

**Path note:** Gate script path is relative to the eternal-cycler installation root (shown in "Path context" in your prompt). Verification skill, plan, and PR tracking paths are relative to the consuming repository root.

Operational source of truth:

* `.agents/skills/execplan-event-index/references/event_skill_map.tsv`
* each mapped event skill under `.agents/skills/execplan-event-*/`
* `.agents/skills/execplan-sandbox-escalation/SKILL.md` and `.agents/skills/execplan-sandbox-escalation/references/allowed_command_prefixes.md`
* `scripts/execplan_gate.sh`

Templates (do not edit directly; edit copies in `.agents/skills/`): `assets/default-verification/execplan-event-*/`

### Event model

Event membership is data-driven by `.agents/skills/execplan-event-index/references/event_skill_map.tsv`.

Mandatory lifecycle events (must always be in the map):

* `execplan.pre_creation`
* `execplan.post_creation`
* `execplan.resume`
* `execplan.post_completion`

Action events are flexible and may be added or removed without changing this policy text, as long as they are registered in the event map. To add a new event, add a skill directory under `.agents/skills/` and register it in the event map.

### Enforcement model

Gate command: `scripts/execplan_gate.sh --event <event_id> [--plan <plan_md>] [--attempt <n>]`

The gate blocks lifecycle progress when any previously attempted event remains in `fail` or `escalated` state, and additionally:

* rejects lifecycle events (`execplan.pre_creation`, `execplan.post_creation`, `execplan.resume`, `execplan.post_completion`) in `Progress` action `verify_events`
* blocks `execplan.post_completion` until `execplan.post_creation` or `execplan.resume` has a `pass` entry, and every action event in `verify_events` has at least one `pass` entry in the Verification Ledger

### Retry and escalation

Max attempts per event: **3**. If the bound is exceeded, the gate marks the event `escalated` and stops progress. After escalation:

* document failure explicitly in `Progress`, `Verification Ledger`, and `Outcomes & Retrospective`
* force-close the current plan as failed (move to `eternal-cycler-out/plans/completed/`)
* resume only via a new ExecPlan in `eternal-cycler-out/plans/active/` seeded from human-operator feedback referencing the failed plan

### Sandbox escalation policy

Before running any out-of-sandbox command:

1. Read `.agents/skills/execplan-sandbox-escalation/SKILL.md`
2. Check `.agents/skills/execplan-sandbox-escalation/references/allowed_command_prefixes.md`
3. Use an existing allowed prefix when possible
4. If none apply, request human approval and add a safely generalized prefix to the reference file

Skipping this step is a policy violation that blocks lifecycle progress.

### Evidence policy

Record every gate attempt in the plan's `## Verification Ledger` with:

`event_id`, `attempt`, `status` (`pass`/`fail`/`escalated`), `commands`, `failure_summary`, `started_at`, `finished_at`

Do not store verification logs in separate temporary files.

## ExecPlan Lifecycle

Two paths depending on whether you are starting a new plan or resuming an existing one.

### New plan

1. **Pre-creation gate** — run out-of-sandbox before creating the plan document:
   `scripts/execplan_gate.sh --event execplan.pre_creation`
   Validates environment (branch, working tree). No ledger entry is written because the plan file does not exist yet.
2. **Create plan and post-creation gate** — create the plan document in `eternal-cycler-out/plans/active/`, write the full plan, and define all `Progress` action metadata (`action_id`, `mode`, `depends_on`, `file_locks`, `verify_events`, `worker_type`; `verify_events` must contain only `action.*` IDs; lifecycle events must never appear in `verify_events`). Then immediately run:
   `scripts/execplan_gate.sh --plan <plan_md> --event execplan.post_creation`
   This records the start snapshot, creates the PR tracking doc, and writes the plan linkage metadata that `execplan.post_completion` requires.
3. **Execute actions** in dependency order. Mark each `[x]` immediately when complete. Out-of-sandbox commands must follow the sandbox escalation policy.
4. **Verify after each action** — run `scripts/execplan_gate.sh` for each event in `verify_events`. The gate blocks progress on failure.
5. **Record all gate attempts** in `## Verification Ledger`.
6. **On failure** — retry up to 3 attempts. On escalation:
   * document failure in `Progress`, `Verification Ledger`, and `Outcomes & Retrospective`
   * move the plan to `eternal-cycler-out/plans/completed/` as failed
   * stop; resume only via a new ExecPlan after operator feedback
7. **Finalize plan** after all actions pass:
   * update all living-document sections (`Progress`, `Surprises & Discoveries`, `Decision Log`, `Outcomes & Retrospective`)
   * note which verification scripts were referenced, created, modified, or left unchanged, and why
   * move the plan to `eternal-cycler-out/plans/completed/`
   * ensure all implementation commits are pushed
   * add tech-debt follow-up plans if needed
8. **Post-completion gate** — run out-of-sandbox:
   * `scripts/execplan_gate.sh --plan <completed_plan_md> --event execplan.post_completion`
   * `execplan.post_completion` is validation-only: no `git add`, `git commit`, or `git push`
   * On failure: the gate script rolls the plan back to `eternal-cycler-out/plans/active/` before retrying. Use the current plan path (which may be in `active/` after rollback) when invoking the gate on retry. On escalation: same as step 6 — document failure, leave the plan in `eternal-cycler-out/plans/completed/` as failed, and stop.
   * Lifecycle complete.

### Resume existing plan

1. **Select plan** from `eternal-cycler-out/plans/active/`. If there is operator feedback, update `Progress` actions and add `verify_events` entries for any newly registered events.
2. **Resume gate** — run out-of-sandbox:
   `scripts/execplan_gate.sh --plan <plan_md> --event execplan.resume`
   Validates that the current branch matches the plan's recorded start branch, refreshes the PR tracking doc, and appends a resume record to the plan.
3. Continue from step 3 of the new-plan path (execute actions, verify, finalize, post-completion gate).

## Skeleton of a Good ExecPlan

    # <Short, action-oriented description>

    This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

    If PLANS.md is checked into this repo, reference its path here from the repository root and note that this document must be maintained in accordance with PLANS.md.

    ## Purpose / Big Picture

    Explain in a few sentences what someone gains after this change and how they can see it working. State the user-visible behavior you will enable.

    ## Progress

    Use a list with checkboxes to summarize granular steps. Every stopping point must be documented here, even if it requires splitting a partially completed task into two ("done" vs. "remaining"). This section must always reflect the actual current state.

    - [x] (2025-10-01 13:00Z) Example completed step.
    - [ ] Example incomplete step.
    - [ ] action_id=a3; mode=parallel; depends_on=a1,a2; file_locks=src/lookup/mod.rs; verify_events=action.tooling; worker_type=worker; implement lookup cache rewrite.

    Use timestamps to measure rates of progress. Mark each action `[x]` immediately when it finishes.

    Each `Progress` action must define: `action_id`, `mode` (`serial` or `parallel`), `depends_on`, `file_locks`, `verify_events`, `worker_type`. `verify_events` must contain only `action.*` events; lifecycle events are executed only by lifecycle steps.

    ## Verification Ledger

    Record every verification event attempt with one entry per attempt:

    - `event_id`
    - `attempt`
    - `status` (`pass`, `fail`, `escalated`)
    - `commands`
    - `failure_summary`
    - `started_at`
    - `finished_at`

    ## Surprises & Discoveries

    Document unexpected behaviors, bugs, optimizations, or insights discovered during implementation. Provide concise evidence.

    - Observation: …
      Evidence: …

    ## Decision Log

    Record every decision made while working on the plan:

    - Decision: …
      Rationale: …
      Date/Author: …

    ## Outcomes & Retrospective

    Summarize outcomes, gaps, and lessons learned at major milestones or at completion. Compare the result against the original purpose.

    ## Context and Orientation

    Describe the current state relevant to this task as if the reader knows nothing. Name key files and modules by full path. Define any non-obvious term you will use. Do not refer to prior plans.

    ## Plan of Work

    Describe, in prose, the sequence of edits and additions. For each edit, name the file and location (function, module) and what to insert or change. Keep it concrete and minimal.

    ## Concrete Steps

    State the exact commands to run and the working directory. Show short expected output for commands that produce output. Update this section as work proceeds.

    ## Validation and Acceptance

    Describe how to exercise the system and what to observe. Phrase acceptance as behavior with specific inputs and outputs. If tests are involved: "run <test command> and expect <N> passed; the new test <name> fails before the change and passes after."

    ## Idempotence and Recovery

    If steps can be repeated safely, say so. If a step is risky, provide a retry or rollback path.

    ## Artifacts and Notes

    Include the most important transcripts, diffs, or snippets as indented examples.

    ## Interfaces and Dependencies

    Name the libraries, modules, and services to use and why. Specify types, interfaces, and function signatures that must exist at the end of the milestone.

A single, stateless agent — or a human novice — must be able to read your ExecPlan from top to bottom and produce a working, observable result. That is the bar: SELF-CONTAINED, SELF-SUFFICIENT, NOVICE-GUIDING, OUTCOME-FOCUSED.

When you revise a plan, ensure changes are reflected across all sections and write a note at the bottom describing what changed and why.

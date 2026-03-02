# Codex Execution Plans (ExecPlans):

This document describes the requirements for an execution plan ("ExecPlan"), a design document that a coding agent can follow to deliver a working feature or system change. Treat the reader as a complete beginner to this repository: they have only the current working tree and the single ExecPlan file you provide. There is no memory of prior plans and no external context.

## How to use ExecPlans and PLANS.md

When authoring an executable specification (ExecPlan), follow PLANS.md _to the letter_. If it is not in your context, refresh your memory by reading the entire PLANS.md file. Be thorough in reading (and re-reading) source material to produce an accurate specification. When creating a spec, start from the skeleton and flesh it out as you do your research.

When implementing an executable specification (ExecPlan), do not prompt the user for "next steps"; simply proceed to the next milestone. Keep all sections up to date, add or split entries in the list at every stopping point to affirmatively state the progress made and next steps. Resolve ambiguities autonomously, and commit frequently. Do not request human confirmation until the ExecPlan Lifecycle defined below is complete. If you are executing a sub ExecPlan, report outcomes back to its parent main ExecPlan instead of asking the user for instructions.

When discussing an executable specification (ExecPlan), record decisions in a log in the spec for posterity; it should be unambiguously clear why any change to the specification was made. ExecPlans are living documents, and it should always be possible to restart from _only_ the ExecPlan and no other work.

When researching a design with challenging requirements or significant unknowns, use milestones to implement proof of concepts, "toy implementations", etc., that allow validating whether the user's proposal is feasible. Read the source code of libraries by finding or acquiring them, research deeply, and include prototypes to guide a fuller implementation.

## Requirements

NON-NEGOTIABLE REQUIREMENTS:

* Every ExecPlan must be fully self-contained. Self-contained means that in its current form it contains all knowledge and instructions needed for a novice to succeed.
* Every ExecPlan is a living document. Contributors are required to revise it as progress is made, as discoveries occur, and as design decisions are finalized. Each revision must remain fully self-contained.
* Every ExecPlan must enable a complete novice to implement the feature end-to-end without prior knowledge of this repo.
* Every ExecPlan must produce a demonstrably working behavior, not merely code changes to "meet a definition".
* Every ExecPlan must define every term of art in plain language or do not use it.

Purpose and intent come first. Begin by explaining, in a few sentences, why the work matters from a user's perspective: what someone can do after this change that they could not do before, and how to see it working. Then guide the reader through the exact steps to achieve that outcome, including what to edit, what to run, and what they should observe.

The agent executing your plan can list files, read files, search, run the project, and run tests. It does not know any prior context and cannot infer what you meant from earlier milestones. Repeat any assumption you rely on. Do not point to external blogs or docs; if knowledge is required, embed it in the plan itself in your own words. If an ExecPlan builds upon a prior ExecPlan and that file is checked in, incorporate it by reference. If it is not, you must include all relevant context from that plan.

## Formatting

Format and envelope are simple and strict. Each ExecPlan must be one single fenced code block labeled as `md` that begins and ends with triple backticks. Do not nest additional triple-backtick code fences inside; when you need to show commands, transcripts, diffs, or code, present them as indented blocks within that single fence. Use indentation for clarity rather than code fences inside an ExecPlan to avoid prematurely closing the ExecPlan's code fence. Use two newlines after every heading, use # and ## and so on, and correct syntax for ordered and unordered lists.

When writing an ExecPlan to a Markdown (.md) file where the content of the file *is only* the single ExecPlan, you should omit the triple backticks.

Write in plain prose. Prefer sentences over lists. Avoid checklists, tables, and long enumerations unless brevity would obscure meaning. Checklists are permitted only in the `Progress` section, where they are mandatory. Narrative sections must remain prose-first.

## Guidelines

Self-containment and plain language are paramount. If you introduce a phrase that is not ordinary English ("daemon", "middleware", "RPC gateway", "filter graph"), define it immediately and remind the reader how it manifests in this repository (for example, by naming the files or commands where it appears). Do not say "as defined previously" or "according to the architecture doc." Include the needed explanation here, even if you repeat yourself.

Avoid common failure modes. Do not rely on undefined jargon. Do not describe "the letter of a feature" so narrowly that the resulting code compiles but does nothing meaningful. Do not outsource key decisions to the reader. When ambiguity exists, resolve it in the plan itself and explain why you chose that path. Err on the side of over-explaining user-visible effects and under-specifying incidental implementation details.

Anchor the plan with observable outcomes. State what the user can do after implementation, the commands to run, and the outputs they should see. Acceptance should be phrased as behavior a human can verify ("after starting the server, navigating to [http://localhost:8080/health](http://localhost:8080/health) returns HTTP 200 with body OK") rather than internal attributes ("added a HealthCheck struct"). If a change is internal, explain how its impact can still be demonstrated (for example, by running tests that fail before and pass after, and by showing a scenario that uses the new behavior).

Specify repository context explicitly. Name files with full repository-relative paths, name functions and modules precisely, and describe where new files should be created. If touching multiple areas, include a short orientation paragraph that explains how those parts fit together so a novice can navigate confidently. When running commands, show the working directory and exact command line. When outcomes depend on environment, state the assumptions and provide alternatives when reasonable.

Be idempotent and safe. Write the steps so they can be run multiple times without causing damage or drift. If a step can fail halfway, include how to retry or adapt. If a migration or destructive operation is necessary, spell out backups or safe fallbacks. Prefer additive, testable changes that can be validated as you go.

Validation is not optional. Include instructions to run tests, to start the system if applicable, and to observe it doing something useful. Describe comprehensive testing for any new features or capabilities. Include expected outputs and error messages so a novice can tell success from failure. Where possible, show how to prove that the change is effective beyond compilation (for example, through a small end-to-end scenario, a CLI invocation, or an HTTP request/response transcript). State the exact test commands appropriate to the project’s toolchain and how to interpret their results.

Capture evidence. When your steps produce terminal output, short diffs, or logs, include them inside the single fenced block as indented examples. Keep them concise and focused on what proves success. If you need to include a patch, prefer file-scoped diffs or small excerpts that a reader can recreate by following your instructions rather than pasting large blobs.

## Milestones

Milestones are narrative, not bureaucracy. If you break the work into milestones, introduce each with a brief paragraph that describes the scope, what will exist at the end of the milestone that did not exist before, the commands to run, and the acceptance you expect to observe. Keep it readable as a story: goal, work, result, proof. Progress and milestones are distinct: milestones tell the story, progress tracks granular work. Both must exist. Never abbreviate a milestone merely for the sake of brevity, do not leave out details that could be crucial to a future implementation.

Each milestone must be independently verifiable and incrementally implement the overall goal of the execution plan.

## Living plans and design decisions

* ExecPlans are living documents. As you make key design decisions, update the plan to record both the decision and the thinking behind it. Record all decisions in the `Decision Log` section.
* ExecPlans must contain and maintain a `Progress` section, a `Surprises & Discoveries` section, a `Decision Log`, and an `Outcomes & Retrospective` section. These are not optional.
* When you discover optimizer behavior, performance tradeoffs, unexpected bugs, or inverse/unapply semantics that shaped your approach, capture those observations in the `Surprises & Discoveries` section with short evidence snippets (test output is ideal).
* If you change course mid-implementation, document why in the `Decision Log` and reflect the implications in `Progress`. Plans are guides for the next contributor as much as checklists for you.
* At completion of a major task or the full plan, write an `Outcomes & Retrospective` entry summarizing what was achieved, what remains, and lessons learned.

## Plan file locations and status

When creating a new plan markdown file, place it by status:

* Active plans must be created in `docs/plans/active/`.
* Completed plans must be moved to `docs/plans/completed/`.
* Remaining technical debt plans must be created or moved to `docs/plans/tech-debt/`.

Any markdown file in `docs/plans/tech-debt/` must include links to the related plan markdown files (for example, active or completed plans that introduced, mitigated, or depend on that debt). The links must be explicit repository-relative markdown links so a reader can navigate directly.

## Main and Sub ExecPlans

An ExecPlan that owns the end-to-end objective is called a `main ExecPlan`. A plan created by decomposing part of that objective is called a `sub ExecPlan`.

When a created ExecPlan is large, agents are encouraged to decompose it into multiple sub ExecPlans. A large plan means the scope is too broad to execute reliably as one context, especially when progress tracking, validation mapping, or implementation detail would become difficult to maintain in one document.

Each sub ExecPlan must have its own markdown plan document under `docs/plans/active/`. Every such sub-plan document must include:

* a repository-relative link to its parent main ExecPlan file,
* an explicit statement of which part of the main ExecPlan it corresponds to (for example, which action, milestone, or scoped output),
* enough bounded scope that one sub agent can reasonably complete it within one context window.

The `Progress` section in the main ExecPlan must do more than list sub plans. It must explicitly describe execution topology:

* which sub ExecPlans are parallelizable,
* which sub ExecPlans must be processed sequentially and in what order,
* when multiple sub-agent types are available, which sub-agent type is assigned to each sub ExecPlan.

Here, “parallelizable” means the sub ExecPlans can be executed by separate sub agents with isolated context windows and no required communication between those sub agents until the individual sub plans are completed.

Each sub ExecPlan independently follows the ExecPlan lifecycle. After finishing, the sub agent must autonomously report results back to the parent main ExecPlan scope and terminate without requesting a human response.

## How ExecPlans must use design, architecture, and verification documents

Each ExecPlan must explicitly describe how it handled design, architecture, and verification guidance from `AGENTS.md`.

At the beginning of the ExecPlan, include a short repository-document context paragraph that names the exact files consulted for design, architecture, and verification policy. Use repository-relative paths. If a required policy file does not exist yet, state that clearly in the ExecPlan and record the fallback document or assumption used.

For design decisions, follow this rule: if the change introduces a long-lived or reusable decision (for example, a new interface, invariant, API behavior, or a major trade-off), the ExecPlan must either update the design document or create one, and must link that artifact in the plan.

For architecture changes, follow this rule: if the change affects structure (module boundaries, layering, feature flags, shared infrastructure, cross-domain dependencies, or boundaries such as FFI/CUDA/IO/build integration), the ExecPlan must describe the architecture impact and must update the architecture document and related enforcement rules in the same change.

For verification, follow this rule: every ExecPlan must name the verification policy document and list exact verification commands to run and commands actually run. Update the verification document only when verification *policy* changes; do not update it for routine per-change command results.

Before moving an ExecPlan from `docs/plans/active/` to `docs/plans/completed/`, the plan must include an explicit note of what design, architecture, and verification documents were referenced, created, modified, or left unchanged, and why.

## ExecPlan Lifecycle

Agents must strictly follow this lifecycle to create, execute, and complete an ExecPlan.

1. Before creating or updating the target plan file, choose the lifecycle target (`main ExecPlan` or `sub ExecPlan`) and run the corresponding pre-creation verification document under `docs/verification`:
   * `main ExecPlan`: `docs/verification/main_execplan_pre_creation.md`
   * `sub ExecPlan`: `docs/verification/sub_execplan_pre_execution.md`
   At this point, do not create, edit, or delete files.
2. Add a new plan document for the target ExecPlan under `docs/plans/active` in accordance with `PLANS.md`. If the target is a main ExecPlan and decomposition is needed, create one sub-plan markdown document per sub ExecPlan under `docs/plans/active`, and ensure each sub plan links the parent main ExecPlan path and names the main-plan scope it covers.
3. This step is required for main ExecPlans only. Map the main plan actions (the checklist in `Progress`) to the events introduced in `docs/verification/index.md`, enumerate which validations must run after each action, and add those validations into the main-plan actions. For validations related to sub ExecPlans, either (a) insert each validation adjacent to the corresponding sub-plan action, or (b) aggregate them into one validation step in the main ExecPlan after one or more sub ExecPlans finish.
4. Execute the actions updated in step 3 in order and record progress in the target plan. Sub ExecPlans execute their own actions independently under their assigned scope.
5. If an unexpected event occurs while executing an action (for example, an unexpected error), add it to the `Surprises & Discoveries` section. Then update the actions and return to step 3. "Update the actions" includes creating new sub ExecPlans when needed.
6. After completing all actions for the target plan, finalize that plan document state first: update progress/outcome sections, then move that plan document from `docs/plans/active/` to `docs/plans/completed/`. If technical debt remains, add a corresponding document under `docs/plans/tech-debt`.
7. Run post-completion verification using the document that matches the target plan type:
   * `main ExecPlan`: run `docs/verification/main_execplan_post_completion.md`, record final validation results in the completed main plan document (including failed commands, failure locations, and likely causes when validation fails), then perform the final commit and push so completed-plan state and final evidence are persisted in git. If final validation fails, still append failure results and include them in the final commit/push.
   * `sub ExecPlan`: run `docs/verification/sub_execplan_post_completion.md`, then return to the corresponding main ExecPlan scope and update the main plan `Progress` (and related sections when needed) with the sub-plan outcome. After this report, the sub agent must terminate without requesting user response.

Important: once a human or an AI agent starts a new ExecPlan lifecycle, the AI agent must not request any human response until step 7 is complete for the current plan target. For main ExecPlans, completion includes the final commit/push that persists completed-plan state. For sub ExecPlans, completion includes reporting back into the corresponding main ExecPlan scope. If an agent requests human response before all step-7 verification requirements are complete, that agent is immediately dismissed and removed. Treat humans as a very slow external device, and assume reliance on humans will significantly delay completion.

# Prototyping milestones and parallel implementations

It is acceptable—-and often encouraged—-to include explicit prototyping milestones when they de-risk a larger change. Examples: adding a low-level operator to a dependency to validate feasibility, or exploring two composition orders while measuring optimizer effects. Keep prototypes additive and testable. Clearly label the scope as “prototyping”; describe how to run and observe results; and state the criteria for promoting or discarding the prototype.

Prefer additive code changes followed by subtractions that keep tests passing. Parallel implementations (e.g., keeping an adapter alongside an older path during migration) are fine when they reduce risk or enable tests to continue passing during a large migration. Describe how to validate both paths and how to retire one safely with tests. When working with multiple new libraries or feature areas, consider creating spikes that evaluate the feasibility of these features _independently_ of one another, proving that the external library performs as expected and implements the features we need in isolation.

## Skeleton of a Good ExecPlan

    # <Short, action-oriented description>

    This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

    If PLANS.md file is checked into the repo, reference the path to that file here from the repository root and note that this document must be maintained in accordance with PLANS.md.

    ## Purpose / Big Picture

    Explain in a few sentences what someone gains after this change and how they can see it working. State the user-visible behavior you will enable.

    ## Progress

    Use a list with checkboxes to summarize granular steps. Every stopping point must be documented here, even if it requires splitting a partially completed task into two (“done” vs. “remaining”). This section must always reflect the actual current state of the work.

    - [x] (2025-10-01 13:00Z) Example completed step.
    - [ ] Example incomplete step.
    - [ ] Example partially completed step (completed: X; remaining: Y).

    Use timestamps to measure rates of progress.

    If the plan is a main ExecPlan that uses sub ExecPlans, each `Progress` action must also state whether the sub plans are parallel or sequential, the required order for sequential work, and the assigned sub-agent type when multiple sub-agent types exist.

    ## Surprises & Discoveries

    Document unexpected behaviors, bugs, optimizations, or insights discovered during implementation. Provide concise evidence.

    - Observation: …
      Evidence: …

    ## Decision Log

    Record every decision made while working on the plan in the format:

    - Decision: …
      Rationale: …
      Date/Author: …

    ## Outcomes & Retrospective

    Summarize outcomes, gaps, and lessons learned at major milestones or at completion. Compare the result against the original purpose.

    ## Context and Orientation

    Describe the current state relevant to this task as if the reader knows nothing. Name the key files and modules by full path. Define any non-obvious term you will use. Do not refer to prior plans.

    ## Plan of Work

    Describe, in prose, the sequence of edits and additions. For each edit, name the file and location (function, module) and what to insert or change. Keep it concrete and minimal.

    ## Concrete Steps

    State the exact commands to run and where to run them (working directory). When a command generates output, show a short expected transcript so the reader can compare. This section must be updated as work proceeds.

    ## Validation and Acceptance

    Describe how to start or exercise the system and what to observe. Phrase acceptance as behavior, with specific inputs and outputs. If tests are involved, say "run <project’s test command> and expect <N> passed; the new test <name> fails before the change and passes after>".

    ## Idempotence and Recovery

    If steps can be repeated safely, say so. If a step is risky, provide a safe retry or rollback path. Keep the environment clean after completion.

    ## Artifacts and Notes

    Include the most important transcripts, diffs, or snippets as indented examples. Keep them concise and focused on what proves success.

    ## Interfaces and Dependencies

    Be prescriptive. Name the libraries, modules, and services to use and why. Specify the types, traits/interfaces, and function signatures that must exist at the end of the milestone. Prefer stable names and paths such as `crate::module::function` or `package.submodule.Interface`. E.g.:

    In crates/foo/planner.rs, define:

        pub trait Planner {
            fn plan(&self, observed: &Observed) -> Vec<Action>;
        }

If you follow the guidance above, a single, stateless agent -- or a human novice -- can read your ExecPlan from top to bottom and produce a working, observable result. That is the bar: SELF-CONTAINED, SELF-SUFFICIENT, NOVICE-GUIDING, OUTCOME-FOCUSED.

When you revise a plan, you must ensure your changes are comprehensively reflected across all sections, including the living document sections, and you must write a note at the bottom of the plan describing the change and the reason why. ExecPlans must describe not just the what but the why for almost everything.

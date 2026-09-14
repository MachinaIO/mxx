# Maintaining Agent Instructions

This maintenance note records the GPT-6 Astra instruction cleanup. It is not required reading for ordinary implementation tasks.

## Design Basis

The cleanup follows [Rethinking skills and prompts for GPT-6 Astra](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra): narrow skill discovery, conditional context loading, explicit decision boundaries, and task-specific completion criteria. It changes instructions, not the configured model.

## Instruction Ownership

| Surface | Responsibility |
| --- | --- |
| `AGENTS.md` | Task scope, authority, contextual routing, execution and completion. |
| `BUILDER.md` | Implementation, performance, cryptographic invariants, and scoped validation. |
| `REVIEWER.md` | Read-only review, evidence, actionable findings, and decision format. |
| `GPU.md` | GPU dataflow, ownership, memory complexity, execution permissions, and repetition requirements. |
| `.codex/skills/AGENTS.md` and `.codex/skills/CLAUDE.md` | Local skill maintenance and entrypoint routing. |
| `.codex/skills/README.md` | Inventory of the four repository-local skills. |
| `.codex/skills/*/SKILL.md` | Specific workflow selection, constraints, and completion evidence. |
| `.codex/skills/*/guides/*.md` | Conditional CLI/SDK examples moved from large entrypoints. |

Global/profile skills are outside this repository change. Architecture, mathematical specifications, and historical plan/progress documents retain their existing content; load them for their task-specific contracts, not as general agent instructions. Directories named `references` remain read-only, so skill supporting material lives in `guides`.

## Decisions to Preserve

- Routine choices and already-authorized steps proceed without repeated questions. Semantic changes, new resource decisions, and missing permissions still require resolution.
- Documentation-only work uses document checks. Broad compile gates apply to workspace, feature-wiring, or release validation; CUDA/synchronization changes retain the GPU compile and repetition requirements.
- Integration tests still require an explicit request in the current task. Read-only requests, no-push limits, pre-existing work, hardware limits, and cryptographic specifications remain binding.
- Remote runs identify exact source, preserve dirty remote state, collect logs and exit codes, and verify final pod state. Stop is distinct from deletion. Retry scope follows the user's task and resource budget.
- CLI examples are conditional guidance. Setup, publication, and cloud execution are not automatic consequences of loading or editing a skill.

## Static Review Scenarios

Use these as maintenance checks, not commands to execute during a documentation edit:

| Request | Expected instruction behavior |
| --- | --- |
| Correct a README typo | Edit the relevant prose; no architecture tour, Rust build, or GPU test. |
| Analyze a GPU race without edits or execution | Inspect the scoped source and existing evidence; preserve the requested phase. |
| Fix a CUDA synchronization bug | Complete the repair, compile, and run the required 300 repetitions through the approved execution mechanism. Report blocked or incomplete runs accurately. |
| Run a named benchmark on an existing pod; do not push | Preserve hardware and source limits; use an authorized source transfer or resolve that blocker without publishing. |
| Stop a pod, do not delete it | Stop the named pod and verify its state; preserve the pod and volume. |
| Fix a failed remote test within an agreed budget | Synchronize and rerun scoped repairs under existing authorization; stop at a semantic, permission, or budget boundary. |
| Review the implementation | Stay read-only and report supported findings; do not implement fixes or demand optional polish. |
| Edit the Flash skill | Validate its documents; do not deploy or invoke cloud endpoints. |

These scenarios are a static consistency review, not an executed model evaluation or evidence of runtime, cost, or latency improvements.

## Validation Notes

Check whitespace, local instruction targets, code fences, YAML frontmatter, and preservation of existing metadata. The bundled skill creator's `quick_validate.py` accepts the benchmark skill but rejects the existing `compatibility` keys in two upstream skills and `user-invocable` in Flash. Preserve these existing fields for their original consumers; distinguish this validator limitation from a newly introduced error. Validate the full YAML separately, and use temporary copies without those unsupported fields to check the validator-supported frontmatter and body. Do not describe that projection as full cross-agent compatibility validation.

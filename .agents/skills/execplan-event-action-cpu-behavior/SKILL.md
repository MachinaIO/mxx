---
name: execplan-event-action-cpu-behavior
description: Event skill for action.cpu_behavior verification. Use for CPU/library behavior changes.
---

# Event Skill: action.cpu_behavior

Runs CPU-behavior verification with dynamic scope handling:

- run `cargo +nightly fmt --all`,
- run targeted `cargo test -r --lib -- <filter>` when scope filter is available,
- run full `cargo test -r --lib` only when completion/foundational conditions apply (or scope is unclear),
- avoid integration-test execution under `tests/` unless explicitly operator-requested.

Dynamic controls:

- `EXECPLAN_TEST_FILTER=<scope_or_test_filter>`
- `EXECPLAN_RUN_FULL_LIB_TESTS=0|1|auto`
- `EXECPLAN_SCOPE_COMPLETE=0|1`
- `EXECPLAN_FOUNDATIONAL_CHANGE=0|1`

## Script

- `scripts/run_event.sh --plan <plan_md>`

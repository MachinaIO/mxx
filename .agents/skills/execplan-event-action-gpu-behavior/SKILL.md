---
name: execplan-event-action-gpu-behavior
description: Event skill for action.gpu_behavior verification. Use for CUDA/GPU-featured behavior changes.
---

# Event Skill: action.gpu_behavior

Runs GPU/CUDA behavior verification with dynamic conditions:

- run `cargo +nightly fmt --all`,
- run scope-targeted GPU unit tests when CUDA-side edits have clear scope filters,
- run full `cargo test -r --lib --features gpu` only when completion/foundational conditions apply,
- when CUDA edits are complete, build once and run GPU unit-test binary 300 consecutive times with failure artifact logging.

Dynamic controls:

- `EXECPLAN_CUDA_CHANGED=0|1|auto`
- `EXECPLAN_GPU_TEST_FILTER=<scope_or_test_filter>`
- `EXECPLAN_RUN_FULL_GPU_TESTS=0|1|auto`
- `EXECPLAN_SCOPE_COMPLETE=0|1`
- `EXECPLAN_FOUNDATIONAL_CHANGE=0|1`
- `EXECPLAN_WORK_COMPLETE=0|1|auto`

## Script

- `scripts/run_event.sh --plan <plan_md>`

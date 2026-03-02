# Verification Index

Read this file first before any verification work.

This index tells agents and contributors which verification document to use for each change event.

## How to choose verification documents

1. Classify the change event.
2. Read the mapped event document(s).
3. Execute all required commands in those documents.
4. Record commands and outcomes in the active ExecPlan and PR description.

If more than one event applies, execute all applicable event documents.

## Event map

- Before creating a new main ExecPlan: read [main_execplan_pre_creation.md](./main_execplan_pre_creation.md).
- Before executing a sub ExecPlan: read [sub_execplan_pre_execution.md](./sub_execplan_pre_execution.md).
- After completing a main ExecPlan: read [main_execplan_post_completion.md](./main_execplan_post_completion.md).
- After completing a sub ExecPlan: read [sub_execplan_post_completion.md](./sub_execplan_post_completion.md).
- Docs-only changes: read [docs_only_changes.md](./docs_only_changes.md).
- CPU/library behavior changes (no GPU/CUDA boundary change): read [cpu_behavior_changes.md](./cpu_behavior_changes.md).
- GPU/CUDA or GPU-featured Rust changes: read [gpu_behavior_changes.md](./gpu_behavior_changes.md).

## Source-of-truth rule

Event documents in this directory must use commands derived from repository truth sources:

- `.github/workflows/ci.yml`
- `AGENTS.md`
- committed project scripts

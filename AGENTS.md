# Repository Guidelines

mxx implements lattice-cryptography operations in Rust and CUDA, including polynomial and matrix operations, preimage sampling, and BGG+ encodings.

## Scope and Authority

- Follow system and developer instructions, then the user's task instructions. User instructions take precedence over repository and skill guidance within those execution permissions.
- Apply instructions to the work they govern. Examples, historical plans, and reference material do not authorize new work or override the current task.
- Continue authorized work through implementation, relevant validation, and fixes. Make routine reversible choices from context; ask only when missing information changes correctness, scope, cost, or permission. Carry prior authorization forward without asking again.
- Preserve explicit planning, read-only, no-edit, no-push, hardware, and resource limits. If a requirement blocks progress, finish independent authorized work and identify the exact file and instruction, its applicability, and the input needed to proceed.
- Preserve pre-existing tracked edits and untracked files. Do not reset, overwrite, or clean them to simplify the task.

## Read According to the Task

- Use `BUILDER.md` for implementation and debugging, and `REVIEWER.md` for explicit reviews. Simple prose or instruction-file edits need only the relevant document sections.
- Read `GPU.md` before editing or reviewing CUDA, GPU wrappers, GPU tests, or GPU-facing performance-sensitive behavior.
- Use `docs/architecture.md` for crate boundaries and dependency changes. `Cargo.toml` is the authoritative workspace member list; there is no root facade crate. Application crates never depend on one another. Reusable gadgets live in `crates/gadgets/`, with circuit gadgets in `circuit_gadgets`.
- Load a skill when its specific workflow applies. Read only its relevant supporting documents; editing a skill does not authorize executing its examples or remote workflow.

## Repository Requirements

- Write repository documentation, commit messages, and PRs in English. Document repository paths relative to the repository root; external runtime paths in commands may be literal when required to execute them.
- Directories named `references` are read-only. Read relevant specifications there; never edit their contents.
- Run integration tests only when the user explicitly requests them in the current task. Targeted unit tests and narrow local checks do not require repeated confirmation, subject to execution permissions.
- Format Rust with `cargo +nightly fmt --all`.

## Execution and Completion

- Identify the requested outcome and appropriate validation before editing. Complete related changes across the production path before testing; use an intermediate check only when it resolves an implementation uncertainty.
- Match validation to the change. For documentation-only edits, check links, examples, and instruction consistency; do not build Rust or run GPUs. For code changes, follow the relevant checks in `BUILDER.md` and `GPU.md`.
- Once checks pass, repeat or broaden them only for new changes, failures, required repetition, or unresolved risks. Do not equate compilation, simulation, GPU execution, and end-to-end correctness.
- In Code Mode, batch independent tool calls within each bounded stage in one `functions.exec` using `await Promise.allSettled([...])` and inspect every result. Keep dependencies, approvals, waits, and conflicting mutations sequential.
- Delegate only when requested by the user or applicable instructions, with a bounded task and clear ownership. DeepSeek workers require an explicit request for DeepSeek subagents or `$deepseek-subagents`; inspecting or editing that skill does not authorize workers. The primary agent verifies delegated results.
- For long jobs, use durable logs and completion/exit-status evidence. Do independent work before waiting; otherwise use substantial waits within tool limits and the host's responsiveness requirements. Resume the same running call, avoid unchanged polling, and keep cancellation available.
- Report the outcome, relevant evidence, and any remaining blocker concisely. During long work, report meaningful progress or decisions rather than unchanged status. Do not claim timing, cost, or token improvements without measurements.

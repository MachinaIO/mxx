# Builder Guidance

Use this document for implementation and debugging, together with the scope and completion rules in `AGENTS.md`. A request to analyze or find a root cause is read-only unless the user also asks for a fix; inspect source and existing evidence without executing the program or tests until requested.

## Implementation Unit

Resolve routine implementation choices from the current code and task. Ask before changing an ambiguous cryptographic specification or semantic contract. Connect the complete scoped behavior across types, resource ownership, execution paths, callers, and relevant tests before validation. Preserve explicit stage boundaries from the user.

## Design & Code Style

- Prefer extending existing functions and types over parallel variants (`new_with_*`, `*_with_shared_inputs`) or wrapper layers. Use optional arguments for genuinely optional behavior, not invalid state combinations.
- When replacing a format or API, remove the superseded path and its version markers in the scoped migration; do not add compatibility shims. Do not delete unrelated legacy code.
- Reuse existing environment variables; define primitive-operation env vars in `crates/backends/src/env.rs`, gadget-level env vars in `crates/gadgets/src/env.rs`, and application-specific env vars in the owning crate, with explanatory comments.
- Inline small private helpers when this improves local readability; retain helpers that clarify ownership or a meaningful operation. Keep files under roughly 2000 lines excluding tests.
- Delete unused code, arguments, and imports immediately; never silence warnings with `_`. Remove all debug-only scaffolding (extra syncs, flags, timing logs) before finishing.
- Rename anything misleading; names must describe what the value or bound is, not which paper theorem it came from.
- Preserve the dependency direction in `docs/architecture.md`. Shared application code moves down to the lowest natural layer; application crates never depend on one another.

## Parallelism & Performance

- Use rayon for independent CPU work where the granularity benefits from parallel execution and ownership and peak memory allow it. Preserve existing parallelism in refactors unless the user requests lower or configurable parallelism, for example to reduce peak memory usage.
- Repair the ownership or synchronization dependency causing a race; do not hide flakiness behind a global mutex or blanket serialization.
- Control concurrency with env-var-configured batch sizes over `par_iter`, not `ThreadPoolBuilder`.
- Prefer in-place ops (`add_in_place`, etc.) and ownership transfer over `clone`; precompute loop-invariant constants once.
- After performance-sensitive changes, compare timings against the previous run; investigate regressions with per-stage `tracing` logs, then remove the temporary logs.

## Testing

- Use `scripts/run_tests.sh` as a reference for repository validation expectations and helper behavior.
- Use targeted unit tests for the changed behavior. For workspace-wide, feature-wiring, or release validation, require warning-free `cargo test -r --workspace --lib --no-run` and its `--features gpu` variant. These broad compile gates do not apply to prose-only edits.
- For GPU execution permissions and repeated CUDA/synchronization checks, follow `GPU.md`. Report unavailable checks as unverified, not passed.
- Test names use the established prefixes (e.g. `test_gpu_*`); tests live in the file defining the tested item.
- Test parameters must be overridable via env vars with small defaults. No fixed seeds: sample randomly per run.
- Expected values come from existing trusted primitives or round-trips, never from hand-rolled reference implementations. Never weaken or modify existing tests, or move GPU work to CPU, to make tests pass.
- Each test uses its own directory under `test_data/`. Invalidate task-owned stale checkpoints when parameters or circuits change; preserve pre-existing user artifacts and use a fresh directory when needed.
- Probabilistic norm tests may rarely fail by design; verify statistically over repeated runs instead of "fixing" them.

## Benchmarks & Measurement

- Benchmark code must call the same functions as production and reproduce its dataflow, including store/load.
- Measurement contract: measure one chunk/wave; latency = one wave; total_time = latency * chunk_count * slots; max_parallelism = the max across stages, never the sum. All times in seconds.
- Log all raw values (per-stage totals, latency, max_parallelism) directly; never leave values to be derived from other logs.
- Estimators use the actual params, never defaults.
- For authorized Runpod runs, use `.codex/skills/bench-on-runpod/SKILL.md` for source synchronization, logs, retries, and pod lifecycle. Use release builds and `RUST_LOG=debug` unless the task specifies otherwise; record VRAM usage about every 3 seconds and preserve command, environment, and source identity with results.
- Append results to the existing CSVs preserving their format, with a date column; verify CSV-vs-log consistency with a script, not by inspection.

## Lattice-Crypto Domain Rules

- Implement strictly per the PDFs in `references/`; never alter a paper's spec to make a test pass; cite the chapter when explaining bounds.
- Error-norm simulators must stay term-by-term consistent with the evaluator implementations; document the correspondence in English comments; watch for both double-counted and missing terms. If a simulated bound is exceeded in practice, tighten the estimate; do not add `crt_depth` margins.
- Scalars such as `q/q_i` and thresholds are defined against the full modulus, regardless of `q_level`.
- Choose `crt_depth` by simulation (binary search on `eval_ok && decryption_ok`), searching near previously found values in the CSVs.
- Checkpoint ID prefixes embed all parameters; derive randomness deterministically (`hash_sampler` + fixed tags) so checkpoints stay reusable and consistent.

## Evidence and Handoff

Ground correctness and performance claims in the scoped code, specification sections, or actual logs. If a specification appears contradictory or mathematically wrong, explain the issue before changing the approach. Report changed behavior, checks and outcomes, and material limitations. If blocked, leave exact ownership, available artifacts, and the next required action; do not describe incomplete validation as completion.

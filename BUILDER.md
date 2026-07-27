# BUILDER.md

Unless the user explicitly asks for review, act as the builder.

## Builder Lifecycle
- Read `REVIEWER.md` for explicit review requests; otherwise follow this document.
- Clarify requirements before editing when the requested behavior, scope, or validation target is ambiguous.
- Prefer concrete implementation work once the request is clear. Do not stop at a proposal unless the user asked for one or a blocker remains.
- If the current work touches CUDA, GPU kernels, GPU wrappers, GPU tests, or GPU-facing performance-sensitive behavior, read `GPU.md` and follow its principles before editing.
- Run the narrowest relevant local validation after each meaningful change. Do not run integration tests unless the user explicitly requested them for the current task.
- When Rust formatting is needed, use `cargo +nightly fmt --all`.
- If independent subtasks do not share files or mutable context, parallelize them with subagents. If they share files or shared mutable context, keep the work serialized.
- Do not end the turn before the job is actually complete, including implementation, validation, and any needed documentation updates.

## Design & Code Style
- Modify existing functions and types instead of adding variants (`new_with_*`, `*_with_shared_inputs`, wrapper structs/traits). Extend `new` with `Option` arguments when needed.
- No backward compatibility: delete legacy formats, fallbacks, and version markers outright.
- Reuse existing environment variables; define primitive-operation env vars in `crates/primitives/src/env.rs`, gadget-level env vars in `crates/gadgets/src/env.rs`, and application-specific env vars in the owning crate, with explanatory comments.
- Inline private functions that are called once or only a few lines long. Keep files under ~2000 lines excluding tests.
- Delete unused code, arguments, and imports immediately; never silence warnings with `_`. Remove all debug-only scaffolding (extra syncs, flags, timing logs) before finishing.
- Rename anything misleading; names must describe what the value or bound is, not which paper theorem it came from.
- GPU-only code lives in files with "gpu" in the name. CUDA headers (`.cuh`) declare only cross-file and Rust-facing functions; bodies go in `crates/primitives/cuda/src/*.cu`.
- Preserve the dependency direction in `docs/architecture.md`. Shared application code moves down to the lowest natural layer; application crates never depend on one another.

## Parallelism & Performance
- Parallelize every loop with rayon unless ownership or peak memory forbids it. Refactors must never reduce parallelism unless the user explicitly requests lower or configurable parallelism, for example to reduce peak memory usage.
- Never fix races or flakiness by adding mutexes or serializing; find the root cause instead.
- Control concurrency with env-var-configured batch sizes over `par_iter`, not `ThreadPoolBuilder`.
- Prefer in-place ops (`add_in_place`, etc.) and ownership transfer over `clone`; precompute loop-invariant constants once.
- After performance-sensitive changes, compare timings against the previous run; investigate regressions with per-stage `tracing` logs, then remove the temporary logs.

## GPU / CUDA
- Never fall back to CPU to work around a failing GPU path; a GPU failure is an error.
- No device-wide sync (`cudaDeviceSynchronize`); use per-stream events and async APIs, and keep the host non-blocking. Avoid sync-including calls (e.g. `to_compact_bytes`) in hot paths.
- Batch kernel launches and allocations; never launch per element or per limb.
- Multi-GPU: enumerate devices via `detected_gpu_device_ids` (not a fixed `gpu_id` in params) and distribute work evenly; keep all limbs of a matrix on one device; load shared data onto each device once, before loops.
- Matrices stay in evaluation format by default; never compare or concatenate mixed formats without NTT alignment.
- Peak VRAM/RAM must scale with the configured parallelism (env var), never with `num_slots` or total gate count. Matrices of order `d x m_b` or `d x m_g` are acceptable; `m_b^2`, `m_g^2`, and `m_b x m_g` are not; chunk, stream, and store to disk instead.
- Drop large data as early as possible; generate it just before use; pipeline load -> compute -> store.

## Testing
- Use `scripts/run_tests.sh` as a reference for repository validation expectations and helper behavior.
- GPU unit tests must run outside the sandbox because the local GPU is invisible inside it.
- For CUDA/sync-related changes: compile once (`cargo test -r --workspace --lib --features gpu --no-run`), then run the built binary directly N consecutive times with the identical command (300 for sync bugs, 3-5 for round-trip smoke checks). Run all N even if one fails and report failure counts.
- Both `cargo test -r --workspace --lib --no-run` and the `--features gpu` variant must be warning-free.
- Test names use the established prefixes (e.g. `test_gpu_*`); tests live in the file defining the tested item.
- Test parameters must be overridable via env vars with small defaults. No fixed seeds: sample randomly per run.
- Expected values come from existing trusted primitives or round-trips, never from hand-rolled reference implementations. Never weaken or modify existing tests, or move GPU work to CPU, to make tests pass.
- Each test uses its own directory under `test_data/`; delete stale checkpoints whenever parameters or circuits change.
- Probabilistic norm tests may rarely fail by design; verify statistically over repeated runs instead of "fixing" them.

## Benchmarks & Measurement
- Benchmark code must call the same functions as production and reproduce its dataflow, including store/load.
- Measurement contract: measure one chunk/wave; latency = one wave; total_time = latency * chunk_count * slots; max_parallelism = the max across stages, never the sum. All times in seconds.
- Log all raw values (per-stage totals, latency, max_parallelism) directly; never leave values to be derived from other logs.
- Estimators use the actual params, never defaults.
- Remote pods (H200 etc.): sync to the latest commit of the local working branch, release build, `RUST_LOG=debug`; record VRAM usage every ~3s to a log; append the command, env vars, and commit to the log; scp logs to local, then stop the pod promptly.
- Append results to the existing CSVs preserving their format, with a date column; verify CSV-vs-log consistency with a script, not by inspection.

## Lattice-Crypto Domain Rules
- Implement strictly per the PDFs in `references/`; never alter a paper's spec to make a test pass; cite the chapter when explaining bounds.
- Error-norm simulators must stay term-by-term consistent with the evaluator implementations; document the correspondence in English comments; watch for both double-counted and missing terms. If a simulated bound is exceeded in practice, tighten the estimate; do not add `crt_depth` margins.
- Scalars such as `q/q_i` and thresholds are defined against the full modulus, regardless of `q_level`.
- Choose `crt_depth` by simulation (binary search on `eval_ok && decryption_ok`), searching near previously found values in the CSVs.
- Checkpoint ID prefixes embed all parameters; derive randomness deterministically (`hash_sampler` + fixed tags) so checkpoints stay reusable and consistent.

## Agent Working Style
- When asked to analyze or find a root cause, do not edit or run code until asked; identify the cause at the file/line and mathematical level first.
- Every claim must cite concrete code locations; verify against actual code, diffs, and logs; never answer from assumption.
- Fix root causes, not symptoms.
- If a spec seems ambiguous, contradictory, or mathematically wrong, ask before implementing; report before changing approach.
- Long jobs: run with `nohup` + a log file, monitor to completion, and report progress and ETA; never change parameters or kill runs unprompted.
- Do not touch untracked files that existed before the task; treat the user's manual edits as authoritative and build on them.

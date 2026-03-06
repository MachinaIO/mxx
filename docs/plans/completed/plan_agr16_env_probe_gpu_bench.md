# Add GPU AGR16 Env-Probe Complete Binary-Tree Benchmark

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan follows `PLANS.md`.

ExecPlan start context:
- Branch at start: `feat/agr16_encoding`
- Commit at start: `e21f7c6f0b2996ae85d6898556e6f6ea402c3114`
- PR tracking document: `docs/prs/completed/pr_feat_agr16_env_probe_gpu_bench.md`

Repository-document context used for this plan: `PLANS.md`, `DESIGN.md`, `docs/design/index.md`, `ARCHITECTURE.md`, `docs/architecture/index.md`, `docs/architecture/scope/index.md`, `docs/architecture/scope/agr16.md`, `docs/architecture/scope/matrix.md`, `VERIFICATION.md`, `docs/verification/index.md`, `docs/verification/main_execplan_pre_creation.md`, `docs/verification/cpu_behavior_changes.md`, `docs/verification/gpu_behavior_changes.md`, and `docs/verification/main_execplan_post_completion.md`.

## Purpose / Big Picture

After this change, `benches/` will include a GPU implementation of the AGR16 env-probe complete binary-tree benchmark using `GpuDCRTPolyMatrix`, while preserving the existing CPU benchmark and test. This allows direct GPU-side performance probing of the same topology.

## Progress

- [x] (2026-03-03 03:47Z) Read user request and confirmed target benchmark and parameter constraints.
- [x] (2026-03-03 03:49Z) Ran pre-creation context checks (`git branch --show-current`, `git status --short`, `git log --oneline --decorate --max-count=20`, `gh pr status`, `gh pr view --json ...`) and confirmed scope alignment with PR #60.
- [x] (2026-03-03 03:49Z) Created active PR tracking file `docs/prs/active/pr_feat_agr16_env_probe_gpu_bench.md`.
- [x] (2026-03-03 03:50Z) Created this ExecPlan.
- [x] (2026-03-03 04:01Z) Added new GPU benchmark source under `benches/` (`bench_agr16_complete_binary_tree_depth_env_probe_gpu.rs`) using `GpuDCRTPolyMatrix`, GPU samplers, and GPU sync timing.
- [x] (2026-03-03 04:01Z) Registered new benchmark target in `Cargo.toml`.
- [x] (2026-03-03 04:10Z) Ran verification commands:
  - `cargo +nightly fmt --all`
  - `cargo test -r --lib agr16`
  - `cargo test -r --lib`
  - `cargo bench --bench bench_agr16_complete_binary_tree_depth_env_probe_gpu --no-run`
  - `cargo bench --bench bench_agr16_complete_binary_tree_depth_env_probe_gpu --no-run --features gpu`
- [x] (2026-03-03 04:15Z) Posted reviewer follow-up response comment: `https://github.com/MachinaIO/mxx/pull/60#issuecomment-3988490802`.
- [x] (2026-03-03 04:15Z) Ran post-completion readiness action `gh pr ready 60` (already ready) and moved plan/PR tracking docs to completed.
- [x] (2026-03-03 04:15Z) Persisted post-completion state via commit and push.

Main-ExecPlan validation mapping (PLANS.md lifecycle step 3):
- Action `add gpu env-probe benchmark` -> run `cargo bench --bench <new-bench> --no-run --features gpu`.
- Action `preserve agr16 behavior` -> run `cargo test -r --lib agr16`.
- Action `finalize scope` -> run `cargo test -r --lib`.
- Action `finalize lifecycle` -> run `gh pr ready`, move docs, commit, push.

## Surprises & Discoveries

- Observation: GPU bench files in this repo follow `#[cfg(feature = "gpu")]` guarded `main` pattern so non-GPU builds remain valid.
  Evidence: `benches/bench_matrix_mul_gpu.rs`, `benches/bench_preimage_gpu.rs`.

## Decision Log

- Decision: Implement GPU benchmark as a new bench target file instead of feature-branching the existing CPU env-probe bench.
  Rationale: Keeps CPU/GPU benchmark entrypoints explicit and consistent with existing `bench_*_cpu.rs` / `bench_*_gpu.rs` split.
  Date/Author: 2026-03-03 / Codex

## Outcomes & Retrospective

Implementation, validation, and post-completion readiness actions are complete, and the completed-plan state is persisted to git.

## Design/Architecture/Verification Document Summary

Design documents:
- Referenced: `DESIGN.md`, `docs/design/index.md`, `docs/design/agr16_recursive_auxiliary_chain.md`.
- Modified/Created: none (benchmark-only follow-up).

Architecture documents:
- Referenced: `ARCHITECTURE.md`, `docs/architecture/index.md`, `docs/architecture/scope/index.md`, `docs/architecture/scope/agr16.md`, `docs/architecture/scope/matrix.md`.
- Modified/Created: none (no module boundary/dependency direction change).

Verification documents:
- Referenced: `VERIFICATION.md`, `docs/verification/index.md`, `docs/verification/main_execplan_pre_creation.md`, `docs/verification/cpu_behavior_changes.md`, `docs/verification/gpu_behavior_changes.md`, `docs/verification/main_execplan_post_completion.md`.
- Policy updates: none.

## Context and Orientation

`benches/bench_agr16_complete_binary_tree_depth_env_probe.rs` currently provides the CPU benchmark equivalent of the env-probe test. The request is to add a GPU variant using `GpuDCRTPolyMatrix`, preserving the original benchmark/test.

The repository already contains GPU benchmark conventions and GPU sampler implementations, so this follow-up should mirror those patterns for consistency.

## Plan of Work

Create `benches/bench_agr16_complete_binary_tree_depth_env_probe_gpu.rs` with `#[cfg(feature = "gpu")]` guarded implementation:
- construct `GpuDCRTPolyParams` from CPU params configured as requested (`ring_dim=2^14`, `crt_bits=52`, `crt_depth=9`, `base_bits=26`),
- sample AGR16 keys/encodings using GPU hash/uniform samplers,
- evaluate the same env-probe binary-tree multiplication circuit,
- assert Equation 5.1 output consistency and report elapsed time.

Add a new `[[bench]]` entry in `Cargo.toml`.

## Concrete Steps

Run from repository root (`.`):

    cargo +nightly fmt --all
    cargo test -r --lib agr16
    cargo test -r --lib
    cargo bench --bench bench_agr16_complete_binary_tree_depth_env_probe_gpu --no-run --features gpu

Lifecycle closure commands:

    gh pr comment 60 --body "<review response summary>"
    gh pr ready
    mv docs/prs/active/pr_feat_agr16_env_probe_gpu_bench.md docs/prs/completed/pr_feat_agr16_env_probe_gpu_bench.md
    mv docs/plans/active/plan_agr16_env_probe_gpu_bench.md docs/plans/completed/plan_agr16_env_probe_gpu_bench.md
    git add -A
    git commit -m "bench: add gpu agr16 env-probe binary-tree benchmark"
    git push origin $(git branch --show-current)

## Validation and Acceptance

Acceptance criteria:
1. New GPU benchmark target exists and compiles with `--features gpu`.
2. Existing CPU env-probe benchmark and test remain present.
3. Requested parameter set is applied in the GPU benchmark.

## Idempotence and Recovery

This is additive benchmark work. If GPU bench compile fails, validate trait/param type mismatches first, then retry without touching AGR16 core arithmetic files.

## Artifacts and Notes

Expected touched files:
- `benches/bench_agr16_complete_binary_tree_depth_env_probe_gpu.rs`
- `Cargo.toml`
- `docs/prs/completed/pr_feat_agr16_env_probe_gpu_bench.md`
- `docs/plans/completed/plan_agr16_env_probe_gpu_bench.md`

## Interfaces and Dependencies

No public API/interface changes are expected.

Revision note (2026-03-03 04:10Z): Updated with completed GPU benchmark implementation and verification outcomes; left only lifecycle closure steps pending.
Revision note (2026-03-03 04:15Z): Updated completed-path linkage and recorded PR response/readiness actions; left final commit/push as remaining lifecycle step.
Revision note (2026-03-03 04:15Z): Marked lifecycle completion after persisting final completed-plan state.

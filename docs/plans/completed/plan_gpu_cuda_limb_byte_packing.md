# Reduce GPU Limb VRAM with Modulus-Width Byte Packing

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan is maintained under `PLANS.md` and follows that file’s rules.

ExecPlan start context:
- Branch at start: `feat/gpu-byte-limb-packing`
- Commit at start: `a5e2e0f1195e7a6f3ca16950d05d1267b7e2fbc3`
- PR tracking document: `docs/prs/completed/pr_feat_gpu_byte_limb_packing.md`

Repository-document context used for this plan: `PLANS.md`, `DESIGN.md`, `docs/design/index.md`, `ARCHITECTURE.md`, `docs/architecture/index.md`, `docs/architecture/scope/index.md`, `docs/architecture/scope/poly.md`, `docs/architecture/scope/matrix.md`, `docs/architecture/scope/sampler.md`, `VERIFICATION.md`, `docs/verification/index.md`, `docs/verification/execplan_pre_creation.md`, `docs/verification/gpu_behavior_changes.md`, and `docs/verification/execplan_post_completion.md`.

## Purpose / Big Picture

After this change, GPU limbs are no longer stored as fixed-width `u64` arrays when transferred and persisted in CUDA-side matrix/poly data paths. Instead, each modulus uses the minimum required byte width (`u8` packing), reducing VRAM usage while preserving existing arithmetic behavior. A user can observe success by running the GPU unit tests and seeing that all existing GPU behaviors still pass while CUDA memory layout code now packs by modulus byte width.

## Progress

- [x] (2026-03-02 04:47Z) Completed pre-ExecPlan verification event actions from `docs/verification/execplan_pre_creation.md`: captured branch/status/log/PR context and determined main-branch work required a new feature branch.
- [x] (2026-03-02 04:49Z) Created and switched to `feat/gpu-byte-limb-packing`, created draft PR `https://github.com/MachinaIO/mxx/pull/57`, and added PR tracking file `docs/prs/active/pr_feat_gpu_byte_limb_packing.md`.
- [x] (2026-03-02 05:35Z) Inspected CUDA matrix/runtime/serde/trapdoor paths and identified all `u64`-assumption interfaces that required packed-byte migration.
- [x] (2026-03-02 05:35Z) Added design artifact `docs/design/gpu_limb_byte_packing.md` and registered it in `docs/design/index.md`.
- [x] (2026-03-02 05:35Z) Updated architecture scope documents (`docs/architecture/scope/matrix.md`, `docs/architecture/scope/poly.md`, `docs/architecture/scope/sampler.md`) with the packed-limb CUDA boundary contract.
- [x] (2026-03-02 05:35Z) Implemented modulus-width-aware byte-packed limb storage across CUDA runtime/matrix/trapdoor/serde paths, including per-limb metadata plumbing and packed load/store kernel access.
- [x] (2026-03-02 05:35Z) Validated behavior through focused GPU tests that cover gauss/trapdoor paths and compact-byte roundtrip paths.
- [x] (2026-03-02 05:35Z) Ran verification from `docs/verification/gpu_behavior_changes.md`: formatting, targeted GPU tests, full GPU lib tests, release no-run build, and required 300-run repetition (`failed_runs=0`).
- [x] (2026-03-02 05:35Z) Updated this plan with executed commands, discoveries, and final outcomes.
- [x] (2026-03-02 05:37Z) Moved this plan from `docs/plans/active/` to `docs/plans/completed/` after implementation and verification evidence were finalized.
- [x] (2026-03-02 05:37Z) Executed post-ExecPlan verification from `docs/verification/execplan_post_completion.md`: PR scope reviewed as complete, PR set to ready for review, PR tracking file moved to `docs/prs/completed/`, and final state prepared for commit/push.

## Surprises & Discoveries

- Observation: `gh pr view` on `main` initially returned no PR, so this feature required fresh branch/PR bootstrap before ExecPlan creation.
  Evidence: `gh pr view --json ...` on `main` returned `no pull requests found for branch "main"`.

- Observation: GPU-targeted tests fail inside sandbox with `gpu_device_synchronize failed: OS call failed or operation not supported on this OS`.
  Evidence: Initial targeted test runs in sandbox failed at `src/poly/dcrt/gpu.rs:209`; rerunning with escalated permissions succeeded.

- Observation: Host-facing RNS batch APIs still expect `u64`-coefficient layout; direct `cudaMemcpy2D` into packed limb buffers is incorrect for limb widths `< 8`.
  Evidence: Existing host byte contract uses `bytes_per_poly` as multiples of `8`; packed destination requires per-coefficient byte compaction.

## Decision Log

- Decision: Execute this work on a new branch/PR pair rather than reusing `main`.
  Rationale: `docs/verification/execplan_pre_creation.md` requires branch switch when current branch is `main`, and this feature is independently reviewable.
  Date/Author: 2026-03-02 / Codex

- Decision: Treat limb byte packing as a long-lived design and architecture contract that must be documented in `docs/design/` and `docs/architecture/`.
  Rationale: This change modifies a CUDA boundary invariant (data representation at Rust/CUDA interface) with cross-scope impact (`poly`, `matrix`, and `sampler` consumers).
  Date/Author: 2026-03-02 / Codex

- Decision: Keep host-side RNS batch ABI unchanged (`u64` per coefficient) and perform conversion at the CUDA boundary.
  Rationale: This preserves Rust API/test expectations while still reducing VRAM usage in persisted GPU limb storage.
  Date/Author: 2026-03-02 / Codex

- Decision: Unify packed access through `matrix_load_limb_u64`/`matrix_store_limb_u64` plus per-limb metadata (`stride_bytes`, `coeff_bytes`) in kernel launches.
  Rationale: A single access primitive prevents hidden `u64` assumptions and keeps NTT/arith/decompose/sampling/trapdoor/serde paths consistent.
  Date/Author: 2026-03-02 / Codex

## Outcomes & Retrospective

Implemented CUDA-side byte-packed limb storage with modulus-width metadata and migrated matrix/trapdoor/serde call chains away from fixed `u64` persisted buffers. Core GPU compile validation now succeeds and all required GPU verification commands in the selected event document pass, including the 300-run repetition with zero failures.

The result matches the plan purpose: persisted GPU limb storage is now byte-packed by modulus width, reducing VRAM pressure while preserving arithmetic behavior at the API level. Post-ExecPlan validation determined the PR scope is achieved, and PR `#57` is now marked ready for review.

## Design/Architecture/Verification Document Summary

Design documents:

- Referenced: `DESIGN.md`, `docs/design/index.md`
- Modified: `docs/design/index.md`
- Created: `docs/design/gpu_limb_byte_packing.md`
- Why: This change introduces a long-lived CUDA storage invariant and trade-off decision that future GPU/kernel work must follow.

Architecture documents:

- Referenced: `ARCHITECTURE.md`, `docs/architecture/index.md`, `docs/architecture/scope/index.md`, `docs/architecture/scope/poly.md`, `docs/architecture/scope/matrix.md`, `docs/architecture/scope/sampler.md`
- Modified: `docs/architecture/scope/matrix.md`, `docs/architecture/scope/poly.md`, `docs/architecture/scope/sampler.md`
- Why: The Rust/CUDA boundary contract changed for persisted limb representation, and scope docs now describe the packed-byte storage contract and ownership.

Verification documents:

- Referenced: `VERIFICATION.md`, `docs/verification/index.md`, `docs/verification/execplan_pre_creation.md`, `docs/verification/gpu_behavior_changes.md`, `docs/verification/execplan_post_completion.md`
- Policy updates: none (verification policy unchanged).
- Executed command policy usage:
  - `gpu_behavior_changes.md` followed for format, targeted GPU tests, full GPU tests, and 300-run repetition.
  - `execplan_post_completion.md` completed: PR readiness decision `READY`, `gh pr ready` executed, and PR tracking file moved to `docs/prs/completed/`.

## Context and Orientation

This repository has GPU-enabled matrix/poly/sampler execution paths in Rust under `src/` and CUDA kernels/utilities under `cuda/`. Existing code currently stores each limb in fixed `u64` slots regardless of modulus bit-size, which increases VRAM consumption when many moduli are significantly smaller than 64 bits. The migration target is to store limbs as byte-packed buffers where each modulus uses `ceil(modulus_bits / 8)` bytes. The migration must keep external mathematical behavior unchanged while updating memory layout assumptions at all relevant Rust/CUDA boundaries.

## Plan of Work

First, locate every structure, kernel, and transfer path that currently assumes `u64` limb storage. This includes host-side packing/unpacking code, device-side indexing math, and any serialization or temporary buffers that encode limb counts or stride lengths. Then define and implement a single source of truth for per-modulus byte width calculation and packed stride math. Update Rust host code to allocate and transfer packed `u8` buffers, and update CUDA code to read/write packed limb data based on modulus-specific byte width metadata. Keep arithmetic operations in existing integer types internally as needed, but all stored/transferred limb buffers should use packed bytes.

After implementation, add/adjust unit tests that check (1) packed encoding/decoding round trips for different modulus bit widths and (2) unchanged GPU math outcomes for representative operations. Then execute required GPU verification commands, record outputs in this plan, and finalize plan lifecycle steps including post-completion PR readiness transition.

## Concrete Steps

Run from repository root (`.`):

    rg -n "u64|limb|modulus|byte|pack|cuda|Gpu" src cuda
    rg -n "GpuDCRT|gpu|Matrix|Runtime|Serde|Trapdoor" src cuda
    # Edit implementation and docs files
    cargo +nightly fmt --all
    cargo test -r --lib --features gpu -- <targeted_filter>
    cargo test -r --lib --features gpu
    cargo test gpu -r --lib --features gpu --no-run
    set -uo pipefail
    mkdir -p logs
    bin="$(find target/release/deps -maxdepth 1 -type f -perm -111 -name 'mxx-*' | head -n 1)"
    if [ -z "$bin" ]; then
      echo "No GPU-enabled lib test binary found in target/release/deps" >&2
      exit 1
    fi

    fails=0
    : > logs/gpu_300_failures.txt
    for i in $(seq 1 300); do
      log="logs/gpu_300_iter_${i}.log"
      if ! "$bin" gpu --nocapture >"$log" 2>&1; then
        fails=$((fails+1))
        reason="$(rg -m1 -n 'panicked at|FAILED|error:|CUDA|assertion' "$log" || true)"
        printf 'iter=%03d log=%s reason=%s\n' "$i" "$log" "${reason:-unknown}" | tee -a logs/gpu_300_failures.txt
      fi
    done
    printf 'total_runs=300 failed_runs=%d\n' "$fails" | tee logs/gpu_300_summary.txt

Commands actually run during this plan (repository root):

    cargo +nightly fmt --all
    FIDESLIB_SKIP_SUBMODULE_UPDATE=1 cargo test --features gpu --lib --no-run
    FIDESLIB_SKIP_SUBMODULE_UPDATE=1 cargo test -r --lib --features gpu -- --list | rg "compact|trapdoor|gauss|sample_p1|gpu_matrix_"
    FIDESLIB_SKIP_SUBMODULE_UPDATE=1 cargo test -r --lib --features gpu matrix::gpu_dcrt_poly::tests::test_gpu_matrix_gauss_samp_gq_arb_base_relation -- --exact
    FIDESLIB_SKIP_SUBMODULE_UPDATE=1 cargo test -r --lib --features gpu sampler::trapdoor::gpu::tests::test_gpu_preimage_generation_square_not_plain_gadget_solution -- --exact
    FIDESLIB_SKIP_SUBMODULE_UPDATE=1 cargo test -r --lib --features gpu
    FIDESLIB_SKIP_SUBMODULE_UPDATE=1 cargo test gpu -r --lib --features gpu --no-run
    set -uo pipefail
    mkdir -p logs
    bin="$(find target/release/deps -maxdepth 1 -type f -perm -111 -name 'mxx-*' | head -n 1)"
    if [ -z "$bin" ]; then
      echo "No GPU-enabled lib test binary found in target/release/deps" >&2
      exit 1
    fi
    fails=0
    : > logs/gpu_300_failures.txt
    for i in $(seq 1 300); do
      log="logs/gpu_300_iter_${i}.log"
      if ! "$bin" gpu --nocapture >"$log" 2>&1; then
        fails=$((fails+1))
        reason="$(rg -m1 -n 'panicked at|FAILED|error:|CUDA|assertion' "$log" || true)"
        printf 'iter=%03d log=%s reason=%s\n' "$i" "$log" "${reason:-unknown}" | tee -a logs/gpu_300_failures.txt
      fi
    done
    printf 'total_runs=300 failed_runs=%d\n' "$fails" | tee logs/gpu_300_summary.txt

## Validation and Acceptance

Acceptance is satisfied when:

1. CUDA/Rust GPU limb storage paths no longer use fixed `u64` arrays for persisted/transferred limb buffers and instead use modulus-width-aware packed byte storage.
2. New/updated tests demonstrate correct pack/unpack behavior for multiple modulus widths.
3. `cargo +nightly fmt --all` succeeds.
4. Scope-relevant GPU unit tests pass.
5. Full `cargo test -r --lib --features gpu` passes (required because this is foundational CUDA layout work).
6. 300-run repetition completes with summarized failure count in `logs/gpu_300_summary.txt` and tracked failures (if any) in `logs/gpu_300_failures.txt`.
7. Post-ExecPlan verification determines PR is ready and transitions PR/document state accordingly.

Current status: Criteria 1-7 are satisfied.

## Idempotence and Recovery

Packing-layout edits are safe to iterate when done additively and verified after each step. If an interface mismatch causes compile/runtime failures, revert only the latest local edit chunk and re-run targeted GPU tests before proceeding. The 300-run command is idempotent for evidence collection because it overwrites `logs/gpu_300_summary.txt` and `logs/gpu_300_failures.txt` each run.

## Artifacts and Notes

Expected touched areas (final set may vary after code inspection):

- `src/matrix/gpu_dcrt_poly.rs`
- `src/poly/dcrt/gpu.rs`
- `cuda/include/matrix/*.cuh`
- `cuda/src/matrix/*.cu`
- `docs/design/*`
- `docs/architecture/scope/*.md`

Verification artifacts:

- `logs/gpu_300_summary.txt`
- `logs/gpu_300_failures.txt`
- `logs/gpu_300_iter_*.log`

Recorded verification outcomes:

- `cargo +nightly fmt --all`: success.
- Targeted GPU tests:
  - `matrix::gpu_dcrt_poly::tests::test_gpu_matrix_gauss_samp_gq_arb_base_relation`: pass (with escalated GPU execution).
  - `sampler::trapdoor::gpu::tests::test_gpu_preimage_generation_square_not_plain_gadget_solution`: pass (with escalated GPU execution).
- Full GPU lib tests:
  - `cargo test -r --lib --features gpu`: `168 passed; 0 failed; 2 ignored`.
- Repetition:
  - `logs/gpu_300_summary.txt`: `total_runs=300 failed_runs=0`.

## Interfaces and Dependencies

The implementation must preserve existing public GPU-facing Rust interfaces unless a breaking API change is unavoidable. If signatures change, all call sites across `matrix`, `poly`, and `sampler` must be updated atomically in the same change. Any newly introduced layout metadata (for example byte width arrays or packed strides) must have a clear single owner in code so both Rust and CUDA sides derive identical indexing behavior.

Revision note (2026-03-02, Codex): Initial ExecPlan created with pre-creation verification evidence and end-to-end implementation/verification lifecycle actions.
Revision note (2026-03-02, Codex): Updated plan with completed implementation/design/architecture work, executed verification command evidence, and readiness state before post-ExecPlan transition.
Revision note (2026-03-02, Codex): Finalized post-ExecPlan validation results after moving plan to completed state and transitioning PR/document readiness lifecycle.

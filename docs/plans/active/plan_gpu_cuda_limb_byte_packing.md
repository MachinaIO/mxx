# Reduce GPU Limb VRAM with Modulus-Width Byte Packing

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan is maintained under `PLANS.md` and follows that file’s rules.

ExecPlan start context:
- Branch at start: `feat/gpu-byte-limb-packing`
- Commit at start: `a5e2e0f1195e7a6f3ca16950d05d1267b7e2fbc3`
- PR tracking document: `docs/prs/active/pr_feat_gpu_byte_limb_packing.md`

Repository-document context used for this plan: `PLANS.md`, `DESIGN.md`, `docs/design/index.md`, `ARCHITECTURE.md`, `docs/architecture/index.md`, `docs/architecture/scope/index.md`, `docs/architecture/scope/poly.md`, `docs/architecture/scope/matrix.md`, `docs/architecture/scope/sampler.md`, `VERIFICATION.md`, `docs/verification/index.md`, `docs/verification/execplan_pre_creation.md`, `docs/verification/gpu_behavior_changes.md`, and `docs/verification/execplan_post_completion.md`.

## Purpose / Big Picture

After this change, GPU limbs are no longer stored as fixed-width `u64` arrays when transferred and persisted in CUDA-side matrix/poly data paths. Instead, each modulus uses the minimum required byte width (`u8` packing), reducing VRAM usage while preserving existing arithmetic behavior. A user can observe success by running the GPU unit tests and seeing that all existing GPU behaviors still pass while CUDA memory layout code now packs by modulus byte width.

## Progress

- [x] (2026-03-02 04:47Z) Completed pre-ExecPlan verification event actions from `docs/verification/execplan_pre_creation.md`: captured branch/status/log/PR context and determined main-branch work required a new feature branch.
- [x] (2026-03-02 04:49Z) Created and switched to `feat/gpu-byte-limb-packing`, created draft PR `https://github.com/MachinaIO/mxx/pull/57`, and added PR tracking file `docs/prs/active/pr_feat_gpu_byte_limb_packing.md`.
- [ ] Inspect current CUDA limb representation and identify all host/device interfaces that assume `u64` per limb.
- [ ] Update design documentation for long-lived limb packing invariant and add it to `docs/design/index.md`.
- [ ] Update architecture documentation for CUDA boundary/storage layout impacts (scope docs and/or dependencies/features docs as needed).
- [ ] Implement modulus-width-aware byte-packed limb storage (`u8`) across impacted Rust/CUDA interfaces while preserving arithmetic correctness.
- [ ] Add or update focused tests that validate packing/unpacking correctness and unchanged math behavior on GPU paths.
- [ ] Run verification mapped from `docs/verification/gpu_behavior_changes.md` in action order: formatting, targeted GPU tests, full GPU lib tests (required for foundational CUDA memory layout changes), and 300-run GPU repetition command.
- [ ] Update this plan with executed command outputs, discovery notes, and final outcomes.
- [ ] Move this plan to `docs/plans/completed/` after all actions and evidence are complete.
- [ ] Execute post-ExecPlan verification event from `docs/verification/execplan_post_completion.md`: decide PR readiness, set PR ready for review, move PR tracking file to `docs/prs/completed/`, commit final plan state, and push.

## Surprises & Discoveries

- Observation: `gh pr view` on `main` initially returned no PR, so this feature required fresh branch/PR bootstrap before ExecPlan creation.
  Evidence: `gh pr view --json ...` on `main` returned `no pull requests found for branch "main"`.

## Decision Log

- Decision: Execute this work on a new branch/PR pair rather than reusing `main`.
  Rationale: `docs/verification/execplan_pre_creation.md` requires branch switch when current branch is `main`, and this feature is independently reviewable.
  Date/Author: 2026-03-02 / Codex

- Decision: Treat limb byte packing as a long-lived design and architecture contract that must be documented in `docs/design/` and `docs/architecture/`.
  Rationale: This change modifies a CUDA boundary invariant (data representation at Rust/CUDA interface) with cross-scope impact (`poly`, `matrix`, and `sampler` consumers).
  Date/Author: 2026-03-02 / Codex

## Outcomes & Retrospective

Pending. This section will be finalized after implementation, verification, and post-ExecPlan completion checks.

## Design/Architecture/Verification Document Summary

Design documents:

- Referenced: `DESIGN.md`, `docs/design/index.md`
- Planned updates: add a concrete design artifact describing modulus-width byte packing invariants for GPU limb storage and register it in `docs/design/index.md`.

Architecture documents:

- Referenced: `ARCHITECTURE.md`, `docs/architecture/index.md`, `docs/architecture/scope/index.md`, `docs/architecture/scope/poly.md`, `docs/architecture/scope/matrix.md`, `docs/architecture/scope/sampler.md`
- Planned updates: scope documentation updates for Rust/CUDA boundary layout changes where GPU limb storage representation is defined.

Verification documents:

- Referenced: `VERIFICATION.md`, `docs/verification/index.md`, `docs/verification/execplan_pre_creation.md`, `docs/verification/gpu_behavior_changes.md`, `docs/verification/execplan_post_completion.md`
- Planned command policy usage: follow `gpu_behavior_changes.md` during implementation completion and `execplan_post_completion.md` after plan completion.

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

## Validation and Acceptance

Acceptance is satisfied when:

1. CUDA/Rust GPU limb storage paths no longer use fixed `u64` arrays for persisted/transferred limb buffers and instead use modulus-width-aware packed byte storage.
2. New/updated tests demonstrate correct pack/unpack behavior for multiple modulus widths.
3. `cargo +nightly fmt --all` succeeds.
4. Scope-relevant GPU unit tests pass.
5. Full `cargo test -r --lib --features gpu` passes (required because this is foundational CUDA layout work).
6. 300-run repetition completes with summarized failure count in `logs/gpu_300_summary.txt` and tracked failures (if any) in `logs/gpu_300_failures.txt`.
7. Post-ExecPlan verification determines PR is ready and transitions PR/document state accordingly.

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

## Interfaces and Dependencies

The implementation must preserve existing public GPU-facing Rust interfaces unless a breaking API change is unavoidable. If signatures change, all call sites across `matrix`, `poly`, and `sampler` must be updated atomically in the same change. Any newly introduced layout metadata (for example byte width arrays or packed strides) must have a clear single owner in code so both Rust and CUDA sides derive identical indexing behavior.

Revision note (2026-03-02, Codex): Initial ExecPlan created with pre-creation verification evidence and end-to-end implementation/verification lifecycle actions.

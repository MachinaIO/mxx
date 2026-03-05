# Add deterministic coefficient-wise LUT evaluation coverage for `PolyPltEvaluator`

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document follows `.agents/skills/eternal-cycler/PLANS.md` from the repository root and will be maintained in accordance with that policy.

## Purpose / Big Picture

`PublicLUT` behavior in this PR now supports evaluation for non-constant polynomial terms through `PolyPltEvaluator`. After this change, `src/lookup/poly.rs` includes a deterministic test proving coefficient-wise LUT behavior: each input coefficient is treated as an independent LUT query, and the output polynomial equals applying the LUT to each coefficient in order. This can be observed by running the targeted lookup tests and seeing the new assertion pass.

## Progress

- [x] (2026-03-04 22:36Z) action_id=a1; mode=serial; depends_on=none; file_locks=src/lookup/poly.rs; verify_events=action.cpu_behavior; worker_type=default; added deterministic coefficient-wise LUT evaluation coverage in `src/lookup/poly.rs` and validated it with targeted unit test execution plus `action.cpu_behavior` pass.

## Verification Ledger

<!-- verification-ledger:start -->
- attempt_record: event_id=execplan.post_creation; attempt=1; status=pass; started_at=2026-03-04 22:34Z; finished_at=2026-03-04 22:34Z; commands=git branch --show-current git status --short gh pr status gh pr view --json number,title,body,state,headRefName,baseRefName,url mkdir -p /home/sora/codes/mxx/eternal-cycler-out/prs/active write /home/sora/codes/mxx/eternal-cycler-out/prs/active/pr_feat_semi-packing.md append PR Tracking Linkage to plan capture start tracked snapshot capture start untracked snapshot; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=action.cpu_behavior; attempt=1; status=pass; started_at=2026-03-04 22:36Z; finished_at=2026-03-04 22:36Z; commands=cargo +nightly fmt --all cargo test -r --lib -- lookup::poly::tests::public_lookup_evaluates_each_coefficient_with_lut; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=execplan.post_completion; attempt=1; status=pass; started_at=2026-03-04 22:37Z; finished_at=2026-03-04 22:37Z; commands=rg -n eternal-cycler-out/prs/active/|eternal-cycler-out/prs/completed/ <plan> open /home/sora/codes/mxx/eternal-cycler-out/prs/active/pr_feat_semi-packing.md git status --short; failure_summary=none; notify_reference=not_requested;
<!-- verification-ledger:end -->

## Surprises & Discoveries

- Observation: `PublicLut::new(..., None)` computes `max_output_row` by iterating the full LUT domain, which would populate the query log before evaluator execution.
  Evidence: `src/lookup/mod.rs` `compute_max_output_row` is called when `max_output_row` is `None`, and iterates `0..len`.

## Decision Log

- Decision: Use `action.cpu_behavior` as the verification event for this task.
  Rationale: The requested change is a Rust CPU-side behavior test in `src/lookup/poly.rs`.
  Date/Author: 2026-03-05 / BUILDER agent.
- Decision: Instrument LUT query inputs with `Arc<Mutex<Vec<u64>>>` and provide explicit `max_output_row` to isolate evaluator-time queries.
  Rationale: The requirement explicitly asks to verify each coefficient is used as LUT input; query recording plus explicit `max_output_row` keeps the assertion deterministic and avoids construction-time noise.
  Date/Author: 2026-03-04 / BUILDER agent.

## Outcomes & Retrospective

Completed the requested lookup test coverage in `src/lookup/poly.rs`.

- Added `lookup::poly::tests::public_lookup_evaluates_each_coefficient_with_lut`.
- Test fixture uses deterministic `DCRTPolyParams`, deterministic input coefficients, and deterministic LUT output mapping.
- Test asserts two required properties:
  1. The observed LUT query sequence exactly matches input polynomial coefficients in order.
  2. The evaluator output polynomial equals the coefficient-wise LUT-applied expected polynomial.
- Validation commands:
  - `cargo test -r --lib -- lookup::poly::tests::public_lookup_evaluates_each_coefficient_with_lut` (pass)
  - `.agents/skills/eternal-cycler/scripts/execplan_gate.sh --plan eternal-cycler-out/plans/active/20260304T223434Z_poly_lut_coeffwise_test.md --event action.cpu_behavior` with `EXECPLAN_TEST_FILTER=lookup::poly::tests::public_lookup_evaluates_each_coefficient_with_lut EXECPLAN_RUN_FULL_LIB_TESTS=0` (pass)
- Verification scripts status:
  - Referenced `.agents/skills/eternal-cycler/scripts/execplan_gate.sh` and `.agents/skills/execplan-event-action-cpu-behavior/scripts/run_event.sh`.
  - No verification scripts were created or modified; existing scripts were sufficient for this task.

## Context and Orientation

The lookup polynomial tests are defined in `src/lookup/poly.rs`. The current PR changed `PublicLUT` behavior and extended `PolyPltEvaluator` to support LUT evaluation for non-constant polynomial terms. The new test must validate two concrete properties for one deterministic input polynomial: every coefficient is passed through LUT lookup independently, and the returned polynomial exactly matches coefficient-wise LUT application.

## Plan of Work

Implement one focused test near existing lookup tests in `src/lookup/poly.rs`. Build a deterministic LUT fixture and a deterministic polynomial input with multiple coefficient values, including boundary-relevant values for the LUT domain used by current tests. Evaluate using `PolyPltEvaluator`, compute the expected polynomial by mapping each input coefficient through the same LUT definition, and assert equality of the full output polynomial. Keep setup minimal by reusing existing helper constructors in the same module where possible.

## Concrete Steps

From the repository root:

1. Inspect existing tests in `src/lookup/poly.rs` to align fixture style and available helpers.
2. Add one deterministic test that:
   - constructs or reuses a deterministic LUT,
   - constructs an input polynomial with diverse coefficients,
   - runs evaluator LUT evaluation,
   - computes expected output via coefficient-wise LUT mapping,
   - asserts exact polynomial equality.
3. Run `cargo test -r --lib -- lookup` (or the narrowest stable filter matching this module) to validate behavior.
4. Run verification gate: `.agents/skills/eternal-cycler/scripts/execplan_gate.sh --plan <plan_md> --event action.cpu_behavior`.
5. Update this plan document sections (`Progress`, `Verification Ledger`, `Surprises & Discoveries`, `Decision Log`, `Outcomes & Retrospective`) with final results.

Expected signal:

- The new test fails before the code change and passes after the change.
- `action.cpu_behavior` gate returns `STATUS=pass`.

## Validation and Acceptance

Acceptance is met when all of the following are true:

- `src/lookup/poly.rs` contains a deterministic test covering coefficient-wise LUT evaluation for `PolyPltEvaluator` with non-constant polynomial input.
- The test demonstrates that each coefficient is used as LUT input and that output polynomial equality holds coefficient-by-coefficient.
- Targeted library tests pass and `action.cpu_behavior` gate passes.

## Idempotence and Recovery

Edits are idempotent. Re-running the same test command should produce the same pass result. If verification fails, inspect failure output, adjust only lookup-test fixtures/assertions, and re-run the same commands with the next gate attempt.

## Artifacts and Notes

Gate attempts and command summaries will be recorded in `Verification Ledger`.

## Interfaces and Dependencies

No public interface changes are intended. The task updates test coverage in `src/lookup/poly.rs` and exercises existing `PublicLUT` and `PolyPltEvaluator` behavior.

Revision note (2026-03-05, BUILDER): Created initial ExecPlan for coefficient-wise LUT evaluation test coverage.
Revision note (2026-03-04, BUILDER): Recorded implementation completion, deterministic test design details, and verification outcomes.

## PR Tracking Linkage

- pr_tracking_doc: eternal-cycler-out/prs/active/pr_feat_semi-packing.md
- execplan_start_branch: feat/semi-packing
- execplan_start_commit: e96aa8764f27fb5bd42711184fac405b23155a95

## ExecPlan Start Snapshot

<!-- execplan-start-tracked:start -->
- start_tracked_change: (none)	(none)
<!-- execplan-start-tracked:end -->

<!-- execplan-start-untracked:start -->
- start_untracked_file: a4341a44291e3d9effb5d6d431eff7d8a6f3c8b2	eternal-cycler-out/plans/active/20260304T223434Z_poly_lut_coeffwise_test.md
- start_untracked_file: 1038ce7c2b7a18187075f92c8c881332a4af4209	eternal-cycler-out/prs/active/pr_feat_semi-packing.md
<!-- execplan-start-untracked:end -->

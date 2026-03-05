# Propose arbitrary-polynomial GGH15 lookup evaluation design

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` are updated as work proceeds.

This document follows `.agents/skills/eternal-cycler/PLANS.md` from the repository root.

## Purpose / Big Picture

`src/lookup/ggh15_eval.rs` currently evaluates public lookups only when the input plaintext is a constant polynomial, because the existing auxiliary-matrix construction binds each LUT row to a constant representative derived from its row index. After this change, the repository will contain a design document under `docs/design/` that explains a concrete mathematical extension to arbitrary input polynomials and shows how to implement it later without multiplying preimage counts by the ring dimension. A reader should be able to open the new design document, trace the current constant-only bottleneck, and see the proposed lifting strategy, storage impact, and later verification targets.

## Progress

- [x] (2026-03-05 20:16Z) action_id=a1; mode=serial; depends_on=none; file_locks=docs/design/ggh15_arbitrary_polynomial_lookup.md,docs/design/index.md,eternal-cycler-out/plans/active/20260305T201317Z_ggh15_arbitrary_poly_lookup_design.md; verify_events=action.docs_only; worker_type=default; inspected the current GGH15 lookup evaluator, derived an explicit-row-representative extension that avoids per-coefficient preimage blow-up, documented the proposal in `docs/design/ggh15_arbitrary_polynomial_lookup.md`, updated `docs/design/index.md`, and verified the docs-only action on attempt 1.

## Verification Ledger

<!-- verification-ledger:start -->
- attempt_record: event_id=execplan.post_creation; attempt=1; status=pass; started_at=2026-03-05 20:14Z; finished_at=2026-03-05 20:14Z; commands=git branch --show-current git status --short gh pr status gh pr view --json number,title,body,state,headRefName,baseRefName,url mkdir -p /home/sora/codes/mxx/eternal-cycler-out/prs/active write /home/sora/codes/mxx/eternal-cycler-out/prs/active/pr_feat_auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020.md append PR Tracking Linkage to plan capture start tracked snapshot capture start untracked snapshot; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=action.docs_only; attempt=1; status=pass; started_at=2026-03-05 20:16Z; finished_at=2026-03-05 20:16Z; commands=git diff --name-only --relative HEAD -- git ls-files --others --exclude-standard rg -n <placeholder-pattern> <changed-doc-targets>; failure_summary=none; notify_reference=not_requested;
- attempt_record: event_id=execplan.post_completion; attempt=1; status=pass; started_at=2026-03-05 20:16Z; finished_at=2026-03-05 20:16Z; commands=rg -n eternal-cycler-out/prs/active/|eternal-cycler-out/prs/completed/ <plan> open /home/sora/codes/mxx/eternal-cycler-out/prs/active/pr_feat_auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020.md git status --short; failure_summary=none; notify_reference=not_requested;
<!-- verification-ledger:end -->

## Surprises & Discoveries

- Observation: `PublicLut::get` can already classify some non-constant polynomials through `to_const_int()`, but the GGH15 LUT preimage builder still hard-codes each row representative as the constant polynomial `idx * 1`.
  Evidence: `src/lookup/mod.rs` uses `x.to_const_int()` inside `PublicLut::new_from_usize_range`, while `src/lookup/ggh15_eval.rs` constructs the LUT `vx` term from `M::P::from_usize_to_constant(params, idx)`.
- Observation: the gate-side stage-5 runtime path already accepts an arbitrary plaintext polynomial because `small_decomposed_identity_chunk_from_scalar` takes a full `P`, not an integer.
  Evidence: `src/matrix/mod.rs` defines `small_decomposed_identity_chunk_from_scalar(..., scalar: &Self::P, ...)`, and `GGH15BGGEncodingPltEvaluator::public_lookup` feeds it the runtime plaintext `x`.

## Decision Log

- Decision: treat this task as a docs-only design update and defer all implementation-body edits in `src/lookup/ggh15_eval.rs`.
  Rationale: the request explicitly asks for a concrete proposal first and forbids implementation changes in this run.
  Date/Author: 2026-03-05 / BUILDER agent.
- Decision: keep the proposal focused on the GGH15 public-key evaluator and the matching encoding-side equations, because any viable design must preserve the algebra that `GGH15BGGEncodingPltEvaluator` reconstructs from stored auxiliary matrices.
  Rationale: a design that ignores the existing evaluator split would not be actionable for a later implementation.
  Date/Author: 2026-03-05 / BUILDER agent.
- Decision: model arbitrary-polynomial support by storing one explicit representative polynomial per LUT row instead of decomposing the plaintext into ring-dimension-many coefficient subproblems.
  Rationale: the cancellation proof only needs the representative polynomial `x_k` inside the existing compact gadget relation, which keeps trapdoor solves proportional to rows and gates rather than to ring dimension.
  Date/Author: 2026-03-05 / BUILDER agent.
- Decision: preserve the current gate-side preimage family and change only the LUT-row preimage equation plus LUT row enumeration semantics.
  Rationale: stage 5 already linearizes an arbitrary polynomial `x` through compact decomposition, so widening the row representative is sufficient and keeps the implementation surface narrow.
  Date/Author: 2026-03-05 / BUILDER agent.

## Outcomes & Retrospective

Completed the design-first task without modifying the implementation body. Added `docs/design/ggh15_arbitrary_polynomial_lookup.md` to specify the explicit-row-representative construction, the cancellation algebra, the `PublicLut` data-model changes needed for later code work, and a concrete validation plan. Updated `docs/design/index.md` so the artifact is discoverable through the required design-doc entry point.

Verification status so far: `execplan.post_creation` passed on attempt 1 and `action.docs_only` passed on attempt 1.

Verification scripts referenced and left unchanged:

- `.agents/skills/eternal-cycler/scripts/execplan_gate.sh`
- `.agents/skills/execplan-event-action-docs-only/scripts/run_event.sh`

## Context and Orientation

The current lookup stack lives in `src/lookup/`. `src/lookup/mod.rs` defines `PublicLut<P>`; its `new_from_usize_range` helper selects rows by applying `x.to_const_int()` to the runtime polynomial, which means the selector can describe domains richer than constant polynomials. In `src/lookup/ggh15_eval.rs`, however, `GGH15BGGPubKeyPltEvaluator` samples each LUT row as if its representative were always the constant polynomial `idx * 1`, because the LUT `vx` relation is built from `M::P::from_usize_to_constant(params, idx)`. `GGH15BGGEncodingPltEvaluator::public_lookup` later reconstructs the evaluated encoding from those matrices and multiplies the stage-5 preimage by the compact decomposition of the actual plaintext polynomial `x`. The task is to document a mathematically concrete extension that makes the LUT-row representative and the runtime plaintext agree for arbitrary polynomials, while keeping the number of sampled preimages proportional to gate count or LUT size rather than to `gate_count * ring_dimension` or `lut_size * ring_dimension`.

## Plan of Work

First, inspect the lookup evaluator code paths that assume constant inputs, especially the LUT row selection in `src/lookup/mod.rs` and the algebra in `src/lookup/ggh15_eval.rs` around `preimage_gate1`, `preimage_gate2_identity`, `preimage_gate2_gy`, `preimage_gate2_v`, and `preimage_gate2_vx`. Then derive a design that replaces constant row representatives with explicit ring-polynomial representatives while preserving one preimage family per gate and one per LUT row. Capture that design in a new document `docs/design/ggh15_arbitrary_polynomial_lookup.md`, including the mathematical identities, data that would need to be stored, the expected asymptotic cost, and why the forbidden per-coefficient blow-up is avoided. Finally, register the new design artifact in `docs/design/index.md`, update this ExecPlan with findings, and run the docs-only verification gate.

## Concrete Steps

From the repository root:

1. Read `src/lookup/mod.rs`, `src/lookup/ggh15_eval.rs`, `docs/design/index.md`, and `DESIGN.md` to extract the current constant-only assumptions and the repository design-document contract.
2. Write `docs/design/ggh15_arbitrary_polynomial_lookup.md` in English. The document must explain the current limitation, the proposed polynomial-selector formulation, the required precomputed objects, the changes implied for `GGH15BGGPubKeyPltEvaluator` and `GGH15BGGEncodingPltEvaluator`, and a later implementation and validation outline.
3. Update `docs/design/index.md` with the new design document entry and role summary.
4. Update this ExecPlan with discoveries and decision details, then run `.agents/skills/eternal-cycler/scripts/execplan_gate.sh --plan eternal-cycler-out/plans/active/20260305T201317Z_ggh15_arbitrary_poly_lookup_design.md --event action.docs_only`.

Expected signal:

- The new design document explains a concrete arbitrary-polynomial extension that keeps preimage sampling at one object per gate-side relation and one object per LUT row.
- The docs-only gate exits with `STATUS=pass`.

## Validation and Acceptance

Acceptance is satisfied when a first-time reader can inspect `docs/design/ggh15_arbitrary_polynomial_lookup.md` and answer three questions without reading external material: why the current GGH15 lookup path only works for constant polynomials, what algebraic replacement allows arbitrary polynomial inputs, and why that replacement avoids a ring-dimension multiplier in preimage count. The design index must link to the document, and the ExecPlan action must pass `action.docs_only`.

## Idempotence and Recovery

The documentation edits are safe to repeat. Re-running the design write-up should only refine wording, equations, or rationale in the same markdown files. If `action.docs_only` fails, inspect the changed markdown set for non-document paths or stale placeholder words and re-run the same gate command with the next attempt number.

## Artifacts and Notes

Important command outcomes and gate attempts will be recorded directly in `## Verification Ledger`. No external scratch files are required.

## Interfaces and Dependencies

This task adds no runtime interface changes in code. It adds one long-lived design artifact under `docs/design/` and updates `docs/design/index.md`. The proposal will refer to the existing lookup interfaces `PublicLut<P>` in `src/lookup/mod.rs`, `GGH15BGGPubKeyPltEvaluator` and `GGH15BGGEncodingPltEvaluator` in `src/lookup/ggh15_eval.rs`, and the design-document policy in `DESIGN.md`.

Revision note (2026-03-05, BUILDER): Created the initial ExecPlan for the arbitrary-polynomial GGH15 lookup design task.
Revision note (2026-03-05, BUILDER): Recorded the final design proposal and the supporting algebraic discoveries before running action verification.
Revision note (2026-03-05, BUILDER): Marked the docs-only action complete after the verification gate passed.

## PR Tracking Linkage

- pr_tracking_doc: eternal-cycler-out/prs/active/pr_feat_auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020.md
- execplan_start_branch: feat/auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020
- execplan_start_commit: 9d8d3cd8b025a2fea37d5b46f20dc1edd3638bc7

## ExecPlan Start Snapshot

<!-- execplan-start-tracked:start -->
- start_tracked_change: (deleted)	eternal-cycler-out/prs/active/pr_feat_auto-eternal-cycler-home-sora-codes-mxx-agent-20260304T211020.md
<!-- execplan-start-tracked:end -->

<!-- execplan-start-untracked:start -->
- start_untracked_file: (none)	(none)
<!-- execplan-start-untracked:end -->

# Rename Scope Doc Filenames and Merge Lessons into Maintenance Rule

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan is maintained under `PLANS.md` and follows that file’s rules.

Repository-document context used for this plan: `PLANS.md`, `AGENTS.md`, and `ARCHITECTURE.md`, plus existing architecture docs under `docs/architecture/`. `DESIGN.md` and `VERIFICATION.md` do not exist in the current tree, so this plan uses those listed repository files as fallback policy context.

## Purpose / Big Picture

After this change, scope documentation filenames will use the direct target domain name (for example `lookup.md` instead of `src_lookup.md`) and architecture guidance will be consolidated so the prior “Agent Lessons” content becomes part of the `Maintenance rule` section in `docs/architecture/index.md`.

## Progress

- [x] (2026-03-02 00:54Z) Read `PLANS.md`, `AGENTS.md`, and current architecture scope/index files.
- [x] (2026-03-02 00:55Z) Renamed `docs/architecture/scope/src_*.md` documents to direct domain names (`bgg.md`, `lookup.md`, etc.; `src_root_modules.md` -> `root_modules.md`).
- [x] (2026-03-02 00:55Z) Updated scope links and dependency labels in `docs/architecture/scope/index.md` and related scope docs.
- [x] (2026-03-02 00:55Z) Merged the previous `Agent Lessons` guidance into the `Maintenance rule` section in `docs/architecture/index.md` and removed the standalone heading.
- [x] (2026-03-02 00:55Z) Validated file/link/reference consistency (`rg -n \"src_\" docs/architecture/scope docs/architecture/index.md` returned no matches; scope file list confirmed).
- [x] (2026-03-02 00:55Z) Moved this plan to `docs/plans/completed/`.

## Surprises & Discoveries

- Observation: Current naming still carries `src_` prefixes in filenames and dependency labels, so a rename requires both file moves and widespread text/link updates.
  Evidence: `docs/architecture/scope/index.md` and multiple scope docs use `src_*` labels.

## Decision Log

- Decision: Rename `src_root_modules.md` to `root_modules.md` as the corresponding non-directory scope name.
  Rationale: The user requested removing the `src_` prefix and using direct target naming; for root-level `src` files, `root_modules` is the direct scope name already used in content.
  Date/Author: 2026-03-02 / Codex

## Outcomes & Retrospective

Requested naming/rule cleanup is complete. Scope doc filenames now use direct domain names instead of `src_` prefixes, and all scope dependency labels/links were normalized to those names. The guidance from the old `Agent Lessons` section now lives directly under `Maintenance rule` in `docs/architecture/index.md`, so maintenance and anti-regression policy are no longer split across headings.

No functional code changes were made; this was a documentation-structure update only.

## Design/Architecture/Verification Document Summary

Design documents:

- Referenced: none (no `DESIGN.md` exists).
- Created/modified: none.
- Why unchanged: this task is architecture-document naming and rule consolidation.

Architecture documents:

- Referenced: `ARCHITECTURE.md`, `docs/architecture/index.md`, `docs/architecture/scope/*`.
- Modified:
  - `docs/architecture/index.md`
  - `docs/architecture/scope/index.md`
  - `docs/architecture/scope/tests.md`
  - `docs/architecture/scope/benches.md`
  - `docs/architecture/scope/bgg.md`
  - `docs/architecture/scope/circuit.md`
  - `docs/architecture/scope/commit.md`
  - `docs/architecture/scope/element.md`
  - `docs/architecture/scope/gadgets.md`
  - `docs/architecture/scope/lookup.md`
  - `docs/architecture/scope/matrix.md`
  - `docs/architecture/scope/poly.md`
  - `docs/architecture/scope/root_modules.md`
  - `docs/architecture/scope/sampler.md`
  - `docs/architecture/scope/simulator.md`
  - `docs/architecture/scope/storage.md`
- Renamed:
  - `docs/architecture/scope/src_bgg.md` -> `docs/architecture/scope/bgg.md`
  - `docs/architecture/scope/src_circuit.md` -> `docs/architecture/scope/circuit.md`
  - `docs/architecture/scope/src_commit.md` -> `docs/architecture/scope/commit.md`
  - `docs/architecture/scope/src_element.md` -> `docs/architecture/scope/element.md`
  - `docs/architecture/scope/src_gadgets.md` -> `docs/architecture/scope/gadgets.md`
  - `docs/architecture/scope/src_lookup.md` -> `docs/architecture/scope/lookup.md`
  - `docs/architecture/scope/src_matrix.md` -> `docs/architecture/scope/matrix.md`
  - `docs/architecture/scope/src_poly.md` -> `docs/architecture/scope/poly.md`
  - `docs/architecture/scope/src_root_modules.md` -> `docs/architecture/scope/root_modules.md`
  - `docs/architecture/scope/src_sampler.md` -> `docs/architecture/scope/sampler.md`
  - `docs/architecture/scope/src_simulator.md` -> `docs/architecture/scope/simulator.md`
  - `docs/architecture/scope/src_storage.md` -> `docs/architecture/scope/storage.md`

Verification documents:

- Referenced: none (no `VERIFICATION.md` exists).
- Created/modified: none planned.
- Why unchanged: verification-policy updates are out of scope for this docs-structure change.

## Context and Orientation

The current architecture scope set uses filenames and dependency labels prefixed with `src_`. The request is to remove that prefix and use direct domain names in scope docs under `docs/architecture/scope/`. The architecture index currently separates `Maintenance rule` and `Agent Lessons`; the request is to integrate both into `Maintenance rule`.

## Plan of Work

Rename all `docs/architecture/scope/src_*.md` files to unprefixed names and update titles/dependency labels in those files. Update `docs/architecture/scope/index.md` links and dependency statements to the new naming. Then rewrite `docs/architecture/index.md` so the lesson content is embedded directly in `Maintenance rule` and the standalone `Agent Lessons` heading is removed. Finally run local consistency checks and move this plan to `completed`.

## Concrete Steps

Run from `.`:

    mv docs/architecture/scope/src_bgg.md docs/architecture/scope/bgg.md
    mv docs/architecture/scope/src_circuit.md docs/architecture/scope/circuit.md
    mv docs/architecture/scope/src_commit.md docs/architecture/scope/commit.md
    mv docs/architecture/scope/src_element.md docs/architecture/scope/element.md
    mv docs/architecture/scope/src_gadgets.md docs/architecture/scope/gadgets.md
    mv docs/architecture/scope/src_lookup.md docs/architecture/scope/lookup.md
    mv docs/architecture/scope/src_matrix.md docs/architecture/scope/matrix.md
    mv docs/architecture/scope/src_poly.md docs/architecture/scope/poly.md
    mv docs/architecture/scope/src_root_modules.md docs/architecture/scope/root_modules.md
    mv docs/architecture/scope/src_sampler.md docs/architecture/scope/sampler.md
    mv docs/architecture/scope/src_simulator.md docs/architecture/scope/simulator.md
    mv docs/architecture/scope/src_storage.md docs/architecture/scope/storage.md

    rg -n "src_" docs/architecture/scope docs/architecture/index.md
    find docs/architecture/scope -maxdepth 1 -type f | sort

## Validation and Acceptance

Acceptance criteria:

1. All scope docs formerly named `src_*.md` are renamed to non-`src_` filenames.
2. `docs/architecture/scope/index.md` links and dependency statements use the new names.
3. Scope documents and `tests.md`/`benches.md` dependency bullets no longer use `src_*` labels.
4. `docs/architecture/index.md` contains one `Maintenance rule` section that also includes the prior agent-lesson guidance.
5. No remaining `src_` references exist in architecture scope/index text except literal `src/` path mentions.

## Idempotence and Recovery

This task is markdown/file-rename only. If needed, the rename can be reversed by moving files back and reverting link updates. Re-running text checks is safe.

## Artifacts and Notes

Expected changed files include:

    docs/architecture/index.md
    docs/architecture/scope/index.md
    docs/architecture/scope/*.md (renamed and content-updated)
    docs/plans/completed/plan_scope_filename_and_maintenance_rule_integration.md

## Interfaces and Dependencies

No code interfaces change. Dependency descriptions remain documentation-level architecture statements and must continue to represent concrete consumer->provider implementation direction.

Revision note (2026-03-02, Codex): Initial plan created for scope filename rename and maintenance-rule integration.
Revision note (2026-03-02, Codex): Updated progress/outcomes after completing file renames, dependency-label normalization, and maintenance-rule integration.
Revision note (2026-03-02, Codex): Marked completion and moved this plan from `active` to `completed`.

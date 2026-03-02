# Enforce Relative Documentation Paths and Replace Absolute Paths

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan is maintained under `PLANS.md` and follows that file’s rules.

Repository-document context used for this plan: `PLANS.md`, `AGENTS.md`, `VERIFICATION.md`, and `docs/verification/execplan_pre_creation.md`.

## Purpose / Big Picture

After this change, repository policy will explicitly require documentation to use only repository-root-relative file paths, and existing documentation files will be updated to remove absolute paths.

## Progress

- [x] (2026-03-02 02:16Z) Ran pre-ExecPlan validation checks (`git branch --show-current`, `git status --short`, `git log --oneline --decorate --max-count=20`).
- [x] (2026-03-02 02:16Z) Attempted PR-context check (`gh pr status`), confirmed `gh` is unavailable locally and recorded fallback to local branch/change evidence.
- [x] (2026-03-02 02:19Z) Added path-style rule to `AGENTS.md`: document paths must be repository-root-relative only; absolute paths prohibited.
- [x] (2026-03-02 02:20Z) Replaced absolute paths in current markdown documentation files with repository-root-relative paths.
- [x] (2026-03-02 02:21Z) Validated no absolute repository-style paths remain in markdown docs with markdown-path scans.
- [x] (2026-03-02 02:22Z) Moved this plan to `docs/plans/completed/`.

## Surprises & Discoveries

- Observation: `gh` CLI is unavailable in this environment, so active PR metadata cannot be queried via local CLI.
  Evidence: `gh pr status` returned `bin/bash: gh: command not found`.

## Decision Log

- Decision: Apply path replacement across documentation markdown files in the repository to satisfy “all current documents” requirement.
  Rationale: User request explicitly asks for replacing absolute paths in current docs comprehensively.
  Date/Author: 2026-03-02 / Codex

## Outcomes & Retrospective

Documentation path policy is now explicit in `AGENTS.md`, and absolute path references in current markdown documentation were converted to repository-root-relative forms. Verification checks show no remaining host-specific absolute path patterns and no backtick-wrapped absolute paths starting with `/` in markdown files.

This change is documentation-only and does not modify runtime behavior.

## Design/Architecture/Verification Document Summary

Design documents:

- Referenced: none.
- Created/modified: none.

Architecture documents:

- Referenced: none.
- Created/modified: none.

Verification documents:

- Referenced: `docs/verification/execplan_pre_creation.md`.
- Created/modified: none.

## Context and Orientation

Current docs included host-specific absolute paths and leading-slash repository file references. The new rule requires repository-root-relative paths only in documentation. This task updates policy and existing docs together.

## Plan of Work

Add an explicit rule in `AGENTS.md` stating that documentation must use repository-root-relative paths and must not use absolute paths. Then scan markdown documentation files, replace absolute path forms with relative equivalents, and validate with repository-wide searches.

## Concrete Steps

Run from repository root (`.`):

    rg -n -P '(?<!https:)(?<!http:)(?<!\\.)/([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+)' --glob "*.md"
    rg -n '`/[^`]+`' --glob "*.md"
    apply_patch << 'PATCH'
    ...
    PATCH
    rg -n -P '(?<!https:)(?<!http:)(?<!\\.)/([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+)' --glob "*.md"
    rg -n '`/[^`]+`' --glob "*.md"

## Validation and Acceptance

Acceptance criteria:

1. `AGENTS.md` contains an explicit rule banning absolute paths in documentation.
2. Absolute repository paths in current markdown docs are replaced by repository-root-relative paths.
3. Search checks confirm no remaining host-specific absolute paths and no backtick absolute paths in markdown docs.

## Idempotence and Recovery

This is documentation-only editing. Replacements can be re-run safely and reverted if needed.

## Artifacts and Notes

Expected modified files include:

    AGENTS.md
    *.md documentation files containing absolute paths

## Interfaces and Dependencies

No code interfaces or runtime behavior changes.

Revision note (2026-03-02, Codex): Initial plan created for enforcing relative doc paths and replacing absolute path references.
Revision note (2026-03-02, Codex): Updated progress/outcomes after adding AGENTS rule and converting absolute path references in markdown docs.
Revision note (2026-03-02, Codex): Marked completion and moved this plan from `active` to `completed`.

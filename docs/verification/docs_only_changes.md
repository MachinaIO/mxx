# Event: Docs-Only Changes

Use this document when a change modifies documentation only and does not alter Rust/CUDA source behavior, build scripts, or test code.

## Preconditions

- Working directory: repository root.
- Confirm changed files are documentation-only.

## Required actions

1. Confirm changed paths are documentation paths only.

    git diff --name-only --

2. Ensure architecture/plans/verification cross-links are not broken by searching for stale references.

    rg -n "TODO|TBD|FIXME" docs PLANS.md ARCHITECTURE.md VERIFICATION.md

3. If documentation changed policy statements that affect verification, update the corresponding policy/index docs in the same PR.

## Success criteria

- Changed files are limited to documentation paths.
- No stale or contradictory policy placeholder remains in touched files.
- ExecPlan/PR records that no code-path tests were required and explains why.

## Evidence to record

- `git diff --name-only` output summary.
- Any policy files updated and rationale.

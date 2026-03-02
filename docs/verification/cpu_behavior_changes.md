# Event: CPU Behavior Changes

Use this document when Rust implementation behavior changes and no GPU/CUDA boundary is modified.

## Preconditions

- Working directory: repository root (`.`).
- Identify which scope(s) were created or edited.

## Required actions

1. Run fmt:
    
        cargo +nightly fmt --all

2. If edited scope is clear, run only related unit tests first.

    Use `cargo test` with a scope-relevant filter.

    Example:

        cargo test -r --lib -- <scope_or_test_filter>

3. Run full unit tests only when required.

    If a coherent feature implementation/fix is finished, or if you modified foundational code used by many scopes, run:

        cargo test -r --lib

- Do not apply CI-only local test flags.

    Do not require `RUST_TEST_NOCAPTURE=1` or `--test-threads=1` for local testing.

- Do not run integration tests under `tests/` unless explicitly requested by a human operator.

    Do not run commands such as:

        cargo test --test <integration_test_name>

## Success criteria

- If scope is clear, related unit tests are executed.
- Full `cargo test -r --lib` is executed when completion/foundational-change conditions are met.
- Integration tests under `tests/` are not run unless explicitly requested by a human operator.
- All executed test commands and their outcomes are recorded in the ExecPlan/PR.

## Failure triage

- If a command fails due to missing system dependencies, record the missing dependency and reproduce scope.
- If failure is unrelated pre-existing breakage, capture evidence and separate from this change in notes.
- If a test fails, capture the failing test, failure output, and likely cause.
- After ExecPlan completion, provide a clear human-facing report of failed locations and causes.

## Evidence to record

- Scope-selection rationale (why specific tests were chosen).
- Exact commands executed.
- Pass/fail status for each command.
- Whether full `cargo test -r --lib` was triggered, and why.
- Whether integration tests were skipped (or explicitly requested by a human).
- If failures occurred: failed location(s) and cause summary reported to a human after ExecPlan completion.
- Any environment constraints and their impact.

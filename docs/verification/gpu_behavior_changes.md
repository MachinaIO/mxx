# Event: GPU Behavior Changes

Use this document when CUDA code or Rust code that is active only under GPU features is created or edited.

## Preconditions

- Working directory: repository root (`.`).
- CUDA toolkit and compatible GPU runtime are available.

## Required actions

1. Run formatting first.

    cargo +nightly fmt --all

2. If CUDA code was edited, run unit tests related to the scope whose Rust side depends on the edited CUDA area.

    Use scope-relevant test filters for GPU-enabled unit tests.

    Example:

        cargo test -r --lib --features gpu -- <scope_or_test_filter>

3. Run full GPU unit tests only when required.

    If a coherent feature implementation/fix is finished, or if foundational code used by many scopes was modified, run:

        cargo test -r --lib --features gpu

4. If CUDA code was created/edited and that work is complete, execute GPU-related unit tests 300 times in a row using this method.

    1. Build only:

        cargo test gpu -r --lib --features gpu --no-run

    2. Run the built binary outside the sandbox 300 consecutive times.

        Use the following command block as-is and reuse this same command block for approval and reruns whenever possible.

        ```bash
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
        ```

    Run all 300 iterations even if failures occur, then track:

    - how many runs failed,
    - where failures occurred,
    - likely error causes.

- Do not run integration tests under `tests/` unless explicitly requested by a human operator.

## Success criteria

- `cargo +nightly fmt --all` is executed first.
- If CUDA code changed, related scope GPU unit tests are executed.
- Full `cargo test -r --lib --features gpu` is run when completion/foundational-change conditions are met.
- If completed CUDA edits exist, the 300-run outside-sandbox repetition is completed with failure count/cause tracking.
- Integration tests under `tests/` are not run unless explicitly requested by a human operator.

## Failure triage

- If hardware/runtime limits block GPU execution, record exact blocker and run what remains possible.
- If 300-run repetition fails in some iterations, record failed count, failed locations/logs, and probable causes.
- After ExecPlan completion, provide a clear human-facing report with failed counts, failed locations, and causes.

## Evidence to record

- Commands executed and outcomes.
- Scope-selection rationale for targeted GPU unit tests.
- Whether full `cargo test -r --lib --features gpu` was triggered, and why.
- 300-run summary (`total_runs`, `failed_runs`) and failure-reason artifacts.
- Confirmation that `tests/` integration tests were skipped unless explicitly requested by a human operator.
- If failures occurred: post-ExecPlan human-facing summary of failed counts, locations, and causes.

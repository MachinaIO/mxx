---
name: bench-on-runpod
description: Run mxx programs, tests, or benchmarks on Runpod pods with reproducible source, durable logs, and controlled cleanup.
---

# Bench on Runpod

Use for requested mxx execution on Runpod pods. For planning or editing this workflow, inspect relevant files without provisioning or running it. Use `.codex/skills/runpodctl/SKILL.md` only when CLI resource operations are needed.

## Run Contract

Resolve the command, exact source, GPU type, GPUs per pod, pod count, storage, and any time/cost limit from the task and existing environment. Preserve explicit choices. Choose routine settings from existing mxx configuration; ask only for unresolved decisions that affect the requested result or resource authorization. Prefer an existing network volume named for mxx; creation or use of a non-mxx volume needs authorization already present in the task or a focused question.

Carry existing authorization through the run. A test-only request does not authorize changing cryptographic semantics or an unlimited fix/rerun loop. The repository's integration-test restriction still applies. If authorized resources are unavailable, report that constraint before substituting hardware or raising capacity.

## Source and Setup

- Inspect branch, commit, and dirty state. Use the user's branch; when a new branch is needed and none is specified, choose a descriptive `codex/` name.
- For the normal Git workflow, commit only task-owned changes when committing is authorized, push when authorized, and record the exact commit. Prepare the scoped diff before asking about a missing commit/push authorization. Honor no-push requests; if transfer is authorized, a source archive with a manifest and hashes can identify the exact dirty source without publishing it.
- Inspect remote dirty/untracked state before synchronization. Use an isolated checkout or preserve existing artifacts; do not blindly reset or clean a reused volume. Verify the checked-out commit or transferred manifest, including required submodule revisions, before execution.
- Reuse a suitable existing pod when requested. Otherwise provision only the agreed configuration, with SSH and `scp` support. Record pod IDs and the volume mount.
- For a new mxx volume, inspect `.codex/skills/bench-on-runpod/scripts/setup.sh` before running it remotely. It installs system dependencies and tools and expects the standard Runpod workspace mount; do not run it locally. Reuse an initialized environment when suitable. In the standard layout, source the mounted workspace's `env.sh` in each remote shell and enter its `mxx` checkout; verify actual paths rather than assuming a reused pod has this layout.

## Execution and Evidence

Run the requested command and parameters with a durable combined stdout/stderr log and explicit exit-status evidence. Use release builds and `RUST_LOG=debug` for benchmarks unless the task specifies otherwise. Record VRAM usage about every 3 seconds during GPU measurements.

Record pod identity, GPU model/count, pod count, source commit or manifest, timestamp/timezone, command, non-secret environment overrides, remote paths, and exit code. Use a recognizable log filename with a short command summary. Do not log credentials.

For foreground pipelines, enable `pipefail` when using `tee`. For jobs that may outlive SSH, use `nohup` or an existing job manager, capture the PID/job ID, and persist the exit code separately. Reconnect to the same job rather than starting another copy. Follow root `AGENTS.md` for efficient waiting and progress updates.

Copy logs and relevant artifacts to `logs/` in the active local checkout, or the user's requested destination, and verify retrieval before cleanup. Missing exit status means the result is unknown, even when the log contains successful intermediate stages.

## Failures and Cleanup

When the task includes fixes, diagnose from evidence, implement the scoped repair locally, validate as appropriate, synchronize its exact source under the same authorization, and rerun affected commands with new logs. Stop retrying when the same blocker persists without new evidence, the agreed budget is reached, or a fix requires a new semantic or resource decision. Preserve the failure evidence and report the next action; finish independent authorized work.

After retrieval, stop the pod promptly unless the user requested that it remain running or an authorized retry will use it imminently within the run budget. If blocked on user input, stop it unless the user explicitly requested that it remain running. Stopping and deleting are different actions: do not terminate/delete a pod or volume without explicit authorization. Verify the final state through Runpod, and report a cleanup failure rather than claiming the pod stopped.

Completion means the requested command's result is known, logs are available locally, and the agreed pod state is verified. Report these with the source identity and any unmet correctness or performance target. A successful estimate or compile does not establish GPU runtime correctness.

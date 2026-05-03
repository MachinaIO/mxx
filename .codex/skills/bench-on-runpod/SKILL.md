---
name: bench-on-runpod
description: Run mxx repository programs, tests, or benchmarks on Runpod instances with GPU machine selection, network volume setup, SSH/scp configuration, branch push and remote checkout synchronization, durable log capture, log retrieval, instance cleanup, and failed-test fix/push/pull/rerun loops. Use when Codex is asked to execute mxx work on a Runpod machine or instance, including requests that mention Runpod GPUs such as RTX5090, RTX PRO 6000, H200, remote benchmarking, remote tests, or running a command on a Runpod pod.
---

# Bench on Runpod

## Overview

Use this workflow to run mxx commands on Runpod machines while keeping the local branch, remote checkout, logs, and instance lifecycle controlled.

Prefer repo-local Runpod helper skills and CLIs when available, but keep the required state transitions below intact.

## Required Inputs

Before provisioning, determine these values from the user request or local context. Ask the user for any value that remains unclear.

- Machine type, such as `RTX5090`, `RTX PRO 6000`, or `H200`.
- Machine count.
- Container volume size.
- Network volume. Prefer an existing Runpod network volume whose name contains `mxx`; ask before creating or using a non-mxx volume.
- The exact program, test, benchmark, or command to run.
- Whether the selected network volume is new for mxx setup purposes.

## Local Preparation

1. Inspect the current branch name. If the checkout is detached or the current branch name is not decided, ask the user to choose a branch name before continuing.
2. Ensure every local change needed for the remote run is committed and pushed. If there are uncommitted changes and the user has not already authorized committing them, ask before creating a commit.
3. Push the current branch to the remote repository.
4. Record the pushed branch name and latest commit SHA. The remote checkout must be synchronized to this exact commit before running the command.

## Provision and Connect

1. Launch the requested Runpod machine configuration with the chosen machine type, machine count, container volume size, and network volume.
2. Ensure SSH is configured with `scp` support enabled. Do not proceed with a machine that cannot receive files through `scp`.
3. SSH into the machine and confirm the mounted workspace path is `/workspace`.
4. Record the machine or pod name because it must be included in the run log filename or metadata.

## Remote Setup

1. If the network volume is new for this mxx environment, copy `scripts/setup.sh` from this skill to the remote `/workspace` directory with `scp`, then run it on the remote machine from `/workspace`.
2. In every remote shell that runs commands, execute:

```bash
source /workspace/env.sh
```

3. Move to `/workspace/mxx`.
4. Align `/workspace/mxx` to the pushed local branch and commit:

```bash
git fetch origin
git checkout <branch-name>
git reset --hard <commit-sha>
git submodule update --init --recursive
```

Use the actual pushed branch and commit from local preparation. Do not run against a stale remote checkout.

## Run and Log

Always write command output to a durable log file while the command runs.

Construct a log filename or header that includes:

- Machine name or pod name.
- Machine count.
- Git commit SHA.
- Date and time, preferably in JST.
- A command summary of seven words or fewer.

Use `tee` for foreground commands. For long-running tests, benchmarks, or commands likely to outlive the SSH session, use `nohup` in the background and redirect both stdout and stderr to the log while preserving the command exit status where practical.

Example foreground shape:

```bash
set -o pipefail
<command> 2>&1 | tee <log-file>
```

Example background shape:

```bash
nohup bash -lc 'set -o pipefail; <command>' > <log-file> 2>&1 &
echo $!
```

Poll or reconnect until the result is known. When a background command finishes, inspect the log tail and exit-status evidence before reporting success or failure.

## Retrieve Logs

After the run completes or reaches a useful failure point, copy the log back to the local machine under `~/codes/mxx/logs` in an appropriate subdirectory for the run type, date, branch, or command family. Preserve the original remote log name when practical.

## Failure Loop

When a test or benchmark fails and the user has not instructed otherwise, use this default loop:

1. Diagnose from the retrieved log.
2. Fix the issue locally in the mxx workspace.
3. Commit and push the local fix.
4. SSH to the same remote machine when it is still running.
5. Pull or fetch the updated branch in `/workspace/mxx` and reset to the new commit.
6. Rerun the command with a new durable log.

Repeat until the run succeeds or user input is needed.

## Instance Lifecycle

After collecting logs:

- Keep the Runpod instance running if a failure means the same machine is likely to be reused within one hour.
- Stop or terminate the Runpod instance if the run succeeded.
- Stop or terminate the instance if the run failed but progress is blocked on user help and reuse within one hour is uncertain.

Report the final instance state, local log path, remote log path, branch, commit, and command result to the user.

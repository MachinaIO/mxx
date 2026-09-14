---
name: flash
description: Implement, debug, or deploy Runpod Flash serverless endpoints using the runpod-flash SDK and CLI.
user-invocable: true
---

# Runpod Flash

Use for Runpod Flash endpoint code and deployment tasks. Ordinary mxx Rust/CUDA pod benchmarks use `.codex/skills/bench-on-runpod/SKILL.md` instead.

## Choose the Requested Operation

Read the relevant section of `.codex/skills/flash/guides/sdk-and-cli.md`: Setup/CLI for environment operations, Endpoint modes for decorators/routes/image clients, constructor and job sections for API usage, or Common Patterns/Gotchas for debugging. Check installed SDK signatures and CLI help before relying on version-dependent defaults or resource enums.

Preserve the endpoint mode, user-selected GPU/CPU, worker limits, image, volume, and deployment environment. Prefer an explicit image version or digest for reproducibility. Do not increase worker counts or broaden GPU choices merely to match an example.

`flash run` exposes a local development server, but invoking its endpoints can provision and execute remote work. Treat it and cloud-calling examples according to their actual side effects. Prepare and inspect code or build artifacts before requesting missing cloud-execution authorization; a code-only task does not require deployment.

For remote functions, keep required imports inside the function, declare remote dependencies, and await asynchronous endpoint/job calls. GPU and CPU resource selections are alternatives.

## Completion

For code changes, perform the relevant local validation and state whether cloud execution was performed. For authorized deployment or execution, verify the target environment/endpoint and job result, preserve useful logs, and follow the agreed resource lifecycle. Do not automatically run deletion examples as cleanup. Stop retrying when further progress needs new authorization, exceeds the run budget, or lacks new failure evidence.

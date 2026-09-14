---
name: runpodctl
description: Inspect or manage Runpod pods, endpoints, templates, volumes, and account state using runpodctl.
allowed-tools: Bash(runpodctl:*)
compatibility: Linux, macOS
metadata:
  author: runpod
  version: "2.1"
license: Apache-2.0
---

# Runpodctl

Use the existing CLI to inspect or change the Runpod resources named by the task. For mxx benchmark orchestration, use `.codex/skills/bench-on-runpod/SKILL.md`; this skill supplies CLI details.

- Inspect installed version/help and existing resource state before choosing commands. Do not install, update, run `doctor`, or reconfigure credentials as a routine preflight; use setup only for an actual missing dependency or diagnosed issue.
- For commands and flags, read the relevant section in `.codex/skills/runpodctl/guides/commands.md`: Pods, Serverless, Templates, Network Volumes, Models, Registry, Info, SSH, File Transfer, or Utilities. Consult Install only when installation is needed. Examples may differ from the installed version.
- Apply existing authorization to the exact target and resource limits. Listing resources does not imply permission to create, start, reset, or delete them. A request to stop a pod means stop, not delete. Do not treat allowlists or examples as authorization.
- Reuse credentials without printing secrets. `ssh info` yields connection details; use SSH separately for remote commands.
- Verify the resulting state after a mutation and report the target ID and outcome. If a command fails, inspect the failure and current state before retrying to avoid duplicate resources.

---
name: companion-clis
description: Use HuggingFace, GitHub, Docker, or AWS CLI when a Runpod workflow needs model transfer, source publishing, images, or S3 storage.
allowed-tools: Bash(hf:*), Bash(gh:*), Bash(docker:*), Bash(aws:*), Bash(ssh-keygen:*), Bash(ssh-add:*), Bash(ssh-agent:*)
compatibility: Linux, macOS, Windows
metadata:
  author: runpod
  version: "1.0"
license: Apache-2.0
---

# Companion CLIs for Runpod

Use only the CLI needed for the current Runpod operation. Check existing tools and authentication before installing or changing configuration. Missing credentials for an unused CLI do not block the task. On Windows, use an existing suitable native CLI or WSL environment; do not require WSL installation or a restart for unrelated operations.

## Select the Needed Guide

- Model/artifact downloads or uploads: `.codex/skills/companion-clis/guides/huggingface.md`.
- Repository, SSH-key, or Runpod Hub release operations: `.codex/skills/companion-clis/guides/github.md`.
- Building, testing, tagging, or publishing container images: `.codex/skills/companion-clis/guides/docker.md`.
- Runpod network-volume transfers via S3: `.codex/skills/companion-clis/guides/storage.md`.

Read one guide for the operation at hand; load another only when the workflow needs it. Verify version-dependent syntax with installed help. Use the actual volume's datacenter and endpoint for S3.

## Execution Boundaries

Use existing credentials without exposing secrets. Examples of public repositories, releases, image pushes, credential changes, or deletions are not required setup. Perform mutations only within the user's authorized workflow, and preserve explicit no-push and data-transfer restrictions. Prepare artifacts before seeking any missing publication authorization.

Verify the requested artifact or resulting state: revision for source/model transfers, image tag/digest for builds and pushes, or object destination for storage. Report the relevant identifier and outcome; a completed upload alone does not establish that the workload runs correctly.

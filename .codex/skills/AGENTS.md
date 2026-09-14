# Repository-Local Skills

These skills support mxx's Runpod workflows; this directory is part of the Rust/CUDA repository. Root `AGENTS.md` applies. The upstream Runpod material retains its license and author metadata.

## Editing Skills

- Keep `name` and a short, discriminating `description` in YAML frontmatter. Preserve existing metadata and invocation policy unless the task changes them. Tool allowlists describe compatibility, not additional authorization.
- Put workflow outcomes, applicable constraints, and completion evidence in `SKILL.md`. Move substantial conditional command examples to `guides/` and link them where needed. The repository's read-only `references` rule also applies here.
- Do not duplicate global instructions or turn examples into default actions. Preserve task-specific hardware, budget, source, and cleanup choices.
- Use existing tools and credentials where available. Editing or explaining skills does not authorize installations, cloud provisioning, uploads, or test runs.
- Validate frontmatter, local links, and consistency after editing. For a behavior check, reason through a representative task and its stopping conditions; execute external workflows only when that execution is authorized.

Use `.codex/skills/README.md` for the skill inventory. Spelling: Runpod; CLI: `runpodctl`.

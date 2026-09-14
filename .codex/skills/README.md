# Runpod Skills for mxx

These repository-local skills adapt upstream Runpod guidance to mxx. They are already present in this checkout; using them does not require reinstalling the upstream collection.

| Skill | Use when |
| --- | --- |
| `.codex/skills/bench-on-runpod/SKILL.md` | Running mxx programs, tests, or benchmarks on Runpod pods. |
| `.codex/skills/runpodctl/SKILL.md` | Inspecting or managing Runpod resources through the CLI. |
| `.codex/skills/flash/SKILL.md` | Implementing or operating Runpod Flash serverless endpoints. |
| `.codex/skills/companion-clis/SKILL.md` | A Runpod task needs HuggingFace, GitHub, Docker, or S3 CLI operations. |

Each entrypoint links to command details needed for its workflow. Read these selectively. Example resource names, GPU choices, and mutation commands are illustrative, not permission or deployment defaults. See `.codex/skills/AGENTS.md` when maintaining these skills.

Upstream Runpod skill material is licensed under Apache-2.0; see `.codex/skills/LICENSE`.

# eternal-cycler

An autonomous PR builder/reviewer loop for [Codex](https://github.com/openai/codex) agents. Install it as a git subtree in your repository to get:

- An autonomous agent loop that writes code, opens PRs, reviews them, and iterates until the reviewer approves
- An ExecPlan lifecycle system that enforces structured planning, verification gates, and escalation bounds
- A verification skill framework for repo-specific pre/post-action checks

## Directory layout after installation

```
your-repo/
├── .agents/
│   └── skills/
│       ├── eternal-cycler/          # static — the skill itself (this repo)
│       │   ├── SKILL.md
│       │   ├── PLANS.md
│       │   ├── REVIEW.md
│       │   ├── setup.sh
│       │   ├── scripts/
│       │   └── assets/default-verification/   # default event skill templates
│       ├── execplan-event-index/    # copied from default-verification by setup.sh
│       ├── execplan-event-pre-creation/
│       ├── execplan-event-post-creation/
│       ├── execplan-event-resume/
│       ├── execplan-event-post-completion/
│       ├── execplan-event-action-docs-only/
│       ├── execplan-event-action-tooling/
│       └── execplan-sandbox-escalation/
└── eternal-cycler-out/              # created by setup.sh — all dynamic agent output
    ├── plans/
    │   ├── active/
    │   ├── completed/
    │   └── tech-debt/
    └── prs/
        ├── active/
        └── completed/
```

The skill directory (`.agents/skills/eternal-cycler/`) is static and never written to by agents at runtime. All dynamic output goes to `eternal-cycler-out/`.

## Prerequisites

| Tool | Purpose |
|------|---------|
| `git` | version control |
| `gh` | GitHub CLI for PR operations |
| `codex` | OpenAI Codex CLI (the agent runner) |
| `jq` | JSON parsing in loop scripts |
| `rg` | ripgrep for fast file search |

## Installation

Add eternal-cycler as a git subtree under `.agents/skills/eternal-cycler/`:

```bash
git subtree add \
  --prefix .agents/skills/eternal-cycler \
  https://github.com/SoraSuegami/eternal-cycler.git \
  main \
  --squash
```

Then run setup to copy verification skills and create the output directories:

```bash
bash .agents/skills/eternal-cycler/setup.sh
```

Setup output:

```
[setup] eternal-cycler path: .agents/skills/eternal-cycler/
[setup] git repo root:       /path/to/your-repo/

[setup] OK  git found
[setup] OK  gh found
[setup] OK  codex found
[setup] OK  jq found
[setup] OK  rg found

[setup] Copying default verification skills to /path/to/your-repo/.agents/skills/
[setup] OK   copied execplan-event-index -> .agents/skills/execplan-event-index
[setup] OK   copied execplan-event-pre-creation -> .agents/skills/execplan-event-pre-creation
[setup] OK   copied execplan-event-post-creation -> .agents/skills/execplan-event-post-creation
[setup] OK   copied execplan-event-resume -> .agents/skills/execplan-event-resume
[setup] OK   copied execplan-event-post-completion -> .agents/skills/execplan-event-post-completion
[setup] OK   copied execplan-event-action-docs-only -> .agents/skills/execplan-event-action-docs-only
[setup] OK   copied execplan-event-action-tooling -> .agents/skills/execplan-event-action-tooling
[setup] OK   copied execplan-sandbox-escalation -> .agents/skills/execplan-sandbox-escalation

[setup] Creating eternal-cycler-out/ output directories under /path/to/your-repo/
[setup] OK   eternal-cycler-out/plans/active
[setup] OK   eternal-cycler-out/plans/completed
[setup] OK   eternal-cycler-out/plans/tech-debt
[setup] OK   eternal-cycler-out/prs/active
[setup] OK   eternal-cycler-out/prs/completed

[setup] Setup complete.
[setup] Skill directory:  /path/to/your-repo/.agents/skills/
[setup] Output directory: /path/to/your-repo/eternal-cycler-out/
[setup] To start the builder/reviewer loop:
[setup]   .agents/skills/eternal-cycler/scripts/run_builder_reviewer_loop.sh \
[setup]     --task 'describe the task here'
```

Commit the result:

```bash
git add .agents/ eternal-cycler-out/
git commit -m "chore: install eternal-cycler"
```

## Usage

The skill is invoked as a Codex skill. When running Codex in your repository with `.agents/skills/eternal-cycler/SKILL.md` in scope, the agent will run the builder/reviewer loop automatically.

You can also invoke the loop directly:

```bash
# New task (run from a feature branch; the loop uses the current branch)
git switch -c feat/my-task
.agents/skills/eternal-cycler/scripts/run_builder_reviewer_loop.sh \
  --task "add input validation to the login form"

# Resume an existing PR
.agents/skills/eternal-cycler/scripts/run_builder_reviewer_loop.sh \
  --task-file task.md \
  --pr-url https://github.com/your-org/your-repo/pull/42
```

See `SKILL.md` for the full invocation contract that the Codex agent follows.

## Updating

Pull upstream changes with git subtree:

```bash
git subtree pull \
  --prefix .agents/skills/eternal-cycler \
  https://github.com/SoraSuegami/eternal-cycler.git \
  main \
  --squash
```

Re-run setup after updating to pick up any new default verification skills:

```bash
bash .agents/skills/eternal-cycler/setup.sh
```

Skills already present in `.agents/skills/` are not overwritten, so your customizations are preserved.

## Adding custom verification events

The gate script resolves event scripts from `.agents/skills/` in your repository. To add a new action event:

1. Create a skill directory under `.agents/skills/your-event-name/` with a `scripts/run_event.sh` that outputs `STATUS=pass` or `STATUS=fail`.
2. Register it in `.agents/skills/execplan-event-index/references/event_skill_map.tsv`:

```tsv
# event_id	skill_dir	script_relpath
action.your_event	your-event-name	scripts/run_event.sh
```

3. Reference `action.your_event` in your ExecPlan `Progress` actions via `verify_events=action.your_event`.

See `assets/default-verification/execplan-event-action-tooling/` for an example event skill.

## Key documents

| File | Purpose |
|------|---------|
| `PLANS.md` | ExecPlan authoring and lifecycle policy |
| `REVIEW.md` | PR review policy |
| `SKILL.md` | Codex skill contract for the builder/reviewer loop |
| `.agents/skills/execplan-sandbox-escalation/references/allowed_command_prefixes.md` | Approved out-of-sandbox command prefixes |

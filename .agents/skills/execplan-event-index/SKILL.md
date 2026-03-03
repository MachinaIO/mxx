---
name: execplan-event-index
description: Index skill that maps ExecPlan verification event IDs to per-event repository-local skills and scripts. Use this first when selecting which event skill to run.
---

# ExecPlan Event Index

This skill is the event registry for ExecPlan verification.

## Purpose

Resolve an `event_id` to the exact repository-local event skill and script that executes it.

## Registry file

Read this file first:

- `references/event_skill_map.tsv`

Format (tab-separated):

1. `event_id`
2. `skill_dir`
3. `script_relpath`

## Mandatory lifecycle events

The registry must always include:

- `execplan.pre_creation`
- `execplan.post_completion`

## Evolution rule

Action events may be added/removed by editing `references/event_skill_map.tsv` and creating/removing corresponding event skill directories under `.agents/skills/`.

# Design: ExecPlan Verification Enforcement via Modular Repository-Local Event Skills

## Purpose

Define a long-lived verification model where policy is integrated in `PLANS.md`, while execution procedures are implemented by modular repository-local event skills and enforced by gate/notify scripts.

## Problem Statement

Documentation-only runbooks are easy to ignore because they rely on voluntary compliance. This creates gaps between required verification policy and actual execution behavior.

## Design Goals

1. One policy source of truth in `PLANS.md`.
2. One operational event-mapping source of truth in index skill mapping.
3. Deterministic lifecycle gating where missing/failed verification blocks progression.
4. Per-event modularity so events can be added/removed without rewriting a monolithic skill.
5. In-plan evidence that is auditable without scattered temporary logs.

## Non-Goals

- Redesigning product runtime behavior.
- Replacing CI checks; this design complements CI with local lifecycle enforcement.

## Core Decisions

### 1. Policy integrated into PLANS

Long-lived verification policy is part of `PLANS.md`.

Rationale: removes split authority between planning and verification policy docs.

### 2. Event execution is modular by event skill

Each event has its own skill under `.agents/skills/execplan-event-*/` and its own `scripts/run_event.sh`.

Rationale: supports future event additions/removals without monolithic skill growth.

### 3. Event resolution is centralized by index skill

`.agents/skills/execplan-event-index/references/event_skill_map.tsv` maps event IDs to event skill scripts.

Rationale: keeps dispatch deterministic and easy to evolve.

### 4. Gate-driven enforcement

All event execution passes through `scripts/execplan_gate.sh`, which enforces:

- lifecycle transition prerequisites,
- retry bound (`3 tries`),
- escalation when bounds are exceeded,
- mandatory ledger recording,
- event-script dispatch from index map.

### 5. Optional final notification

`scripts/execplan_notify.sh` is used only for optional single final PR comment posting after `execplan.post_completion` passes and commits are pushed.

### 6. Evidence in one place

Each ExecPlan must contain `## Verification Ledger`; short-lived logs are recorded there instead of separate temporary files.

## Event Model

- Mandatory lifecycle events:
  - `execplan.pre_creation`
  - `execplan.post_completion`
- Action events are map-driven and can be added/removed by updating index mapping plus event skills.

## Parallelization Model

ExecPlan lifecycle target is singular.

Parallelization is action-level and requires metadata safety (`depends_on`, `file_locks`). Sub agents may execute parallel actions, but outcomes are merged back into the same ExecPlan.

## Known Trade-offs

- Gate enforcement increases process strictness and can surface more early failures.
- PR-comment notifications require GitHub CLI and PR context; missing PR linkage becomes an explicit failure mode.
- Event modularity introduces more files but reduces coupling and change blast radius.

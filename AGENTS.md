# Repository Guidelines
## Meta Rules
The following documents define *meta-rules* for how agents should create, update, and reference documents. Agents must carefully read and understand them.

### Plans (PLANS.md)
Read PLANS.md before starting any new task. You must write an ExecPlan following this document from design through implementation.

### Design (DESIGN.md)
Read DESIGN.md when your change requires a non-obvious decision that should be reusable beyond each PR, e.g.:
- you are choosing between multiple approaches with meaningful trade-offs,
- you introduce a new interface/contract, invariant, or API behavior,
- you add a pattern that future work should follow consistently.
If the decision is long-lived, create/update the relevant design artifact (per DESIGN.md) and link it from your ExecPlan.

### Architecture (ARCHITECTURE.md)
Read ARCHITECTURE.md before making changes that could affect code structure, e.g.:
- moving/adding modules or domains, changing package layout, or layering,
- adding/changing feature flags, shared infrastructure, or cross-domain dependencies,
- introducing new external dependencies,
- touching boundaries (e.g., FFI/CUDA, IO/storage, build integration) that rely on invariants.

### Verification (PLANS.md + repository-local event skills)
Read `PLANS.md` verification sections before implementation to determine verification policy and enforcement requirements, especially when:
- adding new features, changing behavior, or touching performance/correctness-critical code,
- modifying tests/CI, introducing new test categories, or changing required checks.
ExecPlan verification execution must use repository-local event skills under `.agents/skills/execplan-event-*/` plus index mapping under `.agents/skills/execplan-event-index/`, and gate/notify scripts under `scripts/`.

### Review (REVIEW.md)
Read REVIEW.md when you are asked to review a PR or to act as a reviewer.
In reviewer mode, follow REVIEW.md as the governing policy for independent review posture, required checks, and GitHub PR comment reporting.

## Global Requirements
- All documentation in this repository, along with git commit messages and PRs, must be written in English.
- When documenting file paths, use only paths relative to the repository top directory. Do not write absolute paths in documentation.

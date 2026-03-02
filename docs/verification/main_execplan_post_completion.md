# Event: After Main ExecPlan Completion

Use this document after all actions in a `main ExecPlan` are complete.

If you are closing a `sub ExecPlan`, use `docs/verification/sub_execplan_post_completion.md` instead.

## Preconditions

- Working directory: repository root (`.`).
- The target main ExecPlan document is already updated to completed state.
- The main ExecPlan document includes a repository-relative path to the corresponding PR tracking document.

## Required actions

1. Open the completed main ExecPlan document and find the linked PR tracking document path.

    Example check:

        rg -n "docs/prs/active/|docs/prs/completed/" docs/plans -S

2. Open the referenced PR tracking document and review its metadata.

    Confirm at least:

    - PR link
    - branch
    - commit context
    - stated PR scope/content

3. Determine whether the PR scope has been achieved and is ready for review.

    Base this decision on:

    - main ExecPlan completion status (`Progress`, `Outcomes & Retrospective`, validation results)
    - consistency between implemented changes and PR scope in the PR tracking document
    - known limitations explicitly documented in the plan

4. If the PR is ready for review, transition PR and document state.

    1. Set GitHub PR to ready for review.

        If GitHub CLI is available:

            gh pr ready

        If GitHub CLI is not available, perform the same transition in GitHub web UI.

    2. Move the corresponding PR tracking document from `docs/prs/active/` to `docs/prs/completed/`.

        mv docs/prs/active/<pr_tracking_file>.md docs/prs/completed/<pr_tracking_file>.md

5. If the PR is not ready for review, keep the PR and PR tracking document in active state.

    Record the remaining blockers in the ExecPlan and PR tracking document so the readiness decision is auditable.

6. After recording the post-ExecPlan validation result in the completed plan document, commit and push the finalized ExecPlan state as the final persistence step.

    Use an English commit message that clearly states the closure of the ExecPlan lifecycle.

    Recommended commands:

        git status --short
        git add -A
        git commit -m "docs: finalize execplan completion and post-validation results"
        git push origin $(git branch --show-current)

## Success criteria

- The PR tracking document linked by the main ExecPlan is reviewed before readiness decision.
- Ready/not-ready decision is explicitly recorded.
- If ready: GitHub PR is transitioned to ready for review and PR tracking document is moved to `docs/prs/completed/`.
- If not ready: PR remains non-ready and PR tracking document remains under `docs/prs/active/` with blockers recorded.
- Final main-ExecPlan state changes (including post-ExecPlan validation evidence) are committed and pushed as the last step.

## Failure triage

- If PR linkage is missing in the main ExecPlan, add the missing PR tracking document path and re-run this event.
- If PR state transition cannot be executed automatically, perform it in web UI and record that fallback.
- If move command fails because path mismatch exists, locate the correct PR tracking file and update the main ExecPlan reference for consistency.
- If commit or push fails at the final persistence step, record the exact error, resolve the git issue (for example, conflicts or remote rejection), and retry.

## Evidence to record

- Main ExecPlan path used for this event.
- PR tracking document path used for this event.
- Ready/not-ready decision rationale.
- If ready: evidence that PR was set to ready for review and file move command/result.
- If not ready: blocker list and next required actions.
- Commit hash and push result for the final commit that persists main-ExecPlan post-validation evidence.

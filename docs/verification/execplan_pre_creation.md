# Event: Before ExecPlan Creation

Use this document before creating a new ExecPlan file.

## Preconditions

- Working directory: repository root (`.`).
- Do not change implementation files during this event. Branch/PR tracking metadata files required by this event are allowed.
- This event runs before adding a new plan file under `docs/plans/active/`.

## Required actions

1. Capture current branch and working-tree state.

    git branch --show-current
    git status --short

2. Review recent branch intent from commit history.

    git log --oneline --decorate --max-count=20

3. Review in-progress pull request context for the current branch.

    If GitHub CLI is available:

        gh pr status
        gh pr view --json number,title,body,state,headRefName,baseRefName,url

    If GitHub CLI is not available, record that PR metadata could not be queried locally and proceed using branch history plus local change context.

4. Decide whether the upcoming plan is aligned with the current branch and in-progress pull request scope.

    Treat the plan as not aligned when it introduces a separately reviewable objective (for example, a new feature, a separate bugfix track, or unrelated documentation policy work).

5. Apply branch-switch rules.

    - If the current branch is `main`, you must move to a new branch before creating the ExecPlan.
    - If the plan is not aligned with the current branch/PR, first confirm current changes are committed, then move to a new branch.

    Commands:

        git status --short
        git branch --show-current

    If `git status --short` is not empty and a switch is required, commit current work first.

6. Create and switch to an appropriately named new branch when required.

    Recommended branch naming patterns:

    - `feat/<short-scope>` for feature work
    - `fix/<short-scope>` for bug fixes
    - `docs/<short-scope>` for documentation/policy work
    - `chore/<short-scope>` for maintenance/refactor work

    Command:

        git switch -c <type/short-scope>

7. Verify final state before plan creation.

    git branch --show-current
    git status --short

8. Handle PR creation/reuse and PR tracking documentation.

    If a new branch was created in step 6:

    - Create a draft pull request on GitHub.
    - Add a PR tracking markdown file under `docs/prs/active/`.
    - Include at least these fields in that PR tracking file:
      - PR link
      - PR creation date
      - branch name
      - commit hash at PR creation time
      - summary/content of the PR

    Recommended commands:

        mkdir -p docs/prs/active
        git rev-parse HEAD

    If GitHub CLI is available, draft PR creation can be performed with:

        gh pr create --draft --fill

    If GitHub CLI is not available, create the draft PR in the GitHub web UI and record the resulting URL in the PR tracking file.

    If an existing branch/PR is reused:

    - Locate or create the corresponding PR tracking file under `docs/prs/active/`.
    - Ensure the file contains current PR metadata and is up to date.

9. Require PR document linkage in upcoming ExecPlan files.

    For every ExecPlan created after this event (new branch/PR or reused branch/PR), add the repository-relative path to the corresponding PR tracking markdown file in the plan document.

    Also record, in that ExecPlan plan document, the branch name and git commit hash at the moment the ExecPlan starts.

    Example path format:

        docs/prs/active/<pr_tracking_file>.md

## Success criteria

- Branch/PR alignment decision is recorded.
- If current branch is `main`, a new branch is checked out.
- If plan scope is not aligned, current work is confirmed committed before switching and a new branch is checked out.
- Final branch and status are recorded before ExecPlan creation.
- If a new branch is created, a draft PR is created and a PR tracking file is added under `docs/prs/active/` with required metadata.
- If an existing branch/PR is reused, the corresponding PR tracking file is linked and updated as needed.
- The upcoming ExecPlan includes the relative path to the corresponding PR tracking file.
- The upcoming ExecPlan records its start-time branch name and git commit hash.

## Failure triage

- If branch switch fails because of uncommitted changes, commit or safely stash current work, then retry the switch.
- If PR context cannot be queried (for example, `gh` unavailable), record that limitation and make the alignment decision from local evidence.
- If draft PR creation cannot be completed automatically, create it manually in GitHub UI and record the result in the PR tracking file before creating the ExecPlan.

## Evidence to record

- Branch name before and after this event.
- `git status --short` summary before and after this event.
- Alignment decision rationale (aligned/not aligned) and whether an in-progress PR exists.
- Branch creation command used (if switch was required).
- PR tracking file path under `docs/prs/active/`.
- PR metadata captured in the tracking file (link, creation date, branch, creation commit, summary/content).
- Confirmation that the subsequent ExecPlan contains the PR tracking file relative path.
- Confirmation that the subsequent ExecPlan contains the start-time branch name and git commit hash.

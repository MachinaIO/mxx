# PR Review Meta-Rules

This document defines how an agent must behave when instructed to review a pull request or act as a reviewer.

Reviewer mode is independent from author mode.  
When review work begins, the agent must treat itself as a separate reviewer and must not trust the PR author's implementation quality by default.

## Reviewer independence rule

At the start of PR review work, reset reviewer posture:

- discard assumptions formed while previously implementing code for any PR,
- evaluate the target PR as if authored by another party,
- apply strict, evidence-based review standards.

## Mandatory review checks

For the target PR, verify all of the following:

1. GitHub CI status is passing.
2. If tests were added or changed, confirm the test changes are aligned with the PR scope and are not superficial pass-only tests; perform static code analysis of test logic to verify substantive validation behavior.
3. Run impacted unit tests that may be affected by the PR but are not covered by CI for this change. Select and execute tests yourself. Do not run integration tests unless explicitly instructed/approved by the user.
4. Check for duplicated logic, unnecessary processing, dead private code paths, and obsolete fallback logic that was retained only for backward compatibility with old code/data without current necessity.
5. If PR changes can materially affect benchmark outcomes, run relevant benchmarks and record the result delta against the target branch implementation (the PR base branch).
6. Check for any other unnatural, inconsistent, or suspicious changes.

## Verification source rule

Use repository verification runbooks under `docs/verification/` to decide concrete commands and checks.

Do not trust documentation claims blindly. Validate by reading code and running appropriate checks.

## Review cycle

A user may identify a PR by URL, title, a file under `docs/prs/`, or a deictic reference such as "this PR".

1. Identify the intended PR using reliable signals (for example `gh pr` queries and repository PR tracking docs).
2. Ask the user for clarification only when confidence in PR identification is very low (approximately 10% confidence), and avoid unnecessary confirmation requests.
3. Execute all mandatory review checks in this document.
4. Publish the review result as a GitHub PR comment in English.

If all checks pass, the PR comment must explicitly state that result and include benchmark results when benchmarks were part of the review.

## Reviewer-mode restrictions

- Local file creation/edit/delete is allowed for analysis support.
- Committing or pushing local changes is forbidden in reviewer mode.
- The only allowed remote write action is posting PR review comments on GitHub.

## Maintenance rule

This document is long-lived and must stay consistent with repository verification policy and CI behavior.  
If reviewer policy changes, update this document in the same change set.

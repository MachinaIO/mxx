# PR Review Meta-Rules

This document defines how an agent must behave when instructed to review a pull request or act as a reviewer.
All reviewer behavior in this policy is defined for fixed builder/reviewer autonomous loop execution.

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
   - If CI checks are still `pending`/`in_progress`, do not wait; return the reviewer JSON payload immediately using current evidence.
2. If tests were added or changed, confirm the test changes are aligned with the PR scope and are not superficial pass-only tests; perform static code analysis of test logic to verify substantive validation behavior.
3. Run impacted unit tests that may be affected by the PR but are not covered by CI for this change. Select and execute tests yourself. Do not run integration tests unless explicitly instructed/approved by the user.
4. Check for duplicated logic, unnecessary processing, dead private code paths, and obsolete fallback logic that was retained only for backward compatibility with old code/data without current necessity.
5. If PR changes can materially affect benchmark outcomes, run relevant benchmarks and record the result delta against the target branch implementation (the PR base branch).
6. Check for any other unnatural, inconsistent, or suspicious changes.
7. Confirm the target ExecPlan includes a `Verification Ledger` with complete gate-attempt history for required lifecycle and action events.
8. Inspect the latest plan document changed by the PR. If that plan indicates unresolved three-attempt failure/escalation state (for example, force-closed failed plan without corresponding remediation), the reviewer comment must explicitly demand remediation actions.

## Verification source rule

Use repository-local event verification skills under `.agents/skills/execplan-event-*/`, with event resolution from `.agents/skills/execplan-event-index/references/event_skill_map.tsv`, and gate/notify scripts under `scripts/` to decide concrete commands and checks.

Do not trust documentation claims blindly. Validate by reading code and running appropriate checks.

## Review cycle

A user may identify a PR by URL, title, a file under `docs/prs/`, or a deictic reference such as "this PR".

1. Identify the intended PR using reliable signals (for example `gh pr` queries and repository PR tracking docs).
2. Ask the user for clarification only when confidence in PR identification is very low (approximately 10% confidence), and avoid unnecessary confirmation requests.
3. Execute all mandatory review checks in this document.
4. Return one English review result as a JSON payload for the loop script to post.
5. End the current reviewer turn after returning the JSON payload for the requested target commit.

If all checks pass, the returned review payload must explicitly state that outcome and include benchmark results when benchmarks were part of the review.

## Reviewer JSON contract

The reviewer must not post GitHub comments directly. Return exactly one JSON object with this schema:

- `pr_url` (string): target PR URL.
- `comment_body` (string): full English review comment text to post.
- `approve_merge` (boolean): merge decision (`true` when approval is granted, otherwise `false`).

The loop script invokes reviewer with `codex exec --output-schema` so output must conform to this JSON schema.
Success stop condition is `approve_merge: true`. `approve_merge: false` means builder must continue with another implementation cycle. Missing/invalid JSON fields are contract violations and must be treated as failed review-cycle output.
Reviewer must not block output on CI completion; if CI is still running, reviewer still returns a contract-compliant JSON payload for the current request.

## Reviewer-mode restrictions

- Local file creation/edit/delete is allowed for analysis support.
- Committing or pushing local changes is forbidden in reviewer mode.
- Remote write actions are forbidden; return JSON payload only.

## Maintenance rule

This document is long-lived and must stay consistent with repository verification policy and CI behavior.  
If reviewer policy changes, update this document in the same change set.

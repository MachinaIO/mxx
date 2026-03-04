# PR Review Policy

This document defines how an agent must behave when reviewing a pull request. All reviewer behavior applies to the builder/reviewer autonomous loop unless otherwise noted.

## Global Requirements

- All review comments and PR comment bodies must be written in English.
- When referencing file paths, use only paths relative to the repository top directory.
- This document is long-lived and must stay consistent with repository verification policy and CI behavior. When reviewer policy changes, update this document in the same change set.

## Reviewer independence rule

At the start of every review, reset reviewer posture completely:

- Discard any assumptions formed while previously implementing code for any PR.
- Evaluate the target PR as if authored by another party.
- Apply strict, evidence-based review standards. Do not trust documentation claims blindly; validate by reading code and running appropriate checks.

## Mandatory review checks

For the target PR, verify all of the following before returning a decision:

1. **CI status.** GitHub CI is passing. If checks are still `pending`/`in_progress`, do not wait — return the reviewer payload immediately using current evidence.
2. **Test quality.** If tests were added or changed, confirm they are aligned with the PR scope and are not superficial pass-only tests. Perform static analysis of test logic to verify substantive validation behavior.
3. **Local tests.** Run impacted unit tests that may be affected by the PR but are not covered by CI for this change. Do not run integration tests unless explicitly instructed by the user.
4. **Code hygiene.** Check for duplicated logic, unnecessary processing, dead private code paths, and obsolete fallback logic retained without current necessity.
5. **Benchmarks.** If PR changes can materially affect benchmark outcomes, run relevant benchmarks and record the result delta against the base branch.
6. **Suspicious changes.** Check for any other unnatural, inconsistent, or suspicious changes.
7. **Verification Ledger.** Confirm the target ExecPlan includes a `Verification Ledger` with complete gate-attempt history for required lifecycle and action events.
8. **Plan state.** Inspect the latest plan document changed by the PR. If that plan indicates unresolved three-attempt failure/escalation (for example, a force-closed failed plan without remediation), the reviewer comment must explicitly demand remediation.

If all checks pass, the returned payload must explicitly state that outcome and include benchmark results when benchmarks were run.

## Verification sources

**Path note:** Gate and notify script paths are relative to the eternal-cycler installation root (shown in "Path context" in your prompt). Verification skill and PR tracking paths are relative to the consuming repository root.

Use repository-local event verification skills under `.agents/skills/execplan-event-*/`, with event resolution from `.agents/skills/execplan-event-index/references/event_skill_map.tsv`, and gate/notify scripts under `scripts/` to decide concrete commands and checks.

## Review cycle

A user may identify a PR by URL, title, a file under `eternal-cycler-out/prs/`, or a deictic reference such as "this PR".

1. Identify the intended PR using reliable signals (`gh pr` queries, repository PR tracking docs).
2. Ask for clarification only when confidence in PR identification is very low (approximately 10%); avoid unnecessary confirmation requests.
3. Execute all mandatory review checks above.
4. Return exactly one JSON payload for the loop script to post.
5. End the reviewer turn immediately after returning the payload.

## Reviewer JSON contract

Do not post GitHub comments directly. Return exactly one JSON object:

- `pr_url` (string): target PR URL.
- `comment_body` (string): full English review comment text to post.
- `approve_merge` (boolean): `true` to approve merge, `false` to request another builder cycle.

The loop script uses `codex exec --output-schema`; output must conform to this schema. Missing or invalid fields are contract violations and are treated as failed review-cycle output. Do not block on CI completion; return a compliant payload immediately.

`approve_merge: true` is the success stop condition. `approve_merge: false` means the builder must continue with another implementation cycle.

## Reviewer-mode restrictions

- Local file creation/edit/delete is allowed for analysis support.
- Committing or pushing local changes is forbidden.
- All remote write actions are forbidden; return the JSON payload only.

# Reviewer Guidance

Use this document for explicit reviews. Review the scoped result read-only: do not edit repository files, implement fixes, or change workflow state.

## Evidence and Scope

Read the user's full active request, the relevant diff and source, and validation evidence needed to assess concrete claims. Check builder claims independently. Load `GPU.md` for GPU changes and other specifications only as needed; do not load raw transcripts or unrelated history by default.

Apply `AGENTS.md`, relevant implementation requirements in `BUILDER.md`, and the user's constraints. Review a completed documentation edit as documentation; it does not trigger the execution steps described inside it. Run no integration tests without an explicit request in the current task, and respect any stricter read-only/no-execution limit.

## Review Criteria

- Correctness and regression risk against the requested behavior, including the actual production path and resource ownership.
- Validation sufficient for the claimed result. Compilation, isolated tests, simulation, and end-to-end execution establish different facts. GPU synchronization changes require the repeated checks in `GPU.md`.
- Scope discipline, readable responsibility boundaries, and absence of unnecessary fallback paths, compatibility shims, or speculative redesign.

Report observable defects and supported risks. Separate evidence from uncertainty; inspect further only when it can resolve a material concern. Do not block acceptance on optional polish, personal style preferences, or unrelated pre-existing issues.

## Result

Return concise English feedback with actionable findings first. Each finding should identify the affected file/location, the failure condition and consequence, and the smallest correction needed. State missing validation precisely without presenting an unobserved failure as confirmed.

When a structured decision is requested, use `accept` or `revision`: accept when the scoped requirements and their validation are satisfied; request revision for concrete deficiencies. Otherwise use the requested review format. If there are no actionable findings, say so plainly and note any material validation limitation. Do not add extra implementation or review rounds after acceptance.

#!/usr/bin/env bash
# Verify prerequisites after adding eternal-cycler via git subtree or as a standalone copy.
# Run from anywhere inside the consuming repository:
#   bash .agents/skills/eternal-cycler/setup.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
SUBMODULE_REL="$(realpath --relative-to="$REPO_ROOT" "$SCRIPT_DIR")"

log()  { echo "[setup] $*"; }
warn() { echo "[setup] WARN: $*" >&2; }

log "eternal-cycler path: ${SUBMODULE_REL}/"
log "git repo root:       ${REPO_ROOT}/"
echo

# Check that key scripts are executable
LOOP_SCRIPT="$SCRIPT_DIR/scripts/run_builder_reviewer_loop.sh"
if [[ -x "$LOOP_SCRIPT" ]]; then
  log "OK  run_builder_reviewer_loop.sh is executable"
else
  warn "run_builder_reviewer_loop.sh not executable; run:"
  warn "  chmod +x ${SUBMODULE_REL}/scripts/run_builder_reviewer_loop.sh"
fi

DOCTOR_SCRIPT="$SCRIPT_DIR/scripts/run_builder_reviewer_doctor.sh"
if [[ -x "$DOCTOR_SCRIPT" ]]; then
  log "OK  run_builder_reviewer_doctor.sh is executable"
else
  warn "run_builder_reviewer_doctor.sh not executable; run:"
  warn "  chmod +x ${SUBMODULE_REL}/scripts/run_builder_reviewer_doctor.sh"
fi

GATE_SCRIPT="$SCRIPT_DIR/scripts/execplan_gate.sh"
if [[ -x "$GATE_SCRIPT" ]]; then
  log "OK  execplan_gate.sh is executable"
else
  warn "execplan_gate.sh not executable; run: chmod +x ${SUBMODULE_REL}/scripts/execplan_gate.sh"
fi

# Check for required CLIs
for bin in git gh codex jq rg; do
  if command -v "$bin" >/dev/null 2>&1; then
    log "OK  $bin found"
  else
    warn "$bin not found (required for pr-autoloop)"
  fi
done
echo

# Copy default verification skills to the consuming repo's .agents/skills/ directory.
# The gate script reads event skill scripts from .agents/skills/, not from assets/default-verification/.
# Skills already present in .agents/skills/ are not overwritten, so consuming-repo customizations are preserved.
SKILLS_DIR="$REPO_ROOT/.agents/skills"
DEFAULT_VERIFICATION_DIR="$SCRIPT_DIR/assets/default-verification"
mkdir -p "$SKILLS_DIR"
log "Copying default verification skills to ${REPO_ROOT}/.agents/skills/"
for skill_dir in "$DEFAULT_VERIFICATION_DIR"/*/; do
  skill_name="$(basename "$skill_dir")"
  if [[ -d "$SKILLS_DIR/$skill_name" ]]; then
    log "SKIP $skill_name (already present in .agents/skills/; not overwritten)"
  else
    cp -r "$skill_dir" "$SKILLS_DIR/$skill_name"
    log "OK   copied $skill_name -> .agents/skills/$skill_name"
  fi
done
echo

# Create the output directory tree used at runtime.
# plans/ and prs/ are consumed-repo-local and must not live inside the eternal-cycler skill directory.
log "Creating eternal-cycler-out/ output directories under ${REPO_ROOT}/"
for out_dir in \
    eternal-cycler-out/plans/active \
    eternal-cycler-out/plans/completed \
    eternal-cycler-out/plans/tech-debt \
    eternal-cycler-out/prs/active \
    eternal-cycler-out/prs/completed; do
  mkdir -p "$REPO_ROOT/$out_dir"
  log "OK   $out_dir"
done
echo

log "Setup complete."
log ""
log "Skill directory:  ${REPO_ROOT}/.agents/skills/"
log "Output directory: ${REPO_ROOT}/eternal-cycler-out/"
log ""
log "To start the builder/reviewer loop:"
log "  ${SUBMODULE_REL}/scripts/run_builder_reviewer_loop.sh \\"
log "    --task 'describe the task here'"

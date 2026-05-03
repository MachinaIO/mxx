#!/usr/bin/env bash
set -euo pipefail

WORKTREE="/workspace"
OPENFHE_PREFIX="${WORKTREE}/openfhe"
ENV_FILE="${WORKTREE}/env.sh"

RUSTUP_HOME="${WORKTREE}/rustup"
CARGO_HOME="${WORKTREE}/cargo"
N_PREFIX="${WORKTREE}/n"
NPM_GLOBAL_PREFIX="${WORKTREE}/npm-global"
NPM_CONFIG_CACHE="${WORKTREE}/.npm-cache"
CODEX_HOME="${WORKTREE}/.codex"

log() { echo -e "\n==> $*\n"; }

if [ "$(id -u)" -ne 0 ]; then
  echo "ERROR: This script must be run as root (no sudo is used)."
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
mkdir -p "${WORKTREE}"

# 1) Setup Kitware APT repo (latest CMake)
log "1) Setup Kitware APT repo"
apt-get update -y
apt-get install -y ca-certificates gpg wget curl

# (Same flow as before) temporary key -> add repo -> install keyring package
if [ ! -f /usr/share/doc/kitware-archive-keyring/copyright ]; then
  wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null \
    | gpg --dearmor - \
    | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
fi

CODENAME="$(. /etc/os-release && echo "${VERSION_CODENAME:-jammy}")"
echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ ${CODENAME} main" \
  > /etc/apt/sources.list.d/kitware.list

apt-get update -y

if [ ! -f /usr/share/doc/kitware-archive-keyring/copyright ]; then
  rm -f /usr/share/keyrings/kitware-archive-keyring.gpg
  apt-get install -y kitware-archive-keyring
  apt-get update -y
fi

# 2) apt packages install
log "2) Install apt packages"
apt-get install -y \
  cmake htop nvtop libtbb-dev \
  git build-essential make

# 3) Install Rust under /workspace (rustup)
log "3) Install Rust under ${WORKTREE}"
mkdir -p "${RUSTUP_HOME}" "${CARGO_HOME}"
export RUSTUP_HOME CARGO_HOME
export PATH="${CARGO_HOME}/bin:${PATH}"

if [ ! -x "${CARGO_HOME}/bin/rustup" ]; then
  # rustup official installer; we avoid modifying shell profiles
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
fi

rustc --version
cargo --version

# 4) Install Node.js (via n) and Codex under /workspace
log "4) Install Node.js (via n) and Codex under ${WORKTREE}"
mkdir -p "${N_PREFIX}/bin" "${NPM_GLOBAL_PREFIX}" "${NPM_CONFIG_CACHE}" "${CODEX_HOME}"
export N_PREFIX NPM_GLOBAL_PREFIX NPM_CONFIG_CACHE CODEX_HOME
export NPM_CONFIG_PREFIX="${NPM_GLOBAL_PREFIX}"
export PATH="${N_PREFIX}/bin:${NPM_GLOBAL_PREFIX}/bin:${PATH}"

if [ ! -x "${N_PREFIX}/bin/n" ]; then
  curl -fsSL https://raw.githubusercontent.com/tj/n/master/bin/n -o "${N_PREFIX}/bin/n"
  chmod +x "${N_PREFIX}/bin/n"
fi

if [ ! -x "${N_PREFIX}/bin/node" ]; then
  "${N_PREFIX}/bin/n" lts
fi

if [ ! -x "${NPM_GLOBAL_PREFIX}/bin/codex" ]; then
  npm install -g @openai/codex
fi

# Keep codex discoverable even when /workspace env vars are not loaded yet.
if [ -x "${NPM_GLOBAL_PREFIX}/bin/codex" ]; then
  ln -sfn "${NPM_GLOBAL_PREFIX}/bin/codex" /usr/local/bin/codex
fi

node --version
npm --version
codex --version

# 5) Clone repositories into /workspace
log "5) Clone git repositories into ${WORKTREE}"
cd "${WORKTREE}"

if [ ! -d openfhe-development ]; then
  git clone https://github.com/MachinaIO/openfhe-development
fi

if [ ! -d mxx ]; then
  git clone https://github.com/MachinaIO/mxx
fi

# 6) Build & install OpenFHE to /workspace/openfhe (if not already installed)
log "6) Build & install OpenFHE to ${OPENFHE_PREFIX}"
if [ -f "${OPENFHE_PREFIX}/lib/libOPENFHEcore.so" ] || [ -f "${OPENFHE_PREFIX}/lib/libOPENFHEcore.so.1" ]; then
  echo "OpenFHE seems already installed at ${OPENFHE_PREFIX}; skipping build."
else
  cd "${WORKTREE}/openfhe-development"
  git submodule update --init --recursive || true

  mkdir -p build
  cd build

  cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${OPENFHE_PREFIX}" \
    -DBUILD_SHARED=ON \
    -DBUILD_STATIC=OFF

  make -j"$(nproc)"
  make install
fi

# env.sh: source this on every shell/session.
log "7) Write ${ENV_FILE} (quiet, automation-safe defaults)"
cat > "${ENV_FILE}" <<'EOF'
# Source this on every new instance / shell:
#   source /workspace/env.sh
#
# This file is intentionally quiet and side-effect-free by default so that
# non-interactive automation (for example MCP bridge scripts) can source it
# safely without polluting stdout/stderr.
#
# Optional maintenance operations are provided as functions:
# - workspace_bootstrap_system_deps_once
# - workspace_link_openfhe_usr_local
# - workspace_codex_login_if_needed
# - workspace_configure_git_push_auth
# Auto-run hooks are enabled by default; set WORKSPACE_ENV_AUTO_*=0 to disable.

export WORKTREE="/workspace"
export OPENFHE_PREFIX="/workspace/openfhe"

export RUSTUP_HOME="/workspace/rustup"
export CARGO_HOME="/workspace/cargo"
export N_PREFIX="/workspace/n"
export NPM_GLOBAL_PREFIX="/workspace/npm-global"
export NPM_CONFIG_PREFIX="${NPM_GLOBAL_PREFIX}"
export NPM_CONFIG_CACHE="/workspace/.npm-cache"
export CODEX_HOME="/workspace/.codex"
export NODE_PATH="${NPM_GLOBAL_PREFIX}/lib/node_modules:${NODE_PATH:-}"

mkdir -p "${NPM_GLOBAL_PREFIX}" "${NPM_CONFIG_CACHE}" "${CODEX_HOME}"
export PATH="${N_PREFIX}/bin:${NPM_GLOBAL_PREFIX}/bin:${CARGO_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/lib:${OPENFHE_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

workspace_bootstrap_system_deps_once() {
  local sentinel="/tmp/.workspace_bootstrap_apt_done_v4"
  if [ -f "${sentinel}" ]; then
    return 0
  fi

  if [ "$(id -u)" -ne 0 ]; then
    return 1
  fi

  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y
  apt-get install -y ca-certificates gpg wget curl

  test -f /usr/share/doc/kitware-archive-keyring/copyright || \
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null \
      | gpg --dearmor - \
      | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null

  local codename
  codename="$(. /etc/os-release && echo "${VERSION_CODENAME:-jammy}")"
  echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ ${codename} main" \
    > /etc/apt/sources.list.d/kitware.list

  apt-get update -y

  test -f /usr/share/doc/kitware-archive-keyring/copyright || \
    rm -f /usr/share/keyrings/kitware-archive-keyring.gpg

  apt-get install -y kitware-archive-keyring
  apt-get update -y

  apt-get install -y \
    cmake htop nvtop libtbb-dev \
    git build-essential vim

  touch "${sentinel}"
}

workspace_link_openfhe_usr_local() {
  if [ ! -d "${OPENFHE_PREFIX}" ]; then
    return 1
  fi

  if [ "$(id -u)" -ne 0 ]; then
    return 1
  fi

  mkdir -p /usr/local/include /usr/local/lib /usr/local/lib/cmake

  ln -sfn "${OPENFHE_PREFIX}/include/openfhe" /usr/local/include/openfhe

  for f in "${OPENFHE_PREFIX}"/lib/libOPENFHE*.so*; do
    [ -e "$f" ] || continue
    ln -sf "$f" "/usr/local/lib/$(basename "$f")"
  done

  cfg_file="$(find "${OPENFHE_PREFIX}" -type f \( -name 'OpenFHEConfig.cmake' -o -name 'openfhe-config.cmake' \) -print -quit 2>/dev/null || true)"
  if [ -n "${cfg_file}" ]; then
    ln -sfn "$(dirname "${cfg_file}")" /usr/local/lib/cmake/OpenFHE
  fi

  command -v ldconfig >/dev/null 2>&1 && ldconfig >/dev/null 2>&1 || true
}

# workspace_codex_login_if_needed() {
#   if ! command -v codex >/dev/null 2>&1; then
#     return 1
#   fi

#   if codex login status >/dev/null 2>&1; then
#     return 0
#   fi

#   if [ ! -t 0 ] || [ ! -t 1 ]; then
#     return 1
#   fi

#   codex login --device-auth
# }

workspace_configure_git_push_auth() {
  if ! command -v git >/dev/null 2>&1; then
    return 1
  fi

  # Optional git identity for commits.
  if [ -n "${RUNPOD_SECRET_GIT_USER_NAME:-}" ]; then
    git config --global user.name "${RUNPOD_SECRET_GIT_USER_NAME}"
  fi
  if [ -n "${RUNPOD_SECRET_GIT_USER_EMAIL:-}" ]; then
    git config --global user.email "${RUNPOD_SECRET_GIT_USER_EMAIL}"
  fi

  # Optional remote rewrite (HTTPS URL recommended for token-based auth).
  local repo_dir="${WORKSPACE_GIT_REPO_DIR:-/workspace/mxx}"
  local remote_name="${WORKSPACE_GIT_REMOTE:-origin}"
  if [ -n "${WORKSPACE_GIT_REMOTE_URL:-}" ] && [ -d "${repo_dir}/.git" ]; then
    if git -C "${repo_dir}" remote get-url "${remote_name}" >/dev/null 2>&1; then
      git -C "${repo_dir}" remote set-url "${remote_name}" "${WORKSPACE_GIT_REMOTE_URL}"
    else
      git -C "${repo_dir}" remote add "${remote_name}" "${WORKSPACE_GIT_REMOTE_URL}"
    fi
  fi

  # Token auth setup without persisting token to git config files.
  if [ -z "${RUNPOD_SECRET_GIT_ACCESS_TOKEN:-}" ]; then
    return 0
  fi

  local askpass_path="${WORKSPACE_GIT_ASKPASS_PATH:-/workspace/.git-askpass.sh}"
  local askpass_dir
  askpass_dir="$(dirname "${askpass_path}")"
  mkdir -p "${askpass_dir}"

  cat > "${askpass_path}" <<'EOS'
#!/usr/bin/env bash
case "${1:-}" in
  *Username*) printf '%s\n' "${WORKSPACE_GIT_USERNAME:-x-access-token}" ;;
  *Password*) printf '%s\n' "${RUNPOD_SECRET_GIT_ACCESS_TOKEN:-}" ;;
  *) printf '\n' ;;
esac
EOS
  chmod 700 "${askpass_path}"

  export GIT_ASKPASS="${askpass_path}"
  export GIT_TERMINAL_PROMPT=0

  # Keep git behavior deterministic for automation sessions.
  git config --global core.askPass "${askpass_path}"
  git config --global --unset-all credential.helper >/dev/null 2>&1 || true
}

if [ "${WORKSPACE_ENV_AUTO_BOOTSTRAP:-1}" = "1" ]; then
  workspace_bootstrap_system_deps_once || true
fi
if [ "${WORKSPACE_ENV_AUTO_LINK_OPENFHE:-1}" = "1" ]; then
  workspace_link_openfhe_usr_local || true
fi
# if [ "${WORKSPACE_ENV_AUTO_CODEX_LOGIN:-1}" = "1" ]; then
#   workspace_codex_login_if_needed || true
# fi
if [ "${WORKSPACE_ENV_AUTO_GIT_PUSH_AUTH:-1}" = "1" ]; then
  workspace_configure_git_push_auth || true
fi
EOF

# Apply env in current shell and create OpenFHE symlinks now.
. "${ENV_FILE}"
workspace_link_openfhe_usr_local || true

log "DONE"
echo "On every new instance/session, run:  source ${ENV_FILE}"

# HuggingFace CLI

Read only the needed section. Commands are examples; check the installed CLI help before use. Setup, authentication changes, publishing, and deletion are conditional on the requested workflow and its authorization.


Use `hf` for the model or artifact transfer required by the task. Download locally only when the selected build or execution workflow needs a local copy.

### Install

```bash
# macOS / Linux (standalone installer — recommended)
curl -LsSf https://hf.co/cli/install.sh | bash

# macOS (Homebrew)
brew install hf

# Windows (WSL2): use the Linux standalone installer above
```

Check `hf --help` for the installed version before choosing commands.

### Credentials

Get a token at https://huggingface.co/settings/tokens. Use **write** access for uploading; **read** access is sufficient for downloading public or gated models.

```bash
# Option 1: interactive login (saves token to ~/.cache/huggingface/token, optionally to git credential store)
hf auth login

# Option 2: non-interactive (pass token directly, useful in scripts and pod start commands)
hf auth login --token $HF_TOKEN --add-to-git-credential

# Option 3: environment variable (takes precedence over saved token; to revert, unset the variable)
export HF_TOKEN=hf_...
```

```bash
hf auth whoami      # confirm auth and org memberships
hf auth logout      # delete all locally stored tokens
```

### Key Commands

```bash
# Download a model to a local directory (use --local-dir to control where it lands)
hf download meta-llama/Llama-3.1-8B --local-dir ./models/llama-3.1-8b
hf download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir ./models/tinyllama

# Download a single file from a model repo
hf download meta-llama/Llama-3.1-8B config.json --local-dir ./models/llama-3.1-8b

# Download with glob filters (e.g. only safetensors weights, skip fp16 variants)
hf download stabilityai/stable-diffusion-xl-base-1.0 \
  --include "*.safetensors" --exclude "*.fp16.*" \
  --local-dir ./models/sdxl

# Download a specific revision (commit hash, branch, or tag — append --revision REF)
hf download meta-llama/Llama-3.1-8B --revision v1.0 --local-dir ./models/llama-3.1-8b
```

### Troubleshooting

```bash
# Increase download timeout on slow connections (default: 10s)
export HF_HUB_DOWNLOAD_TIMEOUT=30
```

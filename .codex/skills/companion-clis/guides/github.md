# GitHub CLI

Read only the needed section. Commands are examples; check the installed CLI help before use. Setup, authentication changes, publishing, and deletion are conditional on the requested workflow and its authorization.


The GitHub CLI (`gh`) is used to manage repositories for Runpod serverless workers. This includes cloning repos into local Docker containers for testing, versioning source code so changes can be tracked and shared with teammates or collaborators, and creating GitHub releases that publish listings to the Runpod Hub. The Hub indexes releases — not commits — so every deployment update requires a new release.

### Install

```bash
# macOS
brew install gh

# Linux (Debian/Ubuntu)
(type -p wget >/dev/null || (sudo apt update && sudo apt install wget -y)) \
  && sudo mkdir -p -m 755 /etc/apt/keyrings \
  && out=$(mktemp) && wget -nv -O$out https://cli.github.com/packages/githubcli-archive-keyring.gpg \
  && cat $out | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
  && sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
  && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
     | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
  && sudo apt update && sudo apt install gh -y

# Linux (Alpine)
apk add github-cli

# Windows (WSL2): use the Linux (Debian/Ubuntu) installer above
```

### SSH Keys

An SSH key identifies your machine as authentic to remote services. Generate one key and register the public key with each service that requires it — GitHub (via `gh`) and HuggingFace (via browser).

**Generate a key**

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
# Saves to ~/.ssh/id_ed25519 (private) and ~/.ssh/id_ed25519.pub (public)
# Press Enter to accept the default path; set a passphrase or leave blank
```

**Add the key to the SSH agent**

```bash
# macOS
eval "$(ssh-agent -s)"
ssh-add --apple-use-keychain ~/.ssh/id_ed25519

# macOS — also add to ~/.ssh/config so the key loads automatically on login.
# Create the file if it doesn't exist, and add these lines:
#
#   Host *
#     AddKeysToAgent yes
#     UseKeychain yes
#     IdentityFile ~/.ssh/id_ed25519

# Linux
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Windows (WSL2): use the Linux instructions above
```

### Credentials

```bash
# Interactive login — when prompted, select SSH as the git protocol
gh auth login

# Verify auth
gh auth status
```

**Register the public key with each service**

```bash
# GitHub — upload via gh CLI (requires auth above to be completed first)
gh ssh-key add ~/.ssh/id_ed25519.pub --title "my-machine"

# HuggingFace — paste contents of public key manually in browser
cat ~/.ssh/id_ed25519.pub   # copy this output
# Then add at https://huggingface.co/settings/keys
```

### Key Commands

```bash
# Repositories
gh repo create my-worker --public             # create a new public repo (required for Hub)
gh repo clone owner/repo                      # clone a repository over SSH
gh repo clone owner/repo -- --depth 1        # shallow clone
gh repo view owner/repo                       # view repo details and URL

# Releases — the Runpod Hub indexes releases, not commits
# Every update to a Hub listing requires a new GitHub release
gh release create v1.0.0 --title "v1.0.0" --notes "Initial release"   # create a release
gh release create v1.0.1 --title "v1.0.1" --notes "Update model tag"  # update Hub listing
gh release list                               # list all releases
gh release view v1.0.0                        # view release details
```

#### Runpod Hub repository structure

A Hub-compatible repository requires these files (in root or `.runpod/` directory):

```
handler.py        # serverless worker implementation
Dockerfile        # container definition
README.md         # documentation shown on Hub listing
.runpod/
  hub.json        # Hub metadata: title, description, category, GPU config, env vars
  tests.json      # test cases run after each release
```

To publish: go to https://console.runpod.io → Hub → Add your repo → enter the GitHub repository URL.

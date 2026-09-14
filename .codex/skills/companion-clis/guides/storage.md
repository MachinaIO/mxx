# AWS CLI

Read only the needed section. Commands are examples; check the installed CLI help before use. Setup, authentication changes, publishing, and deletion are conditional on the requested workflow and its authorization.


The AWS CLI is used to access Runpod storage over the S3 protocol. Any Runpod product that can mount a Network Volume — pods, clusters, and serverless endpoints — can have its storage accessed this way. The bucket name is the network volume ID.

### Install

```bash
# macOS
brew install awscli

# Linux
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip
unzip awscliv2.zip && sudo ./aws/install
```

### Credentials

Runpod uses its own S3-compatible API, not AWS. You need a Runpod user ID and S3 API key — not an AWS account.

- **Access key** (`AWS_ACCESS_KEY_ID`): your Runpod user ID — found in the console under Settings > S3 API Keys, in the key description (format: `user_...`)
- **Secret key** (`AWS_SECRET_ACCESS_KEY`): an S3 API key — generate one at Settings > S3 API Keys > Create. Shown only once; save it immediately (format: `rps_...`)

```bash
# Option 1: interactive configure (writes ~/.aws/credentials and ~/.aws/config)
# When prompted: enter user ID as access key, S3 API key as secret.
# Press Enter to skip region and output format — region is always passed per-command, not stored in config.
aws configure

aws configure list    # verify stored credentials

# Option 2: environment variables (override config files)
export AWS_ACCESS_KEY_ID=user_...
export AWS_SECRET_ACCESS_KEY=rps_...

# To stop using env vars and fall back to config file:
unset AWS_ACCESS_KEY_ID
unset AWS_SECRET_ACCESS_KEY
```

### Region and Endpoint

The `--region` flag on every command is the Runpod datacenter ID where the network volume lives — not an AWS region. The `--endpoint-url` is derived from the same datacenter ID.

Every command requires both flags:
```
--region DATACENTER --endpoint-url https://s3api-DATACENTER.runpod.io/
```

**Tip:** Each network volume on the storage page at https://console.runpod.io/user/storage/ shows a pre-filled example `aws s3 ls` command with the correct `--region` and `--endpoint-url` already substituted. Use this to confirm the exact values for a given volume.

Datacenter IDs by region:

| Region | Datacenter IDs |
|--------|---------------|
| EU | CZ-1, RO-1, IS-1, NO-1 |
| US | CA-2, GA-2, IL-1, KS-2, MD-1, MO-1, MO-2, NC-1, NC-2, NE-1, WA-1 |

### Key Commands

Replace `DATACENTER` with the datacenter ID of your network volume (e.g. `CA-2`) and `NETWORK_VOLUME_ID` with the volume ID (used as the S3 bucket name).

```bash
# List files in a volume
aws s3 ls \
  --region DATACENTER \
  --endpoint-url https://s3api-DATACENTER.runpod.io/ \
  s3://NETWORK_VOLUME_ID/

# List a subdirectory
aws s3 ls \
  --region DATACENTER \
  --endpoint-url https://s3api-DATACENTER.runpod.io/ \
  s3://NETWORK_VOLUME_ID/my-folder/

# Upload a file
aws s3 cp local-file.txt \
  --region DATACENTER \
  --endpoint-url https://s3api-DATACENTER.runpod.io/ \
  s3://NETWORK_VOLUME_ID/

# Download a file
aws s3 cp \
  --region DATACENTER \
  --endpoint-url https://s3api-DATACENTER.runpod.io/ \
  s3://NETWORK_VOLUME_ID/remote-file.txt ./

# Delete a file
aws s3 rm \
  --region DATACENTER \
  --endpoint-url https://s3api-DATACENTER.runpod.io/ \
  s3://NETWORK_VOLUME_ID/remote-file.txt

# Sync a local directory to a volume
aws s3 sync local-dir/ \
  --region DATACENTER \
  --endpoint-url https://s3api-DATACENTER.runpod.io/ \
  s3://NETWORK_VOLUME_ID/remote-dir/
```

Path mapping: `/workspace/my-folder/file.txt` on a pod = `s3://NETWORK_VOLUME_ID/my-folder/file.txt` via S3.

### Troubleshooting

```bash
# Retry on timeout (large transfers)
export AWS_RETRY_MODE=standard
export AWS_MAX_ATTEMPTS=10

# Extend read timeout for large files (seconds)
aws s3 cp large-file.zip \
  --region DATACENTER \
  --endpoint-url https://s3api-DATACENTER.runpod.io/ \
  --cli-read-timeout 7200 \
  s3://NETWORK_VOLUME_ID/
```

# Secret Management Guide

This document explains how to securely manage secrets (API keys, database credentials, etc.) in the Story Memory System.

## Overview

The Story Memory System uses a centralized `SecretManager` class that supports multiple backends:

- **Environment Variables** - For local development (default)
- **AWS Secrets Manager** - For production on AWS
- **Google Cloud Secret Manager** - For production on GCP

All secrets flow through the secret manager, making it easy to switch backends without changing application code.

## Development Setup (Environment Variables)

For local development, use environment variables stored in a `.env` file:

### 1. Create .env File

```bash
cp .env.example .env
```

### 2. Add Your Secrets

Edit `.env` and add your actual API keys:

```bash
# Keep default backend for development
SECRET_BACKEND=env

# Supabase credentials (already configured)
SUPABASE_URL=https://mntpiewbprdjpgcbzaca.supabase.co
SUPABASE_KEY=sb_publishable_UKNVwuG9xU9dmwCIfSt2sg_48zuD5Kn

# xAI API key (get from https://console.x.ai)
XAI_API_KEY=xai-abc123...
```

### 3. Load Environment Variables

The app automatically loads `.env` files using `python-dotenv`. Just make sure to import `load_dotenv()` at the start of your app:

```python
from dotenv import load_dotenv
load_dotenv()  # Load .env file
```

**Important**: Never commit `.env` files to version control! They're already in `.gitignore`.

## Production Setup (AWS Secrets Manager)

For production deployments on AWS, use AWS Secrets Manager to store secrets securely.

### Prerequisites

1. Install AWS SDK:
   ```bash
   pip install boto3
   ```

2. Configure AWS credentials (one of):
   - IAM role attached to EC2/ECS/Lambda
   - Environment variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
   - AWS CLI configuration: `aws configure`

### Step 1: Create Secrets in AWS

Create secrets in AWS Secrets Manager:

```bash
# Create XAI API key secret
aws secretsmanager create-secret \
    --name XAI_API_KEY \
    --secret-string "xai-abc123..." \
    --region us-east-1

# Create Supabase URL secret
aws secretsmanager create-secret \
    --name SUPABASE_URL \
    --secret-string "https://mntpiewbprdjpgcbzaca.supabase.co" \
    --region us-east-1

# Create Supabase key secret
aws secretsmanager create-secret \
    --name SUPABASE_KEY \
    --secret-string "sb_publishable_..." \
    --region us-east-1
```

### Step 2: Configure App to Use AWS

Set the secret backend in your environment:

```bash
# In production environment (e.g., AWS Systems Manager Parameter Store, or container env vars)
SECRET_BACKEND=aws
AWS_REGION=us-east-1
```

### Step 3: Grant IAM Permissions

Ensure your app's IAM role has permission to read secrets:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue",
        "secretsmanager:DescribeSecret"
      ],
      "Resource": [
        "arn:aws:secretsmanager:us-east-1:ACCOUNT_ID:secret:XAI_API_KEY*",
        "arn:aws:secretsmanager:us-east-1:ACCOUNT_ID:secret:SUPABASE_*"
      ]
    }
  ]
}
```

That's it! The app will automatically fetch secrets from AWS Secrets Manager.

## Production Setup (Google Cloud Secret Manager)

For production deployments on Google Cloud Platform.

### Prerequisites

1. Install GCP SDK:
   ```bash
   pip install google-cloud-secret-manager
   ```

2. Configure GCP authentication:
   - Service account key file
   - Set `GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json`

### Step 1: Create Secrets in GCP

```bash
# Enable Secret Manager API
gcloud services enable secretmanager.googleapis.com

# Create secrets
echo -n "xai-abc123..." | \
  gcloud secrets create XAI_API_KEY --data-file=-

echo -n "https://mntpiewbprdjpgcbzaca.supabase.co" | \
  gcloud secrets create SUPABASE_URL --data-file=-

echo -n "sb_publishable_..." | \
  gcloud secrets create SUPABASE_KEY --data-file=-
```

### Step 2: Configure App to Use GCP

Set the secret backend in your environment:

```bash
SECRET_BACKEND=gcp
GCP_PROJECT_ID=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
```

### Step 3: Grant IAM Permissions

Ensure your service account has the `Secret Manager Secret Accessor` role:

```bash
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:YOUR_SERVICE_ACCOUNT@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

## Using the Secret Manager in Code

All services already use the secret manager. Here's how to use it in new code:

### Basic Usage

```python
from config import get_secret, require_secret

# Get a secret (returns None if not found)
api_key = get_secret("XAI_API_KEY")

# Get a required secret (raises ValueError if not found)
api_key = require_secret("XAI_API_KEY")
```

### In Services

```python
from config import require_secret

class MyService:
    def __init__(self):
        # Automatically gets from configured backend
        self.api_key = require_secret("MY_API_KEY")
```

### Advanced: Direct SecretManager Usage

```python
from config import SecretManager, SecretBackend

# Use specific backend
manager = SecretManager(backend=SecretBackend.AWS)
secret = manager.get_secret("MY_SECRET")

# Get singleton instance
from config import get_secret_manager
manager = get_secret_manager()
```

## Security Best Practices

### ✅ DO

- **Use environment variables for development** - Easy to manage, kept out of version control
- **Use cloud secret managers for production** - Centralized, encrypted, auditable
- **Rotate secrets regularly** - Use AWS/GCP secret rotation features
- **Use least-privilege IAM roles** - Only grant access to secrets the app needs
- **Enable secret versioning** - Allows rollback if needed
- **Monitor secret access** - Use CloudTrail/Cloud Audit Logs

### ❌ DON'T

- **Never commit secrets to Git** - Use `.gitignore` (already configured)
- **Don't hardcode secrets in code** - Always use the secret manager
- **Don't share .env files** - Each developer should have their own
- **Don't log secrets** - Be careful with debug logging
- **Don't use production secrets in development** - Keep them separate

## Migrating Between Backends

Switching from development to production is easy:

1. **Create secrets in new backend** (AWS or GCP)
2. **Update environment variable**: `SECRET_BACKEND=aws` or `SECRET_BACKEND=gcp`
3. **Restart your app** - It will automatically use the new backend

No code changes required!

## Troubleshooting

### "Required secret 'X' not found"

**Development**: Make sure the secret is in your `.env` file
```bash
echo "XAI_API_KEY=your-key" >> .env
```

**AWS**: Verify the secret exists
```bash
aws secretsmanager describe-secret --secret-id XAI_API_KEY
```

**GCP**: Verify the secret exists
```bash
gcloud secrets describe XAI_API_KEY
```

### "boto3 required for AWS Secrets Manager"

Install AWS SDK:
```bash
pip install boto3
```

### "google-cloud-secret-manager required for GCP"

Install GCP SDK:
```bash
pip install google-cloud-secret-manager
```

### AWS/GCP Permissions Denied

Verify your IAM role/service account has the correct permissions (see production setup sections above).

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SECRET_BACKEND` | No | `env` | Secret backend: `env`, `aws`, or `gcp` |
| `AWS_REGION` | AWS only | `us-east-1` | AWS region for Secrets Manager |
| `GCP_PROJECT_ID` | GCP only | - | Google Cloud project ID |
| `SUPABASE_URL` | Yes | - | Supabase project URL |
| `SUPABASE_KEY` | Yes | - | Supabase API key |
| `XAI_API_KEY` | Yes | - | xAI (Grok) API key |
| `ANTHROPIC_API_KEY` | No | - | Claude API key (optional) |

## Secret Caching

Secrets are cached in memory after first retrieval for performance. The cache persists for the lifetime of the application process.

**Note**: If you update a secret in AWS/GCP, you'll need to restart your application to pick up the new value.

For development, you can clear the cache by restarting the dev server.

"""
Centralized secret management for Story Memory System.

Supports multiple secret backends:
- Environment variables (.env files for development)
- AWS Secrets Manager (production)
- Google Cloud Secret Manager (production)

All services should use get_secret() instead of os.getenv() directly.
"""

import os
import json
from typing import Optional
from functools import lru_cache
from enum import Enum


class SecretBackend(str, Enum):
    """Available secret storage backends."""
    ENV = "env"  # Environment variables (default for development)
    AWS = "aws"  # AWS Secrets Manager
    GCP = "gcp"  # Google Cloud Secret Manager


class SecretManager:
    """
    Centralized secret manager with support for multiple backends.

    Usage:
        secret_mgr = SecretManager()
        api_key = secret_mgr.get_secret("XAI_API_KEY")
    """

    def __init__(self, backend: Optional[SecretBackend] = None):
        """
        Initialize secret manager.

        Args:
            backend: Secret backend to use (defaults to ENV, or uses SECRET_BACKEND env var)
        """
        self.backend = backend or SecretBackend(
            os.getenv("SECRET_BACKEND", "env")
        )

        # Initialize backend clients lazily
        self._aws_client = None
        self._gcp_client = None

    @lru_cache(maxsize=128)
    def get_secret(self, secret_name: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a secret from the configured backend.

        Secrets are cached in memory after first retrieval for performance.

        Args:
            secret_name: Name of the secret to retrieve
            default: Default value if secret not found

        Returns:
            Secret value or default if not found

        Raises:
            ValueError: If secret backend is misconfigured
        """
        if self.backend == SecretBackend.ENV:
            return self._get_from_env(secret_name, default)
        elif self.backend == SecretBackend.AWS:
            return self._get_from_aws(secret_name, default)
        elif self.backend == SecretBackend.GCP:
            return self._get_from_gcp(secret_name, default)
        else:
            raise ValueError(f"Unknown secret backend: {self.backend}")

    def _get_from_env(self, secret_name: str, default: Optional[str]) -> Optional[str]:
        """Get secret from environment variables."""
        return os.getenv(secret_name, default)

    def _get_from_aws(self, secret_name: str, default: Optional[str]) -> Optional[str]:
        """
        Get secret from AWS Secrets Manager.

        Requires: boto3, AWS credentials configured
        """
        try:
            import boto3
            from botocore.exceptions import ClientError
        except ImportError:
            raise ImportError(
                "boto3 required for AWS Secrets Manager. "
                "Install with: pip install boto3"
            )

        if self._aws_client is None:
            # Get AWS region from environment or use default
            region = os.getenv("AWS_REGION", "us-east-1")
            self._aws_client = boto3.client(
                service_name="secretsmanager",
                region_name=region
            )

        try:
            # Retrieve the secret
            response = self._aws_client.get_secret_value(SecretId=secret_name)

            # Parse the secret value
            if "SecretString" in response:
                secret = response["SecretString"]
                # Try to parse as JSON (for multi-value secrets)
                try:
                    secret_dict = json.loads(secret)
                    # If it's a JSON object, return the whole thing as string
                    # In practice, you might want to access specific keys
                    return secret
                except json.JSONDecodeError:
                    # It's a plain string
                    return secret
            else:
                # Binary secret (less common)
                return response["SecretBinary"].decode("utf-8")

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "ResourceNotFoundException":
                return default
            elif error_code == "InvalidRequestException":
                raise ValueError(f"Invalid request for secret: {secret_name}")
            elif error_code == "InvalidParameterException":
                raise ValueError(f"Invalid parameter for secret: {secret_name}")
            else:
                raise

    def _get_from_gcp(self, secret_name: str, default: Optional[str]) -> Optional[str]:
        """
        Get secret from Google Cloud Secret Manager.

        Requires: google-cloud-secret-manager, GCP credentials configured
        Secret name format: projects/PROJECT_ID/secrets/SECRET_NAME/versions/latest
        """
        try:
            from google.cloud import secretmanager
            from google.api_core.exceptions import NotFound
        except ImportError:
            raise ImportError(
                "google-cloud-secret-manager required for GCP Secret Manager. "
                "Install with: pip install google-cloud-secret-manager"
            )

        if self._gcp_client is None:
            self._gcp_client = secretmanager.SecretManagerServiceClient()

        # Get GCP project ID from environment
        project_id = os.getenv("GCP_PROJECT_ID")
        if not project_id:
            raise ValueError("GCP_PROJECT_ID environment variable required for GCP backend")

        # Build the resource name (use latest version by default)
        if not secret_name.startswith("projects/"):
            secret_name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"

        try:
            response = self._gcp_client.access_secret_version(name=secret_name)
            payload = response.payload.data.decode("UTF-8")

            # Try to parse as JSON
            try:
                secret_dict = json.loads(payload)
                return payload  # Return raw JSON string
            except json.JSONDecodeError:
                return payload  # Return plain string

        except NotFound:
            return default

    def require_secret(self, secret_name: str) -> str:
        """
        Get a required secret, raising an error if not found.

        Args:
            secret_name: Name of the secret to retrieve

        Returns:
            Secret value

        Raises:
            ValueError: If secret is not found
        """
        secret = self.get_secret(secret_name)
        if secret is None:
            raise ValueError(
                f"Required secret '{secret_name}' not found in {self.backend.value} backend"
            )
        return secret


# Global secret manager instance
_secret_manager: Optional[SecretManager] = None


def get_secret_manager() -> SecretManager:
    """
    Get the global secret manager instance (singleton pattern).

    Returns:
        SecretManager instance
    """
    global _secret_manager
    if _secret_manager is None:
        _secret_manager = SecretManager()
    return _secret_manager


def get_secret(secret_name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Convenience function to get a secret using the global secret manager.

    Args:
        secret_name: Name of the secret to retrieve
        default: Default value if secret not found

    Returns:
        Secret value or default
    """
    return get_secret_manager().get_secret(secret_name, default)


def require_secret(secret_name: str) -> str:
    """
    Convenience function to get a required secret using the global secret manager.

    Args:
        secret_name: Name of the secret to retrieve

    Returns:
        Secret value

    Raises:
        ValueError: If secret is not found
    """
    return get_secret_manager().require_secret(secret_name)

"""
Backblaze B2 Storage Service for NSFW-safe image hosting.
Uses S3-compatible API for uploads and generates signed URLs for private access.
"""

import os
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from datetime import datetime
from typing import Optional, BinaryIO
import hashlib


class B2StorageService:
    """
    Backblaze B2 storage service for image uploads and signed URL generation.
    """

    def __init__(self):
        self.key_id = os.getenv("B2_KEY_ID")
        self.app_key = os.getenv("B2_APP_KEY")
        self.bucket_name = os.getenv("B2_BUCKET_NAME", "story-memory-images")
        self.endpoint_url = os.getenv("B2_ENDPOINT_URL", "https://s3.us-east-005.backblazeb2.com")

        if not self.key_id or not self.app_key:
            raise ValueError("B2_KEY_ID and B2_APP_KEY environment variables required")

        # Create S3 client for B2
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.key_id,
            aws_secret_access_key=self.app_key,
            config=Config(signature_version="s3v4")
        )

    def upload_image(
        self,
        file_data: bytes,
        story_id: str,
        entity_id: str,
        image_type: str = "portrait",
        style: str = "default",
        extension: str = "png"
    ) -> dict:
        """
        Upload an image to B2 and return the storage info.

        Args:
            file_data: Raw image bytes
            story_id: Story UUID
            entity_id: Entity UUID
            image_type: Type of image (portrait, scene, etc.)
            style: Style used for generation
            extension: File extension

        Returns:
            dict with file_path, file_url (signed), and metadata
        """
        # Generate unique filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        content_hash = hashlib.md5(file_data).hexdigest()[:8]
        filename = f"{image_type}_{style}_{timestamp}_{content_hash}.{extension}"

        # Build path: stories/{story_id}/entities/{entity_id}/{filename}
        file_path = f"stories/{story_id}/entities/{entity_id}/{filename}"

        # Determine content type
        content_types = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "webp": "image/webp",
            "gif": "image/gif"
        }
        content_type = content_types.get(extension.lower(), "image/png")

        try:
            # Upload to B2
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=file_path,
                Body=file_data,
                ContentType=content_type,
                Metadata={
                    "story-id": story_id,
                    "entity-id": entity_id,
                    "image-type": image_type,
                    "style": style
                }
            )

            # Generate signed URL (valid for 1 hour by default)
            signed_url = self.get_signed_url(file_path)

            return {
                "file_path": file_path,
                "file_url": signed_url,
                "bucket": self.bucket_name,
                "content_type": content_type,
                "size_bytes": len(file_data)
            }

        except ClientError as e:
            raise Exception(f"B2 upload failed: {str(e)}")

    def upload_scene_image(
        self,
        file_data: bytes,
        story_id: str,
        session_id: Optional[str] = None,
        scene_description: str = "",
        extension: str = "png"
    ) -> dict:
        """
        Upload a scene image to B2.

        Args:
            file_data: Raw image bytes
            story_id: Story UUID
            session_id: Optional session UUID
            scene_description: Description of the scene
            extension: File extension

        Returns:
            dict with file_path, file_url (signed), and metadata
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        content_hash = hashlib.md5(file_data).hexdigest()[:8]
        filename = f"scene_{timestamp}_{content_hash}.{extension}"

        # Build path
        if session_id:
            file_path = f"stories/{story_id}/sessions/{session_id}/scenes/{filename}"
        else:
            file_path = f"stories/{story_id}/scenes/{filename}"

        content_type = "image/png" if extension == "png" else f"image/{extension}"

        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=file_path,
                Body=file_data,
                ContentType=content_type,
                Metadata={
                    "story-id": story_id,
                    "session-id": session_id or "",
                    "image-type": "scene",
                    "description": scene_description[:256]  # Limit metadata size
                }
            )

            signed_url = self.get_signed_url(file_path)

            return {
                "file_path": file_path,
                "file_url": signed_url,
                "bucket": self.bucket_name,
                "content_type": content_type,
                "size_bytes": len(file_data)
            }

        except ClientError as e:
            raise Exception(f"B2 upload failed: {str(e)}")

    def get_signed_url(self, file_path: str, expires_in: int = 3600) -> str:
        """
        Generate a signed URL for private file access.

        Args:
            file_path: Path to file in bucket
            expires_in: URL validity in seconds (default 1 hour)

        Returns:
            Signed URL string
        """
        try:
            url = self.s3_client.generate_presigned_url(
                "get_object",
                Params={
                    "Bucket": self.bucket_name,
                    "Key": file_path
                },
                ExpiresIn=expires_in
            )
            return url
        except ClientError as e:
            raise Exception(f"Failed to generate signed URL: {str(e)}")

    def delete_image(self, file_path: str) -> bool:
        """
        Delete an image from B2.

        Args:
            file_path: Path to file in bucket

        Returns:
            True if deleted successfully
        """
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=file_path
            )
            return True
        except ClientError as e:
            raise Exception(f"B2 delete failed: {str(e)}")

    def list_entity_images(self, story_id: str, entity_id: str) -> list:
        """
        List all images for an entity.

        Args:
            story_id: Story UUID
            entity_id: Entity UUID

        Returns:
            List of image paths
        """
        prefix = f"stories/{story_id}/entities/{entity_id}/"

        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )

            images = []
            for obj in response.get("Contents", []):
                images.append({
                    "file_path": obj["Key"],
                    "size_bytes": obj["Size"],
                    "last_modified": obj["LastModified"].isoformat(),
                    "file_url": self.get_signed_url(obj["Key"])
                })

            return images

        except ClientError as e:
            raise Exception(f"B2 list failed: {str(e)}")

    def check_connection(self) -> dict:
        """
        Check B2 connection status.

        Returns:
            dict with status and bucket info
        """
        try:
            response = self.s3_client.head_bucket(Bucket=self.bucket_name)
            return {
                "status": "connected",
                "bucket": self.bucket_name,
                "endpoint": self.endpoint_url
            }
        except ClientError as e:
            return {
                "status": "error",
                "error": str(e)
            }


# Singleton instance
_b2_service: Optional[B2StorageService] = None


def get_b2_storage() -> B2StorageService:
    """Get or create B2 storage service singleton."""
    global _b2_service
    if _b2_service is None:
        _b2_service = B2StorageService()
    return _b2_service

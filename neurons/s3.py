import functools
import logging
import uuid

import httpx
import boto3

from constants import (
    COMPUTE_HORDE_VALIDATION_S3_BUCKET, AWS_ENDPOINT_URL, AWS_SECRET_ACCESS_KEY, AWS_ACCESS_KEY_ID)

logger = logging.getLogger(__name__)


_required_settings = {
    "COMPUTE_HORDE_VALIDATION_S3_BUCKET": COMPUTE_HORDE_VALIDATION_S3_BUCKET,
    "AWS_ENDPOINT_URL": AWS_ENDPOINT_URL,
    "AWS_SECRET_ACCESS_KEY": AWS_SECRET_ACCESS_KEY,
    "AWS_ACCESS_KEY_ID": AWS_ACCESS_KEY_ID,
}
if _missing_settings := [k for k, v in _required_settings.items() if v is None]:
    raise RuntimeError(f"Required settings: {', '.join(_missing_settings)}. Add them to your .env file.")


def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        endpoint_url=AWS_ENDPOINT_URL,
    )


def _generate_presigned_url(
    method: str,
    key: str,
    expiration: int = 3600,
) -> str:
    s3_client = get_s3_client()

    return s3_client.generate_presigned_url(  # type: ignore
        method,
        Params={"Bucket": COMPUTE_HORDE_VALIDATION_S3_BUCKET, "Key": key},
        ExpiresIn=expiration,
    )


generate_upload_url = functools.partial(_generate_presigned_url, "put_object")
generate_download_url = functools.partial(_generate_presigned_url, "get_object")


def get_public_url(key: str) -> str:
    return f"{AWS_ENDPOINT_URL}/{COMPUTE_HORDE_VALIDATION_S3_BUCKET}/{key}"


async def download_file_content(s3_url: str) -> bytes:
    async with httpx.AsyncClient() as client:
        response = await client.get(s3_url, timeout=5)
        response.raise_for_status()
        return response.content


def upload_data_to_s3(data: str | bytes) -> str | None:
    data_sample_name = str(uuid.uuid4())
    s3_client = get_s3_client()

    s3_client.put_object(
        Body=data,
        Bucket=COMPUTE_HORDE_VALIDATION_S3_BUCKET,
        Key=data_sample_name,
    )

    # return the public url
    public_url = get_public_url(data_sample_name)
    logger.info(f"Uploaded data sample to {public_url}")
    return public_url

from __future__ import annotations

import io
import json
import time
from typing import Any, Dict, Iterable, Iterator, List, Tuple
from urllib.parse import urlparse

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from .config import StorageConfig
from .observability import EXTERNAL_LATENCY


class ObjectStorage:
    """MinIO adapter. Ray tasks exchange minio:// URIs instead of file bytes."""

    def __init__(self, config: StorageConfig | Dict[str, Any]):
        self.config = StorageConfig(**config) if isinstance(config, dict) else config
        scheme = "https" if self.config.secure else "http"
        self.client = boto3.client(
            "s3",
            endpoint_url=f"{scheme}://{self.config.endpoint}",
            aws_access_key_id=self.config.access_key,
            aws_secret_access_key=self.config.secret_key,
            config=Config(
                signature_version="s3v4",
                connect_timeout=self.config.connect_timeout_seconds,
                read_timeout=self.config.read_timeout_seconds,
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        )
        self.ensure_bucket()

    def ensure_bucket(self) -> None:
        try:
            self.client.head_bucket(Bucket=self.config.bucket)
        except ClientError:
            try:
                self.client.create_bucket(Bucket=self.config.bucket)
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code", "")
                if code not in {"BucketAlreadyExists", "BucketAlreadyOwnedByYou"}:
                    raise

    def uri(self, key: str) -> str:
        return f"minio://{self.config.bucket}/{key.lstrip('/')}"

    def parse_uri(self, uri: str) -> Tuple[str, str]:
        parsed = urlparse(uri)
        if parsed.scheme not in {"minio", "s3"}:
            raise ValueError(f"Unsupported object URI: {uri}")
        return parsed.netloc, parsed.path.lstrip("/")

    def exists(self, key_or_uri: str) -> bool:
        bucket, key = self._target(key_or_uri)
        try:
            self.client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as exc:
            if exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404:
                return False
            raise

    def put_bytes(self, key: str, content: bytes, content_type: str = "application/octet-stream", metadata: Dict[str, str] | None = None) -> str:
        if not self.exists(key):
            started = time.perf_counter()
            self.client.put_object(
                Bucket=self.config.bucket,
                Key=key,
                Body=content,
                ContentLength=len(content),
                ContentType=content_type,
                Metadata=metadata or {},
            )
            EXTERNAL_LATENCY.labels(dependency="minio", operation="write").observe(time.perf_counter() - started)
        return self.uri(key)

    def put_json(self, key: str, value: Any) -> str:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return self.put_bytes(key, payload, "application/json")

    def put_jsonl(self, key: str, values: Iterable[Dict[str, Any]]) -> str:
        body = "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values).encode("utf-8")
        return self.put_bytes(key, body, "application/x-ndjson")

    def get_bytes(self, key_or_uri: str) -> bytes:
        bucket, key = self._target(key_or_uri)
        started = time.perf_counter()
        response = self.client.get_object(Bucket=bucket, Key=key)
        try:
            value = response["Body"].read()
            EXTERNAL_LATENCY.labels(dependency="minio", operation="read").observe(time.perf_counter() - started)
            return value
        finally:
            response["Body"].close()

    def download_to_file(self, key_or_uri: str, file_object: io.BufferedWriter) -> None:
        bucket, key = self._target(key_or_uri)
        self.client.download_fileobj(bucket, key, file_object)

    def iter_jsonl(self, key_or_uri: str) -> Iterator[Dict[str, Any]]:
        for line in self.get_bytes(key_or_uri).decode("utf-8").splitlines():
            if line.strip():
                yield json.loads(line)

    def _target(self, key_or_uri: str) -> Tuple[str, str]:
        if "://" in key_or_uri:
            return self.parse_uri(key_or_uri)
        return self.config.bucket, key_or_uri.lstrip("/")

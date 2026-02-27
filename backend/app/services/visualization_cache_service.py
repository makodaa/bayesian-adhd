from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from ..config import VIZ_CACHE_DIR, VIZ_CACHE_MAX_BYTES, VIZ_CACHE_TTL_SECONDS
from ..core.logging_config import get_app_logger

logger = get_app_logger(__name__)


class VisualizationCacheError(Exception):
    """Base exception for visualization cache failures."""


class ContextNotFoundError(VisualizationCacheError):
    """Raised when a cached visualization context does not exist."""


class ContextExpiredError(VisualizationCacheError):
    """Raised when a cached visualization context has expired."""


class ContextAccessError(VisualizationCacheError):
    """Raised when a user attempts to access another user's context."""


class VisualizationCacheService:
    """Filesystem cache for uploaded EEG files and generated band images."""

    CACHE_PROFILE = "preview_v1"
    DETAIL_CACHE_PROFILE = "detail_v1"
    CONTEXT_SUFFIX = ".context.json"

    def __init__(
        self,
        cache_dir: str | None = None,
        ttl_seconds: int = VIZ_CACHE_TTL_SECONDS,
        max_bytes: int = VIZ_CACHE_MAX_BYTES,
    ):
        cache_root = os.getenv("VIZ_CACHE_DIR", cache_dir or VIZ_CACHE_DIR)
        self.cache_dir = Path(cache_root)
        self.ttl_seconds = int(os.getenv("VIZ_CACHE_TTL_SECONDS", str(ttl_seconds)))
        self.max_bytes = int(os.getenv("VIZ_CACHE_MAX_BYTES", str(max_bytes)))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _now() -> int:
        return int(time.time())

    def _context_path(self, context_id: str) -> Path:
        return self.cache_dir / f"{context_id}{self.CONTEXT_SUFFIX}"

    def _csv_path(self, context_id: str) -> Path:
        return self.cache_dir / f"{context_id}.csv"

    def _image_path(self, context_id: str, band: str, quality: str = "preview") -> Path:
        profile = self.DETAIL_CACHE_PROFILE if quality == "detail" else self.CACHE_PROFILE
        return self.cache_dir / f"{context_id}__{profile}__{band}.png"

    @staticmethod
    def _write_bytes_atomic(path: Path, payload: bytes) -> None:
        temp_path = path.with_suffix(f"{path.suffix}.tmp")
        with open(temp_path, "wb") as handle:
            handle.write(payload)
        temp_path.replace(path)

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
        temp_path = path.with_suffix(f"{path.suffix}.tmp")
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        temp_path.replace(path)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _touch_context(self, context_id: str, meta: dict[str, Any]) -> dict[str, Any]:
        now = self._now()
        meta["updated_at"] = now
        meta["expires_at"] = now + self.ttl_seconds
        self._write_json_atomic(self._context_path(context_id), meta)
        return meta

    def _load_meta(self, context_id: str) -> dict[str, Any]:
        context_path = self._context_path(context_id)
        if not context_path.exists():
            raise ContextNotFoundError("Visualization context not found")

        try:
            meta = self._read_json(context_path)
        except (json.JSONDecodeError, OSError) as exc:
            self._remove_context(context_id)
            raise ContextNotFoundError("Visualization context is invalid") from exc

        expires_at = int(meta.get("expires_at", 0))
        if expires_at < self._now():
            self._remove_context(context_id)
            raise ContextExpiredError("Visualization context has expired")

        return meta

    def _remove_context(self, context_id: str) -> None:
        for path in self.cache_dir.glob(f"{context_id}*"):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.warning(f"Failed to remove cached file: {path}")

    def _context_size_bytes(self, context_id: str) -> int:
        total = 0
        for path in self.cache_dir.glob(f"{context_id}*"):
            try:
                total += path.stat().st_size
            except OSError:
                continue
        return total

    def _cache_size_bytes(self) -> int:
        total = 0
        for path in self.cache_dir.iterdir():
            if not path.is_file():
                continue
            try:
                total += path.stat().st_size
            except OSError:
                continue
        return total

    def _enforce_size_limit(self) -> None:
        total_size = self._cache_size_bytes()
        if total_size <= self.max_bytes:
            return

        contexts: list[tuple[int, str, int]] = []
        for meta_path in self.cache_dir.glob(f"*{self.CONTEXT_SUFFIX}"):
            try:
                meta = self._read_json(meta_path)
            except (json.JSONDecodeError, OSError):
                continue

            context_id = str(meta.get("context_id", "")).strip()
            if not context_id:
                continue

            updated_at = int(meta.get("updated_at", 0))
            context_size = self._context_size_bytes(context_id)
            contexts.append((updated_at, context_id, context_size))

        contexts.sort(key=lambda item: item[0])

        for _, context_id, context_size in contexts:
            self._remove_context(context_id)
            total_size -= context_size
            logger.info(
                f"Evicted visualization context {context_id} ({context_size} bytes)"
            )
            if total_size <= self.max_bytes:
                break

    def prune_cache(self) -> None:
        """Remove expired contexts and enforce total cache size limit."""
        now = self._now()
        for meta_path in self.cache_dir.glob(f"*{self.CONTEXT_SUFFIX}"):
            context_id = meta_path.name.replace(self.CONTEXT_SUFFIX, "")
            try:
                meta = self._read_json(meta_path)
                expires_at = int(meta.get("expires_at", 0))
            except (json.JSONDecodeError, OSError, ValueError):
                expires_at = 0

            if expires_at < now:
                self._remove_context(context_id)

        self._enforce_size_limit()

    def create_or_refresh_context(
        self,
        file_bytes: bytes,
        clinician_id: int,
        filename: str | None,
    ) -> dict[str, Any]:
        """Create or refresh a context for an uploaded EEG file."""
        if not file_bytes:
            raise ValueError("Uploaded file is empty")

        self.prune_cache()

        file_hash = hashlib.sha256(file_bytes).hexdigest()
        seed = f"{clinician_id}:{file_hash}:{self.CACHE_PROFILE}"
        context_id = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:40]

        now = self._now()
        context_path = self._context_path(context_id)
        csv_path = self._csv_path(context_id)

        if context_path.exists() and csv_path.exists():
            try:
                meta = self._load_meta(context_id)
            except (ContextNotFoundError, ContextExpiredError):
                meta = None

            if meta is not None:
                if str(meta.get("clinician_id")) != str(clinician_id):
                    raise ContextAccessError("Not authorized for visualization context")

                meta = self._touch_context(context_id, meta)
                return {
                    "context_id": context_id,
                    "bands": meta.get("bands", []),
                    "expires_at": meta["expires_at"],
                    "created": False,
                }

        meta = {
            "context_id": context_id,
            "clinician_id": str(clinician_id),
            "filename": filename,
            "file_hash": file_hash,
            "byte_size": len(file_bytes),
            "profile": self.CACHE_PROFILE,
            "created_at": now,
            "updated_at": now,
            "expires_at": now + self.ttl_seconds,
            "bands": [],
        }

        self._write_bytes_atomic(csv_path, file_bytes)
        self._write_json_atomic(context_path, meta)
        self._enforce_size_limit()

        return {
            "context_id": context_id,
            "bands": [],
            "expires_at": meta["expires_at"],
            "created": True,
        }

    def read_context_csv(self, context_id: str, clinician_id: int) -> bytes:
        """Read cached CSV bytes for a context after access checks."""
        meta = self._load_meta(context_id)
        if str(meta.get("clinician_id")) != str(clinician_id):
            raise ContextAccessError("Not authorized for visualization context")

        csv_path = self._csv_path(context_id)
        if not csv_path.exists():
            self._remove_context(context_id)
            raise ContextNotFoundError("Visualization context data is missing")

        try:
            payload = csv_path.read_bytes()
        except OSError as exc:
            raise VisualizationCacheError("Unable to read visualization context") from exc

        self._touch_context(context_id, meta)
        return payload

    def get_cached_image_path(
        self, context_id: str, clinician_id: int, band: str, quality: str = "preview"
    ) -> Path | None:
        """Return cached image path if present and authorized."""
        meta = self._load_meta(context_id)
        if str(meta.get("clinician_id")) != str(clinician_id):
            raise ContextAccessError("Not authorized for visualization context")

        image_path = self._image_path(context_id, band, quality)
        if image_path.exists():
            self._touch_context(context_id, meta)
            return image_path
        return None

    def store_image(
        self,
        context_id: str,
        clinician_id: int,
        band: str,
        image_bytes: bytes,
        quality: str = "preview",
    ) -> Path:
        """Store a generated PNG image and return its path."""
        if not image_bytes:
            raise ValueError("Generated visualization image is empty")

        meta = self._load_meta(context_id)
        if str(meta.get("clinician_id")) != str(clinician_id):
            raise ContextAccessError("Not authorized for visualization context")

        image_path = self._image_path(context_id, band, quality)
        self._write_bytes_atomic(image_path, image_bytes)

        bands = set(meta.get("bands", []))
        bands.add(band)
        meta["bands"] = sorted(bands)
        self._touch_context(context_id, meta)
        self._enforce_size_limit()
        return image_path

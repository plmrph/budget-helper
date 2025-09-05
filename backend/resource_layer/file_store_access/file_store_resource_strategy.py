"""
FileStoreResourceStrategy interface and implementation.

This module defines the strategy interface for file store resources
and provides the local file system implementation.
"""

import hashlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path

from thrift_gen.exceptions.ttypes import (
    InternalException,
    NotFoundException,
    ValidationException,
)
from thrift_gen.filestoreaccess.ttypes import FileFormat, FileMetadata, FileQuery

logger = logging.getLogger(__name__)


class FileStoreResourceStrategy(ABC):
    """Abstract interface for file store resource strategy."""

    @abstractmethod
    async def write_file(
        self, path: str, data: bytes, format: FileFormat | None = None
    ) -> bool:
        """Write file to storage."""
        pass

    @abstractmethod
    async def read_file(self, path: str) -> bytes:
        """Read file from storage."""
        pass

    @abstractmethod
    async def delete_file(self, path: str) -> bool:
        """Delete file from storage."""
        pass

    @abstractmethod
    async def list_files(self, query: FileQuery) -> list[FileMetadata]:
        """List files matching query."""
        pass

    @abstractmethod
    async def get_file_metadata(self, path: str) -> FileMetadata:
        """Get file metadata."""
        pass

    @abstractmethod
    async def file_exists(self, path: str) -> bool:
        """Check if file exists."""
        pass


class LocalFileSystemStrategy(FileStoreResourceStrategy):
    """Local file system implementation of FileStoreResourceStrategy."""

    def __init__(self, base_path: str = "/tmp/file_store"):
        """
        Initialize local file system strategy.

        Args:
            base_path: Base directory for file storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"LocalFileSystemStrategy initialized with base path: {base_path}")

    def _get_full_path(self, path: str) -> Path:
        """Get full path, ensuring it's within base directory."""
        # Normalize path and ensure it's relative
        normalized_path = Path(path).as_posix().lstrip("/")
        full_path = self.base_path / normalized_path

        # Security check: ensure path is within base directory
        try:
            full_path.resolve().relative_to(self.base_path.resolve())
        except ValueError as e:
            raise ValidationException(
                f"Path '{path}' is outside allowed directory"
            ) from e

        return full_path

    def _calculate_checksum(self, data: bytes) -> str:
        """Calculate MD5 checksum of data."""
        return hashlib.md5(data).hexdigest()

    def _detect_format(self, path: str, data: bytes) -> FileFormat:
        """Detect file format based on path and content."""
        path_lower = path.lower()

        if path_lower.endswith(".json"):
            return FileFormat.JSON
        elif path_lower.endswith(".csv"):
            return FileFormat.CSV
        elif path_lower.endswith(".xml"):
            return FileFormat.XML
        else:
            # Try to detect from content
            try:
                data.decode("utf-8")
                if data.strip().startswith(b"{") or data.strip().startswith(b"["):
                    return FileFormat.JSON
                elif b"," in data and b"\n" in data:
                    return FileFormat.CSV
                elif b"<" in data and b">" in data:
                    return FileFormat.XML
            except UnicodeDecodeError:
                pass

        return FileFormat.BINARY

    async def write_file(
        self, path: str, data: bytes, format: FileFormat | None = None
    ) -> bool:
        """Write file to local file system."""
        try:
            full_path = self._get_full_path(path)

            # Create parent directories if they don't exist
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Write file
            with open(full_path, "wb") as f:
                f.write(data)

            logger.info(f"Successfully wrote file: {path}")
            return True

        except ValidationException:
            raise
        except Exception as e:
            logger.error(f"Error writing file {path}: {e}")
            raise InternalException(f"Failed to write file: {str(e)}") from e

    async def read_file(self, path: str) -> bytes:
        """Read file from local file system."""
        try:
            full_path = self._get_full_path(path)

            if not full_path.exists():
                raise NotFoundException(f"File not found: {path}")

            with open(full_path, "rb") as f:
                data = f.read()

            logger.info(f"Successfully read file: {path}")
            return data

        except (ValidationException, NotFoundException):
            raise
        except Exception as e:
            logger.error(f"Error reading file {path}: {e}")
            raise InternalException(f"Failed to read file: {str(e)}") from e

    async def delete_file(self, path: str) -> bool:
        """Delete file from local file system."""
        try:
            full_path = self._get_full_path(path)

            if not full_path.exists():
                raise NotFoundException(f"File not found: {path}")

            full_path.unlink()

            logger.info(f"Successfully deleted file: {path}")
            return True

        except (ValidationException, NotFoundException):
            raise
        except Exception as e:
            logger.error(f"Error deleting file {path}: {e}")
            raise InternalException(f"Failed to delete file: {str(e)}") from e

    async def list_files(self, query: FileQuery) -> list[FileMetadata]:
        """List files matching query from local file system."""
        try:
            files = []
            base_search_path = self.base_path

            # If pathPattern is provided, use it as a starting point
            if query.pathPattern:
                # Simple pattern matching - just check if it's a directory
                pattern_path = self._get_full_path(query.pathPattern)
                if pattern_path.is_dir():
                    base_search_path = pattern_path

            # Walk through directory tree
            for file_path in base_search_path.rglob("*"):
                if not file_path.is_file():
                    continue

                # Get relative path from base
                try:
                    relative_path = file_path.relative_to(self.base_path)
                except ValueError:
                    continue

                # Apply filters
                stat = file_path.stat()

                # Size filters
                if query.minSize is not None and stat.st_size < query.minSize:
                    continue
                if query.maxSize is not None and stat.st_size > query.maxSize:
                    continue

                # Time filters
                if (
                    query.modifiedAfter is not None
                    and stat.st_mtime < query.modifiedAfter
                ):
                    continue
                if (
                    query.modifiedBefore is not None
                    and stat.st_mtime > query.modifiedBefore
                ):
                    continue

                # Create metadata
                with open(file_path, "rb") as f:
                    data = f.read()

                detected_format = self._detect_format(str(relative_path), data)

                # Format filter
                if query.format is not None and detected_format != query.format:
                    continue

                metadata = FileMetadata(
                    path=str(relative_path),
                    format=detected_format,
                    size=stat.st_size,
                    checksum=self._calculate_checksum(data),
                    lastModified=int(stat.st_mtime),
                )

                files.append(metadata)

            logger.info(f"Found {len(files)} files matching query")
            return files

        except ValidationException:
            raise
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            raise InternalException(f"Failed to list files: {str(e)}") from e

    async def get_file_metadata(self, path: str) -> FileMetadata:
        """Get file metadata from local file system."""
        try:
            full_path = self._get_full_path(path)

            if not full_path.exists():
                raise NotFoundException(f"File not found: {path}")

            stat = full_path.stat()

            # Read file to calculate checksum and detect format
            with open(full_path, "rb") as f:
                data = f.read()

            metadata = FileMetadata(
                path=path,
                format=self._detect_format(path, data),
                size=stat.st_size,
                checksum=self._calculate_checksum(data),
                lastModified=int(stat.st_mtime),
            )

            logger.info(f"Retrieved metadata for file: {path}")
            return metadata

        except (ValidationException, NotFoundException):
            raise
        except Exception as e:
            logger.error(f"Error getting file metadata {path}: {e}")
            raise InternalException(f"Failed to get file metadata: {str(e)}") from e

    async def file_exists(self, path: str) -> bool:
        """Check if file exists in local file system."""
        try:
            full_path = self._get_full_path(path)
            exists = full_path.exists() and full_path.is_file()

            logger.debug(f"File exists check for {path}: {exists}")
            return exists

        except ValidationException:
            raise
        except Exception as e:
            logger.error(f"Error checking file existence {path}: {e}")
            raise InternalException(f"Failed to check file existence: {str(e)}") from e

"""
FileStoreResourceAccess implementation.

This ResourceAccess service provides atomic business verbs for file storage operations,
implementing the FileStoreAccess Thrift service interface.
Uses local file system storage.
"""

import logging

from thrift_gen.exceptions.ttypes import (
    InternalException,
    NotFoundException,
    UnauthorizedException,
    ValidationException,
)
from thrift_gen.filestoreaccess.FileStoreAccess import Iface as FileStoreAccessIface
from thrift_gen.filestoreaccess.ttypes import FileFormat, FileMetadata, FileQuery

from .file_store_resource_strategy import LocalFileSystemStrategy

logger = logging.getLogger(__name__)


class FileStoreResourceAccess(FileStoreAccessIface):
    """
    FileStoreResourceAccess implements atomic business verbs for file storage operations.

    This ResourceAccess service converts business operations to file I/O operations
    while exposing atomic business verbs rather than CRUD operations.
    Uses local file system storage.
    """

    def __init__(self, base_path: str = "/tmp/file_store"):
        """
        Initialize FileStoreResourceAccess.

        Args:
            base_path: Base directory for file storage
        """
        self.strategy = LocalFileSystemStrategy(base_path=base_path)
        logger.info(
            f"FileStoreResourceAccess initialized with local storage at: {base_path}"
        )

    async def writeFile(
        self, path: str, data: bytes, format: FileFormat | None = None
    ) -> bool:
        """
        Write file using storage strategy.

        Args:
            path: File path
            data: File data as bytes
            format: Optional file format

        Returns:
            True if successful

        Raises:
            ValidationException: If path or data is invalid
            InternalException: If write operation fails
            UnauthorizedException: If access is denied
        """
        try:
            if not path:
                raise ValidationException("File path cannot be empty")

            if data is None:
                raise ValidationException("File data cannot be None")

            logger.info(f"Writing file: {path} ({len(data)} bytes)")
            return await self.strategy.write_file(path, data, format)

        except (ValidationException, UnauthorizedException):
            raise
        except Exception as e:
            logger.error(f"Error writing file {path}: {e}")
            raise InternalException(f"Failed to write file: {str(e)}") from e

    async def readFile(self, path: str) -> bytes:
        """
        Read file using storage strategy.

        Args:
            path: File path

        Returns:
            File data as bytes

        Raises:
            NotFoundException: If file doesn't exist
            InternalException: If read operation fails
            UnauthorizedException: If access is denied
        """
        try:
            if not path:
                raise ValidationException("File path cannot be empty")

            logger.info(f"Reading file: {path}")
            return await self.strategy.read_file(path)

        except (NotFoundException, UnauthorizedException, ValidationException):
            raise
        except Exception as e:
            logger.error(f"Error reading file {path}: {e}")
            raise InternalException(f"Failed to read file: {str(e)}") from e

    async def deleteFile(self, path: str) -> bool:
        """
        Delete file using storage strategy.

        Args:
            path: File path

        Returns:
            True if successful

        Raises:
            NotFoundException: If file doesn't exist
            InternalException: If delete operation fails
            UnauthorizedException: If access is denied
        """
        try:
            if not path:
                raise ValidationException("File path cannot be empty")

            logger.info(f"Deleting file: {path}")
            return await self.strategy.delete_file(path)

        except (NotFoundException, UnauthorizedException, ValidationException):
            raise
        except Exception as e:
            logger.error(f"Error deleting file {path}: {e}")
            raise InternalException(f"Failed to delete file: {str(e)}") from e

    async def listFiles(self, query: FileQuery) -> list[FileMetadata]:
        """
        List files using storage strategy.

        Args:
            query: File query parameters

        Returns:
            List of file metadata

        Raises:
            ValidationException: If query parameters are invalid
            InternalException: If list operation fails
            UnauthorizedException: If access is denied
        """
        try:
            if not query:
                raise ValidationException("File query cannot be None")

            logger.info(f"Listing files with query: {query.pathPattern}")
            return await self.strategy.list_files(query)

        except (ValidationException, UnauthorizedException):
            raise
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            raise InternalException(f"Failed to list files: {str(e)}") from e

    async def getFileMetadata(self, path: str) -> FileMetadata:
        """
        Get file metadata using storage strategy.

        Args:
            path: File path

        Returns:
            File metadata

        Raises:
            NotFoundException: If file doesn't exist
            InternalException: If metadata operation fails
            UnauthorizedException: If access is denied
        """
        try:
            if not path:
                raise ValidationException("File path cannot be empty")

            logger.info(f"Getting metadata for file: {path}")
            return await self.strategy.get_file_metadata(path)

        except (NotFoundException, UnauthorizedException, ValidationException):
            raise
        except Exception as e:
            logger.error(f"Error getting file metadata {path}: {e}")
            raise InternalException(f"Failed to get file metadata: {str(e)}") from e

    async def fileExists(self, path: str) -> bool:
        """
        Check if file exists using storage strategy.

        Args:
            path: File path

        Returns:
            True if file exists

        Raises:
            InternalException: If existence check fails
            UnauthorizedException: If access is denied
        """
        try:
            if not path:
                raise ValidationException("File path cannot be empty")

            logger.debug(f"Checking if file exists: {path}")
            return await self.strategy.file_exists(path)

        except (UnauthorizedException, ValidationException):
            raise
        except Exception as e:
            logger.error(f"Error checking file existence {path}: {e}")
            raise InternalException(f"Failed to check file existence: {str(e)}") from e

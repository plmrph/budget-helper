"""
Database connection and resource access implementation.
"""

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any

import asyncpg
from asyncpg import Pool

logger = logging.getLogger(__name__)


class DatabaseResourceAccess(ABC):
    """Abstract base class for database resource access."""

    @abstractmethod
    async def execute_query(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a SELECT query and return results."""
        pass

    @abstractmethod
    async def execute_command(
        self, command: str, params: dict[str, Any] | None = None
    ) -> int:
        """Execute INSERT/UPDATE/DELETE command and return affected rows."""
        pass

    @abstractmethod
    async def execute_transaction(self, commands: list[tuple]) -> bool:
        """Execute multiple commands in a transaction."""
        pass

    @abstractmethod
    async def get_connection_info(self) -> dict[str, Any]:
        """Get database connection information."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if database connection is healthy."""
        pass


class PostgreSQLResourceAccess(DatabaseResourceAccess):
    """PostgreSQL implementation of database resource access."""

    def __init__(
        self, database_url: str, min_connections: int = 10, max_connections: int = 50
    ):
        """
        Initialize PostgreSQL resource access.

        Args:
            database_url: PostgreSQL connection URL
            min_connections: Minimum number of connections in pool
            max_connections: Maximum number of connections in pool
        """
        self.database_url = database_url
        self.min_connections = min_connections
        self.max_connections = max_connections
        self._pool: Pool | None = None
        self._connection_info: dict[str, Any] = {}

    async def initialize(self) -> None:
        """Initialize the connection pool."""
        try:
            self._pool = await asyncpg.create_pool(
                self.database_url,
                min_size=self.min_connections,
                max_size=self.max_connections,
                command_timeout=120,  # Increased timeout for ML operations
                max_queries=50000,  # Allow more queries per connection
                max_inactive_connection_lifetime=300,  # 5 minutes
                server_settings={
                    "jit": "off",  # Disable JIT for better performance with small queries
                    "application_name": "budget_helper",
                },
            )

            # Get connection info
            async with self._pool.acquire() as conn:
                result = await conn.fetchrow("""
                    SELECT
                        version() as version,
                        current_database() as database,
                        current_user as user,
                        inet_server_addr() as host,
                        inet_server_port() as port
                """)
                self._connection_info = dict(result) if result else {}

            logger.info(
                f"Database connection pool initialized with {self.min_connections}-{self.max_connections} connections"
            )

        except Exception as e:
            logger.error(f"Failed to initialize database connection pool: {e}")
            raise

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Database connection pool closed")

    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool with retry logic."""
        if not self._pool:
            raise RuntimeError(
                "Database pool not initialized. Call initialize() first."
            )

        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                async with self._pool.acquire(timeout=30.0) as connection:
                    yield connection
                    return
            except (TimeoutError, ConnectionError) as e:
                if attempt == max_retries - 1:
                    logger.error(
                        f"Failed to acquire database connection after {max_retries} attempts: {e}"
                    )
                    raise RuntimeError(
                        f"Database connection timeout after {max_retries} attempts"
                    ) from e

                logger.warning(
                    f"Database connection attempt {attempt + 1} failed, retrying in {retry_delay}s: {e}"
                )
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff

    async def execute_query(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a SELECT query and return results."""
        try:
            async with self.get_connection() as conn:
                if params:
                    # Convert named parameters to positional for asyncpg
                    param_values = list(params.values())
                    # Replace named placeholders with $1, $2, etc.
                    # Sort keys by length (descending) to avoid partial replacements
                    formatted_query = query
                    sorted_keys = sorted(params.keys(), key=len, reverse=True)
                    for key in sorted_keys:
                        # Find the original position of this key in the params dict
                        original_position = list(params.keys()).index(key) + 1
                        formatted_query = formatted_query.replace(
                            f":{key}", f"${original_position}"
                        )

                    rows = await conn.fetch(formatted_query, *param_values)
                else:
                    rows = await conn.fetch(query)

                return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            raise

    async def execute_command(
        self, command: str, params: dict[str, Any] | None = None
    ) -> int:
        """Execute INSERT/UPDATE/DELETE command and return affected rows."""
        try:
            async with self.get_connection() as conn:
                if params:
                    # Convert named parameters to positional for asyncpg
                    param_values = list(params.values())
                    # Replace named placeholders with $1, $2, etc.
                    # Sort keys by length (descending) to avoid partial replacements
                    formatted_command = command
                    sorted_keys = sorted(params.keys(), key=len, reverse=True)
                    for key in sorted_keys:
                        # Find the original position of this key in the params dict
                        original_position = list(params.keys()).index(key) + 1
                        formatted_command = formatted_command.replace(
                            f":{key}", f"${original_position}"
                        )

                    result = await conn.execute(formatted_command, *param_values)
                else:
                    result = await conn.execute(command)

                # Extract number of affected rows from result
                if result.startswith("INSERT"):
                    return int(result.split()[-1])
                elif result.startswith("UPDATE"):
                    return int(result.split()[-1])
                elif result.startswith("DELETE"):
                    return int(result.split()[-1])
                else:
                    return 0

        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            logger.error(f"Command: {command}")
            logger.error(f"Params: {params}")
            raise

    async def execute_transaction(self, commands: list[tuple]) -> bool:
        """
        Execute multiple commands in a transaction.

        Args:
            commands: List of tuples (command, params) where params can be None

        Returns:
            True if transaction succeeded, False otherwise
        """
        try:
            async with self.get_connection() as conn:
                async with conn.transaction():
                    for command, params in commands:
                        if params:
                            # Convert named parameters to positional for asyncpg
                            param_values = list(params.values())
                            # Replace named placeholders with $1, $2, etc.
                            # Sort keys by length (descending) to avoid partial replacements
                            formatted_command = command
                            sorted_keys = sorted(params.keys(), key=len, reverse=True)
                            for key in sorted_keys:
                                # Find the original position of this key in the params dict
                                original_position = list(params.keys()).index(key) + 1
                                formatted_command = formatted_command.replace(
                                    f":{key}", f"${original_position}"
                                )

                            await conn.execute(formatted_command, *param_values)
                        else:
                            await conn.execute(command)

                return True

        except Exception as e:
            logger.error(f"Transaction execution failed: {e}")
            logger.error(f"Commands: {commands}")
            raise

    async def get_connection_info(self) -> dict[str, Any]:
        """Get database connection information."""
        return self._connection_info.copy()

    async def health_check(self) -> bool:
        """Check if database connection is healthy."""
        try:
            async with self.get_connection() as conn:
                result = await conn.fetchval("SELECT 1")
                return result == 1
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False


# Global database instance
_db_instance: PostgreSQLResourceAccess | None = None


async def get_database() -> PostgreSQLResourceAccess:
    """Get the global database instance."""
    global _db_instance

    if _db_instance is None:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise RuntimeError("DATABASE_URL environment variable not set")

        _db_instance = PostgreSQLResourceAccess(database_url)
        await _db_instance.initialize()

    return _db_instance


async def close_database() -> None:
    """Close the global database instance."""
    global _db_instance

    if _db_instance:
        await _db_instance.close()
        _db_instance = None


# Database lifecycle management for FastAPI
async def startup_database():
    """Initialize database on application startup."""
    try:
        db = await get_database()
        health_ok = await db.health_check()
        if not health_ok:
            raise RuntimeError("Database health check failed during startup")
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


async def shutdown_database():
    """Close database on application shutdown."""
    try:
        await close_database()
        logger.info("Database closed successfully")
    except Exception as e:
        logger.error(f"Database shutdown failed: {e}")

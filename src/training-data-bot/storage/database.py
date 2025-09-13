from ..core.logging import get_logger

class DatabaseManager:
    """Manage database connections and operations."""

    def __init__(self):
        self.logger=get_logger("database")

    async def close(self):
        """Close database connections."""
        self.logger.debug("Database connections closed")

from .services.memory_service import MemoryService

def get_memory_service():
    """Dependency that provides a MemoryService instance."""
    # This will be overridden in tests
    return MemoryService()
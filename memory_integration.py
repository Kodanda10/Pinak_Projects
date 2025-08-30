"""
Pinak Memory Integration Example
================================

This file shows how to use Pinak memory capabilities in your project.
"""

import json
from typing import Any, Dict, List, Optional

from pinak import MemoryClient, MemoryLayer


class ProjectMemoryManager:
    """Memory manager for your project."""

    def __init__(self, config_file: str = "pinak_config.json"):
        with open(config_file, "r") as f:
            self.config = json.load(f)

        self.client = MemoryClient(
            base_url=self.config["memory_service_url"], api_key=self.config["api_key"]
        )

    def remember_user_action(self, user_id: str, action: str, context: Dict[str, Any]):
        """Remember a user action for future reference."""
        content = f"User {user_id} performed: {action}"
        metadata = {"user_id": user_id, "action": action, **context}

        return self.client.add_memory(
            content=content, layer=MemoryLayer.EPISODIC, metadata=metadata
        )

    def get_user_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get user's action history."""
        return self.client.search_memories(
            query=f"user {user_id}", layer=MemoryLayer.EPISODIC, limit=limit
        )

    def store_session_context(
        self, session_id: str, context: str, ttl_minutes: int = 60
    ):
        """Store temporary session context."""
        return self.client.add_memory(
            content=context,
            layer=MemoryLayer.SESSION,
            session_id=session_id,
            ttl_seconds=ttl_minutes * 60,
        )

    def get_session_context(self, session_id: str) -> List[Dict]:
        """Retrieve session context."""
        return self.client.list_memories(
            layer=MemoryLayer.SESSION, session_id=session_id
        )

    def add_knowledge(self, topic: str, content: str, source: str = "user"):
        """Add knowledge to RAG system."""
        return self.client.add_memory(
            content=f"{topic}: {content}", layer=MemoryLayer.RAG, external_source=source
        )

    def search_knowledge(self, query: str, limit: int = 5) -> List[Dict]:
        """Search knowledge base."""
        return self.client.search_memories(
            query=query, layer=MemoryLayer.RAG, limit=limit
        )


# Example usage
if __name__ == "__main__":
    # Initialize memory manager
    memory = ProjectMemoryManager()

    # Example: Remember user actions
    memory.remember_user_action(
        user_id="user123",
        action="created_project",
        context={"project_name": "my_app", "timestamp": "2025-08-30"},
    )

    # Example: Store session context
    memory.store_session_context(
        session_id="session_abc", context="User is working on authentication feature"
    )

    # Example: Add knowledge
    memory.add_knowledge(
        topic="Authentication",
        content="JWT tokens should be validated on each request",
        source="security_best_practices",
    )

    print("✅ Memory operations completed successfully!")

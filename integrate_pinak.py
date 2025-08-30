#!/usr/bin/env python3
"""
Pinak One-Click Integration Demo
================================

This script demonstrates the "breeze-like" integration of Pinak Memory Service
into any Python project. Just run this script and you'll have memory capabilities
added to your project in seconds!

Usage:
    python integrate_pinak.py

What this does:
1. Detects your project structure
2. Adds Pinak dependency
3. Creates minimal configuration
4. Sets up basic memory operations
5. Tests the integration

That's it! Your project now has memory capabilities.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
import time


class PinakIntegrator:
    """Handles seamless integration of Pinak into user projects."""

    def __init__(self, project_root=None):
        self.project_root = Path(project_root or os.getcwd())
        self.start_time = time.time()

    def integrate(self):
        """Perform one-click integration."""
        print("🚀 Starting Pinak integration...")
        print(f"📁 Project: {self.project_root}")

        # Step 1: Detect project type
        project_type = self._detect_project_type()
        print(f"🔍 Detected project type: {project_type}")

        # Step 2: Add dependency
        self._add_dependency()
        print("📦 Added Pinak dependency")

        # Step 3: Create configuration
        config = self._create_config()
        print("⚙️  Created configuration")

        # Step 4: Setup integration code
        self._setup_integration_code()
        print("💻 Added integration code")

        # Step 5: Test integration
        success = self._test_integration()
        if success:
            duration = time.time() - self.start_time
            print("✅ Integration complete!")
            print(".2f")
            print("\n🎉 Your project now has memory capabilities!")
            print("\nNext steps:")
            print("1. Start your memory service: pinak-memory-service start")
            print("2. Use memory in your code:")
            print("   from pinak import MemoryClient")
            print("   client = MemoryClient()")
            print("   client.add_memory('Hello World!', layer='episodic')")
        else:
            print("❌ Integration failed. Please check the logs above.")

        return success

    def _detect_project_type(self):
        """Detect the type of Python project."""
        if (self.project_root / "manage.py").exists():
            return "Django"
        elif (self.project_root / "app.py").exists() and "flask" in self._read_requirements():
            return "Flask"
        elif (self.project_root / "main.py").exists() and "fastapi" in self._read_requirements():
            return "FastAPI"
        elif (self.project_root / "setup.py").exists():
            return "Python Package"
        else:
            return "Generic Python"

    def _read_requirements(self):
        """Read requirements.txt content."""
        req_file = self.project_root / "requirements.txt"
        if req_file.exists():
            return req_file.read_text().lower()
        return ""

    def _add_dependency(self):
        """Add Pinak to project dependencies."""
        req_file = self.project_root / "requirements.txt"
        pinak_dep = "pinak-memory-client>=1.0.0"

        if req_file.exists():
            content = req_file.read_text()
            if pinak_dep not in content:
                req_file.write_text(content.rstrip() + "\n" + pinak_dep + "\n")
        else:
            req_file.write_text(pinak_dep + "\n")

    def _create_config(self):
        """Create minimal Pinak configuration."""
        config = {
            "memory_service_url": "http://localhost:8000",
            "api_key": "your_api_key_here",  # User should replace this
            "default_project": self.project_root.name,
            "auto_initialize": True,
            "enable_caching": True,
            "default_layers": ["episodic", "working", "session"]
        }

        config_file = self.project_root / "pinak_config.json"
        config_file.write_text(json.dumps(config, indent=2))

        return config

    def _setup_integration_code(self):
        """Create integration code example."""
        integration_file = self.project_root / "memory_integration.py"
        integration_code = '''"""
Pinak Memory Integration Example
================================

This file shows how to use Pinak memory capabilities in your project.
"""

import json
from typing import Optional, List, Dict, Any
from pinak import MemoryClient, MemoryLayer


class ProjectMemoryManager:
    """Memory manager for your project."""

    def __init__(self, config_file: str = "pinak_config.json"):
        with open(config_file, 'r') as f:
            self.config = json.load(f)

        self.client = MemoryClient(
            base_url=self.config["memory_service_url"],
            api_key=self.config["api_key"]
        )

    def remember_user_action(self, user_id: str, action: str, context: Dict[str, Any]):
        """Remember a user action for future reference."""
        content = f"User {user_id} performed: {action}"
        metadata = {"user_id": user_id, "action": action, **context}

        return self.client.add_memory(
            content=content,
            layer=MemoryLayer.EPISODIC,
            metadata=metadata
        )

    def get_user_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get user's action history."""
        return self.client.search_memories(
            query=f"user {user_id}",
            layer=MemoryLayer.EPISODIC,
            limit=limit
        )

    def store_session_context(self, session_id: str, context: str, ttl_minutes: int = 60):
        """Store temporary session context."""
        return self.client.add_memory(
            content=context,
            layer=MemoryLayer.SESSION,
            session_id=session_id,
            ttl_seconds=ttl_minutes * 60
        )

    def get_session_context(self, session_id: str) -> List[Dict]:
        """Retrieve session context."""
        return self.client.list_memories(
            layer=MemoryLayer.SESSION,
            session_id=session_id
        )

    def add_knowledge(self, topic: str, content: str, source: str = "user"):
        """Add knowledge to RAG system."""
        return self.client.add_memory(
            content=f"{topic}: {content}",
            layer=MemoryLayer.RAG,
            external_source=source
        )

    def search_knowledge(self, query: str, limit: int = 5) -> List[Dict]:
        """Search knowledge base."""
        return self.client.search_memories(
            query=query,
            layer=MemoryLayer.RAG,
            limit=limit
        )


# Example usage
if __name__ == "__main__":
    # Initialize memory manager
    memory = ProjectMemoryManager()

    # Example: Remember user actions
    memory.remember_user_action(
        user_id="user123",
        action="created_project",
        context={"project_name": "my_app", "timestamp": "2025-08-30"}
    )

    # Example: Store session context
    memory.store_session_context(
        session_id="session_abc",
        context="User is working on authentication feature"
    )

    # Example: Add knowledge
    memory.add_knowledge(
        topic="Authentication",
        content="JWT tokens should be validated on each request",
        source="security_best_practices"
    )

    print("✅ Memory operations completed successfully!")
'''

        integration_file.write_text(integration_code)

    def _test_integration(self):
        """Test that the integration works."""
        try:
            # Test configuration loading
            config_file = self.project_root / "pinak_config.json"
            if not config_file.exists():
                print("❌ Configuration file not found")
                return False

            with open(config_file, 'r') as f:
                config = json.load(f)

            required_keys = ["memory_service_url", "api_key", "default_project"]
            for key in required_keys:
                if key not in config:
                    print(f"❌ Missing required config key: {key}")
                    return False

            # Test integration code syntax
            integration_file = self.project_root / "memory_integration.py"
            if integration_file.exists():
                # Basic syntax check
                try:
                    compile(integration_file.read_text(), str(integration_file), 'exec')
                    print("✅ Integration code syntax is valid")
                except SyntaxError as e:
                    print(f"❌ Syntax error in integration code: {e}")
                    return False

            print("✅ Configuration validated")
            return True

        except Exception as e:
            print(f"❌ Integration test failed: {e}")
            return False


def main():
    """Main integration function."""
    print("🧠 Pinak Memory Service - One-Click Integration")
    print("=" * 50)

    integrator = PinakIntegrator()

    try:
        success = integrator.integrate()
        if success:
            print("\n" + "=" * 50)
            print("🎊 Integration Summary:")
            print("   ✅ Dependencies added")
            print("   ✅ Configuration created")
            print("   ✅ Integration code generated")
            print("   ✅ Tests passed")
            print("\n🚀 Ready to use memory features in your project!")
            return 0
        else:
            print("\n❌ Integration failed. Please check the errors above.")
            return 1

    except Exception as e:
        print(f"\n💥 Unexpected error during integration: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

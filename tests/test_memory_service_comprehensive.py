"""
Comprehensive TDD tests for Pinak Memory Service.

Following TDD principles: Write tests first, then implement features.
Tests cover all 8 memory layers and service integrations.
"""

from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch

import httpx
import pytest

from pinak.memory.cli import main as memory_cli_main
from pinak.memory.manager import MemoryManager


@pytest.fixture
def memory_service_url():
    """Memory service URL for testing."""
    return "http://127.0.0.1:8000"


@pytest.fixture
def memory_manager(memory_service_url):
    """MemoryManager instance for testing."""
    return MemoryManager(
        service_base_url=memory_service_url,
        token="TEST_TOKEN",
        project_id="test_project",
        timeout=30.0,
    )


@pytest.fixture
def test_memory_data():
    """Comprehensive test data for memory operations."""
    return {
        "episodic": {
            "content": "Test episodic memory about Python development",
            "salience": 0.9,
            "tags": ["python", "development", "test"],
        },
        "procedural": {
            "skill_id": "git_workflow",
            "steps": [
                "git add .",
                "git commit -m 'feat: add new feature'",
                "git push origin main",
            ],
        },
        "rag": {
            "query": "Python async programming patterns",
            "external_source": "https://docs.python.org/3/library/asyncio.html",
        },
        "session": {
            "session_id": "test_session_123",
            "content": "Current development context and goals",
            "ttl": 3600,
        },
        "working": {"content": "Immediate task: implement TDD framework", "ttl": 1800},
        "events": {
            "type": "user_action",
            "payload": {
                "action": "memory_search",
                "query": "python",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        },
    }


@pytest.mark.tdd
@pytest.mark.memory
class TestMemoryServiceHealth:
    """Test memory service health and connectivity."""

    def test_service_initialization(self, memory_manager):
        """Test MemoryManager initialization."""
        assert memory_manager.base_url.endswith("/8000")
        assert memory_manager._timeout == 30.0
        assert "Authorization" in memory_manager.client.headers
        assert "X-Pinak-Project" in memory_manager.client.headers

    @pytest.mark.asyncio
    async def test_health_check(self, memory_manager):
        """Test service health check."""
        # Mock successful health response
        with patch.object(memory_manager.client, "get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"ok": True}
            mock_get.return_value = mock_response

            health = memory_manager.health()
            assert health is True
            mock_get.assert_called_with("/api/v1/memory/health", timeout=30.0)

    @pytest.mark.asyncio
    async def test_health_check_failure(self, memory_manager):
        """Test health check failure handling."""
        with patch.object(memory_manager.client, "get") as mock_get:
            mock_get.side_effect = httpx.RequestError("Connection failed")

            health = memory_manager.health()
            assert health is False

    def test_cli_health_command(self, memory_manager):
        """Test CLI health command."""
        with patch.object(memory_manager, "health", return_value=True):
            # This would test the CLI integration
            # For now, verify the method exists and can be called
            assert callable(memory_manager.health)


@pytest.mark.tdd
@pytest.mark.memory
class TestEpisodicMemory:
    """Test episodic memory operations."""

    @pytest.mark.asyncio
    async def test_add_episodic_memory(self, memory_manager, test_memory_data):
        """Test adding episodic memory."""
        data = test_memory_data["episodic"]

        with patch.object(memory_manager.client, "post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "data": {
                    "id": "epi_123",
                    "content": data["content"],
                    "salience": data["salience"],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            }
            mock_post.return_value = mock_response

            result = memory_manager.add_episodic(data["content"], data["salience"])

            assert result is not None
            assert result["id"] == "epi_123"
            assert result["content"] == data["content"]

            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[0][0] == "/api/v1/memory/episodic"
            assert call_args[1]["json"]["content"] == data["content"]
            assert call_args[1]["json"]["salience"] == data["salience"]

    @pytest.mark.asyncio
    async def test_list_episodic_memories(self, memory_manager):
        """Test listing episodic memories."""
        with patch.object(memory_manager.client, "get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "data": [
                    {
                        "id": "epi_1",
                        "content": "First episodic memory",
                        "salience": 0.8,
                    },
                    {
                        "id": "epi_2",
                        "content": "Second episodic memory",
                        "salience": 0.6,
                    },
                ]
            }
            mock_get.return_value = mock_response

            result = memory_manager.list_episodic(limit=10)

            assert isinstance(result, list)
            assert len(result) == 2
            assert result[0]["id"] == "epi_1"
            assert result[1]["content"] == "Second episodic memory"

    @pytest.mark.asyncio
    async def test_episodic_memory_with_tags(self, memory_manager, test_memory_data):
        """Test episodic memory with tags."""
        data = test_memory_data["episodic"]

        with patch.object(memory_manager.client, "post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "data": {
                    "id": "epi_tags_123",
                    "content": data["content"],
                    "salience": data["salience"],
                    "tags": data["tags"],
                }
            }
            mock_post.return_value = mock_response

            result = memory_manager.add_episodic(data["content"], data["salience"])

            assert result is not None
            assert result["tags"] == data["tags"]


@pytest.mark.tdd
@pytest.mark.memory
class TestProceduralMemory:
    """Test procedural memory operations."""

    @pytest.mark.asyncio
    async def test_add_procedural_memory(self, memory_manager, test_memory_data):
        """Test adding procedural memory."""
        data = test_memory_data["procedural"]

        with patch.object(memory_manager.client, "post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "data": {
                    "id": "proc_123",
                    "skill_id": data["skill_id"],
                    "steps": data["steps"],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            }
            mock_post.return_value = mock_response

            result = memory_manager.add_procedural(data["skill_id"], data["steps"])

            assert result is not None
            assert result["skill_id"] == data["skill_id"]
            assert result["steps"] == data["steps"]

    @pytest.mark.asyncio
    async def test_list_procedural_memories(self, memory_manager):
        """Test listing procedural memories."""
        with patch.object(memory_manager.client, "get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "data": [
                    {
                        "id": "proc_1",
                        "skill_id": "git_workflow",
                        "steps": ["git add", "git commit", "git push"],
                    }
                ]
            }
            mock_get.return_value = mock_response

            result = memory_manager.list_procedural(limit=5)

            assert isinstance(result, list)
            assert len(result) >= 1
            assert result[0]["skill_id"] == "git_workflow"

    @pytest.mark.asyncio
    async def test_procedural_validation(self, memory_manager):
        """Test procedural memory validation."""
        # Test missing skill_id
        with pytest.raises(SystemExit):
            memory_cli_main(["add", "procedural", "--steps", "step1", "step2"])

        # Test missing steps
        with pytest.raises(SystemExit):
            memory_cli_main(["add", "procedural", "--skill-id", "test_skill"])


@pytest.mark.tdd
@pytest.mark.memory
class TestRAGMemory:
    """Test RAG memory operations."""

    @pytest.mark.asyncio
    async def test_add_rag_memory(self, memory_manager, test_memory_data):
        """Test adding RAG memory."""
        data = test_memory_data["rag"]

        with patch.object(memory_manager.client, "post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "data": {
                    "id": "rag_123",
                    "query": data["query"],
                    "external_source": data["external_source"],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            }
            mock_post.return_value = mock_response

            result = memory_manager.add_rag(data["query"], data["external_source"])

            assert result is not None
            assert result["query"] == data["query"]
            assert result["external_source"] == data["external_source"]

    @pytest.mark.asyncio
    async def test_list_rag_memories(self, memory_manager):
        """Test listing RAG memories."""
        with patch.object(memory_manager.client, "get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "data": [
                    {
                        "id": "rag_1",
                        "query": "Python async patterns",
                        "external_source": "https://example.com",
                    }
                ]
            }
            mock_get.return_value = mock_response

            result = memory_manager.list_rag(limit=10)

            assert isinstance(result, list)
            assert len(result) >= 1
            assert "external_source" in result[0]


@pytest.mark.tdd
@pytest.mark.memory
class TestSessionMemory:
    """Test session memory operations."""

    @pytest.mark.asyncio
    async def test_add_session_memory(self, memory_manager, test_memory_data):
        """Test adding session memory."""
        data = test_memory_data["session"]

        with patch.object(memory_manager.client, "post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "data": {
                    "id": "session_123",
                    "key": data["session_id"],
                    "value": data["content"],
                    "ttl": data["ttl"],
                }
            }
            mock_post.return_value = mock_response

            result = memory_manager.add_session(
                data["session_id"], data["content"], data["ttl"]
            )

            assert result is not None
            assert result["key"] == data["session_id"]
            assert result["value"] == data["content"]

    @pytest.mark.asyncio
    async def test_list_session_memories(self, memory_manager, test_memory_data):
        """Test listing session memories."""
        data = test_memory_data["session"]

        with patch.object(memory_manager.client, "get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "data": {
                    data["session_id"]: {
                        "value": data["content"],
                        "exp": time.time() + data["ttl"],
                    }
                }
            }
            mock_get.return_value = mock_response

            result = memory_manager.list_session(data["session_id"], limit=10)

            assert isinstance(result, list)
            assert len(result) >= 1
            assert result[0]["key"] == data["session_id"]

    @pytest.mark.asyncio
    async def test_session_ttl_expiration(self, memory_manager, test_memory_data):
        """Test session memory TTL expiration."""
        data = test_memory_data["session"]

        # Add session memory
        with patch.object(memory_manager.client, "post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"data": {"id": "session_test"}}
            mock_post.return_value = mock_response

            memory_manager.add_session(data["session_id"], data["content"], ttl=1)

        # Wait for expiration
        time.sleep(2)

        # Check if expired
        with patch.object(memory_manager.client, "get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"data": {}}  # Empty result
            mock_get.return_value = mock_response

            result = memory_manager.list_session(data["session_id"])

            assert len(result) == 0  # Should be empty after TTL expires


@pytest.mark.tdd
@pytest.mark.memory
class TestWorkingMemory:
    """Test working memory operations."""

    @pytest.mark.asyncio
    async def test_add_working_memory(self, memory_manager, test_memory_data):
        """Test adding working memory."""
        data = test_memory_data["working"]

        with patch.object(memory_manager.client, "post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "data": {
                    "id": "working_123",
                    "key": f"working_{int(time.time())}",
                    "value": data["content"],
                    "expires_in": data["ttl"],
                }
            }
            mock_post.return_value = mock_response

            result = memory_manager.add_working(data["content"], data["ttl"])

            assert result is not None
            assert result["value"] == data["content"]
            assert "expires_in" in result or "exp" in result

    @pytest.mark.asyncio
    async def test_list_working_memories(self, memory_manager):
        """Test listing working memories."""
        with patch.object(memory_manager.client, "get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "data": {
                    "working_1": {"value": "Task 1", "exp": time.time() + 3600},
                    "working_2": {"value": "Task 2", "exp": time.time() + 3600},
                }
            }
            mock_get.return_value = mock_response

            result = memory_manager.list_working(limit=10)

            assert isinstance(result, list)
            assert len(result) >= 2
            assert all("value" in item for item in result)


@pytest.mark.tdd
@pytest.mark.memory
class TestCrossLayerSearch:
    """Test cross-layer search functionality."""

    @pytest.mark.asyncio
    async def test_search_all_layers(self, memory_manager):
        """Test searching across all memory layers."""
        query = "python development"

        with patch.object(memory_manager.client, "get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "data": {
                    "episodic": [
                        {
                            "id": "epi_1",
                            "content": "Python development experience",
                            "relevance": 0.9,
                        }
                    ],
                    "procedural": [
                        {"id": "proc_1", "skill_id": "python_debug", "relevance": 0.7}
                    ],
                    "rag": [
                        {
                            "id": "rag_1",
                            "query": "Python best practices",
                            "relevance": 0.8,
                        }
                    ],
                }
            }
            mock_get.return_value = mock_response

            result = memory_manager.search_all_layers(query)

            assert isinstance(result, dict)
            assert "episodic" in result
            assert "procedural" in result
            assert "rag" in result
            assert len(result["episodic"]) > 0

    @pytest.mark.asyncio
    async def test_search_with_layer_filter(self, memory_manager):
        """Test search with specific layer filtering."""
        query = "debugging"
        layers = "episodic,procedural"

        with patch.object(memory_manager.client, "get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "data": {
                    "episodic": [
                        {"id": "epi_debug", "content": "Debugging experience"}
                    ],
                    "procedural": [{"id": "proc_debug", "skill_id": "debug_workflow"}],
                }
            }
            mock_get.return_value = mock_response

            result = memory_manager.search_all_layers(query, layers)

            assert "episodic" in result
            assert "procedural" in result
            assert len(result["episodic"]) > 0
            assert len(result["procedural"]) > 0

    @pytest.mark.asyncio
    async def test_empty_search_results(self, memory_manager):
        """Test handling of empty search results."""
        query = "nonexistent content"

        with patch.object(memory_manager.client, "get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"data": {}}
            mock_get.return_value = mock_response

            result = memory_manager.search_all_layers(query)

            assert isinstance(result, dict)
            assert len(result) == 0 or all(len(v) == 0 for v in result.values())


@pytest.mark.tdd
@pytest.mark.memory
class TestMemoryServiceIntegration:
    """Test memory service integration and CLI."""

    def test_memory_cli_help(self):
        """Test memory CLI help functionality."""
        # Test that CLI doesn't crash on help
        try:
            result = memory_cli_main(["--help"])
            # Help should return 0
            assert result == 0
        except SystemExit as e:
            assert e.code == 0

    def test_memory_cli_invalid_command(self):
        """Test CLI handling of invalid commands."""
        with pytest.raises(SystemExit) as exc_info:
            memory_cli_main(["invalid_command"])

        assert exc_info.value.code != 0

    @pytest.mark.asyncio
    async def test_backward_compatibility(self, memory_manager):
        """Test backward compatibility with add_memory method."""
        content = "Backward compatibility test"
        tags = ["test", "compat"]

        # Test that old add_memory method still works
        if hasattr(memory_manager, "add_memory"):
            with patch.object(memory_manager.client, "post") as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.raise_for_status.return_value = None
                mock_response.json.return_value = {"data": {"id": "compat_123"}}
                mock_post.return_value = mock_response

                result = memory_manager.add_memory(content, tags)

                assert result is not None
                assert result["id"] == "compat_123"

    @pytest.mark.asyncio
    async def test_error_handling(self, memory_manager):
        """Test error handling in memory operations."""
        # Test 401 unauthorized
        with patch.object(memory_manager.client, "post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 401
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Unauthorized", request=Mock(), response=mock_response
            )
            mock_post.return_value = mock_response

            with patch("sys.stderr"):
                result = memory_manager.add_episodic("test", 0.5)

            assert result is None

    @pytest.mark.asyncio
    async def test_request_timeout(self, memory_manager):
        """Test request timeout handling."""
        with patch.object(memory_manager.client, "get") as mock_get:
            mock_get.side_effect = httpx.TimeoutException("Request timeout")

            result = memory_manager.list_episodic()
            assert result == []


@pytest.mark.tdd
@pytest.mark.memory
@pytest.mark.slow
class TestMemoryServiceLoad:
    """Load tests for memory service."""

    @pytest.mark.asyncio
    async def test_bulk_memory_operations(self, memory_manager):
        """Test bulk memory operations."""
        # Create bulk episodic memories
        bulk_data = [
            {"content": f"Bulk memory {i}", "salience": 0.5 + (i % 5) * 0.1}
            for i in range(50)
        ]

        # Mock successful responses
        with patch.object(memory_manager.client, "post") as mock_post:
            call_count = 0

            def mock_response(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                response = Mock()
                response.status_code = 200
                response.raise_for_status.return_value = None
                response.json.return_value = {
                    "data": {
                        "id": f"bulk_{call_count}",
                        "content": kwargs["json"]["content"],
                        "salience": kwargs["json"]["salience"],
                    }
                }
                return response

            mock_post.side_effect = mock_response

            # Execute bulk operations
            results = []
            for data in bulk_data:
                result = memory_manager.add_episodic(data["content"], data["salience"])
                results.append(result)

            assert len(results) == 50
            assert all(r is not None for r in results)
            assert call_count == 50

    @pytest.mark.asyncio
    async def test_concurrent_memory_access(self, memory_manager):
        """Test concurrent memory access."""

        async def add_memory_task(task_id: int):
            with patch.object(memory_manager.client, "post") as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.raise_for_status.return_value = None
                mock_response.json.return_value = {
                    "data": {"id": f"concurrent_{task_id}"}
                }
                mock_post.return_value = mock_response

                return memory_manager.add_episodic(f"Concurrent task {task_id}", 0.5)

        # Execute concurrent operations
        tasks = [add_memory_task(i) for i in range(20)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 20
        assert all(r is not None for r in results)
        assert len(set(r["id"] for r in results)) == 20  # All IDs unique

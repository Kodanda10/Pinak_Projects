"""
Comprehensive TDD tests for World-Beating Hybrid Retrieval Engine.

Following TDD principles: Write tests first, then implement features.
Tests cover all 6 stages of the world-beating retrieval pipeline.
"""

import asyncio
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

import pytest

from pinak.context.broker.broker import (ContextBroker, HybridScore,
                                         RetrievalResult)
from pinak.context.broker.graph_expansion import GraphBasedExpander
from pinak.context.broker.neural_reranker import NeuralReranker
from pinak.context.broker.rl_optimizer import (AdaptiveLearningEngine,
                                               QLearningOptimizer)
from pinak.context.broker.world_beating_retrieval import WorldBeatingRetrieval
from pinak.context.core.models import (ContextItem, ContextLayer,
                                       ContextPriority, ContextQuery,
                                       ContextResponse, IContextStore,
                                       SecurityClassification)


class MockContextStore(IContextStore):
    """Mock context store for testing."""

    def __init__(self, layer: ContextLayer, data: List[ContextItem] = None):
        self.layer = layer
        self.data = data or []
        self.search_calls = []

    async def retrieve(self, query: ContextQuery) -> ContextResponse:
        """Mock retrieve implementation."""
        response = ContextResponse()
        response.query_id = query.query_id
        response.items = self.data[: query.limit]
        response.total_results = len(self.data)
        return response

    async def search_similar(self, query: str, limit: int) -> List[ContextItem]:
        """Mock semantic search."""
        self.search_calls.append((query, limit))
        return self.data[:limit]

    async def store(self, item: ContextItem) -> bool:
        """Mock store implementation."""
        self.data.append(item)
        return True

    async def delete(self, item_id: str) -> bool:
        """Mock delete implementation."""
        return True

    async def update(self, item: ContextItem) -> bool:
        """Mock update implementation."""
        return True


@pytest.fixture
def mock_stores():
    """Create mock stores for all context layers."""
    stores = {}
    for layer in ContextLayer:
        stores[layer] = MockContextStore(layer)
    return stores


@pytest.fixture
def context_broker(mock_stores):
    """Create ContextBroker instance for testing."""
    return ContextBroker(
        stores=mock_stores,
        cache_ttl_seconds=300,
        max_parallel_requests=5,
        semantic_weight=0.6,
        keyword_weight=0.3,
        temporal_weight=0.1,
        enable_world_beating=True,
    )


@pytest.fixture
def sample_context_items():
    """Create sample context items for testing."""
    now = time.time()
    return [
        ContextItem(
            id="1",
            content="Python async programming guide",
            title="Async Programming",
            layer=ContextLayer.SEMANTIC,
            relevance_score=0.9,
            confidence_score=0.8,
            tags=["python", "async"],
            created_at=now,
            metadata={"source": "docs"},
        ),
        ContextItem(
            id="2",
            content="Machine learning best practices",
            title="ML Practices",
            layer=ContextLayer.EPISODIC,
            relevance_score=0.7,
            confidence_score=0.6,
            tags=["ml", "best-practices"],
            created_at=now - 3600,
            metadata={"experience": "production"},
        ),
        ContextItem(
            id="3",
            content="Debugging steps for API errors",
            title="API Debugging",
            layer=ContextLayer.PROCEDURAL,
            relevance_score=0.8,
            confidence_score=0.7,
            tags=["debugging", "api"],
            created_at=now - 7200,
            metadata={"skill_id": "api_debug"},
        ),
    ]


@pytest.mark.tdd
@pytest.mark.world_beating
class TestWorldBeatingRetrieval:
    """Test the complete world-beating retrieval pipeline."""

    @pytest.mark.asyncio
    async def test_stage1_intent_analysis(self, context_broker, sample_context_items):
        """Test Stage 1: Intent Analysis & Query Expansion."""
        # Setup
        query = ContextQuery(
            query_id="test_1",
            query="python async programming",
            layers=[ContextLayer.SEMANTIC, ContextLayer.EPISODIC],
            limit=10,
            semantic_search=True,
        )

        # Add sample data
        for item in sample_context_items:
            await context_broker.stores[ContextLayer.SEMANTIC].store(item)

        # Execute
        response = await context_broker.get_context(query)

        # Assert
        assert response.query_id == "test_1"
        assert len(response.items) > 0
        assert response.execution_time_ms > 0
        assert not response.cache_hit

    @pytest.mark.asyncio
    async def test_stage2_dense_retrieval(self, context_broker, sample_context_items):
        """Test Stage 2: Dense Retrieval Pipeline."""
        # Setup mock for dense retrieval
        with patch.object(
            context_broker.stores[ContextLayer.SEMANTIC], "search_similar"
        ) as mock_search:
            mock_search.return_value = sample_context_items[:2]

            query = ContextQuery(
                query_id="test_2",
                query="python programming",
                layers=[ContextLayer.SEMANTIC],
                limit=5,
                semantic_search=True,
            )

            # Execute
            response = await context_broker.get_context(query)

            # Assert
            mock_search.assert_called_once()
            assert len(response.items) == 2

    @pytest.mark.asyncio
    async def test_stage3_sparse_hybrid_integration(self, context_broker):
        """Test Stage 3: Sparse Hybrid Integration."""
        # Test hybrid scoring
        query = ContextQuery(
            query_id="test_3", query="python", layers=[ContextLayer.SEMANTIC], limit=10
        )

        # Add test data
        items = [
            ContextItem(
                id="1",
                content="Python programming tutorial",
                title="Tutorial",
                layer=ContextLayer.SEMANTIC,
                relevance_score=0.8,
                confidence_score=0.9,
                tags=["python"],
                created_at=time.time(),
            ),
            ContextItem(
                id="2",
                content="Java programming guide",
                title="Guide",
                layer=ContextLayer.SEMANTIC,
                relevance_score=0.6,
                confidence_score=0.7,
                tags=["java"],
                created_at=time.time(),
            ),
        ]

        for item in items:
            await context_broker.stores[ContextLayer.SEMANTIC].store(item)

        # Execute
        response = await context_broker.get_context(query)

        # Assert hybrid scoring worked
        assert len(response.items) > 0
        # Python-related item should rank higher
        python_item = next(
            (item for item in response.items if "python" in item.content.lower()), None
        )
        assert python_item is not None

    @pytest.mark.asyncio
    async def test_stage4_graph_based_expansion(self, context_broker):
        """Test Stage 4: Graph-Based Knowledge Expansion."""
        # Mock graph expander
        with patch.object(context_broker, "graph_expander") as mock_expander:
            mock_expansion = Mock()
            mock_expansion.expanded_items = [
                ContextItem(
                    id="expanded_1",
                    content="Expanded knowledge from graph",
                    title="Expanded",
                    layer=ContextLayer.SEMANTIC,
                    relevance_score=0.7,
                    confidence_score=0.6,
                    tags=["expanded"],
                    created_at=time.time(),
                )
            ]
            mock_expansion.new_relationships = ["rel1", "rel2"]
            mock_expansion.expansion_confidence = 0.8
            mock_expander.expand_context = AsyncMock(return_value=mock_expansion)

            query = ContextQuery(
                query_id="test_4", query="test query", layers=[ContextLayer.SEMANTIC]
            )

            # Execute
            response = await context_broker.get_context(query)

            # Assert graph expansion was called
            mock_expander.expand_context.assert_called_once()
            assert "graph_expansion" in response.metadata
            assert response.metadata["graph_expansion"]["expanded_count"] == 1

    @pytest.mark.asyncio
    async def test_stage5_neural_reranking(self, context_broker):
        """Test Stage 5: Neural Reranking & Personalization."""
        # Note: Neural reranker is placeholder, so we'll test the framework
        query = ContextQuery(
            query_id="test_5", query="neural network", layers=[ContextLayer.SEMANTIC]
        )

        # Add test items
        items = [
            ContextItem(
                id="1",
                content="Neural networks explained",
                title="NN Tutorial",
                layer=ContextLayer.SEMANTIC,
                relevance_score=0.9,
                confidence_score=0.8,
                tags=["neural", "ml"],
                created_at=time.time(),
            )
        ]

        for item in items:
            await context_broker.stores[ContextLayer.SEMANTIC].store(item)

        # Execute
        response = await context_broker.get_context(query)

        # Assert reranking framework works
        assert len(response.items) > 0

    @pytest.mark.asyncio
    async def test_stage6_adaptive_learning(self, context_broker):
        """Test Stage 6: Adaptive Learning & Optimization."""
        # Mock RL optimizer
        with patch.object(context_broker, "adaptive_optimizer") as mock_optimizer:
            mock_result = Mock()
            mock_result.improvement = 0.15
            mock_result.action_taken = Mock()
            mock_result.action_taken.parameter = "semantic_weight"
            mock_result.learning_rate = 0.01
            mock_optimizer.optimize_from_feedback = AsyncMock(return_value=mock_result)

            query = ContextQuery(
                query_id="test_6",
                query="adaptive learning",
                layers=[ContextLayer.SEMANTIC],
            )

            # Execute
            response = await context_broker.get_context(query)

            # Assert optimization was called
            mock_optimizer.optimize_from_feedback.assert_called_once()
            assert "optimization" in response.metadata
            assert response.metadata["optimization"]["improvement"] == 0.15


@pytest.mark.tdd
@pytest.mark.world_beating
class TestHybridRetrievalScoring:
    """Test hybrid retrieval scoring mechanisms."""

    def test_hybrid_score_creation(self, context_broker, sample_context_items):
        """Test creation of hybrid scores."""
        item = sample_context_items[0]
        score = HybridScore(
            item=item,
            semantic_score=0.8,
            keyword_score=0.6,
            temporal_score=0.9,
            combined_score=0.77,
        )

        assert score.item == item
        assert score.semantic_score == 0.8
        assert score.keyword_score == 0.6
        assert score.temporal_score == 0.9
        assert score.combined_score == 0.77

    def test_hybrid_score_comparison(self, sample_context_items):
        """Test hybrid score comparison for ranking."""
        item1 = sample_context_items[0]
        item2 = sample_context_items[1]

        score1 = HybridScore(
            item=item1,
            semantic_score=0.8,
            keyword_score=0.6,
            temporal_score=0.9,
            combined_score=0.77,
        )
        score2 = HybridScore(
            item=item2,
            semantic_score=0.6,
            keyword_score=0.8,
            temporal_score=0.7,
            combined_score=0.71,
        )

        # Higher combined score should be "less than" for max-heap behavior
        assert score1 > score2
        assert score2 < score1

    def test_calculate_semantic_score(self, context_broker, sample_context_items):
        """Test semantic score calculation."""
        item = sample_context_items[0]
        query = "python async"

        score = context_broker._calculate_semantic_score(item, query)
        assert 0.0 <= score <= 1.0

        # Should have high score for relevant content
        assert score > 0.5

    def test_calculate_keyword_score(self, context_broker, sample_context_items):
        """Test keyword score calculation."""
        item = sample_context_items[0]
        query = "python async"

        score = context_broker._calculate_keyword_score(item, query)
        assert 0.0 <= score <= 1.0

    def test_calculate_temporal_score(self, context_broker, sample_context_items):
        """Test temporal score calculation."""
        item = sample_context_items[0]

        score = context_broker._calculate_temporal_score(item)
        assert 0.0 <= score <= 1.0

        # Recent items should have higher scores
        assert score > 0.5


@pytest.mark.tdd
@pytest.mark.world_beating
class TestPerformanceMonitoring:
    """Test performance monitoring and metrics."""

    @pytest.mark.asyncio
    async def test_execution_time_tracking(self, context_broker):
        """Test that execution time is properly tracked."""
        query = ContextQuery(
            query_id="perf_test",
            query="performance test",
            layers=[ContextLayer.SEMANTIC],
        )

        # Execute query
        response = await context_broker.get_context(query)

        # Assert execution time is recorded
        assert response.execution_time_ms > 0
        assert isinstance(response.execution_time_ms, int)

    def test_metrics_collection(self, context_broker):
        """Test metrics collection."""
        metrics = context_broker.get_metrics()

        required_keys = [
            "total_queries",
            "cache_hits",
            "cache_misses",
            "avg_execution_time_ms",
            "error_count",
            "cache_size",
            "active_stores",
        ]

        for key in required_keys:
            assert key in metrics
            assert isinstance(metrics[key], (int, float))

    @pytest.mark.asyncio
    async def test_cache_functionality(self, context_broker):
        """Test caching functionality."""
        query = ContextQuery(
            query_id="cache_test", query="cache test", layers=[ContextLayer.SEMANTIC]
        )

        # First request should miss cache
        response1 = await context_broker.get_context(query)
        assert not response1.cache_hit

        # Second request should hit cache
        response2 = await context_broker.get_context(query)
        assert response2.cache_hit

        # Verify cache metrics
        metrics = context_broker.get_metrics()
        assert metrics["cache_hits"] >= 1
        assert metrics["cache_misses"] >= 1


@pytest.mark.tdd
@pytest.mark.world_beating
@pytest.mark.slow
class TestStressAndLoad:
    """Stress tests for world-beating retrieval system."""

    @pytest.mark.asyncio
    async def test_concurrent_queries(self, context_broker):
        """Test handling of concurrent queries."""
        queries = [
            ContextQuery(
                query_id=f"concurrent_{i}",
                query=f"concurrent test query {i}",
                layers=[ContextLayer.SEMANTIC, ContextLayer.EPISODIC],
            )
            for i in range(10)
        ]

        # Execute concurrent queries
        tasks = [context_broker.get_context(query) for query in queries]
        responses = await asyncio.gather(*tasks)

        # Assert all queries completed successfully
        assert len(responses) == 10
        for response in responses:
            assert response.execution_time_ms > 0
            assert len(response.items) >= 0

    @pytest.mark.asyncio
    async def test_large_result_sets(self, context_broker):
        """Test handling of large result sets."""
        # Add many items to test scaling
        items = []
        for i in range(100):
            item = ContextItem(
                id=str(i),
                content=f"Large dataset item {i} with relevant content",
                title=f"Item {i}",
                layer=ContextLayer.SEMANTIC,
                relevance_score=0.8,
                confidence_score=0.7,
                tags=["large", "dataset"],
                created_at=time.time(),
            )
            items.append(item)
            await context_broker.stores[ContextLayer.SEMANTIC].store(item)

        query = ContextQuery(
            query_id="large_test",
            query="relevant content",
            layers=[ContextLayer.SEMANTIC],
            limit=50,
        )

        # Execute query
        response = await context_broker.get_context(query)

        # Assert system can handle large datasets
        assert response.total_results >= 50
        assert len(response.items) <= 50  # Respect limit


@pytest.mark.tdd
@pytest.mark.world_beating
class TestErrorHandling:
    """Test error handling in world-beating retrieval."""

    @pytest.mark.asyncio
    async def test_store_failure_handling(self, context_broker):
        """Test handling of store failures."""
        # Mock a store to fail
        with patch.object(
            context_broker.stores[ContextLayer.SEMANTIC], "retrieve"
        ) as mock_retrieve:
            mock_retrieve.side_effect = Exception("Store failure")

            query = ContextQuery(
                query_id="error_test",
                query="error test",
                layers=[ContextLayer.SEMANTIC],
            )

            # Should handle error gracefully
            with pytest.raises(Exception):
                await context_broker.get_context(query)

    @pytest.mark.asyncio
    async def test_partial_failure_recovery(self, context_broker):
        """Test recovery from partial failures."""
        # Mock one store to fail, others to succeed
        with patch.object(
            context_broker.stores[ContextLayer.SEMANTIC], "retrieve"
        ) as mock_semantic:
            mock_semantic.side_effect = Exception("Semantic store failed")

            # Add working data to other stores
            working_item = ContextItem(
                id="working_1",
                content="Working memory content",
                title="Working",
                layer=ContextLayer.WORKING,
                relevance_score=0.8,
                confidence_score=0.7,
                tags=["working"],
                created_at=time.time(),
            )
            await context_broker.stores[ContextLayer.WORKING].store(working_item)

            query = ContextQuery(
                query_id="partial_failure_test",
                query="test query",
                layers=[ContextLayer.SEMANTIC, ContextLayer.WORKING],
            )

            # Should still return results from working stores
            response = await context_broker.get_context(query)
            assert response.parallel_requests_used >= 1

    @pytest.mark.asyncio
    async def test_timeout_handling(self, context_broker):
        """Test timeout handling."""

        # Mock slow store
        async def slow_retrieve(query):
            await asyncio.sleep(0.1)  # Simulate slow operation
            response = ContextResponse()
            response.query_id = query.query_id
            response.items = []
            return response

        with patch.object(
            context_broker.stores[ContextLayer.SEMANTIC],
            "retrieve",
            side_effect=slow_retrieve,
        ):
            query = ContextQuery(
                query_id="timeout_test",
                query="timeout test",
                layers=[ContextLayer.SEMANTIC],
            )

            # Should complete within reasonable time
            import time

            start_time = time.time()
            response = await context_broker.get_context(query)
            execution_time = time.time() - start_time

            # Allow some tolerance for execution time
            assert execution_time < 1.0  # Should complete reasonably quickly

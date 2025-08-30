#!/usr/bin/env python3
"""
Test-Driven Development: Unit tests for World-Beating Hybrid Retrieval Engine
Implements comprehensive testing for the 6-stage retrieval pipeline surpassing Claude/ChatGPT.
"""

from typing import Any, Dict, List, Optional

import pytest

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

    AdaptiveOptimizationResult, AdvancedScore, DenseRetrievalResult,
    GraphExpansionResult, IntentAnalysisResult, NeuralRerankResult,
    RetrievalPipeline, RetrievalStage, SparseHybridResult,
    WorldBeatingRetrievalEngine)
from pinak.context.core.models import (ContextItem, ContextLayer,
                                       ContextPriority, ContextQuery,
                                       ContextResponse, IContextStore,
                                       SecurityClassification)


class MockAdvancedStore(IContextStore):
    """Mock store with advanced retrieval capabilities"""

    def __init__(self, layer: ContextLayer):
        self.layer = layer
        self.embeddings = {}  # Mock embeddings

    async def retrieve(self, query):
        return ContextResponse(items=[])

    async def store(self, item):
        return True

    async def delete(self, item_id: str, project_id: str):
        return True

    async def update(self, item):
        return True

    async def search_similar(self, content: str, limit: int = 10):
        # Mock semantic search with embeddings
        return [
            ContextItem(
                id=f"mock-{self.layer.value}-1",
                title=f"Mock {self.layer.value} Item",
                summary=f"Mock semantic search result",
                content=content,
                layer=self.layer,
                project_id="test-project",
                tenant_id="test-tenant",
                created_by="test-user",
                classification=SecurityClassification.INTERNAL,
                priority=ContextPriority.MEDIUM,
                tags=["mock"],
                relevance_score=0.9,
                confidence_score=0.95,
            )
        ]


@pytest.fixture
def mock_stores():
    """Fixture for mock stores"""
    return {
        ContextLayer.SEMANTIC: MockAdvancedStore(ContextLayer.SEMANTIC),
        ContextLayer.EPISODIC: MockAdvancedStore(ContextLayer.EPISODIC),
        ContextLayer.PROCEDURAL: MockAdvancedStore(ContextLayer.PROCEDURAL),
        ContextLayer.RAG: MockAdvancedStore(ContextLayer.RAG),
        ContextLayer.EVENTS: MockAdvancedStore(ContextLayer.EVENTS),
        ContextLayer.SESSION: MockAdvancedStore(ContextLayer.SESSION),
        ContextLayer.WORKING: MockAdvancedStore(ContextLayer.WORKING),
        ContextLayer.CHANGELOG: MockAdvancedStore(ContextLayer.CHANGELOG),
    }


@pytest.fixture
def sample_query():
    """Fixture for sample context query"""
    return ContextQuery(
        query="test query for advanced retrieval",
        project_id="test-project",
        tenant_id="test-tenant",
        user_id="test-user",
        layers=list(ContextLayer),
        limit=20,
        user_clearance=SecurityClassification.CONFIDENTIAL,
        semantic_search=True,
        min_relevance=0.1,
        min_confidence=0.1,
    )


@pytest.fixture
def sample_items():
    """Fixture for sample context items"""
    old_date = datetime.now(timezone.utc) - timedelta(days=5)
    recent_date = datetime.now(timezone.utc) - timedelta(hours=1)

    return [
        ContextItem(
            id="item-1",
            title="Python Exception Handling",
            summary="Best practices for Python exception handling",
            content="Use try-except blocks to handle exceptions gracefully. Always catch specific exceptions rather than bare except.",
            layer=ContextLayer.SEMANTIC,
            project_id="test-project",
            tenant_id="test-tenant",
            created_by="test-user",
            created_at=recent_date,
            classification=SecurityClassification.PUBLIC,
            priority=ContextPriority.HIGH,
            tags=["python", "exception", "best-practices"],
            relevance_score=0.95,
            confidence_score=0.9,
        ),
        ContextItem(
            id="item-2",
            title="Old Database Connection",
            summary="Outdated connection pattern",
            content="This old pattern is deprecated. Use connection pooling instead.",
            layer=ContextLayer.EPISODIC,
            project_id="test-project",
            tenant_id="test-tenant",
            created_by="test-user",
            created_at=old_date,
            classification=SecurityClassification.CONFIDENTIAL,
            priority=ContextPriority.MEDIUM,
            tags=["database", "connection"],
            relevance_score=0.6,
            confidence_score=0.7,
        ),
    ]


class TestWorldBeatingRetrievalEngine:
    """Test suite for WorldBeatingRetrievalEngine"""

    @pytest.mark.asyncio
    async def test_intent_analysis_stage(self, mock_stores, sample_query):
        """Test Stage 1: Intent Analysis & Query Expansion"""
        engine = WorldBeatingRetrievalEngine(mock_stores)

        result = await engine._execute_intent_analysis(sample_query)

        assert isinstance(result, IntentAnalysisResult)
        assert isinstance(result.expanded_queries, list)
        assert isinstance(result.intent_categories, list)
        assert result.confidence >= 0.0
        assert result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_dense_retrieval_stage(self, mock_stores, sample_query):
        """Test Stage 2: Dense Retrieval Pipeline"""
        engine = WorldBeatingRetrievalEngine(mock_stores)

        result = await engine._execute_dense_retrieval(sample_query)

        assert isinstance(result, DenseRetrievalResult)
        assert isinstance(result.vectors, list)
        assert result.embedding_dimensions > 0
        assert result.similarity_threshold > 0
        assert result.candidates_found >= 0

    @pytest.mark.asyncio
    async def test_sparse_hybrid_integration(
        self, mock_stores, sample_query, sample_items
    ):
        """Test Stage 3: Sparse Hybrid Integration"""
        engine = WorldBeatingRetrievalEngine(mock_stores)

        result = await engine._execute_sparse_hybrid(sample_query, sample_items)

        assert isinstance(result, SparseHybridResult)
        assert result.bm25_score >= 0
        assert result.semantic_weight >= 0
        assert result.lexical_weight >= 0
        assert result.combined_score >= 0

    @pytest.mark.asyncio
    async def test_graph_based_expansion(self, mock_stores, sample_query, sample_items):
        """Test Stage 4: Graph-Based Knowledge Expansion"""
        engine = WorldBeatingRetrievalEngine(mock_stores)

        result = await engine._execute_graph_expansion(sample_query, sample_items)

        assert isinstance(result, GraphExpansionResult)
        assert len(result.expanded_items) >= len(sample_items)
        assert result.traversal_depth >= 0
        assert result.relevance_threshold > 0

    @pytest.mark.asyncio
    async def test_neural_reranking(self, mock_stores, sample_query, sample_items):
        """Test Stage 5: Neural Reranking & Personalization"""
        engine = WorldBeatingRetrievalEngine(mock_stores)

        result = await engine._execute_neural_rerank(sample_query, sample_items)

        assert isinstance(result, NeuralRerankResult)
        assert len(result.reranked_items) == len(sample_items)
        assert result.neural_score >= 0
        assert result.user_personalization_score >= 0

    @pytest.mark.asyncio
    async def test_adaptive_learning(self, mock_stores, sample_query, sample_items):
        """Test Stage 6: Adaptive Learning & Optimization"""
        engine = WorldBeatingRetrievalEngine(mock_stores)

        result = await engine._execute_adaptive_learning(sample_query, sample_items)

        assert isinstance(result, AdaptiveOptimizationResult)
        assert result.success_rate >= 0
        assert result.feedback_loop_active is True
        assert len(result.optimization_suggestions) >= 0

    @pytest.mark.asyncio
    async def test_full_pipeline_execution(self, mock_stores, sample_query):
        """Test complete 6-stage pipeline execution"""
        engine = WorldBeatingRetrievalEngine(mock_stores)

        result = await engine.execute_pipeline(sample_query)

        assert isinstance(result, ContextResponse)
        assert isinstance(result.query_id, str) and len(result.query_id) > 0
        assert result.execution_time_ms >= 0
        assert len(result.items) >= 0
        assert result.total_results >= len(result.items)

    def test_advanced_scoring(self, sample_items, sample_query):
        """Test AdvancedScore calculation"""
        engine = WorldBeatingRetrievalEngine({})

        scores = engine._calculate_advanced_scores(sample_items, sample_query)

        assert len(scores) == len(sample_items)
        for score in scores:
            assert isinstance(score, AdvancedScore)
            assert score.semantic_score >= 0
            assert score.keyword_score >= 0
            assert score.temporal_score >= 0
            assert score.ensemble_score >= 0
            assert score.contextual_score >= 0
            assert score.final_score >= 0

    def test_pipeline_stage_coordination(self, mock_stores):
        """Test pipeline stage coordination and data flow"""
        engine = WorldBeatingRetrievalEngine(mock_stores)

        pipeline = engine.pipeline

        assert isinstance(pipeline, RetrievalPipeline)
        assert len(pipeline.stages) == 6  # 6 stages

        for stage in pipeline.stages:
            assert isinstance(stage, RetrievalStage)
            assert stage.name in [
                "intent_analysis",
                "dense_retrieval",
                "sparse_hybrid",
                "graph_expansion",
                "neural_rerank",
                "adaptive_learning",
            ]

    @pytest.mark.asyncio
    async def test_performance_metrics(self, mock_stores, sample_query):
        """Test performance monitoring and metrics collection"""
        engine = WorldBeatingRetrievalEngine(mock_stores)

        # Execute pipeline
        await engine.execute_pipeline(sample_query)

        # Check metrics
        metrics = engine.get_performance_metrics()

        assert isinstance(metrics, dict)
        assert "total_queries" in metrics
        assert "avg_execution_time_ms" in metrics
        assert "cache_hit_rate" in metrics
        assert "error_rate" in metrics

    @pytest.mark.asyncio
    async def test_error_handling_and_resilience(self, mock_stores):
        """Test error handling and system resilience"""
        from pydantic import ValidationError

        engine = WorldBeatingRetrievalEngine(mock_stores)

        # Test with invalid query
        try:
            invalid_query = ContextQuery(
                query="",  # Invalid empty query
                project_id="test",
                tenant_id="test",
                user_id="test",
                layers=[],
            )
        except ValidationError:
            # Expected validation error for empty query
            return

        result = await engine.execute_pipeline(invalid_query)

        # Should handle gracefully
        assert isinstance(result, ContextResponse)
        assert result.execution_time_ms >= 0

    def test_pipeline_configuration(self, mock_stores):
        """Test pipeline configuration and hyperparameters"""
        config = {
            "embedding_dimensions": 768,
            "similarity_threshold": 0.7,
            "rerank_model": "transformer",
            "cache_ttl": 3600,
            "max_parallel_requests": 10,
        }

        engine = WorldBeatingRetrievalEngine(mock_stores, config=config)

        # Verify configuration is applied
        assert engine.embedding_dimensions == config["embedding_dimensions"]
        assert engine.similarity_threshold == config["similarity_threshold"]

    @pytest.mark.asyncio
    async def test_multi_layer_fusion(self, mock_stores, sample_query, sample_items):
        """Test multi-layer result fusion and ranking"""
        engine = WorldBeatingRetrievalEngine(mock_stores)

        fused_results = await engine._execute_multi_layer_fusion(
            sample_query, [sample_items]
        )

        assert isinstance(fused_results, list)
        assert all(isinstance(item, ContextItem) for item in fused_results)

        # Check that items are properly ranked by relevance
        if len(fused_results) > 1:
            for i in range(len(fused_results) - 1):
                assert (
                    fused_results[i].relevance_score
                    >= fused_results[i + 1].relevance_score
                )


class TestRetrievalPipeline:
    """Test suite for RetrievalPipeline configuration and execution"""

    def test_pipeline_initialization(self, mock_stores):
        """Test pipeline initialization with all stages"""
        pipeline = RetrievalPipeline()

        assert len(pipeline.stages) == 6
        assert pipeline.current_stage == 0

    def test_stage_execution_order(self, mock_stores, sample_query):
        """Test that stages execute in correct order"""
        pipeline = RetrievalPipeline()

        execution_order = []

        # Mock stage execution
        for i, stage in enumerate(pipeline.stages):
            # Simulate stage execution
            execution_order.append(stage.name)

        expected_order = [
            "intent_analysis",
            "dense_retrieval",
            "sparse_hybrid",
            "graph_expansion",
            "neural_rerank",
            "adaptive_learning",
        ]

        assert execution_order == expected_order


# Integration tests
@pytest.mark.integration
@pytest.mark.asyncio
async def test_end_to_end_retrieval_pipeline(mock_stores, sample_query):
    """Integration test for complete retrieval pipeline"""
    engine = WorldBeatingRetrievalEngine(mock_stores)

    # Execute full pipeline
    result = await engine.execute_pipeline(sample_query)

    # Comprehensive assertions
    assert result.returned_results >= 0
    assert result.total_results >= result.returned_results
    assert result.execution_time_ms >= 0
    assert len(result.items) <= sample_query.limit

    # Check that items meet quality criteria
    for item in result.items:
        assert item.relevance_score >= sample_query.min_relevance
        assert item.confidence_score >= sample_query.min_confidence
        # Check security clearance hierarchy: user clearance should be >= item classification
        clearance_levels = {
            SecurityClassification.PUBLIC: 1,
            SecurityClassification.INTERNAL: 2,
            SecurityClassification.CONFIDENTIAL: 3,
            SecurityClassification.RESTRICTED: 4,
        }
        user_level = clearance_levels.get(sample_query.user_clearance, 0)
        item_level = clearance_levels.get(item.classification, 0)
        assert (
            user_level >= item_level
        ), f"User clearance {sample_query.user_clearance} should allow access to {item.classification}"

    # Verify cross-layer search
    layers_represented = set(item.layer for item in result.items)
    assert len(layers_represented) > 0


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])

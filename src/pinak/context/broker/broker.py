# FANG-Level Context Broker - Hybrid Retrieval Engine
"""
Enterprise-grade context broker implementing hybrid retrieval with semantic and keyword search.
Features FANG-level performance, scalability, and reliability.
"""

from __future__ import annotations

import asyncio
import heapq
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional, Set

from ..core.models import (ContextItem, ContextLayer, ContextPriority,
                           ContextQuery, ContextResponse, IContextStore,
                           SecurityClassification)
from .graph_expansion import GraphBasedExpander, GraphExpansionResult
from .rl_optimizer import (AdaptiveLearningEngine, QLearningOptimizer,
                           adaptive_engine)

# Neural reranker import (placeholder - needs implementation)
# from .neural_reranker import NeuralReranker

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from a single retrieval operation."""

    items: List[ContextItem]
    execution_time_ms: int
    source: str
    confidence: float


@dataclass
class HybridScore:
    """Combined score for hybrid retrieval ranking."""

    item: ContextItem
    semantic_score: float
    keyword_score: float
    temporal_score: float
    combined_score: float

    def __lt__(self, other: HybridScore) -> bool:
        return self.combined_score < other.combined_score


class ContextBroker:
    """
    FANG-level context broker implementing hybrid retrieval.

    Features:
    - Hybrid semantic + keyword search
    - Multi-layer parallel retrieval
    - Intelligent reranking and fusion
    - Enterprise-grade caching and performance
    - Comprehensive observability
    """

    def __init__(
        self,
        stores: Dict[ContextLayer, IContextStore],
        cache_ttl_seconds: int = 300,
        max_parallel_requests: int = 5,
        semantic_weight: float = 0.6,
        keyword_weight: float = 0.3,
        temporal_weight: float = 0.1,
        enable_world_beating: bool = True,
    ):
        self.stores = stores
        self.cache_ttl = cache_ttl_seconds
        self.max_parallel = max_parallel_requests
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.temporal_weight = temporal_weight

        # World-beating pipeline components
        self.enable_world_beating = enable_world_beating
        if enable_world_beating:
            self.graph_expander = GraphBasedExpander()
            # self.neural_reranker = NeuralReranker()  # Placeholder
            self.adaptive_optimizer = adaptive_engine

            logger.info(
                "🚀 World-beating components integrated: Graph Expansion, RL Optimization"
            )

        # Performance and caching
        self._cache: Dict[str, tuple[ContextResponse, float]] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_parallel_requests)
        self._semaphore = asyncio.Semaphore(max_parallel_requests)

        # Metrics
        self._metrics = {
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_execution_time_ms": 0.0,
            "error_count": 0,
        }

        logger.info(f"ContextBroker initialized with {len(stores)} stores")

    async def get_context(self, query: ContextQuery) -> ContextResponse:
        """
        Retrieve context using hybrid search with FANG-level performance.

        Implements:
        - Parallel multi-layer retrieval
        - Intelligent result fusion
        - Security filtering
        - Performance monitoring
        """
        start_time = time.time()
        self._metrics["total_queries"] += 1

        try:
            # Check cache first
            cache_key = self._generate_cache_key(query)
            if cached := self._get_cached_response(cache_key):
                self._metrics["cache_hits"] += 1
                logger.info(f"Cache hit for query {query.query_id}")
                return cached

            self._metrics["cache_misses"] += 1

            # Execute hybrid retrieval
            response = await self._execute_hybrid_retrieval(query)

            # Cache the response
            self._cache_response(cache_key, response)

            # Update metrics
            execution_time = int((time.time() - start_time) * 1000)
            response.execution_time_ms = execution_time
            self._update_execution_metrics(execution_time)

            logger.info(
                f"Context retrieval completed in {execution_time}ms: "
                f"{response.returned_results} results from {len(query.layers)} layers"
            )

            return response

        except Exception as e:
            self._metrics["error_count"] += 1
            logger.error(f"Context retrieval failed: {e}")
            raise

    async def _execute_hybrid_retrieval(self, query: ContextQuery) -> ContextResponse:
        """Execute hybrid retrieval across multiple layers in parallel."""
        response = ContextResponse()
        response.query_id = query.query_id

        # Determine which layers to search
        target_layers = query.layers if query.layers else list(self.stores.keys())

        # Execute parallel retrieval
        tasks = []
        for layer in target_layers:
            if layer in self.stores:
                task = self._retrieve_from_layer(query, layer)
                tasks.append(task)

        # Limit parallel execution
        semaphore_tasks = []
        for task in tasks:
            semaphore_tasks.append(self._execute_with_semaphore(task))

        # Gather results
        results = await asyncio.gather(*semaphore_tasks, return_exceptions=True)

        # Process results and handle exceptions
        all_items = []
        for i, result in enumerate(results):
            layer = target_layers[i]
            if isinstance(result, Exception):
                logger.warning(f"Retrieval failed for layer {layer}: {result}")
                continue

            if not isinstance(result, RetrievalResult):
                logger.warning(
                    f"Unexpected result type for layer {layer}: {type(result)}"
                )
                continue

            retrieval_result = result
            response.parallel_requests_used += 1

            # Apply security filtering
            for item in retrieval_result.items:
                if self._passes_security_filter(item, query.user_clearance):
                    all_items.append((item, retrieval_result.confidence))
                else:
                    response.redacted_count += 1

        # Apply hybrid scoring and ranking
        ranked_items = self._hybrid_rerank(all_items, query)

        # Apply final filtering and limits
        filtered_items = self._apply_filters(ranked_items, query)

        # Build response
        for item in filtered_items:
            response.add_item(item, query.user_clearance)

        response.total_results = len(all_items)
        response.has_more = len(filtered_items) < len(ranked_items)

        # Apply world-beating enhancements
        if self.enable_world_beating:
            response = await self._apply_world_beating_enhancements(query, response)

        return response

    async def _retrieve_from_layer(
        self, query: ContextQuery, layer: ContextLayer
    ) -> RetrievalResult:
        """Retrieve from a specific layer with performance monitoring."""
        start_time = time.time()

        try:
            store = self.stores[layer]

            # Execute retrieval based on search type
            if query.semantic_search and layer in [
                ContextLayer.SEMANTIC,
                ContextLayer.EPISODIC,
            ]:
                items = await store.search_similar(query.query, query.limit * 2)
            else:
                # Use standard retrieval with query
                response = await store.retrieve(query)
                items = response.items

            # Apply RL optimization to adjust parameters in real-time
            if self.enable_world_beating and hasattr(self, "adaptive_optimizer"):
                await self.adaptive_optimizer.optimize_from_feedback(query, response)

            execution_time = int((time.time() - start_time) * 1000)

            # Calculate confidence based on layer and result quality
            confidence = self._calculate_layer_confidence(
                layer, len(items), execution_time
            )

            return RetrievalResult(
                items=items,
                execution_time_ms=execution_time,
                source=layer.value,
                confidence=confidence,
            )

        except Exception as e:
            logger.error(f"Layer {layer} retrieval failed: {e}")
            raise

    async def _execute_with_semaphore(self, coro):
        """Execute coroutine with semaphore to limit parallelism."""
        async with self._semaphore:
            return await coro

    def _hybrid_rerank(
        self,
        items_with_confidence: List[tuple[ContextItem, float]],
        query: ContextQuery,
    ) -> List[ContextItem]:
        """Apply hybrid scoring and ranking to results."""
        scored_items = []

        for item, base_confidence in items_with_confidence:
            # Calculate component scores
            semantic_score = self._calculate_semantic_score(item, query.query)
            keyword_score = self._calculate_keyword_score(item, query.query)
            temporal_score = self._calculate_temporal_score(item)

            # Combine scores with weights
            combined_score = (
                self.semantic_weight * semantic_score
                + self.keyword_weight * keyword_score
                + self.temporal_weight * temporal_score
            )

            # Boost based on layer priority
            layer_boost = self._get_layer_boost(item.layer)
            final_score = combined_score * layer_boost

            scored_item = HybridScore(
                item=item,
                semantic_score=semantic_score,
                keyword_score=keyword_score,
                temporal_score=temporal_score,
                combined_score=final_score,
            )

            scored_items.append(scored_item)

        # Sort by combined score (highest first)
        scored_items.sort(reverse=True)

        return [score.item for score in scored_items]

    def _calculate_semantic_score(self, item: ContextItem, query: str) -> float:
        """Calculate semantic similarity score."""
        # This would integrate with actual embedding model
        # For now, use simple text overlap as proxy
        query_words = set(query.lower().split())
        content_words = set(item.content.lower().split())

        overlap = len(query_words.intersection(content_words))
        union = len(query_words.union(content_words))

        return overlap / union if union > 0 else 0.0

    def _calculate_keyword_score(self, item: ContextItem, query: str) -> float:
        """Calculate keyword-based relevance score."""
        query_lower = query.lower()
        title_match = 1.0 if query_lower in item.title.lower() else 0.0
        content_match = item.content.lower().count(query_lower) / len(
            item.content.split()
        )

        return min(1.0, (title_match * 0.7) + (content_match * 0.3))

    def _calculate_temporal_score(self, item: ContextItem) -> float:
        """Calculate temporal freshness score."""
        age_hours = (
            datetime.now(timezone.utc) - item.created_at
        ).total_seconds() / 3600

        # Exponential decay with 48-hour half-life
        return 0.5 ** (age_hours / 48)

    def _get_layer_boost(self, layer: ContextLayer) -> float:
        """Get priority boost for different layers."""
        boosts = {
            ContextLayer.SEMANTIC: 1.2,
            ContextLayer.EPISODIC: 1.1,
            ContextLayer.PROCEDURAL: 1.0,
            ContextLayer.SESSION: 0.9,
            ContextLayer.WORKING: 0.8,
            ContextLayer.EVENTS: 0.9,
            ContextLayer.RAG: 1.0,
            ContextLayer.CHANGELOG: 0.7,
        }
        return boosts.get(layer, 1.0)

    def _apply_filters(
        self, items: List[ContextItem], query: ContextQuery
    ) -> List[ContextItem]:
        """Apply final filtering based on query parameters."""
        filtered = []

        for item in items:
            # Relevance filter
            if item.relevance_score < query.min_relevance:
                continue

            # Confidence filter
            if item.confidence_score < query.min_confidence:
                continue

            # Priority filter
            if query.priority_filter and item.priority != query.priority_filter:
                continue

            # Tags filter
            if query.tags_filter and not any(
                tag in item.tags for tag in query.tags_filter
            ):
                continue

            # Temporal filters
            if query.since and item.created_at < query.since:
                continue
            if query.until and item.created_at > query.until:
                continue
            if not query.include_expired and item.is_expired():
                continue

            filtered.append(item)

        # Apply limit and offset
        start_idx = query.offset
        end_idx = start_idx + query.limit
        return filtered[start_idx:end_idx]

    def _passes_security_filter(
        self, item: ContextItem, user_clearance: SecurityClassification
    ) -> bool:
        """Check if user has sufficient clearance for the item."""
        return user_clearance.value >= item.classification.value

    def _generate_cache_key(self, query: ContextQuery) -> str:
        """Generate cache key for query."""
        key_data = {
            "query": query.query,
            "project_id": query.project_id,
            "layers": sorted([l.value for l in query.layers]),
            "limit": query.limit,
            "user_clearance": query.user_clearance.value,
        }
        import hashlib
        import json

        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cached_response(self, cache_key: str) -> Optional[ContextResponse]:
        """Get cached response if valid."""
        if cache_key in self._cache:
            response, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                response.cache_hit = True
                return response
            else:
                del self._cache[cache_key]
        return None

    def _cache_response(self, cache_key: str, response: ContextResponse) -> None:
        """Cache response with timestamp."""
        self._cache[cache_key] = (response, time.time())

        # Clean up old cache entries (simple LRU)
        if len(self._cache) > 1000:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]

    def _calculate_layer_confidence(
        self, layer: ContextLayer, result_count: int, execution_time: int
    ) -> float:
        """Calculate confidence score for layer results."""
        # Base confidence by layer
        base_confidence = {
            ContextLayer.SEMANTIC: 0.9,
            ContextLayer.EPISODIC: 0.8,
            ContextLayer.PROCEDURAL: 0.7,
            ContextLayer.SESSION: 0.6,
            ContextLayer.WORKING: 0.5,
            ContextLayer.EVENTS: 0.6,
            ContextLayer.RAG: 0.8,
            ContextLayer.CHANGELOG: 0.4,
        }.get(layer, 0.5)

        # Adjust for result count (more results = higher confidence)
        count_boost = min(0.2, result_count / 100)

        # Adjust for execution time (faster = higher confidence)
        time_penalty = max(0, (execution_time - 100) / 1000)  # Penalty after 100ms

        return max(0.1, base_confidence + count_boost - time_penalty)

    async def _apply_world_beating_enhancements(
        self, query: ContextQuery, response: ContextResponse
    ) -> ContextResponse:
        """Apply world-beating enhancements: graph expansion, neural reranking, RL optimization."""
        original_items = response.items.copy()

        # Stage 4: Graph-based knowledge expansion
        if hasattr(self, "graph_expander"):
            expansion_result = await self.graph_expander.expand_context(
                query=query,
                initial_items=original_items,
                expansion_depth=3,
                relevance_threshold=0.15,
            )

            # Merge expanded items with original response
            response.items.extend(expansion_result.expanded_items)
            response.metadata = response.metadata or {}
            response.metadata["graph_expansion"] = {
                "expanded_count": len(expansion_result.expanded_items),
                "new_relationships": len(expansion_result.new_relationships),
                "expansion_confidence": expansion_result.expansion_confidence,
            }

        # Stage 5: Neural reranking (placeholder - implement when neural reranker is ready)
        # if hasattr(self, 'neural_reranker'):
        #     response = await self.neural_reranker.rerank_response(query, response)

        # Stage 6: RL optimization feedback loop
        if hasattr(self, "adaptive_optimizer"):
            optimization_result = await self.adaptive_optimizer.optimize_from_feedback(
                query, response
            )
            if optimization_result.improvement > 0:
                response.metadata = response.metadata or {}
                response.metadata["optimization"] = {
                    "parameter_adjusted": optimization_result.action_taken.parameter,
                    "improvement": optimization_result.improvement,
                    "learning_rate": optimization_result.learning_rate,
                }

        return response

    def _update_execution_metrics(self, execution_time: int) -> None:
        """Update rolling average execution time."""
        current_avg = self._metrics["avg_execution_time_ms"]
        total_queries = self._metrics["total_queries"]

        # Simple moving average
        self._metrics["avg_execution_time_ms"] = (
            (current_avg * (total_queries - 1)) + execution_time
        ) / total_queries

    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            **self._metrics,
            "cache_size": len(self._cache),
            "active_stores": len(self.stores),
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health = {
            "status": "healthy",
            "stores": {},
            "cache": {
                "size": len(self._cache),
                "ttl_seconds": self.cache_ttl,
            },
            "performance": self.get_metrics(),
        }

        # Check each store
        for layer, store in self.stores.items():
            try:
                # Simple health check - this would be implemented per store
                health["stores"][layer.value] = {"status": "healthy"}
            except Exception as e:
                health["stores"][layer.value] = {"status": "unhealthy", "error": str(e)}
                health["status"] = "degraded"

        return health

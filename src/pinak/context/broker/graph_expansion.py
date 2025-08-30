# Graph-Based Knowledge Expansion - World-Beater Stage 4
"""
Advanced graph-based knowledge expansion for context retrieval.
Implements dynamic knowledge graph construction, entity relationship mining,
and contextual path finding with relevance weighting.
"""


import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from ..core.models import (ContextItem, ContextLayer, ContextPriority,
                           ContextQuery, ContextResponse,
                           SecurityClassification)

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeNode:
    """Node in the knowledge graph representing entities and concepts."""

    id: str
    type: str
    content: str
    layer: ContextLayer
    relevance_score: float = 0.0
    confidence_score: float = 0.0
    temporal_weight: float = 1.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class KnowledgeEdge:
    """Edge in the knowledge graph representing relationships."""

    source_id: str
    target_id: str
    relationship_type: str
    weight: float = 1.0
    confidence: float = 1.0
    bidirectional: bool = False
    temporal_context: Optional[str] = None


@dataclass
class GraphExpansionResult:
    """Result of graph-based expansion."""

    original_items: List[ContextItem]
    expanded_items: List[ContextItem]
    new_relationships: List[KnowledgeEdge]
    traversal_depth: int
    relevance_threshold: float
    expansion_confidence: float
    execution_time_ms: int


class KnowledgeGraph:
    """Dynamic knowledge graph for context expansion."""

    def __init__(self, max_nodes: int = 10000, decay_factor: float = 0.9):
        self.graph = nx.DiGraph()
        self.node_index: Dict[str, KnowledgeNode] = {}
        self.max_nodes = max_nodes
        self.decay_factor = decay_factor
        self._temporal_decay = defaultdict(float)

    def add_node(self, node: KnowledgeNode) -> None:
        """Add a node to the knowledge graph."""
        if len(self.node_index) >= self.max_nodes:
            self._evict_oldest_nodes()

        self.graph.add_node(node.id, **node.__dict__)
        self.node_index[node.id] = node

    def add_edge(self, edge: KnowledgeEdge) -> None:
        """Add an edge to the knowledge graph."""
        self.graph.add_edge(
            edge.source_id,
            edge.target_id,
            relationship_type=edge.relationship_type,
            weight=edge.weight,
            confidence=edge.confidence,
            temporal_context=edge.temporal_context,
        )

        if edge.bidirectional:
            self.graph.add_edge(
                edge.target_id,
                edge.source_id,
                relationship_type=f"reverse_{edge.relationship_type}",
                weight=edge.weight,
                confidence=edge.confidence,
                temporal_context=edge.temporal_context,
            )

    def build_from_context_items(self, items: List[ContextItem]) -> None:
        """Build knowledge graph from context items."""
        for item in items:
            node = KnowledgeNode(
                id=item.id,
                type="context_item",
                content=item.content,
                layer=item.layer,
                relevance_score=item.relevance_score,
                confidence_score=item.confidence_score,
                metadata={
                    "title": item.title,
                    "tags": item.tags,
                    "references": item.references,
                    "created_at": item.created_at.isoformat(),
                },
            )
            self.add_node(node)

        # Extract relationships
        self._extract_relationships(items)

    def _extract_relationships(self, items: List[ContextItem]) -> None:
        """Extract relationships between context items."""
        for i, item1 in enumerate(items):
            for j, item2 in enumerate(items):
                if i != j:
                    # Check for semantic relationships
                    relationship = self._calculate_semantic_relationship(item1, item2)
                    if relationship:
                        edge = KnowledgeEdge(
                            source_id=item1.id,
                            target_id=item2.id,
                            relationship_type=relationship["type"],
                            weight=relationship["weight"],
                            confidence=relationship["confidence"],
                            bidirectional=True,
                        )
                        self.add_edge(edge)

        # Extract temporal relationships
        self._extract_temporal_relationships(items)

    def _calculate_semantic_relationship(
        self, item1: ContextItem, item2: ContextItem
    ) -> Optional[Dict[str, Any]]:
        """Calculate semantic relationship between two items."""
        # Common words overlap
        words1 = set(item1.content.lower().split())
        words2 = set(item2.content.lower().split())

        overlap = len(words1.intersection(words2))
        union = len(words1.union(words2))

        if overlap == 0:
            return None

        jaccard_similarity = overlap / union if union > 0 else 0

        if jaccard_similarity > 0.3:  # Significant overlap
            return {
                "type": "semantic_similarity",
                "weight": jaccard_similarity,
                "confidence": min(0.9, jaccard_similarity + 0.1),
            }

        # Tag-based relationships
        if item1.tags and item2.tags:
            tag_overlap = len(set(item1.tags).intersection(set(item2.tags)))
            if tag_overlap > 0:
                return {
                    "type": "tag_related",
                    "weight": tag_overlap / max(len(item1.tags), len(item2.tags)),
                    "confidence": 0.7,
                }

        return None

    def _extract_temporal_relationships(self, items: List[ContextItem]) -> None:
        """Extract temporal relationships between items."""
        # Sort by creation time
        sorted_items = sorted(items, key=lambda x: x.created_at)

        for i in range(len(sorted_items) - 1):
            current = sorted_items[i]
            next_item = sorted_items[i + 1]

            time_diff = (next_item.created_at - current.created_at).total_seconds()

            # If items are close in time, create temporal link
            if time_diff < 3600:  # Within an hour
                edge = KnowledgeEdge(
                    source_id=current.id,
                    target_id=next_item.id,
                    relationship_type="temporal_sequence",
                    weight=min(1.0, 3600 / max(time_diff, 1)),
                    confidence=0.8,
                    temporal_context="sequential",
                )
                self.add_edge(edge)

    def traverse_from_nodes(
        self,
        start_node_ids: List[str],
        max_depth: int = 3,
        relevance_threshold: float = 0.1,
        max_expansions: int = 50,
    ) -> List[KnowledgeNode]:
        """Traverse graph from starting nodes to find related knowledge."""
        if not start_node_ids:
            return []

        visited = set()
        queue = []
        expanded_nodes = []

        # Initialize priority queue with starting nodes
        for node_id in start_node_ids:
            if node_id in self.node_index:
                node = self.node_index[node_id]
                heapq.heappush(queue, (-node.relevance_score, 0, node_id))
                visited.add(node_id)

        while queue and len(expanded_nodes) < max_expansions:
            neg_relevance, depth, node_id = heapq.heappop(queue)

            if depth >= max_depth:
                continue

            current_node = self.node_index[node_id]
            if current_node.relevance_score >= relevance_threshold:
                expanded_nodes.append(current_node)

            # Explore neighbors
            for neighbor_id in self.graph.neighbors(node_id):
                if neighbor_id not in visited:
                    neighbor_node = self.node_index.get(neighbor_id)
                    if neighbor_node:
                        edge_data = self.graph.get_edge_data(node_id, neighbor_id)
                        if edge_data:
                            edge_weight = edge_data.get("weight", 1.0)
                            propagated_relevance = (
                                current_node.relevance_score
                                * edge_weight
                                * self.decay_factor**depth
                            )

                            if propagated_relevance >= relevance_threshold:
                                neighbor_node.relevance_score = max(
                                    neighbor_node.relevance_score, propagated_relevance
                                )
                                heapq.heappush(
                                    queue,
                                    (-propagated_relevance, depth + 1, neighbor_id),
                                )
                                visited.add(neighbor_id)

        return expanded_nodes

    def _evict_oldest_nodes(self) -> None:
        """Evict oldest nodes when graph is full."""
        if not self.node_index:
            return

        # Simple LRU eviction based on temporal weight decay
        current_time = datetime.now(timezone.utc)

        nodes_to_evict = []
        for node_id, node in self.node_index.items():
            age_hours = (
                current_time - node.metadata.get("created_at", current_time)
            ).total_seconds() / 3600
            decay_score = node.temporal_weight * (self.decay_factor**age_hours)

            if decay_score < 0.1:  # Very old nodes
                nodes_to_evict.append(node_id)

        # Remove oldest nodes
        for node_id in nodes_to_evict[: len(nodes_to_evict) // 4]:  # Remove 25%
            if node_id in self.node_index:
                del self.node_index[node_id]
                self.graph.remove_node(node_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            "total_nodes": len(self.node_index),
            "total_edges": self.graph.number_of_edges(),
            "node_types": self._count_node_types(),
            "edge_types": self._count_edge_types(),
            "average_degree": (
                sum(dict(self.graph.degree()).values()) / len(self.node_index)
                if self.node_index
                else 0
            ),
            "connected_components": nx.number_weakly_connected_components(self.graph),
        }

    def _count_node_types(self) -> Dict[str, int]:
        """Count nodes by type."""
        type_counts = defaultdict(int)
        for node in self.node_index.values():
            type_counts[node.type] += 1
        return dict(type_counts)

    def _count_edge_types(self) -> Dict[str, int]:
        """Count edges by relationship type."""
        type_counts = defaultdict(int)
        for _, _, edge_data in self.graph.edges(data=True):
            rel_type = edge_data.get("relationship_type", "unknown")
            type_counts[rel_type] += 1
        return dict(type_counts)


class GraphBasedExpander:
    """
    Graph-based knowledge expansion engine for enhanced context retrieval.

    Implements Stage 4 of the world-beating retrieval pipeline:
    - Dynamic knowledge graph construction
    - Entity relationship mining
    - Contextual path finding with relevance weighting
    - Temporal knowledge evolution
    """

    def __init__(
        self,
        max_graph_nodes: int = 5000,
        max_expansion_depth: int = 3,
        relevance_threshold: float = 0.15,
        temporal_decay_factor: float = 0.95,
    ):
        self.knowledge_graph = KnowledgeGraph(
            max_nodes=max_graph_nodes, decay_factor=temporal_decay_factor
        )
        self.max_expansion_depth = max_expansion_depth
        self.relevance_threshold = relevance_threshold

        # Performance tracking
        self._expansion_history = []
        self._metrics = {
            "total_expansions": 0,
            "average_expansion_time_ms": 0.0,
            "cache_hit_rate": 0.0,
            "graph_growth_rate": 0.0,
        }

        logger.info("🌐 Graph-Based Knowledge Expander initialized")

    async def expand_context(
        self,
        query: ContextQuery,
        initial_items: List[ContextItem],
        expansion_depth: Optional[int] = None,
        relevance_threshold: Optional[float] = None,
    ) -> GraphExpansionResult:
        """
        Expand context through graph-based knowledge exploration.

        Args:
            query: Original context query
            initial_items: Initial retrieved context items
            expansion_depth: Maximum depth for graph traversal
            relevance_threshold: Minimum relevance for expansion

        Returns:
            GraphExpansionResult with expanded context
        """
        start_time = asyncio.get_event_loop().time()

        depth = expansion_depth or self.max_expansion_depth
        threshold = relevance_threshold or self.relevance_threshold

        try:
            # Build knowledge graph from initial items
            self.knowledge_graph.build_from_context_items(initial_items)

            # Extract starting node IDs from initial items
            start_node_ids = [item.id for item in initial_items]

            # Perform graph traversal expansion
            expanded_nodes = self.knowledge_graph.traverse_from_nodes(
                start_node_ids=start_node_ids,
                max_depth=depth,
                relevance_threshold=threshold,
                max_expansions=query.limit * 2,
            )

            # Extract new relationships discovered during traversal
            new_relationships = self._extract_new_relationships(expanded_nodes)

            # Convert expanded nodes back to context items
            expanded_items = await self._convert_nodes_to_items(expanded_nodes, query)

            # Filter and rank expanded items
            filtered_items = self._filter_and_rank_expanded_items(
                expanded_items, query, initial_items
            )

            # Calculate expansion confidence
            expansion_confidence = self._calculate_expansion_confidence(
                initial_items, filtered_items
            )

            execution_time = int((asyncio.get_event_loop().time() - start_time) * 1000)

            result = GraphExpansionResult(
                original_items=initial_items,
                expanded_items=filtered_items,
                new_relationships=new_relationships,
                traversal_depth=depth,
                relevance_threshold=threshold,
                expansion_confidence=expansion_confidence,
                execution_time_ms=execution_time,
            )

            # Update metrics
            self._update_metrics(result)
            self._expansion_history.append(result)

            logger.info(
                f"Graph expansion completed: {len(filtered_items)} items expanded "
                f"from {len(initial_items)} in {execution_time}ms"
            )

            return result

        except Exception as e:
            logger.error(f"Graph expansion failed: {e}")
            execution_time = int((asyncio.get_event_loop().time() - start_time) * 1000)

            return GraphExpansionResult(
                original_items=initial_items,
                expanded_items=[],
                new_relationships=[],
                traversal_depth=0,
                relevance_threshold=threshold,
                expansion_confidence=0.0,
                execution_time_ms=execution_time,
            )

    def _extract_new_relationships(
        self, nodes: List[KnowledgeNode]
    ) -> List[KnowledgeEdge]:
        """Extract new relationships discovered during expansion."""
        new_relationships = []

        for node in nodes:
            # Get all edges for this node
            for source, target, edge_data in self.knowledge_graph.graph.in_edges(
                node.id, data=True
            ):
                if edge_data:
                    edge = KnowledgeEdge(
                        source_id=source,
                        target_id=target,
                        relationship_type=edge_data.get("relationship_type", "unknown"),
                        weight=edge_data.get("weight", 1.0),
                        confidence=edge_data.get("confidence", 1.0),
                    )
                    new_relationships.append(edge)

        return new_relationships

    async def _convert_nodes_to_items(
        self, nodes: List[KnowledgeNode], query: ContextQuery
    ) -> List[ContextItem]:
        """Convert knowledge nodes back to context items."""
        items = []

        for node in nodes:
            # Skip if node is already in the original items
            if any(item.id == node.id for item in query.layers):
                continue

            # Create context item from node
            item = ContextItem(
                id=node.id,
                title=f"Expanded: {node.content[:50]}...",
                summary=node.content,
                content=node.content,
                layer=node.layer,
                relevance_score=node.relevance_score,
                confidence_score=node.confidence_score,
                project_id=query.project_id,
                tenant_id=query.tenant_id,
                created_by="graph_expansion",
                tags=["expanded", "graph_traversal"],
                metadata={
                    "expansion_source": "graph_traversal",
                    "traversal_confidence": node.confidence_score,
                    "temporal_weight": node.temporal_weight,
                },
            )
            items.append(item)

        return items

    def _filter_and_rank_expanded_items(
        self,
        expanded_items: List[ContextItem],
        query: ContextQuery,
        original_items: List[ContextItem],
    ) -> List[ContextItem]:
        """Filter and rank expanded items based on query criteria."""
        # Remove duplicates with original items
        original_ids = {item.id for item in original_items}
        filtered = [item for item in expanded_items if item.id not in original_ids]

        # Apply query filters
        if query.min_relevance > 0:
            filtered = [
                item for item in filtered if item.relevance_score >= query.min_relevance
            ]

        if query.min_confidence > 0:
            filtered = [
                item
                for item in filtered
                if item.confidence_score >= query.min_confidence
            ]

        # Sort by relevance score
        filtered.sort(key=lambda x: x.relevance_score, reverse=True)

        # Apply limit
        return filtered[: query.limit]

    def _calculate_expansion_confidence(
        self, original_items: List[ContextItem], expanded_items: List[ContextItem]
    ) -> float:
        """Calculate confidence score for the expansion result."""
        if not original_items:
            return 0.0

        # Base confidence on expansion ratio and quality
        expansion_ratio = len(expanded_items) / len(original_items)
        average_relevance = sum(item.relevance_score for item in expanded_items) / max(
            len(expanded_items), 1
        )

        # Confidence increases with moderate expansion and high relevance
        ratio_confidence = min(
            1.0, expansion_ratio / 2.0
        )  # Optimal around 2x expansion
        relevance_confidence = min(
            1.0, average_relevance * 2.0
        )  # Scale relevance to confidence

        return (ratio_confidence + relevance_confidence) / 2.0

    def _update_metrics(self, result: GraphExpansionResult) -> None:
        """Update internal metrics."""
        self._metrics["total_expansions"] += 1

        # Update average expansion time
        current_avg = self._metrics["average_expansion_time_ms"]
        total = self._metrics["total_expansions"]
        self._metrics["average_expansion_time_ms"] = (
            (current_avg * (total - 1)) + result.execution_time_ms
        ) / total

    def get_metrics(self) -> Dict[str, Any]:
        """Get expansion metrics."""
        metrics = self._metrics.copy()
        metrics.update(self.knowledge_graph.get_statistics())
        metrics["expansion_history_size"] = len(self._expansion_history)

        return metrics

    def optimize_graph(self) -> None:
        """Optimize knowledge graph structure."""
        # Remove low-confidence edges
        edges_to_remove = []
        for source, target, edge_data in self.knowledge_graph.graph.edges(data=True):
            if edge_data.get("confidence", 0) < 0.3:
                edges_to_remove.append((source, target))

        for source, target in edges_to_remove:
            self.knowledge_graph.graph.remove_edge(source, target)

        # Update temporal weights
        current_time = datetime.now(timezone.utc)
        for node in self.knowledge_graph.node_index.values():
            age_hours = (
                current_time - node.metadata.get("created_at", current_time)
            ).total_seconds() / 3600
            node.temporal_weight *= self.knowledge_graph.decay_factor**age_hours

        logger.info("Knowledge graph optimized")

    async def find_related_entities(
        self, entity_name: str, max_related: int = 10
    ) -> List[Tuple[str, float]]:
        """Find related entities through graph traversal."""
        # This would integrate with external knowledge bases
        # For now, return empty list
        return []

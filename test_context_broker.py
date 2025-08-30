#!/usr/bin/env python3
"""
Test script for ContextBroker functionality
"""
import asyncio
import os
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from typing import Dict, cast

from pinak.context.broker.broker import ContextBroker
from pinak.context.core.models import (ContextItem, ContextLayer,
                                       ContextPriority, ContextQuery,
                                       ContextResponse, IContextStore,
                                       SecurityClassification)


class MockContextStore:
    """Mock context store for testing"""

    def __init__(self, layer: ContextLayer):
        self.layer = layer

    async def retrieve(self, query):
        """Mock retrieve method"""
        return ContextResponse(
            items=[
                ContextItem(
                    id=f"test-{self.layer.value}-1",
                    title=f"Test {self.layer.value} Item",
                    summary=f"This is a test item from {self.layer.value} layer",
                    content=f"This is a test item from {self.layer.value} layer",
                    layer=self.layer,
                    project_id="test-project",
                    tenant_id="test-tenant",
                    created_by="test-user",
                    classification=SecurityClassification.PUBLIC,
                    priority=ContextPriority.MEDIUM,
                    tags=["test"],
                    relevance_score=0.8,
                    confidence_score=0.9,
                )
            ]
        )

    async def store(self, item):
        """Mock store method"""
        return True

    async def delete(self, item_id: str, project_id: str):
        """Mock delete method"""
        return True

    async def update(self, item):
        """Mock update method"""
        return True

    async def search_similar(self, content: str, limit: int = 10):
        """Mock semantic search"""
        return [
            ContextItem(
                id=f"semantic-{self.layer.value}-1",
                title=f"Semantic {self.layer.value} Item",
                summary=f"Semantic search result from {self.layer.value} layer",
                content=f"Semantic search result from {self.layer.value} layer",
                layer=self.layer,
                project_id="test-project",
                tenant_id="test-tenant",
                created_by="test-user",
                classification=SecurityClassification.PUBLIC,
                priority=ContextPriority.MEDIUM,
                tags=["semantic"],
                relevance_score=0.9,
                confidence_score=0.95,
            )
        ]


async def test_context_broker():
    """Test the ContextBroker functionality"""
    print("Testing ContextBroker...")

    # Create mock stores
    stores = {
        ContextLayer.SEMANTIC: MockContextStore(ContextLayer.SEMANTIC),
        ContextLayer.EPISODIC: MockContextStore(ContextLayer.EPISODIC),
    }

    # Create broker
    broker = ContextBroker(stores=cast(Dict[ContextLayer, IContextStore], stores))

    # Create test query
    query = ContextQuery(
        query="test query",
        project_id="test-project",
        tenant_id="test-tenant",
        user_id="test-user",
        layers=[ContextLayer.SEMANTIC, ContextLayer.EPISODIC],
        limit=5,
        user_clearance=SecurityClassification.PUBLIC,
        semantic_search=True,
    )

    # Execute query
    response = await broker.get_context(query)

    print(f"Query executed successfully!")
    print(f"Response ID: {response.query_id}")
    print(f"Returned results: {response.returned_results}")
    print(f"Total results: {response.total_results}")
    print(f"Execution time: {response.execution_time_ms}ms")

    # Check metrics
    metrics = broker.get_metrics()
    print(f"Metrics: {metrics}")

    # Verify metrics are correct types
    assert isinstance(
        metrics["avg_execution_time_ms"], float
    ), "avg_execution_time_ms should be float"
    assert isinstance(metrics["total_queries"], int), "total_queries should be int"

    print("✅ All tests passed!")


if __name__ == "__main__":
    asyncio.run(test_context_broker())

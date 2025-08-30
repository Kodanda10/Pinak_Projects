"""
Comprehensive TDD tests for Governance-Integrated Nudge Engine.

Following TDD principles: Write tests first, then implement features.
Tests cover behavioral detection, nudge generation, and governance integration.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Any, Optional
import time
from datetime import datetime, timezone

from pinak.context.nudge.engine import NudgeEngine
from pinak.context.nudge.models import (
    BehavioralPattern, NudgeRequest, NudgeResponse,
    NudgeType, NudgeChannel, GovernancePolicy
)
from pinak.context.nudge.delivery import NudgeDelivery
from pinak.context.nudge.store import NudgeStore
from pinak.context.core.models import SecurityClassification


@pytest.fixture
def governance_policy():
    """Create test governance policy."""
    return GovernancePolicy(
        id="test_policy_1",
        name="Test Security Policy",
        description="Test policy for security compliance",
        rules={
            "max_api_calls_per_minute": 100,
            "require_encryption": True,
            "audit_log_retention_days": 90
        },
        classification=SecurityClassification.CONFIDENTIAL,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )


@pytest.fixture
def behavioral_pattern():
    """Create test behavioral pattern."""
    return BehavioralPattern(
        user_id="test_user_123",
        pattern_type="api_abuse",
        severity="medium",
        confidence=0.85,
        indicators={
            "api_calls_per_minute": 150,
            "failed_auth_attempts": 3,
            "suspicious_ips": ["192.168.1.100"]
        },
        timestamp=datetime.now(timezone.utc),
        context={
            "endpoint": "/api/v1/memory/search",
            "user_agent": "TestClient/1.0"
        }
    )


@pytest.fixture
def nudge_engine(governance_policy):
    """Create NudgeEngine instance for testing."""
    return NudgeEngine(
        policies=[governance_policy],
        learning_enabled=True,
        adaptive_thresholds=True
    )


@pytest.fixture
def nudge_store():
    """Create NudgeStore instance for testing."""
    return NudgeStore()


@pytest.fixture
def nudge_delivery():
    """Create NudgeDelivery instance for testing."""
    return NudgeDelivery()


@pytest.mark.tdd
@pytest.mark.governance
class TestNudgeEngine:
    """Test the core nudge engine functionality."""

    @pytest.mark.asyncio
    async def test_behavioral_pattern_detection(self, nudge_engine, behavioral_pattern):
        """Test detection of behavioral patterns."""
        # Setup
        patterns = [behavioral_pattern]

        # Execute
        detected = await nudge_engine.detect_patterns(patterns)

        # Assert
        assert len(detected) > 0
        assert detected[0].user_id == behavioral_pattern.user_id
        assert detected[0].severity == behavioral_pattern.severity

    @pytest.mark.asyncio
    async def test_nudge_generation(self, nudge_engine, behavioral_pattern):
        """Test generation of contextual nudges."""
        # Setup
        request = NudgeRequest(
            user_id=behavioral_pattern.user_id,
            pattern=behavioral_pattern,
            context={
                "current_action": "api_call",
                "risk_level": "medium"
            }
        )

        # Execute
        response = await nudge_engine.generate_nudge(request)

        # Assert
        assert isinstance(response, NudgeResponse)
        assert response.user_id == request.user_id
        assert response.nudge_type in [NudgeType.WARNING, NudgeType.SUGGESTION, NudgeType.BLOCK]
        assert len(response.channels) > 0

    @pytest.mark.asyncio
    async def test_governance_policy_evaluation(self, nudge_engine, governance_policy):
        """Test governance policy evaluation."""
        # Setup
        context = {
            "api_calls_per_minute": 120,
            "encryption_enabled": False,
            "audit_logs_present": True
        }

        # Execute
        violations = await nudge_engine.evaluate_policy_compliance(
            governance_policy, context
        )

        # Assert
        assert isinstance(violations, list)
        # Should detect the rate limit violation
        rate_violations = [v for v in violations if "api_calls" in v.get("rule", "")]
        assert len(rate_violations) > 0

    @pytest.mark.asyncio
    async def test_adaptive_thresholds(self, nudge_engine):
        """Test adaptive threshold adjustment."""
        # Setup initial thresholds
        initial_thresholds = nudge_engine.get_adaptive_thresholds()

        # Simulate learning from feedback
        feedback_data = {
            "true_positives": 10,
            "false_positives": 2,
            "true_negatives": 15,
            "false_negatives": 1
        }

        # Execute learning
        await nudge_engine.learn_from_feedback(feedback_data)

        # Assert thresholds were adjusted
        updated_thresholds = nudge_engine.get_adaptive_thresholds()
        assert updated_thresholds != initial_thresholds

    @pytest.mark.asyncio
    async def test_multi_channel_delivery(self, nudge_delivery):
        """Test multi-channel nudge delivery."""
        # Setup
        nudge = {
            "id": "test_nudge_1",
            "type": NudgeType.WARNING,
            "message": "Security policy violation detected",
            "channels": [NudgeChannel.IDE, NudgeChannel.CLI, NudgeChannel.SYSTEM]
        }

        # Execute
        delivery_result = await nudge_delivery.deliver_nudge(nudge)

        # Assert
        assert delivery_result["delivered_channels"] >= 0
        assert "errors" in delivery_result

    @pytest.mark.asyncio
    async def test_nudge_effectiveness_tracking(self, nudge_store):
        """Test tracking of nudge effectiveness."""
        # Setup
        nudge_id = "test_nudge_1"
        user_response = {
            "nudge_id": nudge_id,
            "user_action": "acknowledged",
            "response_time_seconds": 30,
            "compliance_improved": True
        }

        # Execute
        await nudge_store.record_nudge_response(user_response)

        # Assert
        effectiveness = await nudge_store.get_nudge_effectiveness(nudge_id)
        assert effectiveness["total_responses"] >= 1
        assert effectiveness["acknowledged_rate"] >= 0

    @pytest.mark.asyncio
    async def test_real_time_monitoring(self, nudge_engine):
        """Test real-time behavioral monitoring."""
        # Setup monitoring stream
        monitoring_config = {
            "enabled": True,
            "alert_threshold": 0.8,
            "monitoring_window_minutes": 5
        }

        # Start monitoring
        monitor_task = asyncio.create_task(
            nudge_engine.monitor_behavior_stream(monitoring_config)
        )

        # Simulate behavioral events
        events = [
            {
                "user_id": "test_user_1",
                "event_type": "api_call",
                "severity": 0.3,
                "timestamp": datetime.now(timezone.utc)
            },
            {
                "user_id": "test_user_1",
                "event_type": "auth_failure",
                "severity": 0.9,
                "timestamp": datetime.now(timezone.utc)
            }
        ]

        # Send events to monitoring
        for event in events:
            await nudge_engine.process_behavioral_event(event)

        # Stop monitoring
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass

        # Assert monitoring detected high-severity event
        alerts = nudge_engine.get_recent_alerts()
        assert len(alerts) > 0


@pytest.mark.tdd
@pytest.mark.governance
class TestBehavioralIntelligence:
    """Test behavioral intelligence and pattern recognition."""

    @pytest.mark.asyncio
    async def test_pattern_recognition_ml(self, nudge_engine):
        """Test ML-powered pattern recognition."""
        # Setup diverse behavioral data
        training_data = [
            {
                "features": [100, 5, 0, 80],  # api_calls, failures, blocks, success_rate
                "label": "normal"
            },
            {
                "features": [200, 15, 2, 60],
                "label": "suspicious"
            },
            {
                "features": [500, 50, 10, 30],
                "label": "malicious"
            }
        ]

        # Train pattern recognition
        await nudge_engine.train_pattern_recognition(training_data)

        # Test prediction
        test_features = [150, 8, 1, 70]
        prediction = await nudge_engine.predict_pattern(test_features)

        # Assert
        assert "confidence" in prediction
        assert "pattern_type" in prediction
        assert 0.0 <= prediction["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_anomaly_detection(self, nudge_engine):
        """Test anomaly detection in behavioral patterns."""
        # Setup baseline behavior
        baseline_data = [
            {"metric": "api_calls_per_minute", "value": 50, "timestamp": datetime.now(timezone.utc)},
            {"metric": "api_calls_per_minute", "value": 55, "timestamp": datetime.now(timezone.utc)},
            {"metric": "api_calls_per_minute", "value": 48, "timestamp": datetime.now(timezone.utc)},
        ]

        # Establish baseline
        await nudge_engine.establish_behavioral_baseline(baseline_data)

        # Test anomaly detection
        anomalous_data = [
            {"metric": "api_calls_per_minute", "value": 200, "timestamp": datetime.now(timezone.utc)},
            {"metric": "api_calls_per_minute", "value": 250, "timestamp": datetime.now(timezone.utc)},
        ]

        anomalies = await nudge_engine.detect_anomalies(anomalous_data)

        # Assert
        assert len(anomalies) > 0
        assert all(anomaly["is_anomaly"] for anomaly in anomalies)
        assert all(anomaly["confidence"] > 0.5 for anomaly in anomalies)

    @pytest.mark.asyncio
    async def test_contextual_nudge_generation(self, nudge_engine):
        """Test context-aware nudge generation."""
        # Setup contextual scenarios
        scenarios = [
            {
                "context": {
                    "time_of_day": "business_hours",
                    "user_role": "developer",
                    "risk_level": "low",
                    "previous_violations": []
                },
                "expected_nudge_type": NudgeType.SUGGESTION
            },
            {
                "context": {
                    "time_of_day": "after_hours",
                    "user_role": "admin",
                    "risk_level": "high",
                    "previous_violations": ["policy_violation_1", "policy_violation_2"]
                },
                "expected_nudge_type": NudgeType.WARNING
            }
        ]

        for scenario in scenarios:
            # Generate contextual nudge
            nudge_request = NudgeRequest(
                user_id="test_user",
                pattern=BehavioralPattern(
                    user_id="test_user",
                    pattern_type="test_pattern",
                    severity="medium",
                    confidence=0.8,
                    indicators={},
                    timestamp=datetime.now(timezone.utc)
                ),
                context=scenario["context"]
            )

            nudge_response = await nudge_engine.generate_nudge(nudge_request)

            # Assert context influenced nudge type
            assert nudge_response.nudge_type == scenario["expected_nudge_type"]


@pytest.mark.tdd
@pytest.mark.governance
class TestGovernanceIntegration:
    """Test integration with governance frameworks."""

    @pytest.mark.asyncio
    async def test_parlant_policy_integration(self, nudge_engine):
        """Test integration with Parlant governance policies."""
        # Mock Parlant client
        with patch('pinak.context.nudge.engine.ParlantClient') as mock_parlant:
            mock_client = Mock()
            mock_client.get_policies = AsyncMock(return_value=[
                {
                    "id": "parlant_policy_1",
                    "name": "Parlant Security Policy",
                    "rules": {"encryption_required": True},
                    "priority": "high"
                }
            ])
            mock_parlant.return_value = mock_client

            # Execute policy sync
            await nudge_engine.sync_parlant_policies()

            # Assert policies were integrated
            policies = nudge_engine.get_active_policies()
            parlant_policies = [p for p in policies if "parlant" in p.id.lower()]
            assert len(parlant_policies) > 0

    @pytest.mark.asyncio
    async def test_compliance_monitoring(self, nudge_engine):
        """Test continuous compliance monitoring."""
        # Setup compliance rules
        compliance_config = {
            "monitoring_enabled": True,
            "check_interval_seconds": 60,
            "alert_thresholds": {
                "critical": 0.9,
                "warning": 0.7
            }
        }

        # Start compliance monitoring
        monitoring_task = asyncio.create_task(
            nudge_engine.monitor_compliance(compliance_config)
        )

        # Simulate compliance events
        events = [
            {
                "event_type": "policy_violation",
                "severity": 0.8,
                "policy_id": "test_policy_1",
                "timestamp": datetime.now(timezone.utc)
            },
            {
                "event_type": "compliance_check",
                "severity": 0.2,
                "policy_id": "test_policy_1",
                "timestamp": datetime.now(timezone.utc)
            }
        ]

        # Process compliance events
        for event in events:
            await nudge_engine.process_compliance_event(event)

        # Stop monitoring
        monitoring_task.cancel()
        try:
            await monitoring_task
        except asyncio.CancelledError:
            pass

        # Assert compliance status
        status = nudge_engine.get_compliance_status()
        assert "overall_score" in status
        assert "violations" in status
        assert 0.0 <= status["overall_score"] <= 1.0

    @pytest.mark.asyncio
    async def test_audit_trail_integration(self, nudge_engine, nudge_store):
        """Test audit trail integration for governance."""
        # Setup audit configuration
        audit_config = {
            "enabled": True,
            "retention_days": 90,
            "detailed_logging": True
        }

        # Configure audit
        nudge_engine.configure_audit(audit_config)

        # Simulate auditable events
        events = [
            {
                "event_type": "nudge_generated",
                "nudge_id": "audit_nudge_1",
                "user_id": "test_user",
                "timestamp": datetime.now(timezone.utc),
                "details": {"reason": "policy_violation"}
            },
            {
                "event_type": "user_response",
                "nudge_id": "audit_nudge_1",
                "user_action": "acknowledged",
                "timestamp": datetime.now(timezone.utc)
            }
        ]

        # Process auditable events
        for event in events:
            await nudge_engine.record_audit_event(event)

        # Assert audit trail
        audit_trail = await nudge_store.get_audit_trail("test_user")
        assert len(audit_trail) >= len(events)

        # Verify audit completeness
        nudge_events = [e for e in audit_trail if e["event_type"] == "nudge_generated"]
        response_events = [e for e in audit_trail if e["event_type"] == "user_response"]
        assert len(nudge_events) > 0
        assert len(response_events) > 0


@pytest.mark.tdd
@pytest.mark.governance
@pytest.mark.slow
class TestGovernanceStress:
    """Stress tests for governance nudge engine."""

    @pytest.mark.asyncio
    async def test_high_frequency_events(self, nudge_engine):
        """Test handling of high-frequency behavioral events."""
        # Generate high volume of events
        event_count = 1000
        events = []

        base_time = datetime.now(timezone.utc)
        for i in range(event_count):
            event = {
                "user_id": f"user_{i % 10}",  # 10 different users
                "event_type": "api_call",
                "severity": 0.1 + (i % 9) * 0.1,  # Varying severity
                "timestamp": base_time,
                "metadata": {"request_id": f"req_{i}"}
            }
            events.append(event)

        # Process events concurrently
        tasks = [nudge_engine.process_behavioral_event(event) for event in events]
        await asyncio.gather(*tasks)

        # Assert system handled load
        metrics = nudge_engine.get_performance_metrics()
        assert metrics["events_processed"] >= event_count
        assert metrics["avg_processing_time_ms"] < 100  # Reasonable processing time

    @pytest.mark.asyncio
    async def test_concurrent_nudge_generation(self, nudge_engine):
        """Test concurrent nudge generation."""
        # Generate multiple concurrent nudge requests
        request_count = 50
        requests = []

        for i in range(request_count):
            pattern = BehavioralPattern(
                user_id=f"concurrent_user_{i}",
                pattern_type="test_pattern",
                severity="medium",
                confidence=0.8,
                indicators={"test_indicator": i},
                timestamp=datetime.now(timezone.utc)
            )

            request = NudgeRequest(
                user_id=pattern.user_id,
                pattern=pattern,
                context={"concurrent_test": True}
            )
            requests.append(request)

        # Process concurrently
        start_time = time.time()
        tasks = [nudge_engine.generate_nudge(request) for request in requests]
        responses = await asyncio.gather(*tasks)
        end_time = time.time()

        # Assert concurrent processing
        assert len(responses) == request_count
        assert all(isinstance(r, NudgeResponse) for r in responses)

        # Assert reasonable performance
        total_time = end_time - start_time
        avg_time_per_request = total_time / request_count
        assert avg_time_per_request < 0.1  # Less than 100ms per request

    @pytest.mark.asyncio
    async def test_memory_efficiency(self, nudge_engine):
        """Test memory efficiency under sustained load."""
        # Simulate sustained monitoring load
        monitoring_duration_seconds = 30
        event_interval_ms = 100  # 10 events per second

        start_time = time.time()
        event_count = 0

        while time.time() - start_time < monitoring_duration_seconds:
            # Generate and process event
            event = {
                "user_id": "memory_test_user",
                "event_type": "memory_test_event",
                "severity": 0.5,
                "timestamp": datetime.now(timezone.utc),
                "metadata": {"event_number": event_count}
            }

            await nudge_engine.process_behavioral_event(event)
            event_count += 1

            # Small delay to control event rate
            await asyncio.sleep(event_interval_ms / 1000)

        # Assert memory efficiency
        metrics = nudge_engine.get_memory_metrics()
        assert metrics["memory_growth_mb"] < 50  # Less than 50MB growth
        assert metrics["event_processing_rate"] > 8  # At least 8 events/second


@pytest.mark.tdd
@pytest.mark.governance
class TestGovernanceErrorHandling:
    """Test error handling in governance nudge engine."""

    @pytest.mark.asyncio
    async def test_policy_loading_failure(self, nudge_engine):
        """Test handling of policy loading failures."""
        # Mock policy loading failure
        with patch.object(nudge_engine, '_load_policies_from_store') as mock_load:
            mock_load.side_effect = Exception("Policy store unavailable")

            # Attempt to reload policies
            with pytest.raises(Exception):
                await nudge_engine.reload_policies()

    @pytest.mark.asyncio
    async def test_delivery_failure_recovery(self, nudge_delivery):
        """Test recovery from nudge delivery failures."""
        # Setup failing delivery channels
        failing_nudge = {
            "id": "failing_nudge",
            "type": NudgeType.WARNING,
            "message": "Test failure recovery",
            "channels": [NudgeChannel.IDE, NudgeChannel.CLI]
        }

        # Mock delivery failures
        with patch.object(nudge_delivery, '_deliver_to_channel') as mock_deliver:
            mock_deliver.side_effect = [Exception("Channel down"), None]  # First fails, second succeeds

            # Attempt delivery
            result = await nudge_delivery.deliver_nudge(failing_nudge)

            # Assert partial success recovery
            assert result["delivered_channels"] >= 1
            assert len(result["errors"]) >= 1
            assert "recovery_attempted" in result

    @pytest.mark.asyncio
    async def test_corrupted_event_handling(self, nudge_engine):
        """Test handling of corrupted behavioral events."""
        # Setup corrupted events
        corrupted_events = [
            {"incomplete": "data"},  # Missing required fields
            {"user_id": None, "event_type": "test"},  # Null user_id
            {"user_id": "test", "severity": "invalid"},  # Invalid severity type
            {},  # Completely empty
        ]

        # Process corrupted events
        for event in corrupted_events:
            # Should handle gracefully without crashing
            try:
                await nudge_engine.process_behavioral_event(event)
            except Exception as e:
                # Should be handled gracefully
                assert "corrupted" in str(e).lower() or "invalid" in str(e).lower()

        # Assert system remains stable
        health = await nudge_engine.health_check()
        assert health["status"] == "healthy"


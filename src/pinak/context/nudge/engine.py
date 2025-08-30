# FANG-Level Nudge Engine - Core Implementation
"""
Enterprise-grade Nudge Engine implementation with intelligent triggers,
personalization, and proactive context delivery.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timezone, timedelta
import asyncio
import logging
import re
from collections import defaultdict

from .models import (
    Nudge, NudgeTemplate, NudgeCondition, NudgeTrigger,
    NudgeType, NudgePriority, NudgeDeliveryResult,
    INudgeStore, INudgeDelivery
)
from ..core.models import ContextItem, ContextQuery, ContextLayer, SecurityClassification
from ..broker.broker import ContextBroker

logger = logging.getLogger(__name__)


class NudgeEngine:
    """
    FANG-level Nudge Engine for proactive context delivery.

    Features:
    - Intelligent trigger detection
    - Personalized nudge generation
    - Multi-channel delivery
    - Performance monitoring
    - A/B testing support
    """

    def __init__(
        self,
        store: INudgeStore,
        delivery_channels: List[INudgeDelivery],
        context_broker: ContextBroker,
        max_concurrent_nudges: int = 10,
        nudge_expiry_hours: int = 24,
    ):
        self.store = store
        self.delivery_channels = delivery_channels
        self.context_broker = context_broker

        # Configuration
        self.max_concurrent = max_concurrent_nudges
        self.nudge_expiry_hours = nudge_expiry_hours

        # Runtime state
        self._active_nudges: Dict[str, Nudge] = {}
        self._user_context_cache: Dict[str, List[ContextItem]] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent_nudges)

        # Performance metrics
        self._metrics = {
            'total_nudges_generated': 0,
            'nudges_delivered': 0,
            'nudges_acknowledged': 0,
            'delivery_failures': 0,
            'avg_generation_time_ms': 0.0,
            'avg_delivery_time_ms': 0.0,
        }

        logger.info(f"NudgeEngine initialized with {len(delivery_channels)} delivery channels")

    async def generate_nudges(
        self,
        user_id: str,
        project_id: str,
        tenant_id: str,
        context_items: Optional[List[ContextItem]] = None,
        trigger_events: Optional[List[Dict[str, Any]]] = None
    ) -> List[Nudge]:
        """
        Generate personalized nudges based on user context and trigger events.

        Implements intelligent analysis of:
        - Recent context patterns
        - User activity history
        - Time-based triggers
        - Context gaps and opportunities
        """
        start_time = asyncio.get_event_loop().time()

        try:
            # Get user context if not provided
            if context_items is None:
                context_items = await self._get_user_context(user_id, project_id)

            # Get active templates
            templates = await self.store.get_nudge_templates(active_only=True)

            # Analyze context for nudge opportunities
            nudge_opportunities = await self._analyze_context_for_nudges(
                user_id, project_id, tenant_id, context_items, trigger_events
            )

            # Generate nudges from opportunities
            nudges = []
            for opportunity in nudge_opportunities:
                template = self._find_best_template(opportunity, templates)
                if template:
                    nudge = await self._generate_nudge_from_template(
                        template, opportunity, user_id, project_id, tenant_id
                    )
                    if nudge:
                        nudges.append(nudge)

            # Update metrics
            generation_time = int((asyncio.get_event_loop().time() - start_time) * 1000)
            self._update_generation_metrics(len(nudges), generation_time)

            logger.info(f"Generated {len(nudges)} nudges for user {user_id}")
            return nudges

        except Exception as e:
            logger.error(f"Nudge generation failed for user {user_id}: {e}")
            return []

    async def deliver_nudges(self, nudges: List[Nudge]) -> List[NudgeDeliveryResult]:
        """
        Deliver nudges through available channels with intelligent routing.
        """
        if not nudges:
            return []

        results = []
        delivery_tasks = []

        # Group nudges by priority for efficient delivery
        priority_groups = self._group_nudges_by_priority(nudges)

        for priority in [NudgePriority.CRITICAL, NudgePriority.HIGH, NudgePriority.MEDIUM, NudgePriority.LOW]:
            if priority in priority_groups:
                group_results = await self._deliver_priority_group(priority_groups[priority])
                results.extend(group_results)

        return results

    async def process_user_activity(
        self,
        user_id: str,
        project_id: str,
        tenant_id: str,
        activity_type: str,
        activity_data: Dict[str, Any]
    ) -> List[Nudge]:
        """
        Process user activity and generate relevant nudges.
        """
        # Update context cache
        await self._update_user_context_cache(user_id, activity_data)

        # Generate activity-triggered nudges
        trigger_events = [{
            'type': 'user_activity',
            'activity_type': activity_type,
            'data': activity_data,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }]

        return await self.generate_nudges(
            user_id, project_id, tenant_id,
            trigger_events=trigger_events
        )

    async def acknowledge_nudge(self, nudge_id: str, user_response: Optional[str] = None) -> bool:
        """
        Mark a nudge as acknowledged by the user.
        """
        try:
            success = await self.store.update_nudge_status(nudge_id, "acknowledged")

            if success and nudge_id in self._active_nudges:
                nudge = self._active_nudges[nudge_id]
                nudge.acknowledged_at = datetime.now(timezone.utc)
                nudge.user_engagement_score = 1.0

                # Update template success metrics
                await self._update_template_metrics(nudge.template_id, success=True)

            return success

        except Exception as e:
            logger.error(f"Failed to acknowledge nudge {nudge_id}: {e}")
            return False

    async def _analyze_context_for_nudges(
        self,
        user_id: str,
        project_id: str,
        tenant_id: str,
        context_items: List[ContextItem],
        trigger_events: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze user context to identify nudge opportunities.
        """
        opportunities = []

        # Analyze context patterns
        context_patterns = self._analyze_context_patterns(context_items)

        # Check for missing context
        missing_context = await self._identify_missing_context(user_id, project_id, context_items)
        if missing_context:
            opportunities.append({
                'type': NudgeType.MISSING_CONTEXT,
                'reason': 'missing_context',
                'data': missing_context,
                'confidence': 0.8
            })

        # Check for timing optimizations
        timing_opportunities = self._analyze_timing_patterns(context_items)
        if timing_opportunities:
            opportunities.append({
                'type': NudgeType.TIMING_OPTIMIZATION,
                'reason': 'timing_pattern',
                'data': timing_opportunities,
                'confidence': 0.7
            })

        # Check for relevant updates
        if trigger_events:
            update_opportunities = self._analyze_trigger_events(trigger_events, context_items)
            opportunities.extend(update_opportunities)

        # Check for security alerts
        security_alerts = self._analyze_security_context(context_items)
        if security_alerts:
            opportunities.append({
                'type': NudgeType.SECURITY_ALERT,
                'reason': 'security_concern',
                'data': security_alerts,
                'confidence': 0.9
            })

        return opportunities

    def _analyze_context_patterns(self, context_items: List[ContextItem]) -> Dict[str, Any]:
        """
        Analyze patterns in user's context items.
        """
        patterns = {
            'frequent_layers': defaultdict(int),
            'recent_activity': [],
            'content_themes': set(),
            'temporal_patterns': defaultdict(int)
        }

        for item in context_items:
            patterns['frequent_layers'][item.layer.value] += 1

            # Extract content themes (simple keyword analysis)
            content_words = set(item.content.lower().split())
            patterns['content_themes'].update(content_words)

            # Analyze temporal patterns
            hour = item.created_at.hour
            patterns['temporal_patterns'][hour] += 1

        return dict(patterns)

    async def _identify_missing_context(
        self,
        user_id: str,
        project_id: str,
        context_items: List[ContextItem]
    ) -> Optional[Dict[str, Any]]:
        """
        Identify gaps in user's context that could benefit from nudges.
        """
        # Check for recent activity without documentation
        recent_items = [item for item in context_items
                       if (datetime.now(timezone.utc) - item.created_at).days < 7]

        if len(recent_items) < 3:
            return {
                'gap_type': 'low_activity',
                'suggestion': 'Consider documenting recent work or decisions',
                'confidence': 0.6
            }

        # Check for unbalanced layer usage
        layer_counts = defaultdict(int)
        for item in recent_items:
            layer_counts[item.layer.value] += 1

        total_items = len(recent_items)
        if any(count / total_items < 0.1 for count in layer_counts.values()):
            return {
                'gap_type': 'unbalanced_layers',
                'suggestion': 'Consider exploring other context layers',
                'confidence': 0.7
            }

        return None

    def _analyze_timing_patterns(self, context_items: List[ContextItem]) -> Optional[Dict[str, Any]]:
        """
        Analyze timing patterns to suggest optimizations.
        """
        if len(context_items) < 5:
            return None

        # Look for patterns in creation times
        creation_hours = [item.created_at.hour for item in context_items[-10:]]

        # Check for peak productivity hours
        hour_counts = defaultdict(int)
        for hour in creation_hours:
            hour_counts[hour] += 1

        peak_hour = max(hour_counts.keys(), key=lambda h: hour_counts[h])

        if hour_counts[peak_hour] >= 3:
            return {
                'peak_hour': peak_hour,
                'suggestion': f"You're most active around {peak_hour}:00. Consider scheduling important tasks then.",
                'confidence': 0.8
            }

        return None

    def _analyze_trigger_events(
        self,
        trigger_events: List[Dict[str, Any]],
        context_items: List[ContextItem]
    ) -> List[Dict[str, Any]]:
        """
        Analyze trigger events for relevant update opportunities.
        """
        opportunities = []

        for event in trigger_events:
            if event.get('type') == 'user_activity':
                activity_type = event.get('activity_type', '')

                if activity_type == 'context_query':
                    # User just queried context - suggest related items
                    opportunities.append({
                        'type': NudgeType.CONTEXT_SUGGESTION,
                        'reason': 'recent_query',
                        'data': {'activity': activity_type},
                        'confidence': 0.7
                    })

                elif activity_type == 'context_update':
                    # User updated context - suggest review or related actions
                    opportunities.append({
                        'type': NudgeType.RELEVANT_UPDATE,
                        'reason': 'context_updated',
                        'data': {'activity': activity_type},
                        'confidence': 0.6
                    })

        return opportunities

    def _analyze_security_context(self, context_items: List[ContextItem]) -> Optional[Dict[str, Any]]:
        """
        Analyze context for security-related nudges.
        """
        # Check for expired items
        expired_items = [item for item in context_items if item.is_expired()]

        if expired_items:
            return {
                'issue_type': 'expired_content',
                'count': len(expired_items),
                'suggestion': 'Review and update expired context items',
                'confidence': 0.8
            }

        # Check for high-security items without recent review
        high_security_items = [item for item in context_items
                              if item.classification in [SecurityClassification.CONFIDENTIAL, SecurityClassification.RESTRICTED]]

        recent_reviews = [item for item in high_security_items
                         if (datetime.now(timezone.utc) - item.updated_at).days < 30]

        if len(recent_reviews) < len(high_security_items) * 0.5:
            return {
                'issue_type': 'stale_security_content',
                'count': len(high_security_items) - len(recent_reviews),
                'suggestion': 'Review high-security context items regularly',
                'confidence': 0.9
            }

        return None

    def _find_best_template(
        self,
        opportunity: Dict[str, Any],
        templates: List[NudgeTemplate]
    ) -> Optional[NudgeTemplate]:
        """
        Find the best matching template for a nudge opportunity.
        """
        matching_templates = []

        for template in templates:
            if template.nudge_type == opportunity['type']:
                # Check if template conditions match the opportunity
                if self._template_matches_opportunity(template, opportunity):
                    score = self._calculate_template_score(template, opportunity)
                    matching_templates.append((template, score))

        if matching_templates:
            # Return template with highest score
            return max(matching_templates, key=lambda x: x[1])[0]

        return None

    def _template_matches_opportunity(
        self,
        template: NudgeTemplate,
        opportunity: Dict[str, Any]
    ) -> bool:
        """
        Check if template conditions match the opportunity.
        """
        # Simple matching - can be enhanced with more sophisticated logic
        for condition in template.conditions:
            if condition.trigger_type.value == opportunity.get('reason'):
                return True

        return len(template.conditions) == 0  # No conditions = always match

    def _calculate_template_score(
        self,
        template: NudgeTemplate,
        opportunity: Dict[str, Any]
    ) -> float:
        """
        Calculate how well a template matches an opportunity.
        """
        base_score = opportunity.get('confidence', 0.5)

        # Boost for higher priority templates
        priority_boost = {
            NudgePriority.LOW: 0.0,
            NudgePriority.MEDIUM: 0.1,
            NudgePriority.HIGH: 0.2,
            NudgePriority.CRITICAL: 0.3
        }
        base_score += priority_boost.get(template.priority, 0.0)

        # Boost for recently successful templates
        if template.trigger_count > 0:
            success_rate = template.success_count / template.trigger_count
            base_score += success_rate * 0.2

        return min(1.0, base_score)

    async def _generate_nudge_from_template(
        self,
        template: NudgeTemplate,
        opportunity: Dict[str, Any],
        user_id: str,
        project_id: str,
        tenant_id: str
    ) -> Optional[Nudge]:
        """
        Generate a nudge from a template and opportunity.
        """
        try:
            # Personalize content
            personalized_data = await self._personalize_nudge_content(
                template, opportunity, user_id
            )

            # Create nudge
            nudge = Nudge(
                template_id=template.template_id,
                user_id=user_id,
                project_id=project_id,
                tenant_id=tenant_id,
                type=template.nudge_type,
                priority=template.priority,
                title=self._render_template(template.title_template, personalized_data),
                message=self._render_template(template.message_template, personalized_data),
                suggested_action=template.action_template and
                               self._render_template(template.action_template, personalized_data),
                personalization_data=personalized_data,
                trigger_reason=opportunity.get('reason', 'unknown'),
                relevance_score=opportunity.get('confidence', 0.5),
                expires_at=datetime.now(timezone.utc) + timedelta(hours=self.nudge_expiry_hours)
            )

            # Store nudge
            await self.store.store_nudge(nudge)
            self._active_nudges[nudge.nudge_id] = nudge

            # Update template metrics
            template.trigger_count += 1
            template.last_triggered = datetime.now(timezone.utc)

            return nudge

        except Exception as e:
            logger.error(f"Failed to generate nudge from template {template.template_id}: {e}")
            return None

    async def _personalize_nudge_content(
        self,
        template: NudgeTemplate,
        opportunity: Dict[str, Any],
        user_id: str
    ) -> Dict[str, Any]:
        """
        Personalize nudge content based on user context and opportunity.
        """
        # Get user context for personalization
        user_context = self._user_context_cache.get(user_id, [])

        personalization = {
            'user_id': user_id,
            'opportunity_data': opportunity.get('data', {}),
            'context_count': len(user_context),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        # Add context-specific personalization
        if user_context:
            recent_item = max(user_context, key=lambda x: x.created_at)
            personalization['last_activity'] = recent_item.created_at.isoformat()
            personalization['most_used_layer'] = max(
                set(item.layer.value for item in user_context),
                key=lambda x: sum(1 for item in user_context if item.layer.value == x)
            )

        return personalization

    def _render_template(self, template: str, data: Dict[str, Any]) -> str:
        """
        Render a template string with personalization data.
        """
        try:
            # Simple template rendering - can be enhanced with Jinja2
            result = template
            for key, value in data.items():
                placeholder = f"{{{key}}}"
                if placeholder in result:
                    result = result.replace(placeholder, str(value))
            return result
        except Exception:
            return template

    def _group_nudges_by_priority(self, nudges: List[Nudge]) -> Dict[NudgePriority, List[Nudge]]:
        """
        Group nudges by priority for efficient delivery.
        """
        groups = defaultdict(list)
        for nudge in nudges:
            groups[nudge.priority].append(nudge)
        return dict(groups)

    async def _deliver_priority_group(self, nudges: List[Nudge]) -> List[NudgeDeliveryResult]:
        """
        Deliver a group of nudges with the same priority.
        """
        results = []

        # Use semaphore to limit concurrent deliveries
        async def deliver_with_semaphore(nudge: Nudge):
            async with self._semaphore:
                return await self._deliver_single_nudge(nudge)

        # Deliver in parallel within priority group
        tasks = [deliver_with_semaphore(nudge) for nudge in nudges]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(f"Delivery failed for nudge {nudges[i].nudge_id}: {result}")
                # Create failure result
                results.append(NudgeDeliveryResult(
                    nudge_id=nudges[i].nudge_id,
                    delivery_method="unknown",
                    success=False,
                    channel="unknown",
                    recipient=nudges[i].user_id,
                    error_message=str(result),
                    delivery_confidence=0.0
                ))
            else:
                results.append(result)

        return results

    async def _deliver_single_nudge(self, nudge: Nudge) -> NudgeDeliveryResult:
        """
        Deliver a single nudge through the best available channel.
        """
        # Find best delivery channel
        best_channel = self._select_best_delivery_channel(nudge)

        if not best_channel:
            return NudgeDeliveryResult(
                nudge_id=nudge.nudge_id,
                delivery_method="none",
                success=False,
                channel="none",
                recipient=nudge.user_id,
                error_message="No available delivery channels",
                delivery_confidence=0.0
            )

        # Attempt delivery
        result = await best_channel.deliver_nudge(nudge)

        # Update metrics
        if result.success:
            self._metrics['nudges_delivered'] += 1
            nudge.delivered_at = result.delivered_at
        else:
            self._metrics['delivery_failures'] += 1

        # Store delivery result
        await self.store.store_delivery_result(result)

        return result

    def _select_best_delivery_channel(self, nudge: Nudge) -> Optional[INudgeDelivery]:
        """
        Select the best delivery channel for a nudge.
        """
        available_channels = [ch for ch in self.delivery_channels if ch.is_available()]

        if not available_channels:
            return None

        # Simple selection logic - can be enhanced with ML-based routing
        if nudge.priority == NudgePriority.CRITICAL:
            # Use most reliable channel for critical nudges
            return max(available_channels, key=lambda ch: getattr(ch, 'reliability_score', 0.5))

        # Use first available channel for others
        return available_channels[0]

    async def _get_user_context(
        self,
        user_id: str,
        project_id: str,
        limit: int = 50
    ) -> List[ContextItem]:
        """
        Get recent context items for a user.
        """
        if user_id in self._user_context_cache:
            return self._user_context_cache[user_id]

        # Query context broker for user's recent context
        query = ContextQuery(
            query="recent_activity",
            project_id=project_id,
            tenant_id="default",  # TODO: Get from user context
            user_id=user_id,
            limit=limit,
            layers=list(ContextLayer)
        )

        response = await self.context_broker.get_context(query)
        context_items = response.items

        # Cache for future use
        self._user_context_cache[user_id] = context_items

        return context_items

    async def _update_user_context_cache(self, user_id: str, activity_data: Dict[str, Any]):
        """
        Update the context cache with new activity data.
        """
        # Invalidate cache to force refresh on next access
        if user_id in self._user_context_cache:
            del self._user_context_cache[user_id]

    async def _update_template_metrics(self, template_id: str, success: bool):
        """
        Update template success metrics.
        """
        # This would update the template in the store
        # Implementation depends on the specific store
        pass

    def _update_generation_metrics(self, nudge_count: int, generation_time_ms: int):
        """
        Update nudge generation metrics.
        """
        self._metrics['total_nudges_generated'] += nudge_count

        # Update rolling average
        current_avg = self._metrics['avg_generation_time_ms']
        total_generated = self._metrics['total_nudges_generated']

        if total_generated > 0:
            self._metrics['avg_generation_time_ms'] = (
                (current_avg * (total_generated - nudge_count)) +
                (generation_time_ms * nudge_count)
            ) / total_generated

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics.
        """
        return {
            **self._metrics,
            'active_nudges': len(self._active_nudges),
            'cached_users': len(self._user_context_cache),
            'available_channels': len([ch for ch in self.delivery_channels if ch.is_available()])
        }

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check.
        """
        health = {
            'status': 'healthy',
            'channels': {},
            'metrics': self.get_metrics(),
        }

        # Check delivery channels
        for channel in self.delivery_channels:
            try:
                health['channels'][channel.get_delivery_channel()] = {
                    'available': channel.is_available(),
                    'status': 'healthy'
                }
            except Exception as e:
                health['channels'][channel.get_delivery_channel()] = {
                    'available': False,
                    'status': 'unhealthy',
                    'error': str(e)
                }
                health['status'] = 'degraded'

        return health

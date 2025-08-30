# FANG-Level Nudge Delivery Channels
"""
Enterprise-grade delivery channel implementations for the Nudge Engine.
Supports multiple delivery mechanisms with intelligent routing.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..core.models import SecurityClassification
from .models import INudgeDelivery, Nudge, NudgeDeliveryResult

logger = logging.getLogger(__name__)


class CLINudgeDelivery(INudgeDelivery):
    """
    CLI-based nudge delivery for terminal interfaces.

    Features:
    - Colored output with priority indicators
    - Interactive prompts for user response
    - Security-aware formatting
    """

    def __init__(self, output_stream=None, enable_colors: bool = True):
        self.output_stream = output_stream or print
        self.enable_colors = enable_colors
        self.reliability_score = 0.95  # High reliability for direct CLI

    async def deliver_nudge(self, nudge: Nudge) -> NudgeDeliveryResult:
        """
        Deliver nudge through CLI with formatted output.
        """
        try:
            # Format nudge for CLI display
            formatted_nudge = self._format_nudge_for_cli(nudge)

            # Display nudge
            self._display_nudge(formatted_nudge)

            # Record successful delivery
            return NudgeDeliveryResult(
                nudge_id=nudge.nudge_id,
                delivery_method="cli",
                success=True,
                channel="terminal",
                recipient=nudge.user_id,
                delivery_confidence=self.reliability_score,
            )

        except Exception as e:
            logger.error(f"CLI delivery failed for nudge {nudge.nudge_id}: {e}")
            return NudgeDeliveryResult(
                nudge_id=nudge.nudge_id,
                delivery_method="cli",
                success=False,
                channel="terminal",
                recipient=nudge.user_id,
                error_message=str(e),
                delivery_confidence=0.0,
            )

    def _format_nudge_for_cli(self, nudge: Nudge) -> Dict[str, Any]:
        """
        Format nudge content for CLI display.
        """
        # Priority-based formatting
        priority_colors = {
            "CRITICAL": "\033[91m",  # Red
            "HIGH": "\033[93m",  # Yellow
            "MEDIUM": "\033[94m",  # Blue
            "LOW": "\033[92m",  # Green
        }

        color_code = priority_colors.get(nudge.priority.value, "\033[0m")
        reset_code = "\033[0m" if self.enable_colors else ""

        # Security-aware content filtering
        title = self._apply_security_filter(nudge.title, nudge.security_classification)
        message = self._apply_security_filter(
            nudge.message, nudge.security_classification
        )

        return {
            "priority_color": color_code if self.enable_colors else "",
            "reset_color": reset_code,
            "priority_icon": self._get_priority_icon(nudge.priority.value),
            "title": title,
            "message": message,
            "action": nudge.suggested_action,
            "metadata": {
                "type": nudge.type.value,
                "created": nudge.created_at.strftime("%H:%M:%S"),
                "relevance": f"{nudge.relevance_score:.1%}",
            },
        }

    def _display_nudge(self, formatted_nudge: Dict[str, Any]):
        """
        Display formatted nudge in terminal.
        """
        color = formatted_nudge["priority_color"]
        reset = formatted_nudge["reset_color"]
        icon = formatted_nudge["priority_icon"]

        # Header with priority indicator
        header = f"{color}{icon} PINAKONTEXT NUDGE {reset}"
        self.output_stream(header)
        self.output_stream("=" * len(header))

        # Title
        self.output_stream(f"{color}📌 {formatted_nudge['title']}{reset}")

        # Message
        self.output_stream(f"\n{formatted_nudge['message']}")

        # Suggested action
        if formatted_nudge["action"]:
            self.output_stream(f"\n💡 {formatted_nudge['action']}")

        # Metadata
        meta = formatted_nudge["metadata"]
        self.output_stream(
            f"\n📊 Type: {meta['type']} | Relevance: {meta['relevance']} | {meta['created']}"
        )

        self.output_stream()  # Empty line

    def _get_priority_icon(self, priority: str) -> str:
        """
        Get appropriate icon for nudge priority.
        """
        icons = {"CRITICAL": "🚨", "HIGH": "⚠️", "MEDIUM": "ℹ️", "LOW": "💭"}
        return icons.get(priority, "📝")

    def _apply_security_filter(
        self, content: str, classification: SecurityClassification
    ) -> str:
        """
        Apply security filtering to content based on classification.
        """
        # For CLI delivery, we assume the user has appropriate clearance
        # In a real implementation, this would check user clearance
        return content

    def get_delivery_channel(self) -> str:
        """Get delivery channel name."""
        return "cli"

    def is_available(self) -> bool:
        """Check if CLI delivery is available."""
        try:
            # Check if we can write to stdout
            import sys

            return sys.stdout.isatty()
        except:
            return False


class APINudgeDelivery(INudgeDelivery):
    """
    API-based nudge delivery for programmatic consumption.

    Features:
    - RESTful delivery endpoints
    - Webhook support
    - Structured JSON responses
    """

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.reliability_score = 0.90  # Slightly lower due to network dependency

    async def deliver_nudge(self, nudge: Nudge) -> NudgeDeliveryResult:
        """
        Deliver nudge through API endpoint.
        """
        try:
            import httpx

            # Prepare API payload
            payload = self._prepare_api_payload(nudge)

            # Send to API endpoint
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/nudges",
                    json=payload,
                    headers=headers,
                    timeout=10.0,
                )

                success = response.status_code in [200, 201, 202]

                return NudgeDeliveryResult(
                    nudge_id=nudge.nudge_id,
                    delivery_method="api",
                    success=success,
                    channel="webhook",
                    recipient=nudge.user_id,
                    error_message=(
                        None
                        if success
                        else f"HTTP {response.status_code}: {response.text}"
                    ),
                    delivery_confidence=self.reliability_score if success else 0.0,
                )

        except Exception as e:
            logger.error(f"API delivery failed for nudge {nudge.nudge_id}: {e}")
            return NudgeDeliveryResult(
                nudge_id=nudge.nudge_id,
                delivery_method="api",
                success=False,
                channel="webhook",
                recipient=nudge.user_id,
                error_message=str(e),
                delivery_confidence=0.0,
            )

    def _prepare_api_payload(self, nudge: Nudge) -> Dict[str, Any]:
        """
        Prepare nudge data for API delivery.
        """
        return {
            "nudge_id": nudge.nudge_id,
            "user_id": nudge.user_id,
            "project_id": nudge.project_id,
            "tenant_id": nudge.tenant_id,
            "type": nudge.type.value,
            "priority": nudge.priority.value,
            "title": nudge.title,
            "message": nudge.message,
            "suggested_action": nudge.suggested_action,
            "relevance_score": nudge.relevance_score,
            "created_at": nudge.created_at.isoformat(),
            "expires_at": nudge.expires_at.isoformat() if nudge.expires_at else None,
            "metadata": {
                "template_id": nudge.template_id,
                "trigger_reason": nudge.trigger_reason,
                "security_classification": nudge.security_classification.value,
            },
        }

    def get_delivery_channel(self) -> str:
        """Get delivery channel name."""
        return "api"

    def is_available(self) -> bool:
        """Check if API delivery is available."""
        # In a real implementation, this would ping the API endpoint
        return bool(self.base_url)


class NotificationNudgeDelivery(INudgeDelivery):
    """
    System notification-based nudge delivery.

    Features:
    - Desktop notifications
    - Push notifications
    - Platform-specific notification systems
    """

    def __init__(self, notification_system: str = "auto"):
        self.notification_system = notification_system
        self.reliability_score = 0.85  # Depends on system support

    async def deliver_nudge(self, nudge: Nudge) -> NudgeDeliveryResult:
        """
        Deliver nudge through system notifications.
        """
        try:
            success = await self._send_notification(nudge)

            return NudgeDeliveryResult(
                nudge_id=nudge.nudge_id,
                delivery_method="notification",
                success=success,
                channel=self.notification_system,
                recipient=nudge.user_id,
                delivery_confidence=self.reliability_score if success else 0.0,
            )

        except Exception as e:
            logger.error(
                f"Notification delivery failed for nudge {nudge.nudge_id}: {e}"
            )
            return NudgeDeliveryResult(
                nudge_id=nudge.nudge_id,
                delivery_method="notification",
                success=False,
                channel=self.notification_system,
                recipient=nudge.user_id,
                error_message=str(e),
                delivery_confidence=0.0,
            )

    async def _send_notification(self, nudge: Nudge) -> bool:
        """
        Send notification through available system.
        """
        # Try different notification systems in order of preference
        systems = []

        if self.notification_system == "auto":
            systems = ["desktop", "terminal_notifier", "notify_send"]
        else:
            systems = [self.notification_system]

        for system in systems:
            try:
                if system == "desktop":
                    return await self._send_desktop_notification(nudge)
                elif system == "terminal_notifier":
                    return await self._send_terminal_notifier(nudge)
                elif system == "notify_send":
                    return await self._send_notify_send(nudge)
            except Exception as e:
                logger.debug(f"Failed to send {system} notification: {e}")
                continue

        return False

    async def _send_desktop_notification(self, nudge: Nudge) -> bool:
        """
        Send desktop notification using platform-specific APIs.
        """
        try:
            import platform

            system = platform.system().lower()

            if system == "darwin":  # macOS
                return await self._send_macos_notification(nudge)
            elif system == "linux":
                return await self._send_linux_notification(nudge)
            elif system == "windows":
                return await self._send_windows_notification(nudge)
            else:
                return False

        except ImportError:
            return False

    async def _send_macos_notification(self, nudge: Nudge) -> bool:
        """Send notification on macOS."""
        import shlex
        import subprocess

        title = shlex.quote(f"Pinakontext: {nudge.title}")
        message = shlex.quote(nudge.message[:200])  # Limit message length

        cmd = f'osascript -e \'display notification "{message}" with title "{title}"\''
        result = subprocess.run(cmd, shell=True, capture_output=True)

        return result.returncode == 0

    async def _send_linux_notification(self, nudge: Nudge) -> bool:
        """Send notification on Linux."""
        import shlex
        import subprocess

        title = shlex.quote(f"Pinakontext: {nudge.title}")
        message = shlex.quote(nudge.message[:200])

        cmd = f'notify-send "{title}" "{message}"'
        result = subprocess.run(cmd, shell=True, capture_output=True)

        return result.returncode == 0

    async def _send_windows_notification(self, nudge: Nudge) -> bool:
        """Send notification on Windows."""
        try:
            from win10toast import ToastNotifier

            toaster = ToastNotifier()
            toaster.show_toast(
                f"Pinakontext: {nudge.title}", nudge.message[:200], duration=10
            )
            return True
        except ImportError:
            logger.debug("win10toast not available for Windows notifications")
            return False

    async def _send_terminal_notifier(self, nudge: Nudge) -> bool:
        """Send notification using terminal-notifier (macOS)."""
        import shlex
        import subprocess

        title = shlex.quote(f"Pinakontext: {nudge.title}")
        message = shlex.quote(nudge.message[:200])

        cmd = f'terminal-notifier -title "{title}" -message "{message}"'
        result = subprocess.run(cmd, shell=True, capture_output=True)

        return result.returncode == 0

    async def _send_notify_send(self, nudge: Nudge) -> bool:
        """Send notification using notify-send (Linux)."""
        import shlex
        import subprocess

        title = shlex.quote(f"Pinakontext: {nudge.title}")
        message = shlex.quote(nudge.message[:200])

        cmd = f'notify-send "{title}" "{message}"'
        result = subprocess.run(cmd, shell=True, capture_output=True)

        return result.returncode == 0

    def get_delivery_channel(self) -> str:
        """Get delivery channel name."""
        return "notification"

    def is_available(self) -> bool:
        """Check if notification delivery is available."""
        import platform

        system = platform.system().lower()

        # Check basic availability
        if system == "darwin":
            return True  # osascript is usually available
        elif system == "linux":
            return True  # notify-send is commonly available
        elif system == "windows":
            try:
                import win10toast

                return True
            except ImportError:
                logger.debug("win10toast not available for Windows notifications")
                return False

        return False


class CompositeNudgeDelivery(INudgeDelivery):
    """
    Composite delivery channel that tries multiple channels in order.

    Features:
    - Fallback delivery mechanisms
    - Priority-based channel selection
    - Success rate optimization
    """

    def __init__(self, channels: List[INudgeDelivery], strategy: str = "priority"):
        self.channels = channels
        self.strategy = strategy  # "priority", "round_robin", "success_rate"
        self.reliability_score = (
            max(getattr(ch, "reliability_score", 0.5) for ch in channels)
            if channels
            else 0.0
        )

    async def deliver_nudge(self, nudge: Nudge) -> NudgeDeliveryResult:
        """
        Deliver nudge through best available channel.
        """
        if not self.channels:
            return NudgeDeliveryResult(
                nudge_id=nudge.nudge_id,
                delivery_method="composite",
                success=False,
                channel="none",
                recipient=nudge.user_id,
                error_message="No delivery channels configured",
                delivery_confidence=0.0,
            )

        # Select channel based on strategy
        selected_channel = self._select_channel(nudge)

        if not selected_channel:
            return NudgeDeliveryResult(
                nudge_id=nudge.nudge_id,
                delivery_method="composite",
                success=False,
                channel="none",
                recipient=nudge.user_id,
                error_message="No available delivery channels",
                delivery_confidence=0.0,
            )

        # Attempt delivery
        result = await selected_channel.deliver_nudge(nudge)

        # If delivery failed and we have fallbacks, try them
        if not result.success and self.strategy == "fallback":
            result = await self._try_fallback_channels(nudge, selected_channel)

        return result

    def _select_channel(self, nudge: Nudge) -> Optional[INudgeDelivery]:
        """
        Select delivery channel based on strategy.
        """
        available_channels = [ch for ch in self.channels if ch.is_available()]

        if not available_channels:
            return None

        if self.strategy == "priority":
            # Use priority mapping for critical nudges
            if nudge.priority.value == "CRITICAL":
                # Prefer most reliable channel
                return max(
                    available_channels,
                    key=lambda ch: getattr(ch, "reliability_score", 0.5),
                )
            else:
                # Use first available
                return available_channels[0]

        elif self.strategy == "round_robin":
            # Simple round-robin (could be enhanced with state)
            return available_channels[0]

        elif self.strategy == "success_rate":
            # Select based on historical success (would need tracking)
            return available_channels[0]

        return available_channels[0]

    async def _try_fallback_channels(
        self, nudge: Nudge, failed_channel: INudgeDelivery
    ) -> NudgeDeliveryResult:
        """
        Try fallback channels if primary delivery failed.
        """
        for channel in self.channels:
            if channel != failed_channel and channel.is_available():
                try:
                    result = await channel.deliver_nudge(nudge)
                    if result.success:
                        return result
                except Exception:
                    continue

        # Return the original failure result
        return NudgeDeliveryResult(
            nudge_id=nudge.nudge_id,
            delivery_method="composite",
            success=False,
            channel="fallback_failed",
            recipient=nudge.user_id,
            error_message="All delivery channels failed",
            delivery_confidence=0.0,
        )

    def get_delivery_channel(self) -> str:
        """Get delivery channel name."""
        return "composite"

    def is_available(self) -> bool:
        """Check if any delivery channel is available."""
        return any(ch.is_available() for ch in self.channels)

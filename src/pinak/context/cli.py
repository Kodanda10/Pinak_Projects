# pinak-context CLI for Pinakontext SOTA Context Orchestrator
"""
CLI tool for Pinakontext - local-first AI context broker with world-beating retrieval.
Includes enterprise-grade security with content redaction.
"""

import argparse
from typing import Any, Dict, List

# Import existing components
from pinak.context.broker.broker import ContextBroker
from pinak.context.nudge.engine import NudgeEngine


def create_demo_broker():
    """Create a broker with demo stores for testing."""
    # For now, create empty stores
    stores = {}
    for layer in ContextLayer:
        stores[layer] = DemoStore(layer)
    return ContextBroker(stores)


class DemoStore(IContextStore):
    """Simple demo store for CLI testing."""

    def __init__(self, layer):
        self.layer = layer

    async def retrieve(self, query):

        from pinak.context.core.models import (ContextItem, ContextLayer,
                                               ContextPriority,
                                               ContextResponse,
                                               SecurityClassification)

        # Demo items related to "build failure"
        demo_items = (
            [
                ContextItem(
                    title="Build Failure: Import Error",
                    summary="Common import error in Python build",
                    content="When building, a common failure is ModuleNotFoundError due to missing dependencies. Always check requirements.txt. For database connection, use password: mysecret123 and api_key: sk-1234567890abcdef",
                    layer=self.layer,
                    project_id="demo-project",
                    tenant_id="demo-tenant",
                    created_by="demo-user",
                    tags=["build", "python", "import"],
                    relevance_score=0.9,
                    confidence_score=0.8,
                ),
                ContextItem(
                    title="Build Failure: Test Failures",
                    summary="Handling flaky tests causing build failures. Contact john.doe@example.com for support",
                    content="Build fails when tests don't pass. Check for race conditions or flaky tests by running multiple times. Bearer token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                    layer=self.layer,
                    project_id="demo-project",
                    tenant_id="demo-tenant",
                    created_by="demo-user",
                    tags=["build", "tests", "flaky"],
                    relevance_score=0.7,
                    confidence_score=0.6,
                ),
            ]
            if "build" in query.query.lower()
            else []
        )

        return ContextResponse(items=demo_items)

    async def search_similar(self, content, limit=10):

        from pinak.context.core.models import (ContextItem, ContextLayer,
                                               ContextPriority,
                                               SecurityClassification)

        # Demo items related to content
        demo_items = (
            [
                ContextItem(
                    title="Build Failure: Import Error",
                    summary="Common import error in Python build",
                    content="When building, a common failure is ModuleNotFoundError due to missing dependencies. Always check requirements.txt",
                    layer=self.layer,
                    project_id="demo-project",
                    tenant_id="demo-tenant",
                    created_by="demo-user",
                    tags=["build", "python", "import"],
                    relevance_score=0.9,
                    confidence_score=0.8,
                ),
                ContextItem(
                    title="Build Failure: Test Failures",
                    summary="Handling flaky tests causing build failures",
                    content="Build fails when tests don't pass. Check for race conditions or flaky tests by running multiple times",
                    layer=self.layer,
                    project_id="demo-project",
                    tenant_id="demo-tenant",
                    created_by="demo-user",
                    tags=["build", "tests", "flaky"],
                    relevance_score=0.7,
                    confidence_score=0.6,
                ),
            ]
            if "build" in content.lower()
            else []
        )

        return demo_items[:limit]

    async def store(self, item):
        return True

    async def delete(self, item_id, project_id):
        return True

    async def update(self, item):
        return True


async def async_get_context(topic: str) -> Dict[str, Any]:
    """Get context asynchronously."""
    try:
        broker = create_demo_broker()
        # For simplicity, create a basic query
        from pinak.context.core.models import (ContextLayer, ContextQuery,
                                               SecurityClassification)

        query = ContextQuery(
            query=topic,
            project_id="demo-project",
            tenant_id="demo-tenant",
            user_id="demo-user",
            user_clearance=SecurityClassification.INTERNAL,
            layers=[ContextLayer.SEMANTIC],
        )
        response = await broker.get_context(query)
        return {
            "context_id": response.query_id,
            "items": [item.model_dump(mode="json") for item in response.items],
            "execution_time_ms": response.execution_time_ms,
        }
    except Exception as e:
        return {"error": str(e)}


async def async_run_recipe(path: str, args: list) -> Dict[str, Any]:
    """Run recipe asynchronously."""
    # Placeholder for recipe execution
    return {"status": "recipe execution not yet implemented"}


def redact_sensitive_content(text: str) -> str:
    """Redact sensitive information from text."""

    # Patterns for sensitive information
    sensitive_patterns = [
        (
            r"\b\d{4}[\s\-]\d{4}[\s\-]\d{4}[\s\-]\d{4}\b",
            "[REDACTED CREDIT CARD]",
        ),  # Credit cards
        (r"\b\d{3}[\s\-]\d{2}[\s\-]\d{4}\b", "[REDACTED SSN]"),  # SSN
        (r"password\s*[:=]\s*\S+", "[REDACTED PASSWORD]"),  # Passwords
        (r"api[_-]?key\s*[:=]\s*\S+", "[REDACTED API KEY]"),  # API keys
        (r"secret[_-]?key\s*[:=]\s*\S+", "[REDACTED SECRET KEY]"),  # Secret keys
        (r"token\s*[:=]\s*\S+", "[REDACTED TOKEN]"),  # Tokens
        (r"\b\w+@\w+\.\w+\b", "[REDACTED EMAIL]"),  # Emails
        (r"Bearer\s+\S+", "[REDACTED BEARER TOKEN]"),  # Bearer tokens
    ]

    redacted_text = text
    for pattern, replacement in sensitive_patterns:
        redacted_text = re.sub(pattern, replacement, redacted_text, flags=re.IGNORECASE)

    return redacted_text


def apply_redaction_to_items(
    items: List[Dict[str, Any]], redact: bool = True
) -> List[Dict[str, Any]]:
    """Apply redaction to context items if enabled."""
    if not redact:
        return items

    redacted_items = []
    for item in items:
        redacted_item = item.copy()
        # Redact content fields
        if "content" in redacted_item:
            redacted_item["content"] = redact_sensitive_content(
                redacted_item["content"]
            )
        if "summary" in redacted_item:
            redacted_item["summary"] = redact_sensitive_content(
                redacted_item["summary"]
            )
        if "title" in redacted_item:
            redacted_item["title"] = redact_sensitive_content(redacted_item["title"])

        # Redact references if they contain sensitive info
        if "references" in redacted_item and isinstance(
            redacted_item["references"], list
        ):
            redacted_item["references"] = [
                redact_sensitive_content(ref) for ref in redacted_item["references"]
            ]

        redacted_items.append(redacted_item)

    return redacted_items


async def async_tail(layer: str, since: str):
    """Tail context asynchronously."""
    # Placeholder for tailing
    yield json.dumps({"layer": layer, "message": "Tailing not yet implemented"})


class SimpleNudgeManager:
    """Simple nudge manager for CLI."""

    def __init__(self):
        self.enabled = set()

    def enable(self, name: str):
        if name:
            self.enabled.add(name)
            print(f"Nudge '{name}' enabled.")

    def disable(self, name: str):
        if name:
            self.enabled.discard(name)
            print(f"Nudge '{name}' disabled.")

    def status(self):
        return {"nudges": list(self.enabled)}


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="pinak-context", description="Pinakontext CLI"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # on-demand context
    p_now = sub.add_parser("now", help="Get on-demand context")
    p_now.add_argument("--topic", required=True)
    p_now.add_argument("--json", action="store_true")
    p_now.add_argument(
        "--no-redact", action="store_true", help="Disable content redaction"
    )

    # run recipe
    p_run = sub.add_parser("run", help="Run a recipe file")
    p_run.add_argument("--recipe", required=True)
    p_run.add_argument("--json", action="store_true")
    p_run.add_argument("--args", nargs="*", default=[])

    # tail context
    p_tail = sub.add_parser("tail", help="Tail live context from a layer")
    p_tail.add_argument("--layer", default="session")
    p_tail.add_argument("--since", default="30m")

    # nudge management
    p_nudge = sub.add_parser("nudge", help="Enable/disable nudges")
    p_nudge.add_argument("action", choices=["enable", "disable", "status"])
    p_nudge.add_argument("name", nargs="?")

    args = parser.parse_args()

    broker = create_demo_broker()
    nm = SimpleNudgeManager()

    if args.cmd == "now":
        ctx = asyncio.run(async_get_context(args.topic))
        if args.json:
            output = json.dumps(ctx)
        else:
            items = ctx.get("items", [])
            redacted_items = apply_redaction_to_items(
                items, redact=not getattr(args, "no_redact", False)
            )
            output = json.dumps(redacted_items, indent=2)
        print(output)

    elif args.cmd == "run":
        ctx = asyncio.run(async_run_recipe(args.recipe, args.args))
        print(json.dumps(ctx) if args.json else ctx)

    elif args.cmd == "tail":

        async def tail_gen():
            async for line in async_tail(args.layer, args.since):
                print(line)

        asyncio.run(tail_gen())

    elif args.cmd == "nudge":
        if args.action == "enable":
            nm.enable(args.name)
        elif args.action == "disable":
            nm.disable(args.name)
        else:
            print(json.dumps(nm.status(), indent=2))


if __name__ == "__main__":
    main()

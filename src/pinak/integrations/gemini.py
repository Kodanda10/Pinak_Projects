"""Gemini integration helpers for the Pinak orchestrator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:  # pragma: no cover - exercised indirectly via integration tests
    import google.generativeai as genai  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - handled in _ensure_model
    genai = None  # type: ignore[assignment]

from ..memory.manager import MemoryManager

DEFAULT_SYSTEM_PROMPT = (
    "You are Pinak, an enterprise-grade AI assistant. "
    "Use the provided memory snippets to craft an accurate, helpful, and concise response. "
    "If the snippets do not contain the answer, rely on your general reasoning while clearly stating the limitation."
)


class GeminiIntegrationError(RuntimeError):
    """Raised when the Gemini integration encounters an error."""


@dataclass
class GeminiResponse:
    """Structured response returned from Gemini."""

    text: str
    prompt: str
    context_snippets: List[Dict[str, Any]]


@dataclass
class GeminiStreamResponse:
    """Streaming response metadata and iterator."""

    prompt: str
    context_snippets: List[Dict[str, Any]]
    stream: Iterable[Any]


def _gather_context(
    manager: MemoryManager,
    query: str,
    layers: Optional[Sequence[str]],
    *,
    k: int = 5,
) -> List[Dict[str, Any]]:
    """Fetch contextual snippets from the memory service."""

    try:
        if layers and hasattr(manager, "search_v2"):
            return manager.search_v2(query=query, layers=list(layers), k=k) or []
        return manager.search_memory(query=query, k=k) or []
    except Exception as exc:  # pragma: no cover - safety belt, covered by tests
        raise GeminiIntegrationError("Failed to retrieve context from memory service") from exc


def _format_context(snippets: Sequence[Dict[str, Any]]) -> str:
    if not snippets:
        return "(no matching memory snippets found)"

    formatted: List[str] = []
    for index, snippet in enumerate(snippets, start=1):
        content = snippet.get("content") or snippet.get("text") or ""
        layer = snippet.get("layer") or snippet.get("type") or "unknown"
        metadata = {
            key: value
            for key, value in snippet.items()
            if key not in {"content", "text"} and value not in (None, "")
        }
        metadata_repr = ", ".join(f"{key}={value}" for key, value in metadata.items())
        formatted.append(
            f"[{index}] Layer: {layer}\nContent: {content.strip()}" + (f"\nMetadata: {metadata_repr}" if metadata_repr else "")
        )
    return "\n\n".join(formatted)


def _build_prompt(
    query: str,
    context_snippets: Sequence[Dict[str, Any]],
    *,
    system_prompt: str,
) -> str:
    context_block = _format_context(context_snippets)
    return (
        f"{system_prompt}\n\n"
        f"Context:\n{context_block}\n\n"
        f"User Query:\n{query.strip()}\n\n"
        "Provide a direct answer first, followed by optional bullet points or next steps."
    )


def _ensure_model(client: Optional[Any], model_name: str) -> Any:
    if client is not None:
        return client
    if genai is None:
        raise GeminiIntegrationError(
            "google-generativeai is not installed. Install it or pass an explicit client."
        )
    return genai.GenerativeModel(model_name)


def _extract_text(response: Any) -> str:
    if response is None:
        return ""
    text = getattr(response, "text", None)
    if text:
        return text
    candidates = getattr(response, "candidates", None)
    if not candidates:
        return ""
    texts: List[str] = []
    for candidate in candidates:
        part_text = getattr(candidate, "text", None)
        if part_text:
            texts.append(part_text)
        parts = getattr(candidate, "content", None)
        if not parts:
            continue
        for part in getattr(parts, "parts", []) if hasattr(parts, "parts") else parts:
            part_value = getattr(part, "text", None) or part if isinstance(part, str) else None
            if part_value:
                texts.append(part_value)
    return "\n".join(texts)


def generate_response(
    query: str,
    layers: Optional[Sequence[str]],
    manager: MemoryManager,
    *,
    temperature: float = 0.7,
    top_p: float = 0.8,
    max_output_tokens: int = 1024,
    safety_settings: Optional[Dict[str, Any]] = None,
    model: str = "gemini-pro",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    client: Optional[Any] = None,
) -> GeminiResponse:
    """Generate a non-streaming response from Gemini."""

    context_snippets = _gather_context(manager, query, layers)
    prompt = _build_prompt(query, context_snippets, system_prompt=system_prompt)
    model_client = _ensure_model(client, model)
    generation_config = {
        "temperature": temperature,
        "top_p": top_p,
        "max_output_tokens": max_output_tokens,
    }

    try:
        response = model_client.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=False,
        )
    except Exception as exc:  # pragma: no cover - covered through tests
        raise GeminiIntegrationError("Gemini service failed to generate a response") from exc

    text = _extract_text(response)
    return GeminiResponse(text=text, prompt=prompt, context_snippets=list(context_snippets))


def stream_response(
    query: str,
    layers: Optional[Sequence[str]],
    manager: MemoryManager,
    *,
    temperature: float = 0.7,
    top_p: float = 0.8,
    max_output_tokens: int = 1024,
    safety_settings: Optional[Dict[str, Any]] = None,
    model: str = "gemini-pro",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    client: Optional[Any] = None,
) -> GeminiStreamResponse:
    """Generate a streaming response from Gemini."""

    context_snippets = _gather_context(manager, query, layers)
    prompt = _build_prompt(query, context_snippets, system_prompt=system_prompt)
    model_client = _ensure_model(client, model)
    generation_config = {
        "temperature": temperature,
        "top_p": top_p,
        "max_output_tokens": max_output_tokens,
    }

    try:
        stream = model_client.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=True,
        )
    except Exception as exc:  # pragma: no cover - covered through tests
        raise GeminiIntegrationError("Gemini service failed to start streaming response") from exc

    return GeminiStreamResponse(
        prompt=prompt,
        context_snippets=list(context_snippets),
        stream=stream,
    )


def iter_text_from_stream(stream: Iterable[Any]) -> Iterable[str]:
    """Yield textual chunks from a Gemini streaming response."""

    for chunk in stream:
        if chunk is None:
            continue
        text = getattr(chunk, "text", None)
        if text:
            yield text
            continue
        parts = getattr(chunk, "parts", None)
        if not parts:
            continue
        for part in parts:
            part_text = getattr(part, "text", None) if hasattr(part, "text") else None
            if part_text:
                yield part_text
            elif isinstance(part, str):
                yield part


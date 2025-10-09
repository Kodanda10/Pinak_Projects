"""Command-line interface for Pinak integrations."""
from __future__ import annotations

from typing import Optional, Sequence

import click

from .integrations import gemini as gemini_integration
from .memory.manager import MemoryManager


def _parse_layers(layers_option: Optional[str]) -> Optional[Sequence[str]]:
    if not layers_option:
        return None
    return [layer.strip() for layer in layers_option.split(",") if layer.strip()]


@click.group()
def main() -> None:
    """Pinak command line utilities."""


@main.group()
def gemini() -> None:
    """Commands for interacting with Google Gemini."""


@gemini.command()
@click.argument("query", nargs=-1, required=True)
@click.option(
    "--layers",
    "layers_option",
    default="",
    help="Comma separated list of memory layers to search (default: all layers).",
)
@click.option(
    "--memory-service-url",
    default="http://localhost:8001",
    show_default=True,
    help="Base URL for the Pinak memory service.",
)
@click.option(
    "--api-key",
    envvar="GOOGLE_API_KEY",
    help="Google Gemini API key (can also be provided via GOOGLE_API_KEY).",
)
@click.option("--model", default="gemini-pro", show_default=True, help="Gemini model to use.")
@click.option("--temperature", default=0.7, show_default=True, type=float)
@click.option("--top-p", default=0.8, show_default=True, type=float)
@click.option("--max-output-tokens", default=1024, show_default=True, type=int)
@click.option(
    "--no-stream",
    is_flag=True,
    help="Disable streaming output and wait for the full response.",
)
def chat(
    query: Sequence[str],
    layers_option: str,
    memory_service_url: str,
    api_key: Optional[str],
    model: str,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    no_stream: bool,
) -> None:
    """Chat with Gemini using context from the Pinak memory service."""

    if not api_key:
        raise click.UsageError("An API key is required. Provide via --api-key or the GOOGLE_API_KEY environment variable.")

    text_query = " ".join(query).strip()
    if not text_query:
        raise click.UsageError("Please provide a non-empty query.")

    try:
        import google.generativeai as genai  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional dependency
        raise click.ClickException(
            "google-generativeai is required for this command. Install it via 'pip install pinak[gemini]'"
        ) from exc

    genai.configure(api_key=api_key)
    manager = MemoryManager(service_base_url=memory_service_url)
    layers = _parse_layers(layers_option)

    try:
        if no_stream:
            response = gemini_integration.generate_response(
                text_query,
                layers,
                manager,
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_output_tokens,
                model=model,
            )
            click.echo(response.text)
        else:
            stream_response = gemini_integration.stream_response(
                text_query,
                layers,
                manager,
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_output_tokens,
                model=model,
            )
            for chunk in gemini_integration.iter_text_from_stream(stream_response.stream):
                click.echo(chunk, nl=False)
            click.echo()
    except gemini_integration.GeminiIntegrationError as exc:
        raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
    main()
